//
// Created by berry on 2023/2/28.
//

#ifndef VKLEARN_RT_RENDERER_HPP
#define VKLEARN_RT_RENDERER_HPP

#include "model_data.hpp"
#include <ranges>
#include <vector>
#include <span>

namespace rt_render {
    struct PNCTVertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec2 texCoord;
    };
    struct BlasInput {
        // Data used to build acceleration structure geometry
        std::vector<vk::AccelerationStructureGeometryKHR> asGeometry;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
        vk::BuildAccelerationStructureFlagsKHR flags{0};
    };

    struct NaiveBuffernMemory {
        vk::UniqueBuffer buffer;
        vk::UniqueDeviceMemory mem;
    };

    struct AccelStructObj {
        vk::UniqueAccelerationStructureKHR accel = {};
        NaiveBuffernMemory buffer;
    };

    struct AccelStructBuildInfo {
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        vk::AccelerationStructureBuildSizesInfoKHR sizeInfo{};
        const vk::AccelerationStructureBuildRangeInfoKHR *rangeInfo{};
    };

    std::tuple<
            std::vector<PNCTVertex>,
            std::vector<uint16_t>> generateNormalVectoredModel(
            const std::vector<model_info::PCTVertex> &verts, const std::vector<uint16_t> &vertIdxs) {
        auto resVerts = std::vector<PNCTVertex>{};
        auto resIdxs = std::vector<uint16_t>{};
        auto nTri = vertIdxs.size() / 3;
        resVerts.reserve(nTri * 3);
        resIdxs.reserve(nTri * 3);
        for (size_t idTri = 0; idTri < nTri; ++idTri) {
            std::array<PNCTVertex, 3> destTri{};
            for (size_t idVert = 0; idVert < 3; ++idVert) {
                destTri[idVert].pos = verts[vertIdxs[idTri * 3 + idVert]].pos;
                destTri[idVert].color = verts[vertIdxs[idTri * 3 + idVert]].color;
                destTri[idVert].texCoord = verts[vertIdxs[idTri * 3 + idVert]].texCoord;
            }
            auto normalVec = glm::normalize(
                    glm::cross(destTri[1].pos - destTri[0].pos, destTri[2].pos - destTri[1].pos));
            for (size_t idVert = 0; idVert < 3; ++idVert) {
                destTri[idVert].normal = normalVec;
                resVerts.push_back(destTri[idVert]);
                resIdxs.push_back(idVert + idTri * 3);
            }
        }
        return std::make_tuple(resVerts, resIdxs);
    }

    std::vector<PNCTVertex> transformModel(const glm::mat4 &transform,
                                           const std::vector<PNCTVertex> &verts) {
        auto resVerts = std::vector<PNCTVertex>{};
        resVerts.reserve(verts.size());
        auto normTrans = glm::mat3(glm::transpose(glm::inverse(transform)));
        for (auto &vert: verts) {
            resVerts.push_back({
                                       .pos = glm::vec3{transform * glm::vec4{vert.pos, 1.0}},
                                       .normal = normTrans * vert.normal,
                                       .color = vert.color,
                                       .texCoord = vert.texCoord
                               });
        }
        return resVerts;
    }

    const auto [verticesA, vertexIdx] = generateNormalVectoredModel(model_info::vertices, model_info::vertexIdx);

    const auto verticesB = transformModel(
            glm::translate(glm::scale(glm::identity<glm::mat4>(), {0.5f, 0.5f, 0.5f}), {3.0f, 3.0f, 0.0f}), verticesA);

    struct Model {
        std::vector<PNCTVertex> verts;
        std::vector<uint16_t> vertIdxs;
    };

    const auto hanoiModels = std::array<Model, 2>{
            Model{verticesA, vertexIdx},
            Model{verticesB, vertexIdx}
    };

    AccelStructObj createAcceleration(vk::AccelerationStructureCreateInfoKHR &accel_,
                                      vk::Device &device, uint32_t queueFamilyIdx,
                                      const vk::PhysicalDevice &physicalDevice) {
        AccelStructObj resultAccel;
        // Allocating the buffer to hold the acceleration structure
        NaiveBuffernMemory scratchBuffer = {};
        std::tie(resultAccel.buffer.buffer, resultAccel.buffer.mem) = createBuffernMemory(
                accel_.size, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice);
        // Setting the buffer
        accel_.buffer = resultAccel.buffer.buffer.get();
        // Create the acceleration structure
        {
            auto [result, accelS] = device.createAccelerationStructureKHRUnique(accel_);
            utils::vkEnsure(result);
            resultAccel.accel = std::move(accelS);
        }

        return std::move(resultAccel);
    }

    //--------------------------------------------------------------------------------------------------
// Creating the bottom level acceleration structure for all indices of `buildAs` vector.
// The array of BuildAccelerationStructure was created in buildBlas and the vector of
// indices limits the number of BLAS to create at once. This limits the amount of
// memory needed when compacting the BLAS.
    void cmdCreateBlasBatch(vk::CommandBuffer &cmdBuf,
                            std::span<AccelStructBuildInfo> buildAs,
                            std::span<AccelStructObj> blases,
                            vk::DeviceAddress scratchAddress,
                            vk::QueryPool queryPool,
                            vk::Device &device, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice) {
        if (queryPool)  // For querying the compaction size
            device.resetQueryPool(queryPool, 0, static_cast<uint32_t>(buildAs.size()));
        uint32_t queryCnt{0};

        for (size_t idblas = 0; idblas < buildAs.size(); idblas++) {
            auto &blasInfo = buildAs[idblas];
            // Actual allocation of buffer and acceleration structure.
            vk::AccelerationStructureCreateInfoKHR createInfo{};
            createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            createInfo.size = blasInfo.sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.
            blases[idblas] = createAcceleration(createInfo, device, queueFamilyIdx, physicalDevice);

            // BuildInfo #2 part
            blasInfo.buildInfo.dstAccelerationStructure = blases[idblas].accel.get();  // Setting where the build lands
            blasInfo.buildInfo.scratchData.deviceAddress = scratchAddress;  // All build are using the same scratch buffer

            // Building the bottom-level-acceleration-structure
            cmdBuf.buildAccelerationStructuresKHR(blasInfo.buildInfo, blasInfo.rangeInfo);

            // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
            // is finished before starting the next one.
            vk::MemoryBarrier barrier{};
            barrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
            barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR;
            cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier, {}, {});
            if (queryPool) {
                // Add a query to find the 'real' amount of memory needed, use for compaction
                cmdBuf.writeAccelerationStructuresPropertiesKHR(blasInfo.buildInfo.dstAccelerationStructure,
                                                                vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                                                queryPool, queryCnt++);
            }
        }
    }

//--------------------------------------------------------------------------------------------------
// Create and replace a new acceleration structure and buffer based on the size retrieved by the
// Query. Due to synchronization constraints, the old acceleration structure will be returned.
    std::vector<AccelStructObj> cmdCompactBlasBatch(vk::CommandBuffer cmdBuf,
                                                    std::span<AccelStructBuildInfo> buildAs,
                                                    std::span<AccelStructObj> blases,
                                                    vk::QueryPool queryPool,
                                                    vk::Device &device, uint32_t queueFamilyIdx,
                                                    const vk::PhysicalDevice &physicalDevice) {
        uint32_t queryCtn{0};
        std::vector<AccelStructObj> oldBlases{};
        // Get the compacted size result back
        auto [result, compactSizes] = device.getQueryPoolResults<vk::DeviceSize>(
                queryPool, 0, buildAs.size(),
                sizeof(vk::DeviceSize) * buildAs.size(), sizeof(vk::DeviceSize));
        utils::vkEnsure(result);

        for (size_t idx = 0; idx < buildAs.size(); idx++) {
            //buildAs[idx].cleanupAS                          = std::move(buildAs[idx].as);// previous AS to destroy
            oldBlases.push_back(std::move(blases[idx]));
            buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++];  // new reduced size

            // Creating a compact version of the AS
            vk::AccelerationStructureCreateInfoKHR asCreateInfo{};
            asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
            asCreateInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            blases[idx] = createAcceleration(asCreateInfo, device, queueFamilyIdx, physicalDevice);

            // Copy the original BLAS to a compact version
            vk::CopyAccelerationStructureInfoKHR copyInfo{};
            copyInfo.src = buildAs[idx].buildInfo.dstAccelerationStructure;
            copyInfo.dst = blases[idx].accel.get();
            copyInfo.mode = vk::CopyAccelerationStructureModeKHR::eCompact;
            cmdBuf.copyAccelerationStructureKHR(copyInfo);
        }
        return std::move(oldBlases);
    }

//--------------------------------------------------------------------------------------------------
// Create all the BLAS from the vector of BlasInput
// - There will be one BLAS per input-vector entry
// - There will be as many BLAS as input.size()
// - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
//   and can be referenced by index.
// - if flag has the 'Compact' flag, the BLAS will be compacted
//
    std::vector<AccelStructObj>
    buildBlas(const std::span<const BlasInput> input, vk::BuildAccelerationStructureFlagsKHR flags,
              vk::CommandPool &cmdPool, vk::Device &device, uint32_t queueFamilyIdx,
              const vk::PhysicalDevice &physicalDevice, vk::Queue &queue) {
        //AS uses a large amount of memory, thus it needs a buffer and underlying memory.
        // Since we don't use VMA for now, we wrap AS and its buffer and dedicated memory into AccelStructObj
        std::vector<AccelStructObj> m_blas{};
        auto nbBlas = static_cast<uint32_t>(input.size());
        m_blas.resize(nbBlas);
        vk::DeviceSize asTotalSize{0};     // Memory size of all allocated BLAS
        bool doCompaction = false;
        {
            if ((flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) ==
                vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) {
                doCompaction = true;
            } else {
                for (auto &blasInput: input) {
                    bool inputDoCompaction =
                            (blasInput.flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) ==
                            vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
                    if (inputDoCompaction) {
                        doCompaction = true;
                    } else if (doCompaction) {
                        std::abort();// Don't allow mix of on/off compaction
                    }
                }
            }
        }
        vk::DeviceSize maxScratchSize{0};  // Largest scratch size

        // Preparing info for the acceleration build commands.
        std::vector<AccelStructBuildInfo> buildAs(nbBlas);
        // First skim over the BLAS nodes' info to determine the size of scratch buffer for BLAS build commands
        for (uint32_t idx = 0; idx < nbBlas; idx++) {
            auto numGeom = input[idx].asGeometry.size();
            // Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
            // Other information will be filled in the createBlas (see #2)
            buildAs[idx].buildInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            buildAs[idx].buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
            buildAs[idx].buildInfo.flags = input[idx].flags | flags;

            buildAs[idx].buildInfo.geometryCount = static_cast<uint32_t>(input[idx].asGeometry.size());
            buildAs[idx].buildInfo.pGeometries = input[idx].asGeometry.data();
            buildAs[idx].rangeInfo = input[idx].asBuildOffsetInfo.data();

            // Finding sizes to create acceleration structures and scratch
            std::vector<uint32_t> maxPrimCount(numGeom);
            for (auto tt = 0; tt < numGeom; tt++)
                maxPrimCount[tt] = input[idx].asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
            buildAs[idx].sizeInfo = device.getAccelerationStructureBuildSizesKHR(
                    vk::AccelerationStructureBuildTypeKHR::eDevice, buildAs[idx].buildInfo, maxPrimCount);

            // Extra info
            asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
            maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
        }

        // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
        NaiveBuffernMemory scratchBuffer = {};
        std::tie(scratchBuffer.buffer, scratchBuffer.mem) = createBuffernMemory(
                maxScratchSize,
                vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer |
                vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice);
        vk::BufferDeviceAddressInfo bufferInfo{scratchBuffer.buffer.get()};
        vk::DeviceAddress scratchAddress = device.getBufferAddress(bufferInfo);

        // Allocate a query pool for storing the needed size for every BLAS compaction.
        vk::UniqueQueryPool queryPool{};
        if (doCompaction) {
            vk::QueryPoolCreateInfo qpci{};
            qpci.queryCount = nbBlas;
            qpci.queryType = vk::QueryType::eAccelerationStructureCompactedSizeKHR;
            vk::Result result;
            std::tie(result, queryPool) = device.createQueryPoolUnique(qpci).asTuple();
            utils::vkEnsure(result);
        }

        // Batching creation/compaction of BLAS to allow staying in restricted amount of memory
        std::vector<uint32_t> indices;  // Indices of the BLAS to create
        vk::DeviceSize batchLimit{256'000'000};  // A soft-limit of 256MB
        int blasChunkBegin = 0;
        do {
            auto blasChunkEnd = blasChunkBegin;
            vk::DeviceSize batchSize{0};
            std::span<decltype(buildAs)::value_type> remainBLASes(buildAs.begin() + blasChunkBegin, buildAs.end());
            for (auto &aBlas: remainBLASes) {
                batchSize += aBlas.sizeInfo.accelerationStructureSize;
                blasChunkEnd += 1;
                if (batchSize >= batchLimit) {
                    break;
                }
            }
            std::span<decltype(buildAs)::value_type> blasInfoChunk(buildAs.begin() + blasChunkBegin,
                                                                   buildAs.begin() + blasChunkEnd);
            std::span<decltype(m_blas)::value_type> blasChunk(m_blas.begin() + blasChunkBegin,
                                                              m_blas.begin() + blasChunkEnd);
            {
                utils::SingleTimeCommandBuffer tmpCmdBuf{cmdPool, queue, device};
                cmdCreateBlasBatch(tmpCmdBuf.coBuf, blasInfoChunk, blasChunk, scratchAddress,
                                   queryPool.get(), device, queueFamilyIdx, physicalDevice);
            }
            if (doCompaction) {
                std::vector<AccelStructObj> oldAs{};
                utils::SingleTimeCommandBuffer tmpCmdBuf{cmdPool, queue, device};
                oldAs = cmdCompactBlasBatch(tmpCmdBuf.coBuf, blasInfoChunk, blasChunk,
                                            queryPool.get(), device, queueFamilyIdx, physicalDevice);
            }
            blasChunkBegin = blasChunkEnd;
        } while (blasChunkBegin < buildAs.size());

        return m_blas;
    }

    struct ObjModel {
        uint32_t nbIndices{0};
        uint32_t nbVertices{0};
        NaiveBuffernMemory vertexBuffer;    // Device buffer of all 'Vertex'
        NaiveBuffernMemory indexBuffer;     // Device buffer of the indices forming triangles
        size_t vertexStride{sizeof(PNCTVertex)};
        size_t vertexSize{sizeof(PNCTVertex::pos)};
        size_t indicesStride{sizeof(uint16_t)};
        size_t vertexOffset{offsetof(PNCTVertex, pos)};
    };

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
    auto getVkGeometryKHR(const ObjModel &model, vk::Device &device) {
        // BLAS builder requires raw device addresses.
        auto vertexAddress = device.getBufferAddress(model.vertexBuffer.buffer.get());
        auto indexAddress = device.getBufferAddress(model.indexBuffer.buffer.get());

        uint32_t maxPrimitiveCount = model.nbIndices / 3;

        // Describe buffer as array of VertexObj.
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
        if (model.vertexSize == sizeof(glm::vec3)) {
            triangles.vertexFormat = vk::Format::eR32G32B32Sfloat; // vec3 vertex position data.
        } else if (model.vertexSize == sizeof(glm::vec4)) {
            triangles.vertexFormat = vk::Format::eR32G32B32A32Sfloat;
        } else {
            std::abort();
        }

        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride = model.vertexStride;
        // Describe index data
        if (model.indicesStride == sizeof(uint16_t)) {
            triangles.indexType = vk::IndexType::eUint16;
        } else if (model.indicesStride == sizeof(uint32_t)) {
            triangles.indexType = vk::IndexType::eUint32;
        } else {
            std::abort();
        }
        triangles.indexData.deviceAddress = indexAddress;
        // Indicate identity transform by setting transformData to null device pointer.
        //triangles.transformData = {};
        triangles.maxVertex = model.nbVertices;

        // Identify the above data as containing opaque triangles.
        vk::AccelerationStructureGeometryKHR asGeom{};
        asGeom.geometryType = vk::GeometryTypeKHR::eTriangles;
        asGeom.flags = vk::GeometryFlagBitsKHR::eOpaque;
        asGeom.geometry.triangles = triangles;

        // The entire array will be used to build the BLAS.
        vk::AccelerationStructureBuildRangeInfoKHR offset;
        offset.firstVertex = 0;
        offset.primitiveCount = maxPrimitiveCount;
        offset.primitiveOffset = model.vertexOffset;
        offset.transformOffset = 0;

        // Our blas is made from only one geometry, but could be made of many geometries
        BlasInput input;
        input.asGeometry.emplace_back(asGeom);
        input.asBuildOffsetInfo.emplace_back(offset);

        return input;
    }

    std::vector<AccelStructObj> createBottomLevelAS(const std::span<const ObjModel> models,
                                                    vk::Device &device, vk::CommandPool &commandPool,
                                                    uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice,
                                                    vk::Queue &queue) {
        // Each BlasInput corresponds to the geometry info of a BLAS's node, each node can register multiple geometries.
        std::vector<BlasInput> allBlas;
        // However, BLAS cannot be altered once created, thus only one model per BLAS node is a sensible choice.
        allBlas.reserve(models.size());
        for (const auto &model: models) {
            allBlas.emplace_back(getVkGeometryKHR(model, device));
        }
        return buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                                  vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction,
                         commandPool, device, queueFamilyIdx, physicalDevice, queue);
    }

    struct ObjInstance {
        glm::mat4x3 transform;    // Matrix of the instance
        uint32_t objIndex{0};  // Model index reference
    };

    inline vk::TransformMatrixKHR toTransformMatrixKHR(glm::mat4x3 matrix) {
        // VkTransformMatrixKHR uses a row-major memory layout, while glm
        // uses a column-major memory layout. We transpose the matrix so we can
        // memcpy the matrix's data directly.
        glm::mat4 temp = glm::transpose(matrix);
        vk::TransformMatrixKHR out_matrix;
        std::memcpy(&out_matrix.matrix, &temp, sizeof(vk::TransformMatrixKHR::matrix));
        return out_matrix;
    }

//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
    AccelStructObj cmdCreateTlas(vk::CommandBuffer &cmdBuf,
                                 AccelStructBuildInfo &tlasBuilder,
                                 vk::DeviceAddress scratchAddress,
                                 vk::Device &device, uint32_t queueFamilyIdx,
                                 const vk::PhysicalDevice &physicalDevice) {
        //Generated TLAS
        AccelStructObj m_tlas{};
        // Acquire resources for TLAS
        bool update = false;
        if (update == false) {
            vk::AccelerationStructureCreateInfoKHR createInfo{};
            createInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
            createInfo.size = tlasBuilder.sizeInfo.accelerationStructureSize;
            m_tlas = createAcceleration(createInfo, device, queueFamilyIdx, physicalDevice);
        }
        tlasBuilder.buildInfo.dstAccelerationStructure = m_tlas.accel.get();
        tlasBuilder.buildInfo.scratchData.deviceAddress = scratchAddress;

        // Build the TLAS
        cmdBuf.buildAccelerationStructuresKHR(tlasBuilder.buildInfo, tlasBuilder.rangeInfo);

        vk::MemoryBarrier barrier{};
        barrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
        barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR;
        cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                               vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier, {}, {});
        return std::move(m_tlas);
    }

    // Build TLAS from an array of VkAccelerationStructureInstanceKHR
    // - The resulting TLAS will be returned
    AccelStructObj buildTlas(const std::span<const vk::AccelerationStructureInstanceKHR> instances,
                             vk::BuildAccelerationStructureFlagsKHR flags,
                             vk::CommandPool &cmdPool, vk::Device &device, uint32_t queueFamilyIdx,
                             const vk::PhysicalDevice &physicalDevice, vk::Queue &queue) {
        //bool update = false;

        AccelStructObj tlas{};
        AccelStructBuildInfo tlasBuilder{};

        auto countInstance = static_cast<uint32_t>(instances.size());
        vk::DeviceSize instsSize = countInstance * sizeof(std::decay<decltype(instances)>::type::value_type);
        // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
        NaiveBuffernMemory instancesBuffer{};
        std::tie(instancesBuffer.buffer, instancesBuffer.mem) = createBuffernMemoryFromHostData(
                instsSize, (void *) instances.data(),
                vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, cmdPool, queue);

        vk::BufferDeviceAddressInfo bufferInfo{};
        bufferInfo.buffer = instancesBuffer.buffer.get();
        vk::DeviceAddress instBufferAddr = device.getBufferAddress(bufferInfo);
        // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
        vk::AccelerationStructureGeometryKHR topASGeometry{};
        topASGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
        {
            // Vulkan-Hpp acts a bit funky when using designated initialization. We are not taking risks here.
            vk::AccelerationStructureGeometryInstancesDataKHR instancesVk{};
            instancesVk.data.deviceAddress = instBufferAddr;
            topASGeometry.geometry.instances = instancesVk;
        }

        // Find sizes
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.flags = flags;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &topASGeometry;
        //buildInfo.mode = update ? vk::BuildAccelerationStructureModeKHR::eUpdate : vk::BuildAccelerationStructureModeKHR::eBuild;
        buildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
        buildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
        //buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
        tlasBuilder.buildInfo = buildInfo;
        tlasBuilder.sizeInfo = device.getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuilder.buildInfo, countInstance);
        // Build Offsets info: n instances
        vk::AccelerationStructureBuildRangeInfoKHR buildOffsetInfo{countInstance, 0, 0, 0};
        tlasBuilder.rangeInfo = &buildOffsetInfo;

        // Allocate the scratch memory
        NaiveBuffernMemory scratchBuffer{};
        std::tie(scratchBuffer.buffer, scratchBuffer.mem) = createBuffernMemory(
                tlasBuilder.sizeInfo.buildScratchSize,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                queueFamilyIdx, device, physicalDevice);

        bufferInfo.buffer = scratchBuffer.buffer.get();
        vk::DeviceAddress scratchAddress = device.getBufferAddress(bufferInfo);
        // Update build information
        //tlasBuilder.buildInfo.srcAccelerationStructure  = update ? tlas.accel.get() : VK_NULL_HANDLE;
        tlasBuilder.buildInfo.srcAccelerationStructure = nullptr;

        // Command buffer to create the TLAS
        {
            vk::MemoryBarrier barrier{};
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
            utils::SingleTimeCommandBuffer singleCBuf{cmdPool, queue, device};
            // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
            singleCBuf.coBuf.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                    {}, barrier, {}, {});
            // Creating the TLAS
            tlas = cmdCreateTlas(singleCBuf.coBuf, tlasBuilder, scratchAddress, device, queueFamilyIdx, physicalDevice);
        }
        return std::move(tlas);
    }

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
    vk::DeviceAddress
    getBlasDeviceAddress(uint32_t blasId, const std::span<const AccelStructObj> blas, vk::Device &device) {
        assert(size_t(blasId) < 2);
        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.accelerationStructure = blas[blasId].accel.get();
        return device.getAccelerationStructureAddressKHR(addressInfo);
    }

    AccelStructObj
    createTopLevelAS(const std::span<const ObjInstance> m_instances, const std::span<const AccelStructObj> blas,
                     vk::Device &device, vk::CommandPool &commandPool, uint32_t queueFamilyIdx,
                     const vk::PhysicalDevice &physicalDevice, vk::Queue &queue) {
        // For tlas, much less info are needed, thus no need for structs like TlasInput
        std::vector<vk::AccelerationStructureInstanceKHR> tlas;
        tlas.reserve(m_instances.size());
        for (const ObjInstance &inst: m_instances) {
            vk::AccelerationStructureInstanceKHR rayInst{};
            rayInst.transform = toTransformMatrixKHR(inst.transform);  // Position of the instance
            rayInst.instanceCustomIndex = inst.objIndex;                               // gl_InstanceCustomIndexEXT
            rayInst.accelerationStructureReference = getBlasDeviceAddress(inst.objIndex, blas, device);
            // https://github.com/KhronosGroup/Vulkan-Hpp/issues/1002 wont-fix
            rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            rayInst.mask = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
            rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
            tlas.emplace_back(rayInst);
        }
        return buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace, commandPool, device,
                         queueFamilyIdx, physicalDevice, queue);
    }
    std::vector<ObjModel> loadnPrepModels4RT(vk::Device &device, vk::CommandPool &commandPool, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice, vk::Queue &queue){
        std::vector<ObjModel> modelResources{};
        modelResources.resize(2);
        {
            modelResources[0].nbIndices = vertexIdx.size();
            modelResources[0].nbVertices = verticesA.size();
            {
                auto [buf, mem] = createBuffernMemoryFromHostData(
                        verticesA.size() * sizeof(decltype(verticesA)::value_type), (void *) verticesA.data(),
                        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                        vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool,
                        queue);
                modelResources[0].vertexBuffer.buffer = std::move(buf);
                modelResources[0].vertexBuffer.mem = std::move(mem);
            }
            {
                auto [buf, mem] = createBuffernMemoryFromHostData(
                        vertexIdx.size() * sizeof(decltype(vertexIdx)::value_type), (void *) vertexIdx.data(),
                        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                        vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool,
                        queue);
                modelResources[0].indexBuffer.buffer = std::move(buf);
                modelResources[0].indexBuffer.mem = std::move(mem);
            }
        }
        {
            modelResources[1].nbIndices = vertexIdx.size();
            modelResources[1].nbVertices = verticesB.size();
            {
                auto [buf, mem] = createBuffernMemoryFromHostData(
                        verticesB.size() * sizeof(decltype(verticesB)::value_type), (void *) verticesB.data(),
                        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                        vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool,
                        queue);
                modelResources[1].vertexBuffer.buffer = std::move(buf);
                modelResources[1].vertexBuffer.mem = std::move(mem);
            }
            {
                auto [buf, mem] = createBuffernMemoryFromHostData(
                        vertexIdx.size() * sizeof(decltype(vertexIdx)::value_type), (void *) vertexIdx.data(),
                        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                        vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool,
                        queue);
                modelResources[1].indexBuffer.buffer = std::move(buf);
                modelResources[1].indexBuffer.mem = std::move(mem);
            }
        }
        return std::move(modelResources);
    }

    void setupRTRender(
            const vk::PhysicalDevice &physicalDevice,
            vk::Device &device,
            vk::Extent2D &viewportExtent,
            const uint32_t &queueIdx,
            vk::RenderPass &renderPass,
            vk::CommandPool &commandPool,
            vk::Queue &CGTQueue) {
        //initRayTracing();
        auto loadedModels = loadnPrepModels4RT(device, commandPool, queueIdx, physicalDevice, graphicsQueue);
        std::array<ObjInstance, 3> instances = {{
                                                        {glm::identity<glm::mat4x3>(), 0},
                                                        {glm::translate(glm::scale(glm::identity<glm::mat4>(),
                                                                                   {1.2f, 1.2f, 1.0f}),
                                                                        {-3.0f, -3.0f, -0.05f}), 0},
                                                        {glm::identity<glm::mat4x3>(), 1}}
        };
        // Each RT pipeline works on one Acceleration Structure.
        // AS has one TLAS and multiple BLASes.
        // BLAS can have multiple geometries.
        auto blass = createBottomLevelAS(loadedModels, device, commandPool, queueIdx, physicalDevice, CGTQueue);
        // TLAS can have multiple instances. Each instance is mapped to a single BLAS.
        auto tlas = createTopLevelAS(instances, blass, device, commandPool, queueIdx, physicalDevice, CGTQueue);
//        createRtDescriptorSet();
//        createRtPipeline();
//        createRtShaderBindingTable();
//
//        createPostDescriptor();
//        createPostPipeline();
//        updatePostDescriptorSet();
    }
}


#endif //VKLEARN_RT_RENDERER_HPP
