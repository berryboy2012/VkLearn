//
// Created by berry on 2023/2/28.
//

#ifndef VKLEARN_RT_RENDERER_HPP
#define VKLEARN_RT_RENDERER_HPP
#include "model_data.hpp"

namespace rt_render
{
    struct PNCTVertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec2 texCoord;
    };
    struct BlasInput
    {
        // Data used to build acceleration structure geometry
        std::vector<vk::AccelerationStructureGeometryKHR>       asGeometry;
        std::vector<vk::AccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
        vk::BuildAccelerationStructureFlagsKHR                  flags{0};
    };

    struct NaiveBuffernMemory{
        vk::UniqueBuffer               buffer;
        vk::UniqueDeviceMemory         mem;
    };

    struct AccelKHR
    {
        vk::UniqueAccelerationStructureKHR accel = {};
        NaiveBuffernMemory buffer;
    };

    struct BuildAccelerationStructure
    {
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
        vk::AccelerationStructureBuildSizesInfoKHR sizeInfo{};
        const vk::AccelerationStructureBuildRangeInfoKHR* rangeInfo;
        AccelKHR                                  as;  // result acceleration structure
        AccelKHR                                  cleanupAS;
    };

    std::tuple<
        std::vector<PNCTVertex>,
        std::vector<uint16_t>> generateNormalVectoredModel(
                const std::vector<model_info::PCTVertex> &verts, const std::vector<uint16_t> &vertIdxs){
        auto resVerts = std::vector<PNCTVertex>{};
        auto resIdxs = std::vector<uint16_t>{};
        auto nTri = vertIdxs.size()/3;
        resVerts.reserve(nTri*3);
        resIdxs.reserve(nTri*3);
        for (size_t idTri = 0; idTri < nTri; ++idTri){
            std::array<PNCTVertex, 3> destTri{};
            for (size_t idVert = 0; idVert< 3; ++idVert){
                destTri[idVert].pos = verts[vertIdxs[idTri*3+idVert]].pos;
                destTri[idVert].color = verts[vertIdxs[idTri*3+idVert]].color;
                destTri[idVert].texCoord = verts[vertIdxs[idTri*3+idVert]].texCoord;
            }
            auto normalVec = glm::normalize(glm::cross(destTri[1].pos-destTri[0].pos, destTri[2].pos-destTri[1].pos));
            for (size_t idVert = 0; idVert< 3; ++idVert){
                destTri[idVert].normal = normalVec;
                resVerts.push_back(destTri[idVert]);
                resIdxs.push_back(idVert+idTri*3);
            }
        }
        return std::make_tuple(resVerts, resIdxs);
    }

    std::vector<PNCTVertex> transformModel(const glm::mat4 &transform,
            const std::vector<PNCTVertex> &verts){
        auto resVerts = std::vector<PNCTVertex>{};
        resVerts.reserve(verts.size());
        auto normTrans = glm::mat3(glm::transpose(glm::inverse(transform)));
        for (auto& vert: verts){
            resVerts.push_back({
               .pos = glm::vec3{transform*glm::vec4{vert.pos, 1.0}},
               .normal = normTrans*vert.normal,
               .color = vert.color,
               .texCoord = vert.texCoord
            });
        }
        return resVerts;
    }

    const auto [verticesA, vertexIdx] = generateNormalVectoredModel(model_info::vertices, model_info::vertexIdx);

    const auto verticesB = transformModel(glm::translate(glm::scale(glm::identity<glm::mat4>(), {0.5f, 0.5f, 0.5f}), {3.0f, 3.0f, 0.0f}), verticesA);

    struct Model{
        std::vector<PNCTVertex> verts;
        std::vector<uint16_t> vertIdxs;
    };

    const auto hanoiModels = std::array<Model, 2>{
            Model{verticesA, vertexIdx},
            Model{verticesB, vertexIdx}
    };

    AccelKHR createAcceleration(vk::AccelerationStructureCreateInfoKHR& accel_,
                                vk::Device &device, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice)
    {
        AccelKHR resultAccel;
        // Allocating the buffer to hold the acceleration structure
//        resultAccel.buffer = createBuffer(accel_.size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
//                                                       | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        NaiveBuffernMemory scratchBuffer = {};
        std::tie(resultAccel.buffer.buffer, resultAccel.buffer.mem) = createBuffernMemory(
                accel_.size, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice);
        // Setting the buffer
        accel_.buffer = resultAccel.buffer.buffer.get();
        // Create the acceleration structure
        //vkCreateAccelerationStructureKHR(m_device, &accel_, nullptr, &resultAccel.accel);
        {
            auto [result, blas] = device.createAccelerationStructureKHRUnique(accel_);
            utils::vkEnsure(result);
            resultAccel.accel = std::move(blas);
        }

        return std::move(resultAccel);
    }

    //--------------------------------------------------------------------------------------------------
// Creating the bottom level acceleration structure for all indices of `buildAs` vector.
// The array of BuildAccelerationStructure was created in buildBlas and the vector of
// indices limits the number of BLAS to create at once. This limits the amount of
// memory needed when compacting the BLAS.
    void cmdCreateBlas(vk::CommandBuffer                        &cmdBuf,
                       std::vector<uint32_t>                    indices,
                       std::vector<BuildAccelerationStructure>& buildAs,
                       vk::DeviceAddress                          scratchAddress,
                       vk::QueryPool                              queryPool,
                       vk::Device &device, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice)
    {
        if(queryPool)  // For querying the compaction size
            //vkResetQueryPool(device, queryPool, 0, static_cast<uint32_t>(indices.size()));
            device.resetQueryPool(queryPool, 0, static_cast<uint32_t>(indices.size()));
        uint32_t queryCnt{0};

        for(const auto& idx : indices)
        {
            // Actual allocation of buffer and acceleration structure.
            vk::AccelerationStructureCreateInfoKHR createInfo{};
            createInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            createInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;  // Will be used to allocate memory.
            buildAs[idx].as = createAcceleration(createInfo, device, queueFamilyIdx, physicalDevice);
            //NAME_IDX_VK(buildAs[idx].as.accel, idx);
            //NAME_IDX_VK(buildAs[idx].as.buffer.buffer, idx);

            // BuildInfo #2 part
            buildAs[idx].buildInfo.dstAccelerationStructure  = buildAs[idx].as.accel.get();  // Setting where the build lands
            buildAs[idx].buildInfo.scratchData.deviceAddress = scratchAddress;  // All build are using the same scratch buffer

            // Building the bottom-level-acceleration-structure
            cmdBuf.buildAccelerationStructuresKHR(buildAs[idx].buildInfo, buildAs[idx].rangeInfo);

            // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
            // is finished before starting the next one.
            vk::MemoryBarrier barrier{};
            barrier.srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR;
            barrier.dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR;
//            vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
//                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, barrier, {}, {});
            if(queryPool)
            {
                // Add a query to find the 'real' amount of memory needed, use for compaction
//                vkCmdWriteAccelerationStructuresPropertiesKHR(cmdBuf, 1, &buildAs[idx].buildInfo.dstAccelerationStructure,
//                                                              VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, queryCnt++);
                cmdBuf.writeAccelerationStructuresPropertiesKHR(buildAs[idx].buildInfo.dstAccelerationStructure, vk::QueryType::eAccelerationStructureCompactedSizeKHR, queryPool, queryCnt++);
            }
        }
    }
    //--------------------------------------------------------------------------------------------------
// Create and replace a new acceleration structure and buffer based on the size retrieved by the
// Query.
    void cmdCompactBlas(vk::CommandBuffer                          cmdBuf,
                        std::vector<uint32_t>                    indices,
                        std::vector<BuildAccelerationStructure>& buildAs,
                        vk::QueryPool                              queryPool,
                        vk::Device &device, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice)
    {
        uint32_t queryCtn{0};

        // Get the compacted size result back
        std::vector<vk::DeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
//        vkGetQueryPoolResults(m_device, queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize),
//                              compactSizes.data(), sizeof(VkDeviceSize), VK_QUERY_RESULT_WAIT_BIT);
        vk::Result result{};
        std::tie(result, compactSizes) = device.getQueryPoolResults<decltype(compactSizes)::value_type>(queryPool, 0, (uint32_t)compactSizes.size(), compactSizes.size() * sizeof(VkDeviceSize), sizeof(vk::DeviceSize), vk::QueryResultFlagBits::eWait);

        for(auto idx : indices)
        {
            buildAs[idx].cleanupAS                          = std::move(buildAs[idx].as);           // previous AS to destroy
            buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++];  // new reduced size

            // Creating a compact version of the AS
            vk::AccelerationStructureCreateInfoKHR asCreateInfo{};
            asCreateInfo.size = buildAs[idx].sizeInfo.accelerationStructureSize;
            asCreateInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
            buildAs[idx].as   = createAcceleration(asCreateInfo, device, queueFamilyIdx, physicalDevice);

            // Copy the original BLAS to a compact version
            vk::CopyAccelerationStructureInfoKHR copyInfo{};
            copyInfo.src  = buildAs[idx].buildInfo.dstAccelerationStructure;
            copyInfo.dst  = buildAs[idx].as.accel.get();
            copyInfo.mode = vk::CopyAccelerationStructureModeKHR::eCompact;
            cmdBuf.copyAccelerationStructureKHR(copyInfo);
        }
    }

//--------------------------------------------------------------------------------------------------
// Create all the BLAS from the vector of BlasInput
// - There will be one BLAS per input-vector entry
// - There will be as many BLAS as input.size()
// - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
//   and can be referenced by index.
// - if flag has the 'Compact' flag, the BLAS will be compacted
//
    std::vector<AccelKHR> buildBlas(const std::vector<BlasInput>& input, vk::BuildAccelerationStructureFlagsKHR flags,
                   vk::CommandPool &cmdPool, vk::Device &device, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice, vk::Queue &queue)
    {
        std::vector<AccelKHR> m_blas{};
        //cmdPool.init(m_device, m_queueIndex);
        auto         nbBlas = static_cast<uint32_t>(input.size());
        vk::DeviceSize asTotalSize{0};     // Memory size of all allocated BLAS
        uint32_t     nbCompactions{0};   // Nb of BLAS requesting compaction
        vk::DeviceSize maxScratchSize{0};  // Largest scratch size

        // Preparing the information for the acceleration build commands.
        std::vector<BuildAccelerationStructure> buildAs(nbBlas);
        for(uint32_t idx = 0; idx < nbBlas; idx++)
        {
            // Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
            // Other information will be filled in the createBlas (see #2)
            buildAs[idx].buildInfo.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
            buildAs[idx].buildInfo.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
            buildAs[idx].buildInfo.flags         = input[idx].flags | flags;
            buildAs[idx].buildInfo.geometryCount = static_cast<uint32_t>(input[idx].asGeometry.size());
            buildAs[idx].buildInfo.pGeometries   = input[idx].asGeometry.data();

            // Build range information
            buildAs[idx].rangeInfo = input[idx].asBuildOffsetInfo.data();

            // Finding sizes to create acceleration structures and scratch
            std::vector<uint32_t> maxPrimCount(input[idx].asBuildOffsetInfo.size());
            for(auto tt = 0; tt < input[idx].asBuildOffsetInfo.size(); tt++)
                maxPrimCount[tt] = input[idx].asBuildOffsetInfo[tt].primitiveCount;  // Number of primitives/triangles
            device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice,
                                                   &buildAs[idx].buildInfo, maxPrimCount.data(), &buildAs[idx].sizeInfo);

            // Extra info
            asTotalSize += buildAs[idx].sizeInfo.accelerationStructureSize;
            maxScratchSize = std::max(maxScratchSize, buildAs[idx].sizeInfo.buildScratchSize);
            nbCompactions += (buildAs[idx].buildInfo.flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) == vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
        }

        // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
        NaiveBuffernMemory scratchBuffer = {};
        std::tie(scratchBuffer.buffer, scratchBuffer.mem) = createBuffernMemory(
                maxScratchSize, vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
                vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice);
        vk::BufferDeviceAddressInfo bufferInfo{scratchBuffer.buffer.get()};
        vk::DeviceAddress           scratchAddress = device.getBufferAddress(bufferInfo);

        // Allocate a query pool for storing the needed size for every BLAS compaction.
        vk::UniqueQueryPool queryPool{VK_NULL_HANDLE};
        if(nbCompactions > 0)  // Is compaction requested?
        {
            assert(nbCompactions == nbBlas);  // Don't allow mix of on/off compaction
            vk::QueryPoolCreateInfo qpci{};
            qpci.queryCount = nbBlas;
            qpci.queryType  = vk::QueryType::eAccelerationStructureCompactedSizeKHR;
            {
                auto [result, qP] = device.createQueryPoolUnique(qpci, nullptr);
                utils::vkEnsure(result);
                queryPool = std::move(qP);
            }

        }

        // Batching creation/compaction of BLAS to allow staying in restricted amount of memory
        std::vector<uint32_t> indices;  // Indices of the BLAS to create
        vk::DeviceSize          batchSize{0};
        vk::DeviceSize          batchLimit{256'000'000};  // 256 MB
        for(uint32_t idx = 0; idx < nbBlas; idx++)
        {
            indices.push_back(idx);
            batchSize += buildAs[idx].sizeInfo.accelerationStructureSize;
            // Over the limit or last BLAS element
            if(batchSize >= batchLimit || idx == nbBlas - 1)
            {
                //vk::CommandBuffer cmdBuf = cmdPool.createCommandBuffer();
                {
                    utils::SingleTimeCommandBuffer tmpCmdBuf{cmdPool, queue, device};
                    cmdCreateBlas(tmpCmdBuf.coBuf, indices, buildAs, scratchAddress, queryPool.get(), device,
                                  queueFamilyIdx, physicalDevice);
                }
                //m_cmdPool.submitAndWait(cmdBuf);

                if(queryPool)
                {
                    //VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();
                    {
                        utils::SingleTimeCommandBuffer tmpCmdBuf{cmdPool, queue, device};
                        cmdCompactBlas(tmpCmdBuf.coBuf, indices, buildAs, queryPool.get(),
                                       device, queueFamilyIdx, physicalDevice);
                    }
                    //m_cmdPool.submitAndWait(cmdBuf);  // Submit command buffer and call vkQueueWaitIdle

                    // Destroy the non-compacted version
                    for(auto& i : indices)
                    {
                        buildAs[i].cleanupAS = {};
                    }

                }
                // Reset

                batchSize = 0;
                indices.clear();
            }
        }

        // Keeping all the created acceleration structures
        for(auto& b : buildAs)
        {
            m_blas.emplace_back(std::move(b.as));
        }

        // Clean up
//        vkDestroyQueryPool(m_device, queryPool, nullptr);
//        m_alloc->finalizeAndReleaseStaging();
//        m_alloc->destroy(scratchBuffer);
//        cmdPool.deinit();
        return m_blas;
    }
    // The OBJ model
    struct ObjModel
    {
        uint32_t     nbIndices{0};
        uint32_t     nbVertices{0};
        NaiveBuffernMemory vertexBuffer;    // Device buffer of all 'Vertex'
        NaiveBuffernMemory indexBuffer;     // Device buffer of the indices forming triangles
//        NaiveBuffernMemory matColorBuffer;  // Device buffer of array of 'Wavefront material'
//        NaiveBuffernMemory matIndexBuffer;  // Device buffer of array of 'Wavefront material'
    };
    //--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
    auto objectToVkGeometryKHR(const ObjModel& model, vk::Device &device)
    {
        // BLAS builder requires raw device addresses.
        //VkDeviceAddress vertexAddress = getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
        //VkDeviceAddress indexAddress  = getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
        auto vertexAddress = device.getBufferAddress(model.vertexBuffer.buffer.get());
        auto indexAddress  = device.getBufferAddress(model.indexBuffer.buffer.get());

        uint32_t maxPrimitiveCount = model.nbIndices / 3;

        // Describe buffer as array of VertexObj.
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat; // vec3 vertex position data.
        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride             = sizeof(PNCTVertex);
        // Describe index data (32-bit unsigned int)
        triangles.indexType               = vk::IndexType::eUint16;//VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexAddress;
        // Indicate identity transform by setting transformData to null device pointer.
        //triangles.transformData = {};
        triangles.maxVertex = model.nbVertices;

        // Identify the above data as containing opaque triangles.
        vk::AccelerationStructureGeometryKHR asGeom{};
        asGeom.geometryType       = vk::GeometryTypeKHR::eTriangles;
        asGeom.flags              = vk::GeometryFlagBitsKHR::eOpaque;
        asGeom.geometry.triangles = triangles;

        // The entire array will be used to build the BLAS.
        vk::AccelerationStructureBuildRangeInfoKHR offset;
        offset.firstVertex     = 0;
        offset.primitiveCount  = maxPrimitiveCount;
        offset.primitiveOffset = 0;
        offset.transformOffset = 0;

        // Our blas is made from only one geometry, but could be made of many geometries
        BlasInput input;
        input.asGeometry.emplace_back(asGeom);
        input.asBuildOffsetInfo.emplace_back(offset);

        return input;
    }
    std::vector<AccelKHR> createBottomLevelAS(std::vector<ObjModel> &models,
                             vk::Device &device, vk::CommandPool &commandPool, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice, vk::Queue &queue)
    {
        // BLAS - Storing each primitive in a geometry
        std::vector<BlasInput> allBlas;
        // One model per Blas
        allBlas.reserve(models.size());
        for(const auto& model : models)
        {
            auto blas = objectToVkGeometryKHR(model, device);

            // We could add more geometry in each BLAS, but we add only one for now
            allBlas.emplace_back(blas);
        }
        return buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
                  commandPool, device, queueFamilyIdx, physicalDevice, queue);
    }

    std::vector<ObjModel> loadModels(vk::Device &device, vk::CommandPool &commandPool, uint32_t queueFamilyIdx, const vk::PhysicalDevice &physicalDevice, vk::Queue &queue){
        std::vector<ObjModel> modelResources{};
        modelResources.resize(2);
        modelResources[0].nbIndices = vertexIdx.size();
        modelResources[0].nbVertices = verticesA.size();
        {
            auto [buf, mem] = createBuffernMemoryFromHostData(
                    verticesA.size()*sizeof(decltype(verticesA)::value_type), (void*)verticesA.data(),
                    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool, queue);
            modelResources[0].vertexBuffer.buffer = std::move(buf);
            modelResources[0].vertexBuffer.mem = std::move(mem);
        }
        {
            auto [buf, mem] = createBuffernMemoryFromHostData(
                    vertexIdx.size()*sizeof(decltype(vertexIdx)::value_type), (void*)vertexIdx.data(),
                    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool, queue);
            modelResources[0].indexBuffer.buffer = std::move(buf);
            modelResources[0].indexBuffer.mem = std::move(mem);
        }
        modelResources[1].nbIndices = vertexIdx.size();
        modelResources[1].nbVertices = verticesB.size();
        {
            auto [buf, mem] = createBuffernMemoryFromHostData(
                    verticesB.size()*sizeof(decltype(verticesB)::value_type), (void*)verticesB.data(),
                    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool, queue);
            modelResources[1].vertexBuffer.buffer = std::move(buf);
            modelResources[1].vertexBuffer.mem = std::move(mem);
        }
        {
            auto [buf, mem] = createBuffernMemoryFromHostData(
                    vertexIdx.size()*sizeof(decltype(vertexIdx)::value_type), (void*)vertexIdx.data(),
                    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal, queueFamilyIdx, device, physicalDevice, commandPool, queue);
            modelResources[1].indexBuffer.buffer = std::move(buf);
            modelResources[1].indexBuffer.mem = std::move(mem);
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
            vk::Queue &graphicsQueue){

        //initRayTracing();
        auto loadedModels = loadModels(device, commandPool, queueIdx, physicalDevice, graphicsQueue);
        auto accels = createBottomLevelAS(loadedModels, device, commandPool, queueIdx, physicalDevice, graphicsQueue);
//        createTopLevelAS();
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
