//
// Created by berry on 2023/2/12.
//

#ifndef VKLEARN_RENDERER_HPP
#define VKLEARN_RENDERER_HPP
#include <chrono>
#include <thread>
#include <cmath>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

/* About memory alignment rules in Vulkan:
 * For things like vertex buffer, we specify the offset of each variable when creating a pipeline, thus no additional
 * padding rules are needed.
 *
 * For other buffers and images, we have to make sure the alignments are right. Those rules can be found at
 * https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout
 *
 * A tl;dr explanation can be found at
 * https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements
 "
  Vulkan expects the data in your structure to be aligned in memory in a specific way, for example:
    Scalars have to be aligned by N (= 4 bytes given 32bit floats).
    A vec2 must be aligned by 2N (= 8 bytes)
    A vec3 or vec4 must be aligned by 4N (= 16 bytes)
    A nested structure must be aligned by the base alignment of its members rounded up to a multiple of 16.
    A mat4 matrix must have the same alignment as a vec4.
"
 * */
struct Vertex {
    glm::vec3 pos;
    glm::vec4 color;
};

/* Screen coordinate system for vulkan (Zd$\in$[0,1]):
 * Red, Green and Blue correspond to approximate locations of the first three vertices
 *                [-1]
 *                 |
 *                Red
 *                 |
 * [-1]------------0------------[1]>x
 *                 |
 *      Blue       |    Green
 *                 |
 *                [1]
 *                 v
 *                 y
 *
 * For 3D coordinates, we follow the right hand rules.
 * */
const std::vector<Vertex> vertices = {
        {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
        {{0.0f,0.0f,0.5f}, {0.5f,0.5f,0.5f,0.5f}}
};
/*Face culling convention: In OpenGL, the default values are:
 * glCullFace == GL_BACK; glFrontFace == GL_CCW
 * So here we follow the same convention and adjust parameters in VkPipelineRasterizationStateCreateInfo
 * accordingly.
 * */
const std::vector<uint16_t> vertexIdx = {
        3,1,0,
        3,2,1,
        3,0,2,
        0,1,2
};
/*Coordinate system differences between Vulkan and OpenGL (https://vincent-p.github.io/posts/vulkan_perspective_matrix/)
 *
 * Here comes our convention w.r.t. view and projection matrices (suitable for people not working at the CG industry):
 *
 * 1. The `lookAt(eye, center, up)` function can be thought of as two transformations:
 *   1) Coordinate rotation transformation:
 *      {{CX.x, CX.y, CX.z, 0},
 *       {CY.x, CY.y, CY.z, 0},
 *       {CZ.x, CZ.y, CZ.z, 0},
 *       { 0.0,  0.0,  0.0, 1}}
 *       Where CX means the direction of the camera's X-axis, CY for Y-axis and CZ for Z-axis.
 *       In Vulkan, CZ = normalize(center-eye); CX = normalize(cross(CZ, up)); CY = cross(CZ, CX);
 *   2) Translation, the offset vector is -eye
 *   Thus, the final view matrix is:
 *
 *      {{CX.x, CX.y, CX.z, -dot(CX,eye)},
 *       {CY.x, CY.y, CY.z, -dot(CY,eye)},
 *       {CZ.x, CZ.y, CZ.z, -dot(CZ,eye)},
 *       { 0.0,  0.0,  0.0,            1}}
 *
 * 2. The projection transformation (in general) works as follows:
 *   1) Project X and Y component of the vertex into the near plane as X_p, Y_p
 *   2) Scale X_p, Y_p into Normalized Device Coordinate(NDC) as X_final, Y_final
 *   3) Convert Z component of the vertex into NDC as Zd, during which you can use tricks like inverse Zd etc.
 *  As a result, in default Vulkan setup with the right hand coordinate, without the reverse depth trick,
 *  X_final = X/(Z*tan(vertPOV/2.0f)*aspect); Y_final = Y/(Z*tan(vertPOV/2.0f)); Zd = (Z-near)/(far-near);
 *  However, the expression above cannot be achieved by linear transformation. In order to use matrix multiplication, we
 *  need another form of Zd. Thus far, the matrix can be written as:
 *
 *      {1/(tan(vertPOV/2.0f)*aspect),                 0.0, 0.0, 0.0}
 *      {                         0.0, 1/tan(vertPOV/2.0f), 0.0, 0.0}
 *      {                         0.0,                 0.0,   A,   B}
 *      {                         0.0,                 0.0, 1.0, 0.0}
 *
 *      Zd must be the form of A + B / Z. We also want: A+B/near = 1.0; A+B/far = 0.0; (Reverse Zd is used here,
 *      swap `near` and `far` if not needed)
 *      Thus: A = near/(near-far); B = near*far/(far-near)
 * */
// Push constants seems to be slower than UBO
struct ScenePushConstants {
    glm::mat4 view;
    glm::mat4 proj;
};
struct ModelUBO {
    glm::mat4 model;
};

namespace render{
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    uint32_t queueFamilyIndexGT;

    vk::UniqueDescriptorSetLayout descriptorSetLayoutU{};
    vk::UniqueDescriptorPool descriptorPoolU{};
    std::vector<vk::UniqueDescriptorSet> descriptorSetsU{};
    vk::UniquePipelineLayout graphPipeLayoutU{};
    vk::UniquePipeline graphPipelineU{};

    vk::UniqueBuffer vertexBufferU{};
    vk::UniqueDeviceMemory vertexBufferMemoryU{};
    vk::UniqueBuffer vertexIdxBufferU{};
    vk::UniqueDeviceMemory vertexIdxBufferMemoryU{};
    // Frame dependent objects
    std::vector<vk::UniqueCommandBuffer> commandBuffersU{};
    std::vector<vk::UniqueBuffer> uniformBuffersU{};
    std::vector<vk::UniqueDeviceMemory> uniformBuffersMemoryU{};
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphoresU{};
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphoresU{};
    std::vector<vk::UniqueFence> inFlightFencesU{};

    std::vector<void*> uniformBuffersMapped{};
    std::vector<ScenePushConstants> sceneVPs{};
    std::vector<vk::CommandBuffer> commandBuffers{};
    std::vector<vk::Semaphore> imageAvailableSemaphores{};
    std::vector<vk::Semaphore> renderFinishedSemaphores{};
    std::vector<vk::Fence> inFlightFences{};
}

vk::UniqueDescriptorSetLayout createDescriptorSetLayout(vk::Device &device) {

    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    auto [result, descriptorSetLayout] = device.createDescriptorSetLayoutUnique(layoutInfo);
    utils::vkEnsure(result);
    return std::move(descriptorSetLayout);
}

vk::UniqueDescriptorPool createDescriptorPool(vk::Device &device, const uint32_t MAX_FRAMES_IN_FLIGHT) {
    vk::DescriptorPoolSize poolSize{};
    poolSize.type = vk::DescriptorType::eUniformBuffer;//VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;

    auto [result, descriptorPool] = device.createDescriptorPoolUnique(poolInfo);
    utils::vkEnsure(result);
    return std::move(descriptorPool);
}

std::tuple<vk::UniquePipelineLayout, vk::UniquePipeline>
createGraphicsPipeline(vk::Device &device, vk::Extent2D &viewportExtent, vk::RenderPass &renderPass) {
    auto vertShaderCode = utils::readFile("shaders/shader.vert.spv");
    auto fragShaderCode = utils::readFile("shaders/shader.frag.spv");

    auto vertShaderModule = utils::createShaderModule(vertShaderCode, device);
    auto fragShaderModule = utils::createShaderModule(fragShaderCode, device);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
            {
                    vk::PipelineShaderStageCreateFlags(),
                    vk::ShaderStageFlagBits::eVertex,
                    *vertShaderModule,
                    "main"
            },
            {
                    vk::PipelineShaderStageCreateFlags(),
                    vk::ShaderStageFlagBits::eFragment,
                    *fragShaderModule,
                    "main"
            }
    };

    vk::PipelineVertexInputStateCreateInfo triangleVertexInputInfo = {};
    auto triangleVertexBindDesc = vk::VertexInputBindingDescription{
        0,//.binding
        sizeof(Vertex),//.stride
        vk::VertexInputRate::eVertex//.inputRate
    };
    auto triangleVertexAttrDescs = std::array<vk::VertexInputAttributeDescription,2>{};
    triangleVertexAttrDescs[0] = {
            0,//.location
            0,//.binding
            vk::Format::eR32G32B32Sfloat,//.format
            (uint32_t)offsetof(Vertex, pos)//.offset
    };
    triangleVertexAttrDescs[1] = {
            1,
            0,
            vk::Format::eR32G32B32A32Sfloat,
            (uint32_t)offsetof(Vertex, color)
    };
    triangleVertexInputInfo.vertexBindingDescriptionCount = 1;
    triangleVertexInputInfo.pVertexBindingDescriptions = &triangleVertexBindDesc;
    triangleVertexInputInfo.vertexAttributeDescriptionCount = triangleVertexAttrDescs.size();
    triangleVertexInputInfo.pVertexAttributeDescriptions = triangleVertexAttrDescs.data();
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    vk::Viewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)viewportExtent.width;
    viewport.height = (float)viewportExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vk::Rect2D scissor = {
            {0,0},//offset
            viewportExtent//extent
    };

    vk::PipelineViewportStateCreateInfo viewportState = {};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    vk::PipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::PipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    vk::PushConstantRange sceneVPConstants = {};
    sceneVPConstants.offset = 0;
    sceneVPConstants.size = sizeof(ScenePushConstants);
    sceneVPConstants.stageFlags = vk::ShaderStageFlagBits::eVertex;

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &render::descriptorSetLayoutU.get();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &sceneVPConstants;

    auto [pLResult, pipelineLayout] = device.createPipelineLayoutUnique(pipelineLayoutInfo);
    utils::vkEnsure(pLResult);

    vk::GraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &triangleVertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout.get();
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;

    auto [graphPipeResult, graphicsPipeline] = device.createGraphicsPipelineUnique(nullptr, pipelineInfo);
    utils::vkEnsure(graphPipeResult);
    return std::make_tuple(std::move(pipelineLayout),std::move(graphicsPipeline));
}
uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::PhysicalDevice physicalDevice) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    std::abort();
}

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory> createBuffernMemory(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice){

    auto bufferInfo = vk::BufferCreateInfo{};
    bufferInfo.flags = vk::BufferCreateFlags{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &queueFamilyIdx;
    auto [resultBuffer, buffer] = device.createBufferUnique(bufferInfo);
    utils::vkEnsure(resultBuffer);
    // This has a `2` variant
    auto memReqs = device.getBufferMemoryRequirements(buffer.get());
    auto memAllocInfo = vk::MemoryAllocateInfo{};
    memAllocInfo.allocationSize = memReqs.size;
    auto memPropFlags = properties;
    memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memPropFlags, physicalDevice);
    auto [resultMem, bufferMemory] = device.allocateMemoryUnique(memAllocInfo);
    utils::vkEnsure(resultMem);
    auto resultBind = device.bindBufferMemory(buffer.get(),bufferMemory.get(),0);
    utils::vkEnsure(resultBind);
    return std::make_tuple(std::move(buffer), std::move(bufferMemory));
}

void copyBuffer(vk::Buffer &srcBuffer, vk::Buffer &dstBuffer, const vk::DeviceSize &size,
                vk::CommandPool &commandPool, vk::Device &device, vk::Queue &graphicsQueue){

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    auto [resultBuffer, commandBuffersU] = device.allocateCommandBuffersUnique(allocInfo);
    utils::vkEnsure(resultBuffer);

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    auto resultBegin = commandBuffersU[0]->begin(beginInfo);
    utils::vkEnsure(resultBegin);

    std::vector<vk::BufferCopy> copyRegions;
    vk::BufferCopy copyRegion;
    copyRegion.srcOffset = {};
    copyRegion.dstOffset = {};
    copyRegion.size = size;
    copyRegions.push_back(copyRegion);

    // This has a ver. 2 variant
    commandBuffersU[0]->copyBuffer(srcBuffer, dstBuffer, copyRegions.size(), copyRegions.data());

    auto resultEnd = commandBuffersU[0]->end();
    utils::vkEnsure(resultEnd);

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = commandBuffersU.size();
    auto commandBuffers = uniqueToRaw(commandBuffersU);
    submitInfo.pCommandBuffers = commandBuffers.data();

    auto resultSubmit = graphicsQueue.submit(1, &submitInfo, nullptr);
    utils::vkEnsure(resultSubmit);
    auto resultWait = graphicsQueue.waitIdle();
    utils::vkEnsure(resultWait);

}

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory> createTriangleVertexInputBuffer(
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice,
        vk::CommandPool &commandPool, vk::Queue &graphicsQueue){

    using BufUsage = vk::BufferUsageFlagBits;
    using MemProp = vk::MemoryPropertyFlagBits;
    //TODO: use VMA instead

    auto verticesSize = sizeof(Vertex)*vertices.size();

    auto [stageBuffer, stageBufferMemory] = createBuffernMemory(
            verticesSize,
            BufUsage::eTransferSrc,
            MemProp::eHostVisible | MemProp::eHostCoherent,
            queueFamilyIdx,
            device,
            physicalDevice);
    {
        auto [resultMap, data] = device.mapMemory(stageBufferMemory.get(), 0, verticesSize);
        utils::vkEnsure(resultMap);
        std::memcpy(data, vertices.data(), verticesSize);
        device.unmapMemory(stageBufferMemory.get());
    }
    auto [buffer, bufferMemory] = createBuffernMemory(
            verticesSize,
            BufUsage::eTransferDst | BufUsage::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            queueFamilyIdx,
            device,
            physicalDevice);
    copyBuffer(stageBuffer.get(), buffer.get(), verticesSize, commandPool, device, graphicsQueue);
    return std::make_tuple(std::move(buffer), std::move(bufferMemory));
}

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory> createTriangleVertexIdxBuffer(
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice,
        vk::CommandPool &commandPool, vk::Queue &graphicsQueue){
    using BufUsage = vk::BufferUsageFlagBits;
    using MemProp = vk::MemoryPropertyFlagBits;
    //TODO: use VMA instead

    auto vertIdxSize = sizeof(decltype(vertexIdx)::value_type) * vertexIdx.size();

    auto [stageBuffer, stageBufferMemory] = createBuffernMemory(
            vertIdxSize,
            BufUsage::eTransferSrc,
            MemProp::eHostVisible | MemProp::eHostCoherent,
            queueFamilyIdx,
            device,
            physicalDevice);
    {
        auto [resultMap, data] = device.mapMemory(stageBufferMemory.get(), 0, vertIdxSize);
        utils::vkEnsure(resultMap);
        std::memcpy(data, vertexIdx.data(), vertIdxSize);
        device.unmapMemory(stageBufferMemory.get());
    }
    auto [buffer, bufferMemory] = createBuffernMemory(
            vertIdxSize,
            BufUsage::eTransferDst | BufUsage::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            queueFamilyIdx,
            device,
            physicalDevice);
    copyBuffer(stageBuffer.get(), buffer.get(), vertIdxSize, commandPool, device, graphicsQueue);
    return std::make_tuple(std::move(buffer), std::move(bufferMemory));
}

std::tuple<
    vk::UniqueBuffer,
    vk::UniqueDeviceMemory,
    void*> createModelUniformBuffer(vk::Device &device, const vk::PhysicalDevice &physicalDevice) {

    vk::DeviceSize bufferSize = sizeof(ModelUBO);

    auto [uniformBuffer, uniformBufferMemory] = createBuffernMemory(
            bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
            render::queueFamilyIndexGT, device, physicalDevice);

    auto [result, uniformBufferMapped] = device.mapMemory(
            uniformBufferMemory.get(), vk::DeviceSize{0}, bufferSize, vk::MemoryMapFlags{});
    utils::vkEnsure(result);
    return std::make_tuple(std::move(uniformBuffer), std::move(uniformBufferMemory), uniformBufferMapped);
}

void createModelUniformBuffers(vk::Device &device, const vk::PhysicalDevice &physicalDevice){
    using namespace render;
    uniformBuffersU.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemoryU.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i){
        std::tie(uniformBuffersU[i], uniformBuffersMemoryU[i], uniformBuffersMapped[i]) = createModelUniformBuffer(device, physicalDevice);
    }
}

void recordCommandBuffer(const vk::Framebuffer &framebuffer, const vk::RenderPass &renderPass,
                              const vk::Extent2D &renderExtent, const vk::Pipeline &graphicsPipeline,
                              vk::CommandBuffer &commandBuffer, uint32_t currentFrame) {
    vk::CommandBufferBeginInfo beginInfo = {};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

    auto beginResult = commandBuffer.begin(beginInfo);
    utils::vkEnsure(beginResult);

    vk::RenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffer;
    renderPassInfo.renderArea = vk::Offset2D{ 0, 0 };
    renderPassInfo.renderArea.extent = renderExtent;

    vk::ClearValue clearColor = { std::array<float, 4>{ 1.0f, 1.0f, 1.0f, 1.0f } };
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    //TODO: set viewport and scissor (are these already set when creating pipelines?)

    // This has a non-trivial ver. `2`.
    // Omit some code since here we only have one vertex buffer
    auto vertBufOffset = vk::DeviceSize{0};
    commandBuffer.bindVertexBuffers(0, 1, &render::vertexBufferU.get(), &vertBufOffset);
    commandBuffer.bindIndexBuffer(render::vertexIdxBufferU.get(), 0, vk::IndexTypeValue<decltype(vertexIdx)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, render::graphPipeLayoutU.get(),
                                     0, 1, &render::descriptorSetsU[currentFrame].get(), 0, nullptr);
    commandBuffer.pushConstants(render::graphPipeLayoutU.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(ScenePushConstants), &render::sceneVPs[currentFrame]);

    commandBuffer.drawIndexed(vertexIdx.size(), 1, 0, 0, 0);

    commandBuffer.endRenderPass();

    auto endResult = commandBuffer.end();
    utils::vkEnsure(endResult);
}
void createDescriptorSets(vk::Device &device) {
    std::vector<vk::DescriptorSetLayout> layouts(render::MAX_FRAMES_IN_FLIGHT, render::descriptorSetLayoutU.get());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = render::descriptorPoolU.get();
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    auto [result, descriptorSets] = device.allocateDescriptorSetsUnique(allocInfo);
    utils::vkEnsure(result);
    render::descriptorSetsU = std::move(descriptorSets);

    for (size_t i = 0; i < render::MAX_FRAMES_IN_FLIGHT; i++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = render::uniformBuffersU[i].get();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(ModelUBO);

        vk::WriteDescriptorSet descriptorWrite{};
        descriptorWrite.dstSet = render::descriptorSetsU[i].get();
        descriptorWrite.dstBinding = 0;
        descriptorWrite.dstArrayElement = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pBufferInfo = &bufferInfo;

        device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }
}
std::vector<vk::UniqueCommandBuffer> createCommandBuffers(
        vk::CommandPool &commandPool, vk::Device &device,
        vk::RenderPass &renderPass, vk::Extent2D &renderExtent, vk::Pipeline &graphicsPipeline) {

    auto buffersSize = render::MAX_FRAMES_IN_FLIGHT;

    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)buffersSize;

    auto [buffersResult, commandBuffers] = device.allocateCommandBuffersUnique(allocInfo);
    utils::vkEnsure(buffersResult);

    return std::move(commandBuffers);
}

void createSyncObjects(vk::Device &device) {
    using namespace render;
    imageAvailableSemaphoresU.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphoresU.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFencesU.resize(MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        auto [availResult, availSemaphore] = device.createSemaphoreUnique({});
        utils::vkEnsure(availResult);
        imageAvailableSemaphoresU[i] = std::move(availSemaphore);
        auto [finishedResult, finishedSemaphore] = device.createSemaphoreUnique({});
        utils::vkEnsure(finishedResult);
        renderFinishedSemaphoresU[i] = std::move(finishedSemaphore);
        auto [fenceResult, fence] = device.createFenceUnique({vk::FenceCreateFlagBits::eSignaled});
        inFlightFencesU[i] = std::move(fence);
    }
}

void setupRender(
        const vk::PhysicalDevice &physicalDevice,
        vk::Device &device,
        vk::Extent2D &viewportExtent,
        const uint32_t &queueIdx,
        vk::RenderPass &renderPass,
        vk::CommandPool &commandPool,
        vk::Queue &graphicsQueue){
    render::queueFamilyIndexGT = queueIdx;
    render::descriptorSetLayoutU = createDescriptorSetLayout(device);
    render::descriptorPoolU = createDescriptorPool(device, render::MAX_FRAMES_IN_FLIGHT);
    std::tie(render::graphPipeLayoutU, render::graphPipelineU) = createGraphicsPipeline(device, viewportExtent, renderPass);
    std::tie(render::vertexBufferU, render::vertexBufferMemoryU) = createTriangleVertexInputBuffer(queueIdx, device, physicalDevice, commandPool, graphicsQueue);
    std::tie(render::vertexIdxBufferU, render::vertexIdxBufferMemoryU) = createTriangleVertexIdxBuffer(queueIdx, device, physicalDevice, commandPool, graphicsQueue);
    createModelUniformBuffers(device, physicalDevice);
    render::commandBuffersU = createCommandBuffers(commandPool, device, renderPass, viewportExtent, render::graphPipelineU.get());
    {
        render::sceneVPs.resize(render::MAX_FRAMES_IN_FLIGHT);
    }
    createDescriptorSets(device);
    createSyncObjects(device);
    {
        using namespace render;
        commandBuffers = uniqueToRaw(commandBuffersU);
        imageAvailableSemaphores = uniqueToRaw(imageAvailableSemaphoresU);
        renderFinishedSemaphores = uniqueToRaw(renderFinishedSemaphoresU);
        inFlightFences = uniqueToRaw(inFlightFencesU);
    }
}
void updateModelUBO(const uint32_t currentFrame, const float runTime) {

    ModelUBO ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), runTime * glm::radians(37.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    std::memcpy(render::uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}
void updateFrameData(const uint32_t currentFrame, const vk::Extent2D swapChainExtent){
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float runTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    updateModelUBO(currentFrame, runTime);
    // Update view and projection
    auto sceneVP = ScenePushConstants{};

    sceneVP.view = utils::vkuLookAtRH(glm::vec3(0.65f, 0.65f, 0.65f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    sceneVP.proj = utils::vkuPerspectiveRHReverse_ZO(glm::radians(60.0f),
                                                     swapChainExtent.width / (float) swapChainExtent.height, 0.1f,
                                                     10.0f);
    render::sceneVPs[currentFrame] = sceneVP;
}

decltype(auto) cleanupRenderSync(){
    using namespace render;
    return std::make_tuple(
            std::move(imageAvailableSemaphoresU),
            std::move(renderFinishedSemaphoresU),
            std::move(inFlightFencesU));
}
void cleanupRender4FB(){
    using namespace render;
    {
        auto gPL = std::move(graphPipeLayoutU);
        auto gP = std::move(graphPipelineU);
        auto cBs = std::move(commandBuffersU);
        auto dSL = std::move(descriptorSetLayoutU);
        auto dP = std::move(descriptorPoolU);
        auto dSs = std::move(descriptorSetsU);
        auto uBs = std::move(uniformBuffersU);
        auto uBMs = std::move(uniformBuffersMemoryU);
    }
}
void cleanupRender(){
    cleanupRender4FB();
    auto [iASs, rFSs, iFFs] = cleanupRenderSync();

    auto vertBuff = std::move(render::vertexBufferU);
    auto vertBuffMem = std::move(render::vertexBufferMemoryU);
    auto vertIdxBuff = std::move(render::vertexIdxBufferU);
    auto vertIdxBuffMem = std::move(render::vertexIdxBufferMemoryU);
}

#endif //VKLEARN_RENDERER_HPP
