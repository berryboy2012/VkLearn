//
// Created by berry on 2023/2/12.
//

#ifndef VKLEARN_RENDERER_HPP
#define VKLEARN_RENDERER_HPP
#include <chrono>
#include <thread>
#include "glm/glm.hpp"

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
};
const std::vector<Vertex> vertices = {
        {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}}
};
namespace render{
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    uint32_t queueFamilyIndexGT;

    vk::UniquePipelineLayout graphPipeLayoutU{};
    vk::UniquePipeline graphPipelineU{};

    vk::UniqueBuffer vertexBufferU{};
    vk::UniqueDeviceMemory vertexBufferMemoryU{};

    std::vector<vk::UniqueCommandBuffer> commandBuffersU{};
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphoresU{};
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphoresU{};
    std::vector<vk::UniqueFence> inFlightFencesU{};

    std::vector<vk::CommandBuffer> commandBuffers{};
    std::vector<vk::Semaphore> imageAvailableSemaphores{};
    std::vector<vk::Semaphore> renderFinishedSemaphores{};
    std::vector<vk::Fence> inFlightFences{};
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
            0//.offset
    };
    triangleVertexAttrDescs[1] = {
            1,
            0,
            vk::Format::eR32G32B32Sfloat,
            sizeof(Vertex::color)
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
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eClockwise;
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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

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

void recordCommandBuffer(const vk::Framebuffer &framebuffer, const vk::RenderPass &renderPass,
                              const vk::Extent2D &renderExtent, const vk::Pipeline &graphicsPipeline,
                              vk::CommandBuffer &commandBuffer) {
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

    //TODO: set viewport and scissor

    // This has a non-trivial ver. `2`.
    // Omit some code since here we only have one vertex buffer
    auto vertBufOffset = vk::DeviceSize{0};
    commandBuffer.bindVertexBuffers(0, 1, &render::vertexBufferU.get(), &vertBufOffset);

    commandBuffer.draw(vertices.size(), 1, 0, 0);

    commandBuffer.endRenderPass();

    auto endResult = commandBuffer.end();
    utils::vkEnsure(endResult);
}

std::vector<vk::UniqueCommandBuffer> createCommandBuffers(
        vk::CommandPool &commandPool, vk::Device &device, std::vector<vk::UniqueFramebuffer> &framebuffers,
        vk::RenderPass &renderPass, vk::Extent2D &renderExtent, vk::Pipeline &graphicsPipeline) {

    auto buffersSize = framebuffers.size();

    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)buffersSize;

    auto [buffersResult, commandBuffers] = device.allocateCommandBuffersUnique(allocInfo);
    utils::vkEnsure(buffersResult);

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        recordCommandBuffer(framebuffers[i].get(), renderPass, renderExtent, graphicsPipeline, commandBuffers[i].get());
    }
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
        std::vector<vk::UniqueFramebuffer> &framebuffers,
        vk::Queue &graphicsQueue){
    render::queueFamilyIndexGT = queueIdx;
    std::tie(render::graphPipeLayoutU, render::graphPipelineU) = createGraphicsPipeline(device, viewportExtent, renderPass);
    std::tie(render::vertexBufferU, render::vertexBufferMemoryU) = createTriangleVertexInputBuffer(queueIdx, device, physicalDevice, commandPool, graphicsQueue);
    render::commandBuffersU = createCommandBuffers(commandPool, device, framebuffers, renderPass, viewportExtent, render::graphPipelineU.get());

    createSyncObjects(device);
    {
        using namespace render;
        commandBuffers = uniqueToRaw(commandBuffersU);
        imageAvailableSemaphores = uniqueToRaw(imageAvailableSemaphoresU);
        renderFinishedSemaphores = uniqueToRaw(renderFinishedSemaphoresU);
        inFlightFences = uniqueToRaw(inFlightFencesU);
    }
}

decltype(auto) cleanupRenderSync(){
    using namespace render;
    return std::make_tuple(
            std::move(imageAvailableSemaphoresU),
            std::move(renderFinishedSemaphoresU),
            std::move(inFlightFencesU));
}
decltype(auto) cleanupRender4FB(){
    using namespace render;
    return std::make_tuple(
            std::move(graphPipeLayoutU),
            std::move(graphPipelineU),
            std::move(commandBuffersU));
}
void cleanupRender(){
    auto [gPL, gP, cB] = cleanupRender4FB();
    auto [iASs, rFSs, iFFs] = cleanupRenderSync();
    auto vertBuffMem = std::move(render::vertexBufferMemoryU);
    auto vertBuff = std::move(render::vertexBufferU);
}

#endif //VKLEARN_RENDERER_HPP
