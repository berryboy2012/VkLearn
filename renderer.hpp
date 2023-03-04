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
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "model_data.hpp"
#include "shader_modules.hpp"

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

namespace render{
    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    uint32_t queueFamilyIndexGT;
    vk::Device renderDevice;
    vk::PhysicalDevice renderPhysicalDevice;
    vk::CommandPool renderCommandPool;
    vk::Queue renderQueue;

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

vk::UniqueDescriptorSetLayout createDescriptorSetLayout(std::span<const vk::DescriptorSetLayoutBinding> bindings, vk::Device &device){
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    auto [result, descriptorSetLayout] = device.createDescriptorSetLayoutUnique(layoutInfo);
    utils::vkEnsure(result);
    return std::move(descriptorSetLayout);
}

// TODO: let each resource self register layout
vk::UniqueDescriptorPool createDescriptorPool(vk::Device &device, const uint32_t MAX_FRAMES_IN_FLIGHT) {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;

    auto [result, descriptorPool] = device.createDescriptorPoolUnique(poolInfo);
    utils::vkEnsure(result);
    return std::move(descriptorPool);
}

std::tuple<vk::UniqueDescriptorSetLayout, vk::UniquePipelineLayout, vk::UniquePipeline>
createGraphicsPipeline(vk::Device &device, vk::Extent2D &viewportExtent, vk::RenderPass &renderPass) {
    auto testShaderModule = utils::createShaderModule("shaders/testAttr.vert.spv", device);
    auto vertShader = VertexShader{device}; // shaders/shader.vert
    auto fragShader = FragShader{device}; // shaders/shader.frag

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {{
        {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eVertex,
                vertShader.shaderModule_.get(),
                "main"
        },
        {
                vk::PipelineShaderStageCreateFlags(),
                vk::ShaderStageFlagBits::eFragment,
                fragShader.shaderModule_.get(),
                "main"
        }
    }};

    vk::PipelineVertexInputStateCreateInfo triangleVertexInputInfo = {};

    triangleVertexInputInfo.vertexBindingDescriptionCount = vertShader.inputInfos_.size();
    triangleVertexInputInfo.pVertexBindingDescriptions = vertShader.inputInfos_.data();
    triangleVertexInputInfo.vertexAttributeDescriptionCount = vertShader.attrInfos_.size();
    triangleVertexInputInfo.pVertexAttributeDescriptions = vertShader.attrInfos_.data();

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
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = VK_FALSE;

    vk::PipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = vk::CompareOp::eGreater;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants = {{ 0.0f,0.0f,0.0f,0.0f }};

    vk::UniqueDescriptorSetLayout descLayout{};
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings{};
        for (auto &bind: vertShader.descLayouts_) {
            bindings.push_back(bind);
        }
        for (auto &bind: fragShader.descLayouts_) {
            bindings.push_back(bind);
        }
        descLayout = createDescriptorSetLayout(bindings, device);
    }

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descLayout.get();
    pipelineLayoutInfo.pushConstantRangeCount = vertShader.pushConstInfos_.size();
    pipelineLayoutInfo.pPushConstantRanges = vertShader.pushConstInfos_.data();

    auto [pLResult, pipelineLayout] = device.createPipelineLayoutUnique(pipelineLayoutInfo);
    utils::vkEnsure(pLResult);

    vk::GraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.stageCount = shaderStages.size();
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &triangleVertexInputInfo;
    pipelineInfo.pInputAssemblyState = &vertShader.inputAsmInfo_;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout.get();
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = nullptr;

    auto [graphPipeResult, graphicsPipeline] = device.createGraphicsPipelineUnique(nullptr, pipelineInfo);
    utils::vkEnsure(graphPipeResult);
    return std::make_tuple(std::move(descLayout), std::move(pipelineLayout),std::move(graphicsPipeline));
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
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-vkBindBufferMemory-bufferDeviceAddress-03339
    vk::MemoryAllocateFlagsInfo memAllocFlagInfo{};
    memAllocInfo.pNext = &memAllocFlagInfo;
    if ((usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) == vk::BufferUsageFlagBits::eShaderDeviceAddress){
        memAllocFlagInfo.flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }
    auto [resultMem, bufferMemory] = device.allocateMemoryUnique(memAllocInfo);
    utils::vkEnsure(resultMem);
    auto resultBind = device.bindBufferMemory(buffer.get(),bufferMemory.get(),0);
    utils::vkEnsure(resultBind);
    return std::make_tuple(std::move(buffer), std::move(bufferMemory));
}

std::tuple<vk::UniqueImage, vk::UniqueDeviceMemory> createImagenMemory(
        const vk::Extent3D &extent,
        const vk::Format &format,
        const vk::ImageTiling &tiling,
        const vk::ImageLayout &layout,
        const vk::ImageUsageFlags &usage,
        const vk::MemoryPropertyFlags &properties,
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice){

    auto imageInfo = vk::ImageCreateInfo{};
    imageInfo.flags = vk::ImageCreateFlags{};
    assert(extent.depth == 1);
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent = extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = layout;
    imageInfo.usage = usage;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.queueFamilyIndexCount = 1;
    imageInfo.pQueueFamilyIndices = &queueFamilyIdx;
    auto [resultImage, image] = device.createImageUnique(imageInfo);
    utils::vkEnsure(resultImage);
    // This has a `2` variant
    auto memReqs = device.getImageMemoryRequirements(image.get());
    auto memAllocInfo = vk::MemoryAllocateInfo{};
    memAllocInfo.allocationSize = memReqs.size;
    auto memPropFlags = properties;
    memAllocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memPropFlags, physicalDevice);
    auto [resultMem, imageMemory] = device.allocateMemoryUnique(memAllocInfo);
    utils::vkEnsure(resultMem);
    auto resultBind = device.bindImageMemory(image.get(), imageMemory.get(), 0);
    utils::vkEnsure(resultBind);
    return std::make_tuple(std::move(image), std::move(imageMemory));
}

void copyBuffer(vk::Buffer &srcBuffer, vk::Buffer &dstBuffer, const vk::DeviceSize &size,
                vk::CommandPool &commandPool, vk::Device &device, vk::Queue &graphicsQueue){

    std::vector<vk::BufferCopy> copyRegions;
    vk::BufferCopy copyRegion;
    copyRegion.srcOffset = {};
    copyRegion.dstOffset = {};
    copyRegion.size = size;
    copyRegions.push_back(copyRegion);
    {
        utils::SingleTimeCommandBuffer singleTime{commandPool, graphicsQueue, device};
        // This has a ver. 2 variant
        singleTime.coBuf.copyBuffer(srcBuffer, dstBuffer, copyRegions.size(), copyRegions.data());
    }
}

void copyImageFromBuffer(vk::Buffer &srcBuffer, vk::Image &dstImage, const vk::Extent3D &extent,
                         vk::CommandPool &commandPool, vk::Device &device, vk::Queue &graphicsQueue){
    std::vector<vk::BufferImageCopy> copyRegions;
    vk::BufferImageCopy copyRegion;
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = vk::Offset3D{0, 0, 0};
    copyRegion.imageExtent = extent;
    copyRegions.push_back(copyRegion);
    {
        utils::SingleTimeCommandBuffer singleTime{commandPool, graphicsQueue, device};
        // This has a ver. 2 variant
        singleTime.coBuf.copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegions);
    }
}

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory> createBuffernMemoryFromHostData(
        vk::DeviceSize size,
        void* hostData,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice,
        vk::CommandPool &commandPool,
        vk::Queue &graphicsQueue) {
    using BufUsage = vk::BufferUsageFlagBits;
    using MemProp = vk::MemoryPropertyFlagBits;
    auto [stageBuffer, stageBufferMemory] = createBuffernMemory(
            size,
            BufUsage::eTransferSrc,
            MemProp::eHostVisible | MemProp::eHostCoherent,
            queueFamilyIdx,
            device,
            physicalDevice);
    {
        auto [resultMap, data] = device.mapMemory(stageBufferMemory.get(), 0, size);
        utils::vkEnsure(resultMap);
        std::memcpy(data, hostData, size);
        device.unmapMemory(stageBufferMemory.get());
    }
    auto [buffer, bufferMemory] = createBuffernMemory(
            size,
            BufUsage::eTransferDst | usage,
            properties,
            queueFamilyIdx,
            device,
            physicalDevice);
    copyBuffer(stageBuffer.get(), buffer.get(), size, commandPool, device, graphicsQueue);
    return std::make_tuple(std::move(buffer), std::move(bufferMemory));
}

void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                           vk::Device &device,
                           vk::CommandPool &commandPool,
                           vk::Queue &graphicsQueue) {

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
        std::abort();
    }

    {
        utils::SingleTimeCommandBuffer singleTime{commandPool, graphicsQueue, device};
        singleTime.coBuf.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);
    }

}

std::tuple<vk::UniqueImage, vk::UniqueDeviceMemory> createImagenMemoryFromHostData(
        vk::Extent3D extent,
        vk::DeviceSize size,
        void* hostData,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageLayout layout,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        uint32_t queueFamilyIdx,
        vk::Device &device,
        vk::PhysicalDevice physicalDevice,
        vk::CommandPool &commandPool,
        vk::Queue &graphicsQueue) {
    using BufUsage = vk::BufferUsageFlagBits;
    using ImgUsage = vk::ImageUsageFlagBits;
    using MemProp = vk::MemoryPropertyFlagBits;
    auto [stageBuffer, stageBufferMemory] = createBuffernMemory(
            size,
            BufUsage::eTransferSrc,
            MemProp::eHostVisible | MemProp::eHostCoherent,
            queueFamilyIdx,
            device,
            physicalDevice);
    {
        auto [resultMap, data] = device.mapMemory(stageBufferMemory.get(), 0, size);
        utils::vkEnsure(resultMap);
        std::memcpy(data, hostData, size);
        device.unmapMemory(stageBufferMemory.get());
    }
    auto [image, imageMemory] = createImagenMemory(
            extent, format, tiling, layout,
            ImgUsage::eTransferDst | usage,
            properties,
            queueFamilyIdx,
            device,
            physicalDevice);
    transitionImageLayout(image.get(), format,
                          vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                          device, commandPool, graphicsQueue);
    copyImageFromBuffer(stageBuffer.get(), image.get(), extent, commandPool, device, graphicsQueue);
    transitionImageLayout(image.get(), format,
                          vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                          device, commandPool, graphicsQueue);
    return std::make_tuple(std::move(image), std::move(imageMemory));
}

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory> createTriangleVertexInputBuffer(
        const uint32_t &queueFamilyIdx,
        vk::Device &device,
        const vk::PhysicalDevice &physicalDevice,
        vk::CommandPool &commandPool, vk::Queue &graphicsQueue){

    using BufUsage = vk::BufferUsageFlagBits;
    using MemProp = vk::MemoryPropertyFlagBits;
    //TODO: use VMA instead

    auto verticesSize = sizeof(model_info::PCTVertex) * model_info::vertices.size();
    auto [buffer, bufferMemory] = createBuffernMemoryFromHostData(
            verticesSize, (void*)model_info::vertices.data(),
            BufUsage::eVertexBuffer, MemProp::eDeviceLocal,
            queueFamilyIdx, device, physicalDevice, commandPool, graphicsQueue);

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

    auto vertIdxSize = sizeof(decltype(model_info::vertexIdx)::value_type) * model_info::vertexIdx.size();
    auto [buffer, bufferMemory] = createBuffernMemoryFromHostData(
            vertIdxSize, (void*)model_info::vertexIdx.data(),
            BufUsage::eIndexBuffer, MemProp::eDeviceLocal,
            queueFamilyIdx, device, physicalDevice, commandPool, graphicsQueue);

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

    std::array<vk::ClearValue, 2> clearValues{};
    clearValues[0].color = {std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}};
    clearValues[1].depthStencil.depth = 0.0f;
    clearValues[1].depthStencil.stencil = 0;
    renderPassInfo.clearValueCount = clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    //TODO: set viewport and scissor (are these already set when creating pipelines?)

    // This has a non-trivial ver. `2`.
    // Omit some code since here we only have one vertex buffer
    auto vertBufOffset = vk::DeviceSize{0};
    commandBuffer.bindVertexBuffers(0, 1, &render::vertexBufferU.get(), &vertBufOffset);
    commandBuffer.bindIndexBuffer(render::vertexIdxBufferU.get(), 0, vk::IndexTypeValue<decltype(model_info::vertexIdx)::value_type>::value);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, render::graphPipeLayoutU.get(),
                                     0, 1, &render::descriptorSetsU[currentFrame].get(), 0, nullptr);
    commandBuffer.pushConstants(render::graphPipeLayoutU.get(), vk::ShaderStageFlagBits::eVertex, 0, sizeof(ScenePushConstants), &render::sceneVPs[currentFrame]);

    commandBuffer.drawIndexed(model_info::vertexIdx.size(), 1, 0, 0, 0);

    commandBuffer.endRenderPass();

    auto endResult = commandBuffer.end();
    utils::vkEnsure(endResult);
}

std::vector<vk::UniqueCommandBuffer> createCommandBuffers(
        vk::CommandPool &commandPool, vk::Device &device) {

    auto buffersSize = render::MAX_FRAMES_IN_FLIGHT;

    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)buffersSize;

    auto [buffersResult, commandBuffers] = device.allocateCommandBuffersUnique(allocInfo);
    utils::vkEnsure(buffersResult);

    return std::move(commandBuffers);
}
vk::UniqueSampler createTextureSampler(
        vk::Device &device,
        vk::PhysicalDevice &physicalDevice) {

    auto properties = physicalDevice.getProperties();

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = vk::CompareOp::eAlways;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    auto [result, sampler] = device.createSamplerUnique(samplerInfo);
    utils::vkEnsure(result);
    return std::move(sampler);
}
namespace render {
    class TextureObject {
    public:
        vk::UniqueSampler sampler{};

        vk::UniqueImage image{};

        vk::ImageLayout imageLayout{vk::ImageLayout::eUndefined};
        //vk::MemoryAllocateInfo mem_alloc{};
        vk::UniqueDeviceMemory mem{};
        vk::UniqueImageView view{};

        vk::Extent3D extent{};
        uint32_t texChannels{0};

        explicit TextureObject(const std::string &filePath){
            {
                int width, height, channels;
                int ok;
                ok = stbi_info(filePath.c_str(), &width, &height, &channels);
                if (ok == 1 && channels == 4) {
                    pixels_ = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
                } else {
                    std::abort();
                }
                extent.width = width;
                extent.height = height;
                texChannels = channels;
                extent.depth = 1;
            }
            auto devSize = vk::DeviceSize{extent.width*extent.height*extent.depth*texChannels};
            using ImgUsage = vk::ImageUsageFlagBits;
            using MemProp = vk::MemoryPropertyFlagBits;
            auto format = vk::Format::eR8G8B8A8Srgb;
            auto tiling = vk::ImageTiling::eOptimal;
            auto usage = ImgUsage::eTransferDst | ImgUsage::eSampled;
            auto properties = MemProp::eDeviceLocal;
            std::tie(image, mem) = createImagenMemoryFromHostData(
                    extent, devSize, (void*)pixels_, format, tiling, imageLayout, usage, properties,
                    queueFamilyIndexGT, renderDevice, renderPhysicalDevice, renderCommandPool, renderQueue);
            imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal; // Set by createImagenMemoryFromHostData
            view = std::move(createImageViews(std::span{&image.get(), 1}, format, vk::ImageAspectFlagBits::eColor, renderDevice)[0]);
            sampler = createTextureSampler(renderDevice, renderPhysicalDevice);
        }
        TextureObject() = default;
        TextureObject(const TextureObject &) = delete;
        TextureObject& operator= (const TextureObject &) = delete;
        TextureObject& operator= (TextureObject &&other) noexcept {
            if (this != &other){
                sampler = std::move(other.sampler);
                image = std::move(other.image);
                imageLayout = other.imageLayout;
                mem = std::move(other.mem);
                view = std::move(other.view);
                extent = other.extent;
                texChannels = other.texChannels;
                pixels_ = other.pixels_;
                other.pixels_ = nullptr;
            }
            return *this;
        }
        TextureObject(TextureObject &&other) noexcept{
            *this = std::move(other);
        }
        ~TextureObject(){
            if (pixels_ != nullptr) {
                stbi_image_free(pixels_);
            }
        }
    private:
        stbi_uc* pixels_{};

    };
    TextureObject testTexture{};
}
std::vector<vk::UniqueDescriptorSet> createDescriptorSets(vk::Device &device) {
    std::vector<vk::DescriptorSetLayout> layouts(render::MAX_FRAMES_IN_FLIGHT, render::descriptorSetLayoutU.get());
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = render::descriptorPoolU.get();
    allocInfo.descriptorSetCount = layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    auto [result, descriptorSets] = device.allocateDescriptorSetsUnique(allocInfo);
    utils::vkEnsure(result);
    return std::move(descriptorSets);

}
void updateDescriptorSetBuffer(vk::DescriptorSet &descriptorSet, vk::Buffer &buffer, size_t size, uint32_t binding,
                               vk::Device &device) {
    vk::DescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = size;

    vk::WriteDescriptorSet descriptorWrite{};
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
}
void updateDescriptorSetImage(vk::DescriptorSet &descriptorSet,
                              vk::ImageView &imageView, vk::Sampler &sampler, vk::ImageLayout &layout, uint32_t binding,
                              vk::Device &device) {
    vk::DescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = layout;
    imageInfo.imageView = imageView;
    imageInfo.sampler = sampler;

    vk::WriteDescriptorSet descriptorWrite{};
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pImageInfo = &imageInfo;

    device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
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
    {
        using namespace render;
        queueFamilyIndexGT = queueIdx;
        renderDevice = device;
        renderPhysicalDevice = physicalDevice;
        renderCommandPool = commandPool;
        renderQueue = graphicsQueue;
    }
    render::descriptorPoolU = createDescriptorPool(device, render::MAX_FRAMES_IN_FLIGHT);

    std::tie(render::descriptorSetLayoutU, render::graphPipeLayoutU, render::graphPipelineU) = createGraphicsPipeline(device, viewportExtent, renderPass);
    std::tie(render::vertexBufferU, render::vertexBufferMemoryU) = createTriangleVertexInputBuffer(queueIdx, device, physicalDevice, commandPool, graphicsQueue);
    std::tie(render::vertexIdxBufferU, render::vertexIdxBufferMemoryU) = createTriangleVertexIdxBuffer(queueIdx, device, physicalDevice, commandPool, graphicsQueue);
    createModelUniformBuffers(device, physicalDevice);
    render::commandBuffersU = createCommandBuffers(commandPool, device);
    {
        render::sceneVPs.resize(render::MAX_FRAMES_IN_FLIGHT);
    }
    render::testTexture = render::TextureObject{"textures/test_512.png"};
    render::descriptorSetsU = createDescriptorSets(device);
    for (size_t i = 0; i < render::MAX_FRAMES_IN_FLIGHT; ++i){
        auto descriptorSet = render::descriptorSetsU[i].get();
        updateDescriptorSetBuffer(descriptorSet, render::uniformBuffersU[i].get(), sizeof(ModelUBO), 0, device);
        updateDescriptorSetImage(descriptorSet, render::testTexture.view.get(), render::testTexture.sampler.get(), render::testTexture.imageLayout, 1, device);
    }
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
    auto testTexture = std::move(render::testTexture);
}

#endif //VKLEARN_RENDERER_HPP
