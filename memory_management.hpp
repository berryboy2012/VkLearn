//
// Created by berry on 2023/3/6.
//

#ifndef VKLEARN_MEMORY_MANAGEMENT_HPP
#define VKLEARN_MEMORY_MANAGEMENT_HPP

#include <concepts>
#include "vulkan/vulkan.hpp"
#include "spirv_glsl.hpp"
#include "glm/glm.hpp"
#include <optional>
#include <map>
#include <format>
#include <fstream>
#include <any>
#include <functional>
#include <iostream>

#include "commands_management.h"
#include "vk_mem_alloc.hpp"
#include "utils.h"
#include "global_objects.hpp"
#include "shader_modules.hpp"

// The lifetime of vk::Buffer and vk::Image has some degree of freedom, we need to keep track of some information.
struct VulkanBufferMemory{
    vk::UniqueHandle<vma::Allocation,vma::Dispatcher> mem{};
    vma::AllocationInfo auxInfo{};
    vk::UniqueHandle<vk::Buffer,vma::Dispatcher> resource{};
    vk::BufferCreateInfo resInfo{};
    VulkanBufferMemory() = default;
    VulkanBufferMemory(const VulkanBufferMemory &) = delete;
    VulkanBufferMemory& operator= (const VulkanBufferMemory &) = delete;
    VulkanBufferMemory(VulkanBufferMemory &&other) noexcept{
        *this = std::move(other);
    }
    VulkanBufferMemory& operator= (VulkanBufferMemory &&other) noexcept {
        if (this != &other) [[likely]]{
            mem = std::move(other.mem);
            auxInfo = other.auxInfo;
            resource = std::move(other.resource);
            resInfo = other.resInfo;
        }
        return *this;
    }
};

struct VulkanImageMemory{
    vk::Device device{};
    vk::UniqueHandle<vma::Allocation,vma::Dispatcher> mem{};
    vma::AllocationInfo auxInfo{};
    vk::UniqueHandle<vk::Image,vma::Dispatcher> resource{};
    vk::ImageCreateInfo resInfo{};
    // vk::Image has the complication of vk::ImageView, vk::Sampler can live on its own.
    vk::UniqueImageView view{};
    VulkanImageMemory() = default;
    VulkanImageMemory(const VulkanImageMemory &) = delete;
    VulkanImageMemory& operator= (const VulkanImageMemory &) = delete;
    VulkanImageMemory(VulkanImageMemory &&other) noexcept{
        *this = std::move(other);
    }
    VulkanImageMemory& operator= (VulkanImageMemory &&other) noexcept {
        if (this != &other) [[likely]]{
            device = other.device;
            mem = std::move(other.mem);
            auxInfo = other.auxInfo;
            resource = std::move(other.resource);
            resInfo = other.resInfo;
            view = std::move(other.view);
        }
        return *this;
    }
    void createView(const vk::ImageAspectFlags imageAspect){
        auto createInfo = factory::image_view_info_builder(
                resource.get(),
                resInfo.imageType, resInfo.format, resInfo.mipLevels, resInfo.arrayLayers,
                imageAspect);
        auto [result, imgView] = device.createImageViewUnique(createInfo);
        utils::vk_ensure(result);
        view = std::move(imgView);
    }
};
struct VulkanImageHandle{
    vk::Device device{};
    vk::Image resource{};
    vk::ImageCreateInfo resInfo{};
    // vk::Image has the complication of vk::ImageView, vk::Sampler can live on its own.
    vk::UniqueImageView view{};
    VulkanImageHandle() = default;
    VulkanImageHandle(const VulkanImageHandle &) = delete;
    VulkanImageHandle& operator= (const VulkanImageHandle &) = delete;
    VulkanImageHandle(VulkanImageHandle &&other) noexcept{
        *this = std::move(other);
    }
    VulkanImageHandle& operator= (VulkanImageHandle &&other) noexcept {
        if (this != &other) [[likely]]{
            device = other.device;
            resource = other.resource;
            resInfo = other.resInfo;
            view = std::move(other.view);
        }
        return *this;
    }
    void createView(const vk::ImageAspectFlags imageAspect){
        vk::ImageViewCreateInfo createInfo = {};
        createInfo.image = resource;
        switch (resInfo.imageType) {
            case vk::ImageType::e1D:
                createInfo.viewType = vk::ImageViewType::e1D;
                break;
            case vk::ImageType::e2D:
                createInfo.viewType = vk::ImageViewType::e2D;
                break;
            case vk::ImageType::e3D:
                createInfo.viewType = vk::ImageViewType::e3D;
                break;
        }
        createInfo.format = resInfo.format;
        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;
        createInfo.subresourceRange.aspectMask = imageAspect;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = resInfo.mipLevels;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = resInfo.arrayLayers;
        auto [result, imgView] = device.createImageViewUnique(createInfo);
        utils::vk_ensure(result);
        view = std::move(imgView);
    }
};

class VulkanResourceManager{
    // A thin layer on top of VMA. Along with some helper function for creating usable vk::Buffer and vk::Image etc.
    //  You still need to manage the lifetime of resources by yourself.
public:
    vma::Allocator allocator_{};
    VulkanResourceManager(const VulkanResourceManager &) = delete;
    VulkanResourceManager& operator= (const VulkanResourceManager &) = delete;
    VulkanResourceManager(VulkanResourceManager &&other) noexcept{
        *this = std::move(other);
    }
    VulkanResourceManager& operator= (VulkanResourceManager &&other) noexcept {
        if (this != &other) [[likely]]{
            inst_ = other.inst_;
            physDev_ = other.physDev_;
            device_ = other.device_;
            alloc_ = std::move(other.alloc_);
            allocator_ = other.allocator_;
            other.allocator_ = VK_NULL_HANDLE;
            queueIdxCGTP_ = other.queueIdxCGTP_;
        }
        return *this;
    }
    VulkanResourceManager() = default;
    VulkanResourceManager(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device,
                          utils::QueueStruct queueCGTP, bool isThreaded = true) {
        inst_ = instance;
        physDev_ = physicalDevice;
        device_ = device;
        queueIdxCGTP_ = queueCGTP.queueFamilyIdx;
        queue_ = queueCGTP.queue;
        auto funcs = vma::VulkanFunctions{};//
        if (isThreaded){
            std::lock_guard<std::mutex> lock(gVulkanMutex);
            funcs = vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);
        }
        else{
            funcs = vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);
        }
        vma::AllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorCreateInfo.physicalDevice = physDev_;
        allocatorCreateInfo.device = device_;
        allocatorCreateInfo.instance = inst_;
        allocatorCreateInfo.pVulkanFunctions = &funcs;
        auto [result, alloc] = vma::createAllocatorUnique(allocatorCreateInfo);
        utils::vk_ensure(result);
        alloc_ = std::move(alloc);
        allocator_ = alloc_->get();
        resCmdPool_ = CommandBufferManager{device_, {.queue = queue_, .queueFamilyIdx = queueIdxCGTP_}};
    }
    VulkanBufferMemory createBuffernMemory(
            const vk::DeviceSize bufferSize, const vk::BufferUsageFlags bufferUsage,
            const vma::AllocationCreateFlags vmaFlag = {}, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto){

        auto bufferInfo = vk::BufferCreateInfo{};
        bufferInfo.flags = vk::BufferCreateFlags{};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = bufferUsage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        bufferInfo.queueFamilyIndexCount = 1;
        bufferInfo.pQueueFamilyIndices = &queueIdxCGTP_;
        // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
        vma::AllocationCreateInfo allocInfo{};
        allocInfo.flags = vmaFlag;
        allocInfo.usage = vmaMemUsage;
        allocInfo.requiredFlags = memProps;
        vma::AllocationInfo auxInfo{};
        auto [result, buffer] = allocator_.createBufferUnique(bufferInfo, allocInfo, &auxInfo);
        utils::vk_ensure(result);
        VulkanBufferMemory createdBuf{};
        createdBuf.resource = std::move(buffer.first);
        createdBuf.mem = std::move(buffer.second);
        createdBuf.auxInfo = auxInfo;
        createdBuf.resInfo = bufferInfo;
        createdBuf.resInfo.queueFamilyIndexCount = 0;
        createdBuf.resInfo.pQueueFamilyIndices = nullptr;
        return std::move(createdBuf);
    }
    VulkanImageMemory createImagenMemory(
            const vk::Extent3D extent, const vk::Format format, const vk::ImageTiling tiling,
            const vk::ImageLayout layout, const vk::ImageUsageFlags imageUsage,
            const vma::AllocationCreateFlags vmaFlag = {}, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto){

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
        imageInfo.usage = imageUsage;
        imageInfo.samples = vk::SampleCountFlagBits::e1;
        imageInfo.sharingMode = vk::SharingMode::eExclusive;
        imageInfo.queueFamilyIndexCount = 1;
        imageInfo.pQueueFamilyIndices = &queueIdxCGTP_;
        vma::AllocationCreateInfo allocInfo{};
        allocInfo.flags = vmaFlag;
        allocInfo.usage = vmaMemUsage;
        allocInfo.requiredFlags = memProps;
        vma::AllocationInfo auxInfo{};
        auto [result, image] = allocator_.createImageUnique(imageInfo, allocInfo, &auxInfo);
        utils::vk_ensure(result);
        VulkanImageMemory createdImg{};
        createdImg.device = device_;
        createdImg.resource = std::move(image.first);
        createdImg.mem = std::move(image.second);
        createdImg.auxInfo = auxInfo;
        createdImg.resInfo = imageInfo;
        createdImg.resInfo.queueFamilyIndexCount = 0;
        createdImg.resInfo.pQueueFamilyIndices = nullptr;
        return std::move(createdImg);
    }
    VulkanImageMemory createImagenMemory(
            const vk::Extent2D extent2D, const vk::Format format, const vk::ImageTiling tiling,
            const vk::ImageLayout layout, const vk::ImageUsageFlags imageUsage,
            const vma::AllocationCreateFlags vmaFlag = {}, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto){
        vk::Extent3D extent{extent2D, 1};
        return createImagenMemory(extent, format, tiling, layout, imageUsage, vmaFlag, memProps, vmaMemUsage);
    }
    VulkanBufferMemory createStagingBuffer(const vk::DeviceSize bufferSize){
        using VmaFlagE = vma::AllocationCreateFlagBits;
        return createBuffernMemory(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                VmaFlagE::eHostAccessSequentialWrite|VmaFlagE::eMapped);
    }

    //TODO: implement async copy helper
    void copyBuffer(vk::Buffer srcBuffer, //vk::DeviceSize srcOffset,
                    vk::Buffer dstBuffer, //vk::DeviceSize dstOffset,
                    vk::DeviceSize size){

        std::vector<vk::BufferCopy> copyRegions;
        vk::BufferCopy copyRegion;
        copyRegion.srcOffset = {};//srcOffset;
        copyRegion.dstOffset = {};//dstOffset;
        copyRegion.size = size;
        copyRegions.push_back(copyRegion);
        {
            auto singleTime = resCmdPool_.getSingleTimeCommandBuffer();
            singleTime.coBuf_.copyBuffer(srcBuffer, dstBuffer, copyRegions.size(), copyRegions.data());
        }
    }

    //TODO: implement async copy helper
    void copyImageFromBuffer(vk::Buffer srcBuffer, vk::Image dstImage, vk::Extent3D extent){
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
            auto singleTime = resCmdPool_.getSingleTimeCommandBuffer();
            singleTime.coBuf_.copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegions);
        }
    }

    [[nodiscard("Keep track of imageLayout")]] vk::ImageLayout copyImageFromBuffer(
            vk::Buffer srcBuffer,
            uint32_t mipLevels, uint32_t arrayLayers, vk::ImageAspectFlags imageAspect,
            vk::ImageLayout imageCurrentLayout, vk::Image dstImage, vk::Extent3D extent){
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
        transitionImageLayout(dstImage, mipLevels, arrayLayers, imageAspect, imageCurrentLayout, vk::ImageLayout::eTransferDstOptimal);
        {
            auto singleTime = resCmdPool_.getSingleTimeCommandBuffer();
            singleTime.coBuf_.copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegions);
        }
        if (imageCurrentLayout != vk::ImageLayout::eUndefined){
            transitionImageLayout(dstImage, mipLevels, arrayLayers, imageAspect, vk::ImageLayout::eTransferDstOptimal, imageCurrentLayout);
            return vk::ImageLayout::eTransferDstOptimal;
        }
        return imageCurrentLayout;
    }

    template<class HostElementType>
    VulkanBufferMemory createBuffernMemoryFromHostData(
            const std::span<const HostElementType> hostData,
            const vk::BufferUsageFlags bufferUsage,
            const vma::AllocationCreateFlags vmaFlag = {}, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto) {
        using BufUsage = vk::BufferUsageFlagBits;
        vk::DeviceSize bufferSize = hostData.size_bytes();
        auto stagingBuffer = createStagingBuffer(bufferSize);
        std::memcpy(stagingBuffer.auxInfo.pMappedData, hostData.data(), bufferSize);
        auto finalBuffer = createBuffernMemory(bufferSize, bufferUsage|BufUsage::eTransferDst, vmaFlag, memProps, vmaMemUsage);
        copyBuffer(stagingBuffer.resource.get(), finalBuffer.resource.get(), bufferSize);
        return std::move(finalBuffer);
    }

    void transitionImageLayout(vk::Image image, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageAspectFlags imageAspect, vk::ImageLayout oldLayout, vk::ImageLayout newLayout){
        auto [barrier, sourceStage, destinationStage] = imgLayoutTransitionInfoBuilder(image, mipLevels, arrayLayers, imageAspect, oldLayout, newLayout);

        {
            auto singleTime = resCmdPool_.getSingleTimeCommandBuffer();
            singleTime.coBuf_.pipelineBarrier(sourceStage, destinationStage, {}, 0, nullptr, 0, nullptr, 1, &barrier);
        }
    }
    vk::UniqueSampler createTextureSampler() {
        auto properties = physDev_.getProperties();
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
        auto [result, sampler] = device_.createSamplerUnique(samplerInfo);
        utils::vk_ensure(result);
        return std::move(sampler);
    }

    template<class HostElementType>
    VulkanImageMemory createImagenMemoryFromHostData(
            const std::span<const HostElementType> hostData,
            const vk::Extent3D extent, const vk::Format format, const vk::ImageTiling tiling,
            const vk::ImageLayout layout, const vk::ImageUsageFlags imageUsage,
            const vma::AllocationCreateFlags vmaFlag = {}, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto) {
        using ImgUsage = vk::ImageUsageFlagBits;
        vk::DeviceSize bufferSize = hostData.size_bytes();
        auto stagingBuffer = createStagingBuffer(bufferSize);
        std::memcpy(stagingBuffer.auxInfo.pMappedData, hostData.data(), bufferSize);
        auto finalImage = createImagenMemory(extent, format, tiling, layout, imageUsage|ImgUsage::eTransferDst, vmaFlag, memProps, vmaMemUsage);
        transitionImageLayout(finalImage.resource.get(), finalImage.resInfo.mipLevels, finalImage.resInfo.arrayLayers, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyImageFromBuffer(stagingBuffer.resource.get(), finalImage.resource.get(), extent);
        transitionImageLayout(finalImage.resource.get(), finalImage.resInfo.mipLevels, finalImage.resInfo.arrayLayers, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        return std::move(finalImage);
    }
    VulkanResourceManager getManagerHandle(){
        auto handle = VulkanResourceManager{};
        handle.allocator_ = allocator_;
        handle.inst_ = inst_;
        handle.physDev_ = physDev_;
        handle.device_ = device_;
        handle.alloc_.reset();
        handle.queueIdxCGTP_ = {};
        handle.queue_ = VK_NULL_HANDLE;
        handle.resCmdPool_ = {};
        return handle;
    }
    void setupManagerHandle(utils::QueueStruct queueCGTP){
        assert(!alloc_.has_value());
        queueIdxCGTP_ = queueCGTP.queueFamilyIdx;
        queue_ = queueCGTP.queue;
        resCmdPool_ = CommandBufferManager{device_, {.queue = queue_, .queueFamilyIdx = queueIdxCGTP_}};
    }

private:
    static std::tuple<
            vk::ImageMemoryBarrier,
            vk::PipelineStageFlags,
            vk::PipelineStageFlags> imgLayoutTransitionInfoBuilder(
                    vk::Image image, uint32_t mipLevels, uint32_t arrayLayers, vk::ImageAspectFlags imageAspect, vk::ImageLayout oldLayout, vk::ImageLayout newLayout){
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = imageAspect;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = arrayLayers;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        assert(newLayout != vk::ImageLayout::eUndefined);
        barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite;
        sourceStage = vk::PipelineStageFlagBits::eAllCommands;
        destinationStage = vk::PipelineStageFlagBits::eAllCommands;
        return std::make_tuple(barrier, sourceStage, destinationStage);
    }
    vk::Instance inst_{};
    vk::PhysicalDevice physDev_{};
    vk::Device device_{};
    std::optional<vma::UniqueAllocator> alloc_{};
    uint32_t queueIdxCGTP_{};
    vk::Queue queue_{};
    CommandBufferManager resCmdPool_{};
    // ToDo: Deprecate VulkanResourceManager
    friend class RenderResourceManager;
};

#endif //VKLEARN_MEMORY_MANAGEMENT_HPP
