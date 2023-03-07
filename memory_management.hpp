//
// Created by berry on 2023/3/6.
//

#ifndef VKLEARN_MEMORY_MANAGEMENT_HPP
#define VKLEARN_MEMORY_MANAGEMENT_HPP
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.hpp"
template<typename ResType>
requires std::is_same_v<ResType, vk::Buffer> || std::is_same_v<ResType, vk::Image>
struct VulkanResourceMemory{
    vk::UniqueHandle<ResType,vma::Dispatcher> resource{};
    vk::UniqueHandle<vma::Allocation,vma::Dispatcher> mem{};
    vma::AllocationInfo auxInfo{};
};
class VulkanResourceManager{
    // A thin layer on top of VMA. Along with some helper function for creating usable vk::Buffer and vk::Image etc.
    //  You still need to manually manage both the `memory view` object and underlying `memory allocation` by yourself.
public:
    VulkanResourceManager(const VulkanResourceManager &) = delete;
    VulkanResourceManager& operator= (const VulkanResourceManager &) = delete;
    VulkanResourceManager(VulkanResourceManager &&other) noexcept{
        *this = std::move(other);
    }
    VulkanResourceManager& operator= (VulkanResourceManager &&other) noexcept {
        if (this != &other){
            inst_ = other.inst_;
            physDev_ = other.physDev_;
            device_ = other.device_;
            allocator_ = std::move(other.allocator_);
            queueIdxCGTP_ = other.queueIdxCGTP_;
        }
        return *this;
    }
    VulkanResourceManager() = default;
    VulkanResourceManager(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device,
                          QueueStruct queueCGTP) {
        inst_ = instance;
        physDev_ = physicalDevice;
        device_ = device;
        queueIdxCGTP_ = queueCGTP.queueFamilyIdx;
        queue_ = queueCGTP.queue;
        auto funcs = vma::functionsFromDispatcher(VULKAN_HPP_DEFAULT_DISPATCHER);
        vma::AllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorCreateInfo.physicalDevice = physDev_;
        allocatorCreateInfo.device = device_;
        allocatorCreateInfo.instance = inst_;
        allocatorCreateInfo.pVulkanFunctions = &funcs;
        auto [result, alloc] = vma::createAllocatorUnique(allocatorCreateInfo);
        utils::vkEnsure(result);
        allocator_ = std::move(alloc);
        resCmdPool_ = CommandBufferManager{device_, {.queue = queue_, .queueFamilyIdx = queueIdxCGTP_}};
    }
    VulkanResourceMemory<vk::Buffer> createBuffernMemory(
            const vk::DeviceSize bufferSize, const vk::BufferUsageFlags bufferUsage,
            const vma::AllocationCreateFlags vmaFlag, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto){

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
        auto [result, buffer] = allocator_->createBufferUnique(bufferInfo, allocInfo, &auxInfo);
        utils::vkEnsure(result);
        VulkanResourceMemory<vk::Buffer> createdBuf{};
        createdBuf.resource = std::move(buffer.first);
        createdBuf.mem = std::move(buffer.second);
        createdBuf.auxInfo = auxInfo;
        return std::move(createdBuf);
    }
    VulkanResourceMemory<vk::Image> createImagenMemory(
            const vk::Extent3D extent, const vk::Format format, const vk::ImageTiling tiling,
            const vk::ImageLayout layout, const vk::ImageUsageFlags imageUsage,
            const vma::AllocationCreateFlags vmaFlag, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto){

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
        auto [result, image] = allocator_->createImageUnique(imageInfo, allocInfo, &auxInfo);
        utils::vkEnsure(result);
        VulkanResourceMemory<vk::Image> createdBuf{};
        createdBuf.resource = std::move(image.first);
        createdBuf.mem = std::move(image.second);
        createdBuf.auxInfo = auxInfo;
        return std::move(createdBuf);
    }
    //TODO: implement async copy helper
    static void copyBuffer(vk::Buffer srcBuffer, //vk::DeviceSize srcOffset,
                    vk::Buffer dstBuffer, //vk::DeviceSize dstOffset,
                    vk::DeviceSize size,
                    CommandBufferManager &cmdMgr){

        std::vector<vk::BufferCopy> copyRegions;
        vk::BufferCopy copyRegion;
        copyRegion.srcOffset = {};//srcOffset;
        copyRegion.dstOffset = {};//dstOffset;
        copyRegion.size = size;
        copyRegions.push_back(copyRegion);
        {
            auto singleTime = cmdMgr.getSingleTimeCommandBuffer();
            singleTime.coBuf.copyBuffer(srcBuffer, dstBuffer, copyRegions.size(), copyRegions.data());
        }
    }
    //TODO: implement async copy helper
    static void copyImageFromBuffer(vk::Buffer srcBuffer, vk::Image dstImage, vk::Extent3D extent,
                                    CommandBufferManager &cmdMgr){
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
            auto singleTime = cmdMgr.getSingleTimeCommandBuffer();
            singleTime.coBuf.copyBufferToImage(srcBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegions);
        }
    }

    template<class HostElementType>
    VulkanResourceMemory<vk::Buffer> createBuffernMemoryFromHostData(
            const std::span<const HostElementType> hostData,
            const vk::BufferUsageFlags bufferUsage,
            const vma::AllocationCreateFlags vmaFlag, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto) {
        using BufUsage = vk::BufferUsageFlagBits;
        using VmaFlagE = vma::AllocationCreateFlagBits;
        vk::DeviceSize bufferSize = hostData.size_bytes();
        auto stagingBuffer = createBuffernMemory(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                VmaFlagE::eHostAccessSequentialWrite|VmaFlagE::eMapped);
        std::memcpy(stagingBuffer.auxInfo.pMappedData, hostData.data(), bufferSize);
        auto finalBuffer = createBuffernMemory(bufferSize, bufferUsage|BufUsage::eTransferDst, vmaFlag, memProps, vmaMemUsage);
        copyBuffer(stagingBuffer.resource.get(), finalBuffer.resource.get(), bufferSize, resCmdPool_);
        return std::move(finalBuffer);
    }

    template<class HostElementType>
    VulkanResourceMemory<vk::Image> createImagenMemoryFromHostData(
            const std::span<const HostElementType> hostData,
            const vk::Extent3D extent, const vk::Format format, const vk::ImageTiling tiling,
            const vk::ImageLayout layout, const vk::ImageUsageFlags imageUsage,
            const vma::AllocationCreateFlags vmaFlag, const vk::MemoryPropertyFlags memProps = {}, const vma::MemoryUsage vmaMemUsage = vma::MemoryUsage::eAuto) {
        using ImgUsage = vk::ImageUsageFlagBits;
        using BufUsage = vk::BufferUsageFlagBits;
        using VmaFlagE = vma::AllocationCreateFlagBits;
        vk::DeviceSize bufferSize = hostData.size_bytes();
        auto stagingBuffer = createBuffernMemory(
                bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                VmaFlagE::eHostAccessSequentialWrite|VmaFlagE::eMapped);
        std::memcpy(stagingBuffer.auxInfo.pMappedData, hostData.data(), bufferSize);
        auto finalImage = createImagenMemory(extent, format, tiling, layout, imageUsage|ImgUsage::eTransferDst, vmaFlag, memProps, vmaMemUsage);
        //utils::transitionImageLayout(finalImage.resource.get(), format, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, device_, resCmdPool_)
        copyImageFromBuffer(stagingBuffer.resource.get(), finalImage.resource.get(), bufferSize, resCmdPool_);
        return std::move(finalImage);
    }

private:
    vk::Instance inst_{};
    vk::PhysicalDevice physDev_{};
    vk::Device device_{};
    vma::UniqueAllocator allocator_{};
    uint32_t queueIdxCGTP_{};
    vk::Queue queue_{};
    CommandBufferManager resCmdPool_{};
};

#endif //VKLEARN_MEMORY_MANAGEMENT_HPP
