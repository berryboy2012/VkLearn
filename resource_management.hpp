//
// Created by berry on 2023/3/12.
//

#ifndef VKLEARN_RESOURCE_MANAGEMENT_HPP
#define VKLEARN_RESOURCE_MANAGEMENT_HPP
#include "utils.h"
#include "memory_management.hpp"
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
class TextureObject {
public:
    vk::Image image_{};
    vk::ImageView view_{};
    vk::UniqueSampler sampler_{};

    vk::ImageLayout imageLayout_{vk::ImageLayout::eUndefined};

    vk::Extent3D extent_{};
    uint32_t texChannels_{0};
    struct Pixel{
        uint8_t r, g, b, a;
    };
    [[nodiscard]] const vk::ImageCreateInfo& getImageInfo() const{
        return data_.resInfo;
    }
    [[nodiscard]] std::span<const Pixel> getImageHostData() const{
        return {(Pixel*)pixels_, extent_.width*extent_.height*extent_.depth};
    }

    explicit TextureObject(const std::string &filePath, VulkanResourceManager &resMgrRef){
        {
            int width, height, channels;
            int ok;
            ok = stbi_info(filePath.c_str(), &width, &height, &channels);
            if (ok == 1 && channels == 4) {
                pixels_ = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
            } else {
                std::abort();
            }
            extent_.width = width;
            extent_.height = height;
            texChannels_ = channels;
            extent_.depth = 1;
        }
        using ImgUsage = vk::ImageUsageFlagBits;
        auto format = vk::Format::eR8G8B8A8Srgb;
        auto tiling = vk::ImageTiling::eOptimal;
        auto usage = ImgUsage::eTransferDst | ImgUsage::eSampled;
        std::span<Pixel> pixels{(Pixel*)pixels_, extent_.width*extent_.height*extent_.depth};
        data_ = resMgrRef.createImagenMemoryFromHostData<Pixel>(pixels,
                                                                extent_, format, tiling, imageLayout_, usage);
        image_ = data_.resource.get();
        data_.createView(vk::ImageAspectFlagBits::eColor);
        view_ = data_.view.get();
        imageLayout_ = vk::ImageLayout::eShaderReadOnlyOptimal; // Set by createImagenMemoryFromHostData
        sampler_ = resMgrRef.createTextureSampler();
    }
    TextureObject() = default;
    TextureObject(const TextureObject &) = delete;
    TextureObject& operator= (const TextureObject &) = delete;
    TextureObject& operator= (TextureObject &&other) noexcept {
        if (this != &other) [[likely]]{
            data_ = std::move(other.data_);
            image_ = data_.resource.get();
            view_ = data_.view.get();
            sampler_ = std::move(other.sampler_);
            imageLayout_ = other.imageLayout_;
            extent_ = other.extent_;
            texChannels_ = other.texChannels_;
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
    VulkanImageMemory data_{};
};

class RenderResourceManager{
    /*Time for another rant
     * We want to create different types of resources in a "single" interface, each resource type has different methods.
     * In fact, the only common ground might be "interacting with descriptor". Thus it is not possible to decouple
     * memory management.
     *
     * The need to keep track of vk::ImageLayout is yet another Vulkan verbosity BS.
     * On the one hand, Vulkan introduces too many immutable objects to make the rendering as static as possible.
     * On the other hand, vk::ImageLayout is a performance-related state that will change during the "static" rendering.
     * Things start getting nasty when you are forced to specify accepted vk::ImageLayout when creating the immutable objects.
     * This makes things unnecessary intertwined since resource management have to intercept the rendering process, even though
     * all rendering components are static.
     * However, the real pain is that, you can't query the layout of an image at all. Even though it is impossible for an
     * image to have multiple layouts.
     * */
public:
    using ResourceType = factory::ResourceType;
    using VulkanResource = factory::VulkanResource;
    RenderResourceManager(vk::Device device, VulkanResourceManager &manager): device_(device), memMgr_(manager) {
        queueIdxCGTP_ = memMgr_.queueIdxCGTP_;
    }
    ~RenderResourceManager(){
        destroyAllResources();
    }
    void createBuffer(const std::string &resourceName, size_t bufferByteSize, vk::BufferUsageFlags usage, vma::AllocationCreateFlags vmaFlag){
        assert(!resources_.contains(resourceName));
        resources_.try_emplace(resourceName, VulkanResource(ResourceType::eBuffer, resourceName, queueIdxCGTP_));
        resourceDependencies_[resourceName] = {};
        auto memObj = memMgr_.createBuffernMemory(bufferByteSize, usage|vk::BufferUsageFlagBits::eTransferDst, vmaFlag);
        auto& res = resources_.at(resourceName).getBufferHandle();
        res.resInfo = memObj.resInfo;
        res.resource = memObj.resource.release();
        res.isResourceManaged = true;
        res.memInfo = memObj.auxInfo;
        res.mem = memObj.mem.release();
        res.isMemManaged = true;
    }

    void createImage(const std::string &resourceName, const vk::Extent3D extent,
                     const vk::Format format, const vk::ImageTiling tiling,
                     const vk::ImageUsageFlags usage, vma::AllocationCreateFlags vmaFlag){
        assert(!resources_.contains(resourceName));
        resources_.try_emplace(resourceName, VulkanResource(ResourceType::eImage, resourceName, queueIdxCGTP_));
        resourceDependencies_[resourceName] = {};
        auto memObj = memMgr_.createImagenMemory(extent, format, tiling, vk::ImageLayout::eUndefined, usage|vk::ImageUsageFlagBits::eTransferDst, vmaFlag);
        auto& res = resources_.at(resourceName).getImageHandle();
        res.resInfo = memObj.resInfo;
        res.resource = memObj.resource.release();
        res.isResourceManaged = true;
        res.currentLayout = vk::ImageLayout::eUndefined;
        res.memInfo = memObj.auxInfo;
        res.mem = memObj.mem.release();
        res.isMemManaged = true;
    }

    void createViewForImage(const std::string &resourceName, const vk::ImageAspectFlags imageAspect){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        auto viewInfo = factory::image_view_create_info_builder(res.resource, res.resInfo.imageType, res.resInfo.format,
                                                       imageAspect, res.resInfo.mipLevels, res.resInfo.arrayLayers);
        res.viewInfo = viewInfo;
        auto [result, view] = device_.createImageView(viewInfo);
        utils::vk_ensure(result);
        res.view = view;
        res.isViewManaged = true;
    }

    void createSamplerForImage(const std::string &resourceName, const vk::SamplerCreateInfo &samplerInfo){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        res.samplerInfo = samplerInfo;
        auto [result, sampler] = device_.createSampler(samplerInfo);
        utils::vk_ensure(result);
        res.sampler = sampler;
        res.isSamplerManaged = true;
    }

    template<class HostElementType>
    void copyDataHostToDevice(const std::string &resourceName, const std::span<const HostElementType> hostData){
        auto& resource = resources_.at(resourceName);
        vk::DeviceSize bufferSize = hostData.size_bytes();
        auto stagingBuffer = memMgr_.createStagingBuffer(bufferSize);
        std::memcpy(stagingBuffer.auxInfo.pMappedData, hostData.data(), bufferSize);
        switch (resource.getResourceType()) {
            case ResourceType::eBuffer:{
                auto& res = resource.getBufferHandle();
                memMgr_.copyBuffer(stagingBuffer.resource.get(), res.resource, bufferSize);
            } break;
            case ResourceType::eImage:{
                auto& res = resource.getImageHandle();
                res.currentLayout = memMgr_.copyImageFromBuffer(
                        stagingBuffer.resource.get(),
                        res.resInfo.mipLevels, res.resInfo.arrayLayers, vk::ImageAspectFlagBits::eColor,
                        res.currentLayout, res.resource, res.resInfo.extent);
            } break;
            default:
                assert(("Copy destination can only be buffer or image",false));
        }
    }

    // There is one anomaly that needs special treatment: swapchain image view
    //   As each frame will use a different view.
    void createPhantomView(const std::string &resourceName, const vk::ImageCreateInfo &imageInfo, vk::ImageLayout imageCurrentLayout, const vk::ImageViewCreateInfo &viewInfo){
        assert(!resources_.contains(resourceName));
        resources_.try_emplace(resourceName, VulkanResource(ResourceType::eImage, resourceName, queueIdxCGTP_));
        // Avoid the destroy methods being called on this resource.
        resourceDependencies_[resourceName] = {1, resourceName};
        auto& res = resources_.at(resourceName).getImageHandle();
        res.resInfo = imageInfo;
        res.resource = nullptr;
        res.isResourceManaged = false;
        res.memInfo = vma::AllocationInfo{};
        res.mem = nullptr;
        res.isMemManaged = false;
        res.viewInfo = viewInfo;
        res.view = nullptr;
        res.isViewManaged = false;
    }

    void updatePhantomView(const std::string &resourceName, vk::ImageView view){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        res.view = view;
    }

    [[nodiscard]] vk::ImageLayout getImageLayout(const std::string &resourceName){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        return res.currentLayout;
    }

    void updateImageLayout(const std::string &resourceName, vk::ImageLayout currentLayout){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        res.currentLayout = currentLayout;
    }

    [[nodiscard]] const vk::ImageCreateInfo& queryImageInfo(const std::string &resourceName){
        assert(resources_.contains(resourceName));
        auto& res = resources_.at(resourceName).getImageHandle();
        return res.resInfo;
    }

private:
    void destroyAllResources(){
        bool resourceListChanged;
        do {
            resourceListChanged = false;
            std::vector<std::string> resourceList{};
            for(const auto& res: resources_){
                resourceList.push_back(res.first);
            }
            for (const auto& resName: resourceList){
                if(destroyResource(resName)){
                    resourceListChanged = true;
                }
            }
        } while(resourceListChanged);
    }
    bool destroyResource(const std::string &resourceName){
        // first check dependencies
        for (const auto& depend: resourceDependencies_){
            for (const auto& dependLeaf: depend.second){
                if (dependLeaf == resourceName){
                    return false;
                }
            }
        }
        auto& resource = resources_.at(resourceName);
        switch (resource.getResourceType()) {
            case ResourceType::eBuffer:{
                auto& res = resource.getBufferHandle();
                if (res.isResourceManaged){
                    if (res.isMemManaged){
                        memMgr_.allocator_.destroyBuffer(res.resource, res.mem);
                    } else {
                        device_.destroy(res.resource);
                    }
                }
                resources_.erase(resourceName);
                resourceDependencies_.erase(resourceName);
                return true;
            }
                break;
            case ResourceType::eImage:{
                auto& res = resource.getImageHandle();
                if (res.isSamplerManaged){
                    device_.destroy(res.sampler);
                }
                if (res.isViewManaged){
                    device_.destroy(res.view);
                }
                if (res.isResourceManaged){
                    if (res.isMemManaged){
                        memMgr_.allocator_.destroyImage(res.resource, res.mem);
                    } else {
                        device_.destroy(res.resource);
                    }
                }
                resources_.erase(resourceName);
                resourceDependencies_.erase(resourceName);
                return true;
            }
                break;
            default:
                assert(("Only buffer and image allowed", false));
                std::abort();
        }
    }
    std::unordered_map<std::string, VulkanResource> resources_{};
    std::unordered_map<std::string, std::vector<std::string>> resourceDependencies_{};
    vk::Device device_{};
    VulkanResourceManager &memMgr_;
    uint32_t queueIdxCGTP_{};
};
#endif //VKLEARN_RESOURCE_MANAGEMENT_HPP
