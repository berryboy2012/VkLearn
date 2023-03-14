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
        data = resMgrRef.createImagenMemoryFromHostData<Pixel>(pixels,
                extent_, format, tiling, imageLayout_, usage);
        image_ = data.resource.get();
        data.createView(vk::ImageAspectFlagBits::eColor);
        view_ = data.view.get();
        imageLayout_ = vk::ImageLayout::eShaderReadOnlyOptimal; // Set by createImagenMemoryFromHostData
        sampler_ = resMgrRef.createTextureSampler();
    }
    TextureObject() = default;
    TextureObject(const TextureObject &) = delete;
    TextureObject& operator= (const TextureObject &) = delete;
    TextureObject& operator= (TextureObject &&other) noexcept {
        if (this != &other) [[likely]]{
            data = std::move(other.data);
            image_ = data.resource.get();
            view_ = data.view.get();
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
    VulkanImageMemory data{};
};
#endif //VKLEARN_RESOURCE_MANAGEMENT_HPP
