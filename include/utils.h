//
// Created by berry on 2023/3/7.
//

#ifndef VKLEARN_UTILS_H
#define VKLEARN_UTILS_H
#include <concepts>
#include <iostream>
#include <functional>
#include <any>
#include <fstream>
#include <format>
#include <map>
#include <optional>
#include "spirv_glsl.hpp"
#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT_ON_RESULT
#include "vulkan/vulkan.hpp"
#include "glm/glm.hpp"

namespace utils {
    // Useful for storing a series of vk::Image handles along with their infos
    struct VkImagesPack{
        std::vector<vk::Image> images;
        vk::Format format;
        vk::Extent2D extent;
    };
    VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback( VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                                                VkDebugUtilsMessageTypeFlagsEXT              messageTypes,
                                                                VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
                                                                void * /*pUserData*/ );
    std::vector<char> readFile(const std::string& filePath);
    std::tuple<std::vector<const char*>, uint32_t> stringToVecptrU32(const std::vector<std::string> &strings);
    inline void vkEnsure(const vk::Result &result, const std::optional<std::string> &prompt = std::nullopt);
    void probeLayout(spirv_cross::CompilerGLSL &glsl, spirv_cross::SPIRType &objType, size_t &offset, size_t &size);
    template <typename T>
    concept isGlmType = requires (){
        typename T::value_type;
        // Just introduce glm namespace, will not use it.
        {T::length()} -> std::same_as<glm::length_t>;
    };
    template <typename T>
    requires isGlmType<T>
    consteval vk::Format glmTypeToFormat(){
        if (std::is_same_v<typename T::value_type, float>){
            const auto totalSize = sizeof(T)/sizeof(typename T::value_type);
            switch (totalSize) {
                case 1: return vk::Format::eR32Sfloat;
                case 2: return vk::Format::eR32G32Sfloat;
                case 3: return vk::Format::eR32G32B32Sfloat;
                case 4: return vk::Format::eR32G32B32A32Sfloat;
                default:
                    static_assert(totalSize<5 && totalSize > 0);;//MSVC is acting a bit funny here, don't remove the empty expression
            }
        }
    }
    size_t getSizeofVkFormat(vk::Format format);
    vk::UniqueShaderModule createShaderModule(const std::string &filePath, vk::Device &device);
    vk::UniqueDescriptorSetLayout createDescriptorSetLayout(std::span<const vk::DescriptorSetLayoutBinding> bindings, vk::Device &device);

    template<typename T>
    inline glm::mat<4, 4, T> vkuLookAtRH(glm::vec<3, T> const& eye, glm::vec<3, T> const& center, glm::vec<3, T> const& up)
    {
        glm::vec<3, T> const f(normalize(center - eye));
        glm::vec<3, T> const s(normalize(cross(f, up)));
        glm::vec<3, T> const u(cross(f, s));

        glm::mat<4, 4, T> Result(1);
        Result[0][0] = s.x;
        Result[1][0] = s.y;
        Result[2][0] = s.z;
        Result[0][1] = u.x;
        Result[1][1] = u.y;
        Result[2][1] = u.z;
        Result[0][2] = f.x;
        Result[1][2] = f.y;
        Result[2][2] = f.z;
        Result[3][0] =-dot(s, eye);
        Result[3][1] =-dot(u, eye);
        Result[3][2] =-dot(f, eye);
        return Result;
    }
    template<typename T>
    inline glm::mat<4, 4, T> vkuPerspectiveRHReverse_ZO(T fovy, T aspect, T zNear, T zFar)
    {
        assert(abs(aspect - std::numeric_limits<T>::epsilon()) > static_cast<T>(0));

        T const tanHalfFovy = tan(fovy / static_cast<T>(2));

        glm::mat<4, 4, T> Result(static_cast<T>(0));
        Result[0][0] = static_cast<T>(1) / (aspect * tanHalfFovy);
        Result[1][1] = static_cast<T>(1) / (tanHalfFovy);
        Result[2][2] = zNear / (zNear - zFar);
        Result[2][3] = static_cast<T>(1);
        Result[3][2] = (zFar * zNear) / (zFar - zNear);
        return Result;
    }
    // A trick to manually display type names upon compilation
    template<typename T>
    class TypeDisplayer;
}
#ifndef SHOW_TYPE
#define SHOW_TYPE(obj)      utils::TypeDisplayer<decltype(obj)> LOOK_ME;
#endif
#endif //VKLEARN_UTILS_H
