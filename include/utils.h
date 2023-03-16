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
#include <ranges>
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
    VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                                                  VkDebugUtilsMessageTypeFlagsEXT              messageTypes,
                                                                  VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
                                                                  void * /*pUserData*/ );
    std::vector<char> read_file(const std::string& filePath);
    std::tuple<std::vector<const char*>, uint32_t> stringToVecptrU32(const std::vector<std::string> &strings);
    inline void vk_ensure(const vk::Result &result, const std::optional<std::string> &prompt = std::nullopt);
    template <typename T>
    concept isGlmType = requires (){
        typename T::value_type;
        // Just introduce glm namespace, will not use it.
        {T::length()} -> std::same_as<glm::length_t>;
    };
    template <typename T>
    requires isGlmType<T>
    consteval vk::Format glm_type_to_format(){
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
    size_t get_sizeof_vk_format(vk::Format format);
    std::vector<uint32_t> load_shader_byte_code(const std::string_view &filePath);
    vk::UniqueShaderModule create_shader_module(const std::string_view &filePath, vk::Device &device);
    vk::UniqueDescriptorSetLayout create_descriptor_set_layout(std::span<const vk::DescriptorSetLayoutBinding> bindings, vk::Device &device);

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
    template <typename Enum>
    constexpr inline auto enum_range = [](auto front, auto back) {
        return std::views::iota(static_cast<std::underlying_type_t<Enum>>(front), static_cast<std::underlying_type_t<Enum>>(back) + 1)
               | std::views::transform([](auto e) { return Enum(e); });
    };
    // A trick to manually display type names upon compilation
    template<typename T>
    class TypeDisplayer;
    // Meta programming tricks
    namespace meta_trick{
        // Get number of members (with unique memory addresses) of a struct,
        // yanked from comments at https://stackoverflow.com/a/38575501 ,
        //  usage: struct S3{int a,b;bool c;}; member_count<S3> == 3
        struct Any
        {
            template <typename T> operator T();
        };

        template <typename T, std::size_t I>
        using always_t = T;


        template <typename T, typename ... Args>
        auto is_aggregate_constructible_impl(int) -> decltype(T{std::declval<Args>()...}, void(), std::true_type{});

        template <typename T, typename ... Args>
        auto is_aggregate_constructible_impl(...) -> std::false_type;

        template <typename T, typename ... Args>
        using is_aggregate_constructible = decltype(is_aggregate_constructible_impl<T, Args...>(0));

        template <typename T, typename Seq> struct has_n_member_impl;

        template <typename T, std::size_t ... Is>
        struct has_n_member_impl<T, std::index_sequence<Is...>> : is_aggregate_constructible<T, always_t<Any, Is>...> {};

        template <typename T, std::size_t N>
        using has_n_member = has_n_member_impl<T, std::make_index_sequence<N>>;

        template <typename T, typename Seq> struct member_count_impl;

        template <typename T, std::size_t ... Is>
        struct member_count_impl<T, std::index_sequence<Is...>> : std::integral_constant<std::size_t, std::max({(has_n_member<T, Is>() * Is)...})> {};

        template <typename T>
        using member_count = member_count_impl<T, std::make_index_sequence<1 + sizeof (T)>>;
    }


    struct QueueStruct{
        vk::Queue queue{};
        uint32_t queueFamilyIdx{};
    };
}
#ifndef SHOW_TYPE
#define SHOW_TYPE(obj)      utils::TypeDisplayer<decltype(obj)> LOOK_ME;
#endif
#endif //VKLEARN_UTILS_H
