//
// Created by berry on 2023/2/10.
// Boilerplate codes
//

#ifndef VKLEARN_UTILS_HPP
#define VKLEARN_UTILS_HPP
#include <concepts>
#include <functional>
#include <any>
#include <fstream>

namespace utils {
    struct TextureObject {
        vk::Sampler sampler;

        vk::Image image;
        vk::Buffer buffer;
        vk::ImageLayout imageLayout{vk::ImageLayout::eUndefined};

        vk::MemoryAllocateInfo mem_alloc;
        vk::DeviceMemory mem;
        vk::ImageView view;

        uint32_t tex_width{0};
        uint32_t tex_height{0};
    };
    struct SwapchainImageResources {
        vk::Image image;
        vk::CommandBuffer cmd;
        vk::CommandBuffer graphics_to_present_cmd;
        vk::ImageView view;
        vk::Buffer uniform_buffer;
        vk::DeviceMemory uniform_memory;
        void *uniform_memory_ptr = nullptr;
        vk::Framebuffer framebuffer;
        vk::DescriptorSet descriptor_set;
    };
    // Useful for storing a series of vk::Image handles along with their infos
    struct VkImagesPack{
        std::vector<vk::Image> images;
        vk::Format format;
        vk::Extent2D extent;
    };
    // Yanked from https://github.com/KhronosGroup/Vulkan-Hpp/samples/utils/utils.cpp
    VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback( VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
                                                                VkDebugUtilsMessageTypeFlagsEXT              messageTypes,
                                                                VkDebugUtilsMessengerCallbackDataEXT const * pCallbackData,
                                                                void * /*pUserData*/ )
    {
#if !defined( NDEBUG )
        if ( pCallbackData->messageIdNumber == 648835635 )
        {
            // UNASSIGNED-khronos-Validation-debug-build-warning-message
            return VK_FALSE;
        }
        if ( pCallbackData->messageIdNumber == 767975156 )
        {
            // UNASSIGNED-BestPractices-vkCreateInstance-specialuse-extension
            return VK_FALSE;
        }
#endif

        std::cerr << vk::to_string( static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>( messageSeverity ) ) << ": "
                  << vk::to_string( static_cast<vk::DebugUtilsMessageTypeFlagsEXT>( messageTypes ) ) << ":\n";
        std::cerr << std::string( "\t" ) << "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
        std::cerr << std::string( "\t" ) << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
        std::cerr << std::string( "\t" ) << "message         = <" << pCallbackData->pMessage << ">\n";
        if ( 0 < pCallbackData->queueLabelCount )
        {
            std::cerr << std::string( "\t" ) << "Queue Labels:\n";
            for ( uint32_t i = 0; i < pCallbackData->queueLabelCount; i++ )
            {
                std::cerr << std::string( "\t\t" ) << "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
            }
        }
        if ( 0 < pCallbackData->cmdBufLabelCount )
        {
            std::cerr << std::string( "\t" ) << "CommandBuffer Labels:\n";
            for ( uint32_t i = 0; i < pCallbackData->cmdBufLabelCount; i++ )
            {
                std::cerr << std::string( "\t\t" ) << "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
            }
        }
        if ( 0 < pCallbackData->objectCount )
        {
            std::cerr << std::string( "\t" ) << "Objects:\n";
            for ( uint32_t i = 0; i < pCallbackData->objectCount; i++ )
            {
                std::cerr << std::string( "\t\t" ) << "Object " << i << "\n";
                std::cerr << std::string( "\t\t\t" ) << "objectType   = " << vk::to_string( static_cast<vk::ObjectType>( pCallbackData->pObjects[i].objectType ) )
                          << "\n";
                std::cerr << std::string( "\t\t\t" ) << "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
                if ( pCallbackData->pObjects[i].pObjectName )
                {
                    std::cerr << std::string( "\t\t\t" ) << "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
                }
            }
        }
        return VK_TRUE;
    }

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        auto fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

        file.close();

        return buffer;
    }

    // Extract C-style vector<const char*> and size() from vector<string>
    std::tuple<std::vector<const char*>, uint32_t> stringToVecptrU32(const std::vector<std::string> &strings){
        std::vector<const char*> vecPtr;
        for (const auto& str: strings){
            vecPtr.push_back(str.c_str());
        }
        auto size = static_cast<uint32_t>(strings.size());
        return std::make_tuple(vecPtr,size);
    }

    // Whether T looks like a struct used in Vulkan API
    template<typename T>
    concept IsVulkanStruct = requires{
        T::sType;
        requires std::is_same_v<decltype(T::sType), VULKAN_HPP_NAMESPACE::StructureType>;
        T::pNext;
        requires std::is_same_v<decltype(T::pNext), void*>;
    };

    inline void vkEnsure(const vk::Result &result, const std::optional<std::string> &prompt = std::nullopt){
        if (result != vk::Result::eSuccess){
            if (prompt.has_value()){
                std::cerr<<prompt.value()<<std::endl;
            }
            std::abort();
        }
    }
    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code, vk::Device &device) {
        auto [result, shaderModule] = device.createShaderModuleUnique({vk::ShaderModuleCreateFlags(), code.size(), reinterpret_cast<const uint32_t*>(code.data())});
        utils::vkEnsure(result, "createShaderModule() failed");
        return std::move(shaderModule);
    }
    // A trick to manually display type names upon compilation
    template<typename T>
    class TypeDisplayer;
}
    #ifndef SHOW_TYPE
        #define SHOW_TYPE(obj)      utils::TypeDisplayer<decltype(obj)> LOOK_ME;
    #endif
#endif //VKLEARN_UTILS_HPP
