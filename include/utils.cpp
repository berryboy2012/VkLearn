//
// Created by berry on 2023/2/10.
// Boilerplate codes
//

#ifndef VKLEARN_UTILS_HPP
#define VKLEARN_UTILS_HPP
#include <chrono>
#include <thread>
#include <concepts>
#include <iostream>
#include <functional>
#include <any>
#include <fstream>
#include <format>
#include <map>
#include <optional>
#include "glm/glm.hpp"
#include "spirv_glsl.hpp"
#include "utils.h"
#include "commands_management.h"

namespace utils {
    const std::unordered_map<spirv_cross::SPIRType::BaseType, const std::string> gSpirvTypeNameMap = {
            {spirv_cross::SPIRType::BaseType::Unknown, "Unknown"},
            {spirv_cross::SPIRType::BaseType::Void, "Void"},
            {spirv_cross::SPIRType::BaseType::Boolean, "Boolean"},
            {spirv_cross::SPIRType::BaseType::SByte, "SByte"},
            {spirv_cross::SPIRType::BaseType::UByte, "UByte"},
            {spirv_cross::SPIRType::BaseType::Short, "Short"},
            {spirv_cross::SPIRType::BaseType::UShort, "UShort"},
            {spirv_cross::SPIRType::BaseType::Int, "Int"},
            {spirv_cross::SPIRType::BaseType::UInt, "UInt"},
            {spirv_cross::SPIRType::BaseType::Int64, "Int64"},
            {spirv_cross::SPIRType::BaseType::UInt64, "UInt64"},
            {spirv_cross::SPIRType::BaseType::AtomicCounter, "AtomicCounter"},
            {spirv_cross::SPIRType::BaseType::Half, "Half"},
            {spirv_cross::SPIRType::BaseType::Float, "Float"},
            {spirv_cross::SPIRType::BaseType::Double, "Double"},
            {spirv_cross::SPIRType::BaseType::Struct, "Struct"},
            {spirv_cross::SPIRType::BaseType::Image, "Image"},
            {spirv_cross::SPIRType::BaseType::SampledImage, "SampledImage"},
            {spirv_cross::SPIRType::BaseType::Sampler, "Sampler"},
            {spirv_cross::SPIRType::BaseType::AccelerationStructure, "AccelerationStructure"},
            {spirv_cross::SPIRType::BaseType::RayQuery, "RayQuery"}
    };
    const std::unordered_map<spirv_cross::SPIRType::BaseType, size_t> gSpirvTypeSizeMap = {
            //{spirv_cross::SPIRType::BaseType::Unknown, 0},
            //{spirv_cross::SPIRType::BaseType::Void, 0},
            //{spirv_cross::SPIRType::BaseType::Boolean, 0},
            {spirv_cross::SPIRType::BaseType::SByte, 1},
            {spirv_cross::SPIRType::BaseType::UByte, 1},
            {spirv_cross::SPIRType::BaseType::Short, 2},
            {spirv_cross::SPIRType::BaseType::UShort, 2},
            {spirv_cross::SPIRType::BaseType::Int, 4},
            {spirv_cross::SPIRType::BaseType::UInt, 4},
            {spirv_cross::SPIRType::BaseType::Int64, 8},
            {spirv_cross::SPIRType::BaseType::UInt64, 8},
            //{spirv_cross::SPIRType::BaseType::AtomicCounter, 0},
            {spirv_cross::SPIRType::BaseType::Half, 2},
            {spirv_cross::SPIRType::BaseType::Float, 4},
            {spirv_cross::SPIRType::BaseType::Double, 8},
            {spirv_cross::SPIRType::BaseType::Struct, 0},
            {spirv_cross::SPIRType::BaseType::Image, 0},
            {spirv_cross::SPIRType::BaseType::SampledImage, 0},
            //{spirv_cross::SPIRType::BaseType::Sampler, 0},
            //{spirv_cross::SPIRType::BaseType::AccelerationStructure, 0},
            //{spirv_cross::SPIRType::BaseType::RayQuery, 0}
    };
    struct SwapchainImageResources {
        vk::Image image;
        vk::CommandBuffer cmd;
        vk::CommandBuffer graphicsToPresentCmd;
        vk::ImageView view;
        vk::Buffer uniformBuffer;
        vk::DeviceMemory uniformMemory;
        void *uniformMemoryPtr = nullptr;
        vk::Framebuffer framebuffer;
        vk::DescriptorSet descriptorSet;
    };

    // Yanked from https://github.com/KhronosGroup/Vulkan-Hpp/samples/utils/utils.cpp
    VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT       messageSeverity,
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

    std::vector<char> read_file(const std::string& filePath) {
        std::ifstream file(filePath, std::ios::ate | std::ios::binary);

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
    std::tuple<std::vector<const char*>, uint32_t> stringToVecptrU32(const std::span<const std::string> &strings){
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

    inline void vk_ensure(const vk::Result &result, const std::optional<std::string_view> &prompt, const std::source_location& location){
        if (result != vk::Result::eSuccess) [[unlikely]]{
            std::cerr<<location.file_name() << ':'
                     << location.line() << ' ' << to_string(result);
            if (prompt.has_value()){
                std::cerr<<prompt.value()<<std::endl;
            } else {
                std::cerr<<std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::abort();
        }
    }

    void log_and_pause(const std::optional<std::string_view> &prompt, size_t sleepMs, const std::source_location& location){
        std::cout << "Logging:"
                  << location.file_name() << ':'
                  << location.line() << ' '
                  << (prompt.has_value() ? prompt.value() : "") << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }

    void probeLayout(spirv_cross::CompilerGLSL &glsl, spirv_cross::SPIRType &objType, size_t &offset, size_t &size){
        if (objType.basetype==spirv_cross::SPIRType::BaseType::Struct){
            auto memberSize = objType.member_types.size();
            for (size_t i=0; i<memberSize; ++i){
                auto memberType = glsl.get_type(objType.member_types[i]);
                std::cout<<std::format("{}: ", glsl.get_member_name(objType.self, i));
                probeLayout(glsl, memberType, offset, size);
            }
        } else {
            if (gSpirvTypeSizeMap.contains(objType.basetype)){
                offset = offset + size;
                size = gSpirvTypeSizeMap.at(objType.basetype) * objType.columns * objType.vecsize;
            }
            auto name = std::string{};
            name = gSpirvTypeNameMap.at(objType.basetype);
            if (objType.vecsize != 1) {
                if (objType.columns == 1) {
                    name = std::format("{}vec{}", name, objType.vecsize);
                }
                else {
                    name = std::format("{}mat{}x{}", name, objType.columns, objType.vecsize);
                }
            }
            std::cout<<std::format("{}: Offset: {} B, Size: {} B\n", name, offset, size);
        }
    }

    const std::unordered_map<vk::Format, size_t> sizeofVkFormat = {
        {vk::Format::eR32Sfloat, 4},
        {vk::Format::eR32G32Sfloat, 8},
        {vk::Format::eR32G32B32Sfloat, 12},
        {vk::Format::eR32G32B32A32Sfloat, 16},
        {vk::Format::eR32Sint, 4},
        {vk::Format::eR32G32Sint, 8},
        {vk::Format::eR32G32B32Sint, 12},
        {vk::Format::eR32G32B32A32Sint, 16},
        {vk::Format::eR32Uint, 4},
        {vk::Format::eR32G32Uint, 8},
        {vk::Format::eR32G32B32Uint, 12},
        {vk::Format::eR32G32B32A32Uint, 16},

        {vk::Format::eR64Sfloat, 8},
        {vk::Format::eR64G64Sfloat, 16},
        {vk::Format::eR64G64B64Sfloat, 24},
        {vk::Format::eR64G64B64A64Sfloat, 32},
        {vk::Format::eR64Sint, 8},
        {vk::Format::eR64G64Sint, 16},
        {vk::Format::eR64G64B64Sint, 24},
        {vk::Format::eR64G64B64A64Sint, 32},
        {vk::Format::eR64Uint, 8},
        {vk::Format::eR64G64Uint, 16},
        {vk::Format::eR64G64B64Uint, 24},
        {vk::Format::eR64G64B64A64Uint, 32},

        {vk::Format::eR16Sfloat, 2},
        {vk::Format::eR16G16Sfloat, 4},
        {vk::Format::eR16G16B16Sfloat, 6},
        {vk::Format::eR16G16B16A16Sfloat, 8},
        {vk::Format::eR16Sint, 2},
        {vk::Format::eR16G16Sint, 4},
        {vk::Format::eR16G16B16Sint, 6},
        {vk::Format::eR16G16B16A16Sint, 8},
        {vk::Format::eR16Uint, 2},
        {vk::Format::eR16G16Uint, 4},
        {vk::Format::eR16G16B16Uint, 6},
        {vk::Format::eR16G16B16A16Uint, 8},
        {vk::Format::eUndefined, 0}
    };

    size_t get_sizeof_vk_format(vk::Format format){
        return sizeofVkFormat.at(format);
    }

    std::vector<uint32_t> load_shader_byte_code(const std::string_view &filePath){
        auto irCode = std::vector<uint32_t>();
        const std::string filePathString{filePath};
        {
            std::ifstream file(filePathString, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                utils::log_and_pause(std::format("Cannot open shader file at: {}", filePath));
                std::abort();
            }
            auto fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);

            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();

            auto byteSize = buffer.size()*sizeof(decltype(buffer)::value_type);
            irCode.resize(byteSize/sizeof(decltype(irCode)::value_type));
            std::memcpy(irCode.data(), buffer.data(), byteSize);
        }
        return irCode;
    }

    vk::UniqueShaderModule create_shader_module(const std::string_view &filePath, vk::Device &device) {
        auto irCode = load_shader_byte_code(filePath);
        auto [result, shaderModule] = device.createShaderModuleUnique({vk::ShaderModuleCreateFlags(),
                                                                       irCode.size()*sizeof(decltype(irCode)::value_type),
                                                                       irCode.data()});
        utils::vk_ensure(result, "create_shader_module() failed");
        return std::move(shaderModule);
    }

    vk::UniqueDescriptorSetLayout create_descriptor_set_layout(std::span<const vk::DescriptorSetLayoutBinding> bindings, vk::Device &device){
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

        auto [result, descriptorSetLayout] = device.createDescriptorSetLayoutUnique(layoutInfo);
        utils::vk_ensure(result);
        return std::move(descriptorSetLayout);
    }

}
#endif //VKLEARN_UTILS_CPP
