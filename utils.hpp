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
#include <format>
#include <map>
#include "glm/glm.hpp"
#include "spirv_glsl.hpp"

namespace utils {
    const std::unordered_map<spirv_cross::SPIRType::BaseType, const std::string> spirvTypeNameMap = {
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
    const std::unordered_map<spirv_cross::SPIRType::BaseType, size_t> spirvTypeSizeMap = {
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
            //{spirv_cross::SPIRType::BaseType::Image, 0},
            //{spirv_cross::SPIRType::BaseType::SampledImage, 0},
            //{spirv_cross::SPIRType::BaseType::Sampler, 0},
            //{spirv_cross::SPIRType::BaseType::AccelerationStructure, 0},
            //{spirv_cross::SPIRType::BaseType::RayQuery, 0}
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

    std::vector<char> readFile(const std::string& filePath) {
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

    void probeLayout(spirv_cross::CompilerGLSL &glsl, spirv_cross::SPIRType &objType, size_t &offset, size_t &size){
        if (objType.basetype==spirv_cross::SPIRType::BaseType::Struct){
            auto memberSize = objType.member_types.size();
            for (size_t i=0; i<memberSize; ++i){
                auto memberType = glsl.get_type(objType.member_types[i]);
                std::cout<<std::format("{}: ", glsl.get_member_name(objType.self, i));
                probeLayout(glsl, memberType, offset, size);
            }
        } else {
            if (spirvTypeSizeMap.contains(objType.basetype)){
                offset = offset + size;
                size = spirvTypeSizeMap.at(objType.basetype)*objType.columns*objType.vecsize;
            }
            auto name = std::string{};
            name = spirvTypeNameMap.at(objType.basetype);
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

    const std::unordered_map<vk::Format, size_t> sizeofVkFormat = {
        {vk::Format::eR32Sfloat, 4},
        {vk::Format::eR32G32Sfloat, 8},
        {vk::Format::eR32G32B32Sfloat, 12},
        {vk::Format::eR32G32B32A32Sfloat, 16}
    };

    // TODO: add probing support for SSBO
    vk::UniqueShaderModule createShaderModule(const std::string &filePath, vk::Device &device) {
        std::cout<<std::format("\nReading shader bytecode located at {}\n\n", filePath);
        auto irCode = std::vector<uint32_t>();
        {
            std::ifstream file(filePath, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
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
        spirv_cross::CompilerGLSL glsl(irCode);
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();

        for (auto &sampler : resources.sampled_images){
            unsigned set = glsl.get_decoration(sampler.id, spv::DecorationDescriptorSet);
            unsigned binding = glsl.get_decoration(sampler.id, spv::DecorationBinding);
            std::cout<<std::format("Combined image sampler: {} at set = {}, binding = {}.\n", sampler.name, set, binding);
        }
        {
            size_t offset{0};
            size_t varSize{0};
            for (auto &ubo: resources.uniform_buffers) {
                auto set = glsl.get_decoration(ubo.id, spv::DecorationDescriptorSet);
                auto binding = glsl.get_decoration(ubo.id, spv::DecorationBinding);
                std::cout << std::format("Uniform buffer: {} at set = {}, binding = {};\t", ubo.name, set, binding);
                auto varTypeId = glsl.get_type(ubo.type_id);
                auto size = varTypeId.vecsize * varTypeId.columns;
                if (varTypeId.basetype == spirv_cross::SPIRType::BaseType::Struct){
                    size *= glsl.get_declared_struct_size(varTypeId);
                } else{
                    size *= spirvTypeSizeMap.at(varTypeId.basetype);
                }
                std::cout << std::format("Size = {} B, base type is {}.\n", size,
                                         spirvTypeNameMap.at(varTypeId.basetype));
                probeLayout(glsl, varTypeId, offset, varSize);
            }
        }
        {
            size_t offset{0};
            size_t varSize{0};
            for (auto &pushConst: resources.push_constant_buffers) {
                auto set = glsl.get_decoration(pushConst.id, spv::DecorationDescriptorSet);
                auto binding = glsl.get_decoration(pushConst.id, spv::DecorationBinding);
                std::cout << std::format("Push constant: {} at set = {}, binding = {};\t", pushConst.name, set, binding);
                auto varTypeId = glsl.get_type(pushConst.type_id);
                auto size = varTypeId.vecsize * varTypeId.columns;
                if (varTypeId.basetype == spirv_cross::SPIRType::BaseType::Struct){
                    size *= glsl.get_declared_struct_size(varTypeId);
                } else{
                    size *= spirvTypeSizeMap.at(varTypeId.basetype);
                }
                std::cout << std::format("Size = {} B, base type is {}.\n", size,
                                         spirvTypeNameMap.at(varTypeId.basetype));
                probeLayout(glsl, varTypeId, offset, varSize);
            }
        }
        // `in` and `out` won't be struct, stay cool dude.
        {
            size_t offset{0};
            size_t varSize{0};
            for (auto &inputAttr: resources.stage_inputs) {
                auto set = glsl.get_decoration(inputAttr.id, spv::DecorationDescriptorSet);
                auto binding = glsl.get_decoration(inputAttr.id, spv::DecorationBinding);
                auto location = glsl.get_decoration(inputAttr.id, spv::DecorationLocation);
                std::cout << std::format("Input attribute: {} at set = {}, binding = {}, location = {};\t",
                                         inputAttr.name, set, binding, location);
                auto varTypeId = glsl.get_type(inputAttr.type_id);
                auto size = varTypeId.vecsize * varTypeId.columns;
                if (varTypeId.basetype == spirv_cross::SPIRType::BaseType::Struct) {
                    size *= glsl.get_declared_struct_size(varTypeId);
                } else {
                    size *= spirvTypeSizeMap.at(varTypeId.basetype);
                }
                std::cout << std::format("Size = {} B, base type is {}.\n", size,
                                         spirvTypeNameMap.at(varTypeId.basetype));
                probeLayout(glsl, varTypeId, offset, varSize);
            }
        }
        {
            size_t offset{0};
            size_t varSize{0};
            for (auto &outputAttr: resources.stage_outputs) {
                auto set = glsl.get_decoration(outputAttr.id, spv::DecorationDescriptorSet);
                auto binding = glsl.get_decoration(outputAttr.id, spv::DecorationBinding);
                auto location = glsl.get_decoration(outputAttr.id, spv::DecorationLocation);
                std::cout << std::format("Output attribute: {} at set = {}, binding = {}, location = {};\t",
                                         outputAttr.name, set, binding, location);
                auto varTypeId = glsl.get_type(outputAttr.type_id);
                auto size = varTypeId.vecsize * varTypeId.columns;
                if (varTypeId.basetype == spirv_cross::SPIRType::BaseType::Struct) {
                    size *= glsl.get_declared_struct_size(varTypeId);
                } else {
                    size *= spirvTypeSizeMap.at(varTypeId.basetype);
                }
                std::cout << std::format("Size = {} B, base type is {}.\n", size,
                                         spirvTypeNameMap.at(varTypeId.basetype));
                probeLayout(glsl, varTypeId, offset, varSize);
            }
        }

        auto [result, shaderModule] = device.createShaderModuleUnique({vk::ShaderModuleCreateFlags(),
                                                                       irCode.size()*sizeof(decltype(irCode)::value_type),
                                                                       irCode.data()});
        utils::vkEnsure(result, "createShaderModule() failed");
        return std::move(shaderModule);
    }
    // Submit the recorded vk::CommandBuffer and wait once dtor is called.
    class SingleTimeCommandBuffer {
    public:
        vk::CommandBuffer coBuf{};
        SingleTimeCommandBuffer(const SingleTimeCommandBuffer &) = delete;
        SingleTimeCommandBuffer& operator= (const SingleTimeCommandBuffer &) = delete;
        SingleTimeCommandBuffer(vk::CommandPool &commandPool, vk::Queue &queue, vk::Device &device):
                commandPool_(commandPool),
                queue_(queue),
                device_(device)
        {
            vk::CommandBufferAllocateInfo allocInfo{};
            allocInfo.level = vk::CommandBufferLevel::ePrimary;
            allocInfo.commandPool = commandPool;
            allocInfo.commandBufferCount = 1;
            {
                vk::Result result{};
                std::tie(result, commBuffs_) = device.allocateCommandBuffers(allocInfo);
                vkEnsure(result);
            }
            coBuf = commBuffs_[0];
            vk::CommandBufferBeginInfo beginInfo{};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
            auto result = coBuf.begin(beginInfo);
            vkEnsure(result);
        }
        ~SingleTimeCommandBuffer(){
            auto result = coBuf.end();
            vkEnsure(result);
            vk::SubmitInfo submitInfo{};
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &coBuf;
            auto subResult = queue_.submit(submitInfo);
            vkEnsure(subResult);
            auto waitResult = queue_.waitIdle();
            vkEnsure(waitResult);
            device_.freeCommandBuffers(commandPool_, commBuffs_);
        }
    private:
        std::vector<vk::CommandBuffer> commBuffs_{};
        vk::Device device_{};
        vk::CommandPool commandPool_{};
        vk::Queue queue_{};
    };
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
#endif //VKLEARN_UTILS_HPP
