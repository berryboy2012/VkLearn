//
// Created by berry on 2023/3/3.
//

#ifndef VKLEARN_SHADER_MODULES_HPP
#define VKLEARN_SHADER_MODULES_HPP

#include "global_objects.hpp"
#include "commands_management.h"
#include <iostream>
#include <functional>
#include <any>
#include <fstream>
#include <format>
#include <map>
#include <optional>
#include "glm/glm.hpp"
#include "spirv_glsl.hpp"
#include <concepts>

#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT_ON_RESULT

#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.hpp"
#include "utils.h"
#include "model_data.hpp"
namespace factory {
    typedef size_t RenderpassIdx;
    typedef uint32_t AttachIdx;
    typedef uint32_t DependIdx;
    typedef uint32_t SubpassIdx;
    typedef uint64_t StageMask;
    typedef size_t ShaderIdx;
    typedef uint32_t DescSetIdx;
    typedef uint32_t BindIdx;
    typedef uint32_t LocIdx;
    enum ShaderStage: uint64_t {
        eNone                   = 0b0LLU,
        eVertex                 = 0b1LLU,
//        eTessellationControl    = 0b10LLU,
//        eTessellationEvaluation = 0b100LLU,
//        eGeometry               = 0b1000LLU,
        eFragment               = 0b10000LLU,
        eCompute                = 0b100000LLU,
//        eRaygenKHR              = 0b1000000LLU,
//        eAnyHitKHR              = 0b10000000LLU,
//        eClosestHitKHR          = 0b100000000LLU,
//        eMissKHR                = 0b1000000000LLU,
//        eIntersectionKHR        = 0b10000000000LLU,
//        eCallableKHR            = 0b100000000000LLU,
//        eTaskEXT                = 0b1000000000000LLU,
//        eMeshEXT                = 0b10000000000000LLU,
        eFull                   = 0xFFFFFFFFFFFFFFFFLLU
    };
    struct ShaderDescriptorInfo{
        std::string resName{};
        DescSetIdx set{};
        vk::DescriptorSetLayoutBinding desc{};
        size_t byteSize{};
    };

    struct VertexInputAttrParsedInfo {
        // We need more info to deal with GLSL's unintuitive rules
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-attrib-location
        uint32_t startLocation;
        uint32_t multiplier;
        vk::Format format;
        uint32_t fmtBaseCompByteSize;
    };

    struct AttachmentInfo {
        std::string resName{};
        vk::AttachmentDescription2 description{};
    };

    enum class ResourceType: uint8_t {
        eUndefined = 0,
        eBuffer = 1,
        eImage = 2,
        //eImagelessView = 3,
        //eCombinedImageSampler = 4,
        ePushConst = 3,
        eAccelStruct = 4,
        eCount = 5
    };

    template<ResourceType>
    struct VulkanConcreteResource {
        std::string resName{};
    };

    template<>
    struct VulkanConcreteResource<ResourceType::eBuffer> {
        std::string resName{};
        vma::Allocation mem{};
        vma::AllocationInfo memInfo{};
        bool isMemManaged{};
        vk::Buffer resource{};
        vk::BufferCreateInfo resInfo{};
        bool isResourceManaged{};
        uint32_t queueFamIdx{};
    };

    template<>
    struct VulkanConcreteResource<ResourceType::eImage> {
        std::string resName{};
        vma::Allocation mem{};
        vma::AllocationInfo memInfo{};
        bool isMemManaged{};
        vk::Image resource{};
        vk::ImageCreateInfo resInfo{};
        vk::ImageLayout currentLayout{};
        bool isResourceManaged{};
        uint32_t queueFamIdx{};
        vk::ImageView view{};
        vk::ImageViewCreateInfo viewInfo{};
        bool isViewManaged{};
        vk::Sampler sampler{};
        vk::SamplerCreateInfo samplerInfo{};
        bool isSamplerManaged{};
    };

    template<>
    struct VulkanConcreteResource<ResourceType::ePushConst> {
        std::string resName{};
        size_t byteSize{};
        VulkanConcreteResource(const VulkanConcreteResource& other) {
            *this = other;
        }
        VulkanConcreteResource(VulkanConcreteResource&& other) noexcept {
            *this = std::move(other);
        }
        VulkanConcreteResource& operator=(const VulkanConcreteResource& other) {
            if (this != &other){
                resName = other.resName;
                mem.reset();
                resource = other.resource;
                isMemManaged = false;
                byteSize = other.byteSize;
            }
            return *this;
        }
        VulkanConcreteResource& operator=(VulkanConcreteResource<ResourceType::ePushConst> &&other){
            if (this != &other){
                resName = other.resName;
                mem = std::move(other.mem);
                resource = other.resource;
                other.resource = nullptr;
                isMemManaged = other.isMemManaged;
                byteSize = other.byteSize;
            }
            return *this;
        }
        explicit VulkanConcreteResource(size_t size): byteSize(size), isMemManaged(true) {
            mem = std::make_unique<char[]>(size);
            resource = mem.get();
        }
        bool isMemManaged;
        void* resource{};
        std::unique_ptr<char[]> mem ;
    };

    template<>
    struct VulkanConcreteResource<ResourceType::eAccelStruct> {
        std::string resName{};
        vma::Allocation mem{};
        vma::AllocationInfo memInfo{};
        bool isMemManaged{};
        vk::AccelerationStructureKHR resource{};
        vk::AccelerationStructureCreateInfoKHR resInfo{};
        bool isResourceManaged{};
        uint32_t queueFamIdx{};
        //...
    };

    // Basically just a sugar class for polymorphism, any_cast related stuff should not leak to outside.
    class VulkanResource{
    public:
        VulkanResource() = delete;

        VulkanResource(ResourceType resourceType, const std::string &resourceName, uint32_t queueFamilyIndex): resType_(resourceType){
            switch (resType_) {
                case ResourceType::eBuffer:
                    resStruct_ = VulkanConcreteResource<ResourceType::eBuffer>{.resName = resourceName, .queueFamIdx = queueFamilyIndex};
                    break;
                case ResourceType::eImage:
                    resStruct_ = VulkanConcreteResource<ResourceType::eImage>{.resName = resourceName, .queueFamIdx = queueFamilyIndex};
                    break;
                case ResourceType::ePushConst:
                    // We can't input template parameters directly here.
                    break;
                case ResourceType::eAccelStruct:
                    resStruct_ = VulkanConcreteResource<ResourceType::eAccelStruct>{.resName = resourceName, .queueFamIdx = queueFamilyIndex};
                    break;
                default:
                    resStruct_ = VulkanConcreteResource<ResourceType::eUndefined>{.resName = resourceName};
            }
        }

        template<class ValueType>
        static VulkanResource createPushConst(const std::string &resourceName, uint32_t queueFamilyIndex){
            VulkanResource res{ResourceType::ePushConst, resourceName, queueFamilyIndex};
            res.resStruct_ = std::move(VulkanConcreteResource<ResourceType::ePushConst>{sizeof(ValueType)});
            return std::move(res);
        }

        [[nodiscard]] VulkanConcreteResource<ResourceType::eBuffer>& getBufferHandle(){
            assert(resType_==ResourceType::eBuffer);
            return std::any_cast<VulkanConcreteResource<ResourceType::eBuffer>&>(resStruct_);
        }

        [[nodiscard]] VulkanConcreteResource<ResourceType::eImage>& getImageHandle(){
            assert(resType_==ResourceType::eImage);
            return std::any_cast<VulkanConcreteResource<ResourceType::eImage>&>(resStruct_);
        }

        [[nodiscard]] VulkanConcreteResource<ResourceType::ePushConst>& getPushConstHandle(){
            assert(resType_==ResourceType::ePushConst);
            return std::any_cast<VulkanConcreteResource<ResourceType::ePushConst>&>(resStruct_);
        }

        [[nodiscard]] VulkanConcreteResource<ResourceType::eAccelStruct>& getAccelStructHandle(){
            assert(resType_==ResourceType::eAccelStruct);
            return std::any_cast<VulkanConcreteResource<ResourceType::eAccelStruct>&>(resStruct_);
        }

        [[nodiscard]] ResourceType getResourceType() const{
            return resType_;
        }

    private:
        ResourceType resType_;
        std::any resStruct_;
    };

    typedef vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2> VkSubpassDependency2StructChain;
    // Tested for UBO and SSBO
    size_t get_resource_byte_size(spirv_cross::CompilerGLSL &glsl, const spirv_cross::Resource &res) {
        size_t byteSize{0};
        size_t compByteSize{0};
        size_t numComp{0};
        auto typeInfo = glsl.get_type(res.type_id);
        const auto &baseTypeInfo = glsl.get_type(res.base_type_id);
        if (typeInfo.columns != 1) {
            numComp = typeInfo.vecsize * typeInfo.columns;
        } else {
            numComp = typeInfo.vecsize;
        }
        switch (typeInfo.basetype) {
            case spirv_cross::SPIRType::Double:
            case spirv_cross::SPIRType::Int64:
            case spirv_cross::SPIRType::UInt64:
                compByteSize = 8;
                break;
            case spirv_cross::SPIRType::Float:
            case spirv_cross::SPIRType::Int:
            case spirv_cross::SPIRType::UInt:
                compByteSize = 4;
                break;
            case spirv_cross::SPIRType::Half:
            case spirv_cross::SPIRType::Short:
            case spirv_cross::SPIRType::UShort:
                compByteSize = 2;
                break;
            case spirv_cross::SPIRType::Struct:
                compByteSize = glsl.get_declared_struct_size(baseTypeInfo);
                break;
            default:
                compByteSize = 0;
        }
        byteSize = compByteSize * numComp;
        return byteSize;
    }

    std::unordered_map<std::string, ShaderDescriptorInfo>
    parse_descriptors(const std::vector<uint32_t>& shaderCode, vk::ShaderStageFlags stages){
        auto glsl = spirv_cross::CompilerGLSL(shaderCode);
        std::unordered_map<std::string, ShaderDescriptorInfo> entries{};
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();
        for (auto &ubo: resources.uniform_buffers) {
            const spirv_cross::SPIRType &type = glsl.get_type(ubo.type_id);
            uint32_t count{1};
            for (const auto &length: type.array) {
                count *= length;
            }
            ShaderDescriptorInfo entry{};
            entry.desc.binding = glsl.get_decoration(ubo.id, spv::DecorationBinding);
            entry.desc.descriptorType = vk::DescriptorType::eUniformBuffer;
            entry.desc.stageFlags = stages;
            entry.desc.descriptorCount = count;
            entry.set = glsl.get_decoration(ubo.id, spv::DecorationDescriptorSet);
            entry.byteSize = get_resource_byte_size(glsl, ubo);
            entries[glsl.get_name(ubo.id)] = entry;
        }
        // Here comes the complication: Input attachment needs an entry in descriptor set as well
        // Ignore subpassInputMS for now
        for (auto &inAttach: resources.subpass_inputs){
            const spirv_cross::SPIRType &type = glsl.get_type(inAttach.type_id);
            uint32_t count{1};
            for (const auto &length: type.array) {
                count *= length;
            }
            ShaderDescriptorInfo entry{};
            entry.desc.binding = glsl.get_decoration(inAttach.id, spv::DecorationBinding);
            entry.desc.descriptorType = vk::DescriptorType::eInputAttachment;
            entry.desc.stageFlags = stages;
            entry.desc.descriptorCount = count;
            entry.set = glsl.get_decoration(inAttach.id, spv::DecorationDescriptorSet);
            assert(entries.contains(glsl.get_name(inAttach.id)));
            entry.byteSize = get_resource_byte_size(glsl, inAttach);
            entries[glsl.get_name(inAttach.id)] = entry;
        }
        for (auto &combImg: resources.sampled_images) {
            const spirv_cross::SPIRType &type = glsl.get_type(combImg.type_id);
            uint32_t count{1};
            for (const auto &length: type.array) {
                count *= length;
            }
            ShaderDescriptorInfo entry{};
            entry.desc.binding = glsl.get_decoration(combImg.id, spv::DecorationBinding);
            entry.desc.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            entry.desc.stageFlags = stages;
            entry.desc.descriptorCount = count;
            entry.set = glsl.get_decoration(combImg.id, spv::DecorationDescriptorSet);
            entry.byteSize = get_resource_byte_size(glsl, combImg);
            entries[glsl.get_name(combImg.id)] = entry;
        }
        for (auto &ssbo: resources.storage_buffers) {
            const spirv_cross::SPIRType &type = glsl.get_type(ssbo.type_id);
            uint32_t count{1};
            for (const auto &length: type.array) {
                count *= length;
            }
            ShaderDescriptorInfo entry{};
            entry.desc.binding = glsl.get_decoration(ssbo.id, spv::DecorationBinding);
            entry.desc.descriptorType = vk::DescriptorType::eStorageBuffer;
            entry.desc.stageFlags = stages;
            entry.desc.descriptorCount = count;
            entry.set = glsl.get_decoration(ssbo.id, spv::DecorationDescriptorSet);
            entry.byteSize = get_resource_byte_size(glsl, ssbo);
            entries[glsl.get_name(ssbo.id)] = entry;
        }
        for (auto &acc: resources.acceleration_structures) {
            const spirv_cross::SPIRType &type = glsl.get_type(acc.type_id);
            uint32_t count{1};
            for (const auto &length: type.array) {
                count *= length;
            }
            ShaderDescriptorInfo entry{};
            entry.desc.binding = glsl.get_decoration(acc.id, spv::DecorationBinding);
            entry.desc.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
            entry.desc.stageFlags = stages;
            entry.desc.descriptorCount = count;
            entry.set = glsl.get_decoration(acc.id, spv::DecorationDescriptorSet);
            entry.byteSize = get_resource_byte_size(glsl, acc);
            entries[glsl.get_name(acc.id)] = entry;
        }
        return entries;
    }

    struct PushConstantInfo{
        std::string resName{};
        vk::ShaderStageFlags stages{};
        std::string blockName{};
        typedef std::string MemberName;
        struct PushConstantMemberInfo{
            MemberName name{};
            uint32_t byteOffset{};
            uint32_t byteSize{};
        };
        std::map<MemberName, PushConstantMemberInfo> members{};
        [[nodiscard]] vk::PushConstantRange getPushBlockRange() const{
            // Loop over members, sort by offset, min() is blockOffset,
            // then sort by `offset+size`, max() - blockOffset is blockSize
            vk::PushConstantRange result{};
            result.stageFlags = stages;
            result.offset = std::min_element(
                    members.begin(), members.end(),
                    [](const decltype(members)::value_type& l, const decltype(members)::value_type& r) -> bool {
                        return l.second.byteOffset < r.second.byteOffset; })->second.byteOffset;
            const auto& lastMember = std::max_element(
                    members.begin(), members.end(),
                    [](const decltype(members)::value_type& l, const decltype(members)::value_type& r) -> bool {
                        return l.second.byteOffset+l.second.byteSize < r.second.byteOffset+r.second.byteSize; })->second;
            result.size = lastMember.byteOffset+lastMember.byteSize-result.offset;
            return result;
        }
    };

    //Yup, this is the best I can do in terms of docs:
    // https://raw.githubusercontent.com/KhronosGroup/GLSL/master/extensions/khr/GL_KHR_vulkan_glsl.txt
    PushConstantInfo
    parse_push_constants(const std::vector<uint32_t>& shaderCode, vk::ShaderStageFlags stage){
        auto glsl = spirv_cross::CompilerGLSL(shaderCode);
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();
        // At most one push constant block per stage
        auto result = PushConstantInfo{};
        result.stages = stage;
        for (auto& push: resources.push_constant_buffers){
            // push constant block can't be array, no need to count
            // spirv-cross's API is a bit inconsistent here
            result.blockName = push.name;
            //loop over members, and find their offset, push-constant blocks themselves don't have offset
            const spirv_cross::SPIRType &baseType = glsl.get_type(push.base_type_id);
            for (uint32_t i = 0; i < baseType.member_types.size(); i++){
                const auto& name = glsl.get_member_name(baseType.self, i);
                uint32_t byteSize = glsl.get_declared_struct_member_size(baseType, i);
                uint32_t byteOffset = glsl.type_struct_member_offset(baseType, i);
                result.members[name] = {.name = name, .byteOffset=byteOffset, .byteSize=byteSize};
            }
        }
        return result;
    }

    // the first component means a multiplier (number of rows),
    // the last component means byte size of a component inside the format (4 for R32G32B32Sfloat for example)
    std::tuple<uint32_t, vk::Format, uint32_t>
    get_resource_format(spirv_cross::CompilerGLSL &glsl, const spirv_cross::Resource &res) {
        vk::Format fmt;
        size_t multiplier{1};
        size_t numComp{0};
        uint32_t compByteSize{0};
        auto typeInfo = glsl.get_type(res.type_id);
        const auto &baseTypeInfo = glsl.get_type(res.base_type_id);
        if (typeInfo.columns != 1) {
            multiplier = typeInfo.vecsize;
            numComp = typeInfo.columns;
        } else {
            multiplier = 1;
            numComp = typeInfo.vecsize;
        }
        switch (typeInfo.basetype) {
            case spirv_cross::SPIRType::Float:
                compByteSize = 4;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR32Sfloat;
                        break;
                    case 2:
                        fmt = vk::Format::eR32G32Sfloat;
                        break;
                    case 3:
                        fmt = vk::Format::eR32G32B32Sfloat;
                        break;
                    case 4:
                        fmt = vk::Format::eR32G32B32A32Sfloat;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Int:
                compByteSize = 4;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR32Sint;
                        break;
                    case 2:
                        fmt = vk::Format::eR32G32Sint;
                        break;
                    case 3:
                        fmt = vk::Format::eR32G32B32Sint;
                        break;
                    case 4:
                        fmt = vk::Format::eR32G32B32A32Sint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::UInt:
                compByteSize = 4;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR32Uint;
                        break;
                    case 2:
                        fmt = vk::Format::eR32G32Uint;
                        break;
                    case 3:
                        fmt = vk::Format::eR32G32B32Uint;
                        break;
                    case 4:
                        fmt = vk::Format::eR32G32B32A32Uint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Double:
                compByteSize = 8;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR64Sfloat;
                        break;
                    case 2:
                        fmt = vk::Format::eR64G64Sfloat;
                        break;
                    case 3:
                        fmt = vk::Format::eR64G64B64Sfloat;
                        break;
                    case 4:
                        fmt = vk::Format::eR64G64B64A64Sfloat;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Int64:
                compByteSize = 8;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR64Sint;
                        break;
                    case 2:
                        fmt = vk::Format::eR64G64Sint;
                        break;
                    case 3:
                        fmt = vk::Format::eR64G64B64Sint;
                        break;
                    case 4:
                        fmt = vk::Format::eR64G64B64A64Sint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::UInt64:
                compByteSize = 8;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR64Uint;
                        break;
                    case 2:
                        fmt = vk::Format::eR64G64Uint;
                        break;
                    case 3:
                        fmt = vk::Format::eR64G64B64Uint;
                        break;
                    case 4:
                        fmt = vk::Format::eR64G64B64A64Uint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Half:
                compByteSize = 2;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR16Sfloat;
                        break;
                    case 2:
                        fmt = vk::Format::eR16G16Sfloat;
                        break;
                    case 3:
                        fmt = vk::Format::eR16G16B16Sfloat;
                        break;
                    case 4:
                        fmt = vk::Format::eR16G16B16A16Sfloat;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Short:
                compByteSize = 2;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR16Sint;
                        break;
                    case 2:
                        fmt = vk::Format::eR16G16Sint;
                        break;
                    case 3:
                        fmt = vk::Format::eR16G16B16Sint;
                        break;
                    case 4:
                        fmt = vk::Format::eR16G16B16A16Sint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::UShort:
                compByteSize = 2;
                switch (numComp) {
                    case 1:
                        fmt = vk::Format::eR16Uint;
                        break;
                    case 2:
                        fmt = vk::Format::eR16G16Uint;
                        break;
                    case 3:
                        fmt = vk::Format::eR16G16B16Uint;
                        break;
                    case 4:
                        fmt = vk::Format::eR16G16B16A16Uint;
                        break;
                    default:
                        fmt = vk::Format::eUndefined;
                        break;
                }
                break;
            case spirv_cross::SPIRType::Struct:
            default:
                fmt = vk::Format::eUndefined;
        }
        return std::make_tuple(multiplier, fmt, compByteSize);
    }

    struct VertexInputBindingTable {
        std::string resName{};
        vk::VertexInputBindingDescription bindingDescription{};
        std::vector<vk::VertexInputAttributeDescription> attributesDescription{};
    };

    enum class AttachmentType {
        //TODO: add support for preserveAttachments
        eUnknown = 0,
        eInput= 1,
        eColor = 2,
        eResolve = 3,
        eDepthStencil = 4
    };
    struct AttachmentReferenceInfo {
        // Polymorphism through enum, yay
        AttachmentType attachType{};
        vk::AttachmentReference2 attachRef{};
        // Input attachment only
        typedef uint32_t InAttachIdx;
        InAttachIdx inputAttachIdx{};
        // Color attachment only
        LocIdx location{};
        vk::PipelineColorBlendAttachmentState blendInfo{};
    };

    vk::ImageViewCreateInfo image_view_create_info_builder(
            vk::Image image, vk::ImageType imageType, vk::Format format,
            vk::ImageAspectFlags imageAspect, uint32_t mipLevels, uint32_t arrayLayers){
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        switch (imageType) {
            case vk::ImageType::e1D:
                viewInfo.viewType = vk::ImageViewType::e1D;
                break;
            case vk::ImageType::e2D:
                viewInfo.viewType = vk::ImageViewType::e2D;
                break;
            case vk::ImageType::e3D:
                viewInfo.viewType = vk::ImageViewType::e3D;
                break;
        }
        viewInfo.format = format;
        viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
        viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
        viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
        viewInfo.components.a = vk::ComponentSwizzle::eIdentity;
        viewInfo.subresourceRange.aspectMask = imageAspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = arrayLayers;
        return viewInfo;
    }

vk::ImageViewCreateInfo image_view_info_builder(
        vk::Image image, vk::ImageType imageType, vk::Format format, uint32_t mipLevelCount, uint32_t imgCount,
        const vk::ImageAspectFlags imageAspect){
    vk::ImageViewCreateInfo createInfo = {};
    createInfo.image = image;
    switch (imageType) {
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
    createInfo.format = format;
    createInfo.components.r = vk::ComponentSwizzle::eIdentity;
    createInfo.components.g = vk::ComponentSwizzle::eIdentity;
    createInfo.components.b = vk::ComponentSwizzle::eIdentity;
    createInfo.components.a = vk::ComponentSwizzle::eIdentity;
    createInfo.subresourceRange.aspectMask = imageAspect;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = mipLevelCount;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = imgCount;
    return createInfo;
}
}
// Push constants seems to be slower than UBO
struct ScenePushConstants {
    glm::mat4 view;
    glm::mat4 proj;
};
struct ModelUBO {
    glm::mat4 model;
};
struct SubpassAttachmentReferences {
    //TODO: add support for preserveAttachments
    std::vector<vk::AttachmentReference2> inputAttachments;
    std::vector<vk::AttachmentReference2> colorAttachments;
    std::vector<vk::AttachmentReference2> resolveAttachments;
    vk::AttachmentReference2 depthStencilAttachment;
};

class VertexShader {
public:
    vk::UniqueShaderModule shaderModule_;
    typedef size_t DescSetIdx;
    typedef uint32_t BindIdx;
    //TODO: Add proper resId to connect with handles in the resource part
    std::unordered_map<DescSetIdx, std::unordered_map<std::string, BindIdx>> descResLUT_{};
    std::unordered_map<DescSetIdx, std::vector<vk::DescriptorSetLayoutBinding>> descLayouts_{};
    // Specific to vertex shaders
    std::vector<vk::VertexInputBindingDescription> inputInfos_{};
    std::vector<vk::VertexInputAttributeDescription> attrInfos_{};
    vk::PipelineInputAssemblyStateCreateInfo inputAsmInfo_{};
    // Other less frequent entries
    std::vector<vk::PushConstantRange> pushConstInfos_{};

    VertexShader() = default;

    VertexShader(const VertexShader &) = delete;

    VertexShader &operator=(const VertexShader &) = delete;

    VertexShader(VertexShader &&other) noexcept {
        *this = std::move(other);
    }

    VertexShader &operator=(VertexShader &&other) noexcept {
        if (this != &other) [[likely]] {
            shaderModule_ = std::move(other.shaderModule_);
            descResLUT_ = other.descResLUT_;
            descLayouts_ = other.descLayouts_;
            inputInfos_ = other.inputInfos_;
            attrInfos_ = other.attrInfos_;
            inputAsmInfo_ = other.inputAsmInfo_;
            pushConstInfos_ = other.pushConstInfos_;
            device_ = other.device_;
        }
        return *this;
    }

    explicit VertexShader(vk::Device device) {
        std::string filePath = "shaders/shader.vert.spv";
        device_ = device;
        shaderModule_ = utils::create_shader_module(filePath, device_);
        // filling descriptor set layout info
        {
            DescSetIdx setIndex = 0;
            vk::DescriptorSetLayoutBinding modelUBOInfo{};
            modelUBOInfo.binding = 0;
            // If you want to specify an array of element objects, descriptorCount is the length of such array.
            modelUBOInfo.descriptorCount = 1;
            modelUBOInfo.descriptorType = vk::DescriptorType::eUniformBuffer;
            modelUBOInfo.pImmutableSamplers = nullptr;
            modelUBOInfo.stageFlags = vk::ShaderStageFlagBits::eVertex;
            descLayouts_[setIndex].push_back(modelUBOInfo);
            descResLUT_[setIndex]["modelUBO"] = descLayouts_[setIndex].rbegin()->binding;
        }
        // vertex inputs info
        {
            vk::VertexInputBindingDescription bindTableInfo{};
            bindTableInfo.binding = 0;
            bindTableInfo.stride = sizeof(model_info::PCTVertex);
            bindTableInfo.inputRate = vk::VertexInputRate::eVertex;
            inputInfos_.push_back(bindTableInfo);

            {
                vk::VertexInputAttributeDescription inPositionInfo{};
                inPositionInfo.binding = bindTableInfo.binding;
                inPositionInfo.location = 0;
                inPositionInfo.format = utils::glm_type_to_format<glm::vec3>();
                inPositionInfo.offset = 0;

                vk::VertexInputAttributeDescription inColorInfo{};
                inColorInfo.binding = bindTableInfo.binding;
                inColorInfo.location = 1;
                inColorInfo.format = utils::glm_type_to_format<glm::vec3>();
                inColorInfo.offset =
                        inPositionInfo.offset + static_cast<uint32_t>(utils::get_sizeof_vk_format(inPositionInfo.format));

                vk::VertexInputAttributeDescription inTexCoordInfo{};
                inTexCoordInfo.binding = bindTableInfo.binding;
                inTexCoordInfo.location = 2;
                inTexCoordInfo.format = utils::glm_type_to_format<glm::vec2>();
                inTexCoordInfo.offset =
                        inColorInfo.offset + static_cast<uint32_t>(utils::get_sizeof_vk_format(inColorInfo.format));
                attrInfos_.push_back(inPositionInfo);
                attrInfos_.push_back(inColorInfo);
                attrInfos_.push_back(inTexCoordInfo);
            }
        }
        {
            inputAsmInfo_.topology = vk::PrimitiveTopology::eTriangleList;
            inputAsmInfo_.primitiveRestartEnable = VK_FALSE;
        }
        // Push constants
        {
            vk::PushConstantRange sceneVPInfo{};
            sceneVPInfo.offset = 0;
            sceneVPInfo.size = sizeof(ScenePushConstants);
            sceneVPInfo.stageFlags = vk::ShaderStageFlagBits::eVertex;
            pushConstInfos_.push_back(sceneVPInfo);
        }
    }

private:
    vk::Device device_;
};

struct AttachmentInfoBundle {
    // renderpass
    vk::AttachmentDescription2 description{};
    // framebuffer and image
    vk::Format format{};
    vk::ImageCreateFlags flags{};
    vk::ImageUsageFlags usage{};
    // image only
    vk::ImageTiling tiling{};
    vk::ImageLayout layout{};
    vma::AllocationCreateFlags vmaFlag{};
    // image view only
    vk::ImageAspectFlags aspect{};
    // TODO: Corresponding ID in resource manager
    std::string resId{};
};

class FragShader {
public:
    vk::UniqueShaderModule shaderModule_;
    typedef size_t DescSetIdx;
    typedef size_t AttachIdx;
    typedef uint32_t BindIdx;
    std::unordered_map<DescSetIdx, std::unordered_map<std::string, BindIdx>> descResLUT_{};
    std::unordered_map<DescSetIdx, std::vector<vk::DescriptorSetLayoutBinding>> descLayouts_{};
    // Mandatory for fragment shaders
    std::unordered_map<AttachIdx, AttachmentInfoBundle> attachmentResourceInfos_{};
    SubpassAttachmentReferences attachmentReferences_{};
    // Other less frequent entries

    FragShader() = default;

    FragShader(const FragShader &) = delete;

    FragShader &operator=(const FragShader &) = delete;

    FragShader(FragShader &&other) noexcept {
        *this = std::move(other);
    }

    FragShader &operator=(FragShader &&other) noexcept {
        if (this != &other) [[likely]] {
            shaderModule_ = std::move(other.shaderModule_);
            descResLUT_ = other.descResLUT_;
            descLayouts_ = other.descLayouts_;
            device_ = other.device_;
            attachmentResourceInfos_ = other.attachmentResourceInfos_;
            attachmentReferences_ = other.attachmentReferences_;
        }
        return *this;
    }

    explicit FragShader(vk::Device device) {
        std::string filePath = "shaders/shader.frag.spv";
        device_ = device;
        shaderModule_ = utils::create_shader_module(filePath, device_);
        // filling descriptor set layout info
        {
            DescSetIdx setIndex = 0;
            vk::DescriptorSetLayoutBinding texSamplerInfo{};
            texSamplerInfo.binding = 1;
            texSamplerInfo.descriptorCount = 1;
            texSamplerInfo.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            texSamplerInfo.pImmutableSamplers = nullptr;
            texSamplerInfo.stageFlags = vk::ShaderStageFlagBits::eFragment;
            descLayouts_[setIndex].push_back(texSamplerInfo);
            descResLUT_[setIndex]["texSampler"] = descLayouts_[setIndex].rbegin()->binding;
        }
        // filling attachment info, only need to provide attachments first used by this shader
        {
            AttachIdx attachIndex = 0;
            AttachmentInfoBundle swapchainImageAttachmentInfo{};
            // For swapchain image, we use the format determined by main thread
            swapchainImageAttachmentInfo.description.format = {};
            swapchainImageAttachmentInfo.description.samples = vk::SampleCountFlagBits::e1;
            swapchainImageAttachmentInfo.description.loadOp = vk::AttachmentLoadOp::eClear;
            swapchainImageAttachmentInfo.description.storeOp = vk::AttachmentStoreOp::eStore;
            swapchainImageAttachmentInfo.description.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            swapchainImageAttachmentInfo.description.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            swapchainImageAttachmentInfo.description.initialLayout = vk::ImageLayout::eUndefined;
            swapchainImageAttachmentInfo.description.finalLayout = vk::ImageLayout::ePresentSrcKHR;
            // For swapchain images, we need some image info for framebuffer creation
            swapchainImageAttachmentInfo.flags = {};
            // Required by VK_KHR_imageless_framebuffer if this is the swapchain Image
            swapchainImageAttachmentInfo.usage =
                    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment;
            swapchainImageAttachmentInfo.format = {};
            swapchainImageAttachmentInfo.tiling = vk::ImageTiling::eOptimal;
            swapchainImageAttachmentInfo.layout = vk::ImageLayout::ePresentSrcKHR;
            swapchainImageAttachmentInfo.vmaFlag = {};
            // Image view info are not needed as well
            swapchainImageAttachmentInfo.aspect = vk::ImageAspectFlagBits::eColor;
            swapchainImageAttachmentInfo.resId = "swapchainIMG";
            attachmentResourceInfos_[attachIndex] = swapchainImageAttachmentInfo;
        }
        {
            AttachIdx attachIndex = 1;
            AttachmentInfoBundle depthImageAttachmentInfo{};
            // For depth image, we use the format determined by main thread
            depthImageAttachmentInfo.description.format = {};
            depthImageAttachmentInfo.description.samples = vk::SampleCountFlagBits::e1;
            depthImageAttachmentInfo.description.loadOp = vk::AttachmentLoadOp::eClear;
            depthImageAttachmentInfo.description.storeOp = vk::AttachmentStoreOp::eDontCare;
            depthImageAttachmentInfo.description.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            depthImageAttachmentInfo.description.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            depthImageAttachmentInfo.description.initialLayout = vk::ImageLayout::eUndefined;
            depthImageAttachmentInfo.description.finalLayout = vk::ImageLayout::eDepthAttachmentOptimal;
            // For depth image, some image info are provided by the main thread
            depthImageAttachmentInfo.flags = {};
            depthImageAttachmentInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
            depthImageAttachmentInfo.format = {};
            depthImageAttachmentInfo.tiling = vk::ImageTiling::eOptimal;
            depthImageAttachmentInfo.layout = vk::ImageLayout::eUndefined;
            depthImageAttachmentInfo.vmaFlag = {};
            // Image view info are needed
            depthImageAttachmentInfo.aspect = vk::ImageAspectFlagBits::eDepth;
            depthImageAttachmentInfo.resId = "depthIMG";
            attachmentResourceInfos_[attachIndex] = depthImageAttachmentInfo;
        }
        // filling attachment references
        {
            attachmentReferences_.inputAttachments = {};
            attachmentReferences_.resolveAttachments = {};
            vk::AttachmentReference2 swapchainImageAttachmentRef{};
            swapchainImageAttachmentRef.attachment = 0;
            swapchainImageAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
            swapchainImageAttachmentRef.aspectMask = vk::ImageAspectFlagBits::eColor;
            attachmentReferences_.colorAttachments.push_back(swapchainImageAttachmentRef);
            vk::AttachmentReference2 depthImageAttachmentRef{};
            depthImageAttachmentRef.attachment = 1;
            depthImageAttachmentRef.layout = vk::ImageLayout::eDepthAttachmentOptimal;
            depthImageAttachmentRef.aspectMask = vk::ImageAspectFlagBits::eDepth;
            attachmentReferences_.depthStencilAttachment = depthImageAttachmentRef;
        }
    }

private:
    vk::Device device_;
};

// General rules about what goes where:
//  Classes have direct counterparts in Vulkan should be able to spit out related Vulkan API info themselves
//  Those instances should be 'immutable' outside of related Factory classes
//  Infos that can be finalized at current level but unrelated to Vulkan's counterpart shouldn't stay at factories.
//  But should be visible to related factory classes.
//  By finalized at current level, we mean the following preconditions:
//    1. All resource info are already known
//    2. No forms of resource aliased modification are possible at higher levels (up to renderpass)
//   By those two rules, things like binding vertex input attributes with host data structures is allowed to stay at
//   shader level, while binding attachment references with attachments must stay at renderpass level.
//  For information that can only be partially filled at current level, the factories are responsible to propagate the
//  info to high levels, however, only higher levels are responsible for filling the remaining info.
//  The incomplete info shouldn't be kept inside factories.
//
//  Due to intertwined nature of Vulkan API, they can only be released from factories after nearly
//  all levels of info are filled. Also, the public side of non-factory classes merely serves as an interface for inspection.
// For factory classes, things are layered. Again constrained by Vulkan's hierarchy, higher-level factories have to hold
// handles of lower-level factory instances. Factory classes should only have methods that set information 'known' at
// their respective layers. Which means there will be lots of things in the top few levels.
// Giving out usable Vulkan API information shouldn't be factories' job. (By usable we mean no modification needed)
class ShaderBase {
public:
    using BindIdx = factory::BindIdx;
    using ShaderDescriptorInfo = factory::ShaderDescriptorInfo;
    vk::ShaderModule getShaderModule(){
        if (!shaderModule_.get()){
            auto [result, shaderModule] = device_.createShaderModuleUnique({vk::ShaderModuleCreateFlags(),
                                                                            irCode_.size()*sizeof(decltype(irCode_)::value_type),
                                                                            irCode_.data()});
            utils::vk_ensure(result, "create_shader_module() failed");
            shaderModule_ = std::move(shaderModule);
        }
        return shaderModule_.get();
    }
    // Methods for getting usable Vulkan API info
    [[nodiscard]] const std::string& getEntryPointName() const{
        return entryPoint_;
    }
    // Methods for propagating info to higher levels
    [[nodiscard]] const std::unordered_map<std::string, ShaderDescriptorInfo>& propagateDescriptors() const{
        return descriptors_;
    }
    [[nodiscard]] const factory::PushConstantInfo& propagatePushConstantBlock() const{
        return pushConst_;
    }
    explicit ShaderBase(vk::Device device){
        device_ = device;
    }
    ShaderBase() = default;
protected:
    // Incomplete info, will be propagated to higher levels
    std::unordered_map<std::string, ShaderDescriptorInfo> descriptors_{};
    factory::PushConstantInfo pushConst_{};
    // Info modified by current level factories only
    std::vector<uint32_t> irCode_{};
    std::string name_{};
    std::string entryPoint_{};
private:
    vk::Device device_{};
    vk::UniqueShaderModule shaderModule_{};
};

class VertexShaderBase : public ShaderBase {
public:
    using VertexInputAttrParsedInfo = factory::VertexInputAttrParsedInfo;
    using VertexInputBindingTable = factory::VertexInputBindingTable;
    // Methods for getting usable Vulkan API info
    [[nodiscard]] vk::PipelineInputAssemblyStateCreateInfo getInputAssemblyInfo() const{
        return inputAsmInfo_;
    }
    [[nodiscard]] vk::PipelineRasterizationStateCreateInfo getRasterizationInfo() const{
        return rasterizationInfo_;
    }
    // For Vulkan API info that involves pointers, we introduce a two-step process: first store getFooData(), then call getFoo()
    // The type of valued return by getFooData() is implementation-defined.
    [[nodiscard]] std::tuple<vk::PipelineVertexInputStateCreateInfo,
            std::vector<vk::VertexInputBindingDescription>, std::vector<vk::VertexInputAttributeDescription>> getVertexInputInfoData() const{
        vk::PipelineVertexInputStateCreateInfo inputInfo{};
        std::vector<vk::VertexInputBindingDescription> inputTables{inputBindTables_.size()};
        std::vector<vk::VertexInputAttributeDescription> inputAttrs{};
        for (const auto& table: inputBindTables_){
            inputTables[table.first] = table.second.bindingDescription;
            for (const auto& attr: table.second.attributesDescription){
                inputAttrs.push_back(attr);
            }
        }
        return std::make_tuple(inputInfo, inputTables, inputAttrs);
    }
    static vk::PipelineVertexInputStateCreateInfo getVertexInputInfo(
            const std::tuple<vk::PipelineVertexInputStateCreateInfo,
                    std::vector<vk::VertexInputBindingDescription>, std::vector<vk::VertexInputAttributeDescription>>& data){
        auto result = std::get<0>(data);
        result.setVertexBindingDescriptions(std::get<1>(data));
        result.setVertexAttributeDescriptions(std::get<2>(data));
        return result;
    }
    // Methods for propagating info to higher levels
    [[nodiscard]] const std::map<BindIdx, VertexInputBindingTable>& propagateInputBindTables() const{
        return inputBindTables_;
    }
    explicit VertexShaderBase(vk::Device device) : ShaderBase(device) {}
    VertexShaderBase() = default;
protected:
    // Incomplete info, will be propagated to higher levels
    std::map<BindIdx, VertexInputBindingTable> inputBindTables_{};
    // Info modified by current level factories only
    std::unordered_map<std::string, VertexInputAttrParsedInfo> parsedVertexInputInfo_{};
    vk::PipelineInputAssemblyStateCreateInfo inputAsmInfo_{};
    vk::PipelineRasterizationStateCreateInfo rasterizationInfo_{};
private:
    friend class VertexShaderFactory;
};

class FragmentShaderBase : public ShaderBase {
public:
    using LocIdx = factory::LocIdx;
    using AttachmentReferenceInfo = factory::AttachmentReferenceInfo;
    // Methods for getting usable Vulkan API info
    [[nodiscard]] vk::PipelineDepthStencilStateCreateInfo getDepthStencilInfo() const{
        return depthStencilInfo_;
    }
    [[nodiscard]] vk::PipelineMultisampleStateCreateInfo getMSAAInfo() const {
        return msaaInfo_;
    }
    // Methods for propagating info to higher levels
    [[nodiscard]] const std::unordered_map<std::string, AttachmentReferenceInfo>& propagateAttachmentInfo() const{
        return attachmentInfo_;
    }
    [[nodiscard]] vk::PipelineColorBlendStateCreateInfo propagateColorBlendInfo() const{
        return colorBlendInfo_;
    }
    explicit FragmentShaderBase(vk::Device device): ShaderBase(device) {}
    FragmentShaderBase() = default;
protected:
    // Incomplete info, will be propagated to higher levels
    std::unordered_map<std::string, AttachmentReferenceInfo> attachmentInfo_{};
    vk::PipelineColorBlendStateCreateInfo colorBlendInfo_{};
    // Info modified by current level factories only
    vk::PipelineDepthStencilStateCreateInfo depthStencilInfo_{};
    vk::PipelineMultisampleStateCreateInfo msaaInfo_{};
private:
    friend class FragmentShaderFactory;
};

class FragmentShaderFactory {
public:
    using ShaderIdx = factory::ShaderIdx;
    using DescSetIdx = factory::DescSetIdx;
    using BindIdx = factory::BindIdx;
    using LocIdx = factory::LocIdx;

    explicit FragmentShaderFactory(vk::Device device) {
        device_ = device;
    }

    ShaderIdx registerShader(const std::string_view shaderName) {
        ShaderIdx index = shaders_.empty() ? 0 : shaders_.rbegin()->first + 1;
        FragmentShaderBase shader{device_};
        shader.name_ = shaderName;
        std::string nameStr{shaderName};
        shaderLUT_[nameStr] = index;
        shaders_.emplace(std::make_pair(index, std::move(shader)));
        return index;
    }

    [[nodiscard]] FragmentShaderBase popShader(ShaderIdx index) {
        if (shaders_.contains(index)) {
            auto result = FragmentShaderBase{};
            std::swap(shaders_.at(index), result);
            shaders_.erase(index);
            shaderLUT_.erase(result.name_);
            return result;
        }
        std::abort();
    }

    [[nodiscard]] const FragmentShaderBase& propagateShader(ShaderIdx index) const{
        return shaders_.at(index);
    }

    void loadShaderModule(ShaderIdx shaderIdx, const std::string_view filePath, const std::string &entryPointName) {
        auto &shader = shaders_.at(shaderIdx);
        shader.irCode_ = utils::load_shader_byte_code(filePath);
        shader.entryPoint_ = entryPointName;
        shader.descriptors_ = factory::parse_descriptors(shader.irCode_, vk::ShaderStageFlagBits::eFragment);
        shader.attachmentInfo_ = parseAttachments(shader);
        shader.pushConst_ = factory::parse_push_constants(shader.irCode_, vk::ShaderStageFlagBits::eFragment);
    }

    void setDepthStencilInfo(ShaderIdx shaderIdx){
        // Hard-code for now
        auto &shader = shaders_.at(shaderIdx);
        shader.depthStencilInfo_.depthTestEnable = true;
        shader.depthStencilInfo_.depthWriteEnable = true;
        shader.depthStencilInfo_.depthCompareOp = vk::CompareOp::eGreater;
        shader.depthStencilInfo_.depthBoundsTestEnable = false;
        shader.depthStencilInfo_.stencilTestEnable = false;
        // Handle the implicit depth attachment
        if (shader.depthStencilInfo_.depthTestEnable & shader.depthStencilInfo_.depthWriteEnable){
            AttachmentReferenceInfo attachInfo{};
            attachInfo.attachType = AttachmentType::eDepthStencil;
            attachInfo.attachRef.layout = vk::ImageLayout::eDepthAttachmentOptimal;
            attachInfo.attachRef.aspectMask = vk::ImageAspectFlagBits::eDepth;
            shader.attachmentInfo_["__depthShaderVariable"] = attachInfo;
        } else{
            if (shader.attachmentInfo_.contains("__depthShaderVariable")){
                shader.attachmentInfo_.erase("__depthShaderVariable");
            }
        }
    }

    void setAttachmentProperties(ShaderIdx shaderIdx, const std::string& variableName,
                                 vk::ImageAspectFlags aspects, vk::ImageLayout layout){
        auto& shader = shaders_.at(shaderIdx);
        auto &attach = shader.attachmentInfo_.at(variableName);
        attach.attachRef.aspectMask = aspects;
        attach.attachRef.layout = layout;
    }

    void setColorAttachmentBlendInfo(ShaderIdx shaderIdx, const std::string& variableName){
        auto& shader = shaders_.at(shaderIdx);
        auto &attach = shader.attachmentInfo_.at(variableName);
        assert(attach.attachType == factory::AttachmentType::eColor);
        attach.blendInfo.blendEnable = false;
        attach.blendInfo.colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA;
    }

    void setColorBlendInfo(ShaderIdx shaderIdx){
        // Hard-code for now
        auto &shader = shaders_.at(shaderIdx);
        shader.colorBlendInfo_.logicOpEnable = false;
        shader.colorBlendInfo_.logicOp = vk::LogicOp::eCopy;
        shader.colorBlendInfo_.blendConstants = {{ 0.0f,0.0f,0.0f,0.0f }};
    }

    void setMSAAInfo(ShaderIdx shaderIdx){
        // Hard-code for now
        auto &shader = shaders_.at(shaderIdx);
        shader.msaaInfo_.sampleShadingEnable = false;
        shader.msaaInfo_.rasterizationSamples = vk::SampleCountFlagBits::e1;
    }

    vk::ShaderModule getShaderModule(ShaderIdx shaderIdx){
        auto &shader = shaders_.at(shaderIdx);
        return shader.getShaderModule();
    }

private:
    vk::Device device_{};
    std::unordered_map<std::string, ShaderIdx> shaderLUT_{};
    std::map<ShaderIdx, FragmentShaderBase> shaders_{};
    // Very little Vulkan API related info can be parsed from shader code
    // We don't allow name-aliasing here
    using AttachmentReferenceInfo = factory::AttachmentReferenceInfo;
    using AttachmentType = factory::AttachmentType;
    static std::unordered_map<std::string, AttachmentReferenceInfo> parseAttachments(FragmentShaderBase &shader){
        auto glsl = spirv_cross::CompilerGLSL(shader.irCode_);
        std::unordered_map<std::string, AttachmentReferenceInfo> attaches{};
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();
        // We ignore MSAA-related attachments for now.
        for (auto &inputAttach: resources.subpass_inputs){
            AttachmentReferenceInfo attachInfo{};
            attachInfo.attachType = AttachmentType::eInput;
            attachInfo.inputAttachIdx = glsl.get_decoration(inputAttach.id, spv::DecorationInputAttachmentIndex);
            // These has to be filled by shader's user, can be color or depth
            attachInfo.attachRef.layout = vk::ImageLayout::eAttachmentOptimal;
            attachInfo.attachRef.aspectMask = vk::ImageAspectFlagBits::eNone;
            // Input attachment needs descriptor as well
            assert(shader.descriptors_.contains(inputAttach.name));
            attaches[inputAttach.name] = attachInfo;
        }
        for (auto &colorAttach: resources.stage_outputs){
            AttachmentReferenceInfo attachInfo{};
            attachInfo.attachType = AttachmentType::eColor;
            attachInfo.attachRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
            attachInfo.attachRef.aspectMask = vk::ImageAspectFlagBits::eColor;
            attachInfo.location = glsl.get_decoration(colorAttach.id, spv::DecorationLocation);
            attaches[colorAttach.name] = attachInfo;
        }
        // Depth attachment is implicit. Handled at setDepthStencilInfo()
        return attaches;
    };
};

/* One possible use pattern:

    auto vertexShaderFactory = VertexShaderFactory{device.get()};
    auto testShaderIdx = vertexShaderFactory.registerShader("test");
    vertexShaderFactory.loadShaderModule(testShaderIdx, "shaders/shader.vert.spv", "main");
    std::vector<std::string> testVertAttrs = {"inPosition", "inColor", "inTexCoord"};
    vertexShaderFactory.setVertexInputAttributeTable<model_info::PCTVertex>(testShaderIdx, vk::VertexInputRate::eVertex, 0,
                                                                            testVertAttrs, "foo_ModelGeometry");
    vertexShaderFactory.setInputAssemblyState(testShaderIdx, vk::PrimitiveTopology::eTriangleList);
    vertexShaderFactory.setRasterizationInfo(testShaderIdx);

 * */
class VertexShaderFactory {
public:
    using ShaderIdx = factory::ShaderIdx;
    using DescSetIdx = factory::DescSetIdx;
    using BindIdx = factory::BindIdx;

    explicit VertexShaderFactory(vk::Device device) {
        device_ = device;
    }

    ShaderIdx registerShader(const std::string_view shaderName) {
        ShaderIdx index = shaders_.empty() ? 0 : shaders_.rbegin()->first + 1;
        VertexShaderBase shader{device_};
        shader.name_ = shaderName;
        std::string nameStr{shaderName};
        shaderLUT_[nameStr] = index;
        shaders_.emplace(std::make_pair(index, std::move(shader)));
        return index;
    }

    // Do we even need this?
    [[nodiscard]] VertexShaderBase popShader(ShaderIdx index) {
        if (shaders_.contains(index)) {
            auto result = VertexShaderBase{};
            std::swap(shaders_.at(index), result);
            shaders_.erase(index);
            shaderLUT_.erase(result.name_);
            return result;
        }
    }
    [[nodiscard]] const VertexShaderBase& propagateShader(ShaderIdx index) const{
        return shaders_.at(index);
    }

    void loadShaderModule(ShaderIdx shaderIdx, const std::string_view filePath, const std::string &entryPointName) {
        auto &shader = shaders_.at(shaderIdx);
        shader.irCode_ = utils::load_shader_byte_code(filePath);
        shader.entryPoint_ = entryPointName;
        shader.descriptors_ = factory::parse_descriptors(shader.irCode_, vk::ShaderStageFlagBits::eVertex);
        shader.parsedVertexInputInfo_ = parseVertexInputAttrs(shader);
        shader.pushConst_ = factory::parse_push_constants(shader.irCode_, vk::ShaderStageFlagBits::eVertex);
    }

    void setRasterizationInfo(ShaderIdx shaderIdx){
        // Hard-code for now
        auto &shader = shaders_.at(shaderIdx);
        shader.rasterizationInfo_.depthClampEnable = false;
        shader.rasterizationInfo_.rasterizerDiscardEnable = false;
        shader.rasterizationInfo_.polygonMode = vk::PolygonMode::eFill;
        shader.rasterizationInfo_.lineWidth = 1.0f;
        shader.rasterizationInfo_.cullMode = vk::CullModeFlagBits::eBack;
        shader.rasterizationInfo_.frontFace = vk::FrontFace::eCounterClockwise;
        shader.rasterizationInfo_.depthBiasEnable = false;
    }

    template<class BindType>
    void setVertexInputAttributeTable(ShaderIdx shaderIdx,
                                       vk::VertexInputRate rate, BindIdx binding,
                                       const std::span<const std::string> attrNames, const std::string &resourceName) {
        auto &shader = shaders_.at(shaderIdx);
        auto &inputInfo = shader.parsedVertexInputInfo_;
        auto numAttrs = attrNames.size();
        assert(utils::meta_trick::member_count<BindType>() == numAttrs);
        vk::VertexInputBindingDescription bindTable{};
        bindTable.binding = binding;
        bindTable.stride = sizeof(BindType);
        bindTable.inputRate = rate;
        std::vector<vk::VertexInputAttributeDescription> tableAttrs{};
        uint32_t byteOffset = 0;
        for (const auto &name: attrNames) {
            vk::VertexInputAttributeDescription attr{};
            attr.binding = binding;
            attr.location = inputInfo.at(name).startLocation;
            attr.offset = byteOffset;
            attr.format = inputInfo.at(name).format;
            byteOffset += utils::get_sizeof_vk_format(inputInfo.at(name).format) * inputInfo.at(name).multiplier;
            tableAttrs.push_back(attr);
        }
        assert(bindTable.stride == byteOffset);
        shader.inputBindTables_[binding].resName = resourceName;
        shader.inputBindTables_[binding].bindingDescription = bindTable;
        shader.inputBindTables_[binding].attributesDescription = tableAttrs;
    }

    void setInputAssemblyState(ShaderIdx shaderIdx, vk::PrimitiveTopology topology) {
        auto &shader = shaders_.at(shaderIdx);
        shader.inputAsmInfo_.flags = {}; // Reserved
        shader.inputAsmInfo_.topology = topology;
        shader.inputAsmInfo_.primitiveRestartEnable = false; // Future
    }

    vk::ShaderModule getShaderModule(ShaderIdx shaderIdx){
        auto &shader = shaders_.at(shaderIdx);
        return shader.getShaderModule();
    }

private:
    vk::Device device_{};
    std::unordered_map<std::string, ShaderIdx> shaderLUT_{};
    std::map<ShaderIdx, VertexShaderBase> shaders_{};
    using VertexInputAttrParsedInfo = VertexShaderBase::VertexInputAttrParsedInfo;

    static std::unordered_map<std::string, VertexInputAttrParsedInfo> parseVertexInputAttrs(VertexShaderBase &shader) {
        auto glsl = spirv_cross::CompilerGLSL(shader.irCode_);
        std::unordered_map<std::string, VertexInputAttrParsedInfo> attrs{};
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();
        for (auto &vertInput: resources.stage_inputs) {
            VertexInputAttrParsedInfo attrParsedInfo{};
            // Only location is determined by shader code.
            attrParsedInfo.startLocation = glsl.get_decoration(vertInput.id, spv::DecorationLocation);
            std::tie(
                    attrParsedInfo.multiplier,
                    attrParsedInfo.format,
                    attrParsedInfo.fmtBaseCompByteSize) = factory::get_resource_format(glsl, vertInput);
            attrs[vertInput.name] = attrParsedInfo;
        }
        return attrs;
    }
};

#endif //VKLEARN_SHADER_MODULES_HPP
