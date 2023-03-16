//
// Created by berry on 2023/3/3.
//

#ifndef VKLEARN_SHADER_MODULES_HPP
#define VKLEARN_SHADER_MODULES_HPP
#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT_ON_RESULT

#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.hpp"
#include "utils.h"
#include "model_data.hpp"
namespace shader {
    typedef size_t ShaderIdx;
    typedef uint32_t DescSetIdx;
    typedef uint32_t BindIdx;
    typedef uint32_t LocIdx;

    struct ShaderDescriptorInfo{
        DescSetIdx set{};
        vk::DescriptorSetLayoutBinding desc{};
        size_t byteSize{};
    };

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
            entries[ubo.name] = entry;
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
            assert(entries.contains(inAttach.name));
            entry.byteSize = get_resource_byte_size(glsl, inAttach);
            entries[inAttach.name] = entry;
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
            entries[combImg.name] = entry;
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
            entries[ssbo.name] = entry;
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
            entries[acc.name] = entry;
        }
        return entries;
    }

    // the first component means a multiplier, the last component means byte size of a component inside the format (4 for R32G32B32Sfloat for example)
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

struct AttachmentInfo {
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
    std::unordered_map<AttachIdx, AttachmentInfo> attachmentResourceInfos_{};
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
            AttachmentInfo swapchainImageAttachmentInfo{};
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
            AttachmentInfo depthImageAttachmentInfo{};
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

struct VertexInputBindingTable {
    vk::VertexInputBindingDescription bindingDescription{};
    std::vector<vk::VertexInputAttributeDescription> attributesDescription{};
};

class ShaderBase {
public:
    using DescSetIdx = shader::DescSetIdx;
    using BindIdx = shader::BindIdx;
    using ShaderDescriptorInfo = shader::ShaderDescriptorInfo;
    vk::UniqueShaderModule shaderModule_{};
    std::unordered_map<std::string, ShaderDescriptorInfo> descriptors_{};
    // Other info
    std::string name_{};
    std::vector<uint32_t> irCode_{};
};

class VertexShaderBase : public ShaderBase {
public:
    // Specific to vertex shaders
    std::unordered_map<BindIdx, VertexInputBindingTable> inputBindTables_{};
    vk::PipelineInputAssemblyStateCreateInfo inputAsmInfo_{};
};

class FragmentShaderBase : public ShaderBase {
public:
    SubpassAttachmentReferences attachmentReferences_{};
};

class FragmentShaderFactory {
public:
    using ShaderIdx = shader::ShaderIdx;
    using DescSetIdx = shader::DescSetIdx;
    using BindIdx = shader::BindIdx;
    using LocIdx = shader::LocIdx;

    explicit FragmentShaderFactory(vk::Device device) {
        device_ = device;
    }

    ShaderIdx registerShader(const std::string_view shaderName) {
        ShaderIdx index = shaders_.empty() ? 0 : shaders_.rbegin()->first + 1;
        shaders_[index] = {};
        shaders_[index].name_ = shaderName;
        std::string nameStr{shaderName};
        shaderLUT_[nameStr] = index;
        return index;
    }

    [[nodiscard]] FragmentShaderBase popShader(ShaderIdx index) {
        if (shaders_.contains(index)) {
            auto result = FragmentShaderBase{};
            std::swap(shaders_.at(index), result);
            shaders_.erase(index);
            shaderLUT_.erase(result.name_);
            shaderParsedAttachmentInfo_.erase(index);
            return result;
        }
        std::abort();
    }

    void loadShaderModule(ShaderIdx shaderIdx, const std::string_view filePath) {
        auto &shader = shaders_.at(shaderIdx);
        shader.irCode_ = utils::load_shader_byte_code(filePath);
        shader.shaderModule_ = utils::create_shader_module(filePath, device_);
        shader.descriptors_ = shader::parse_descriptors(shader.irCode_, vk::ShaderStageFlagBits::eFragment);
        shaderParsedAttachmentInfo_[shaderIdx] = parseAttachments(shader);
    }

    void setAttachmentProperties(ShaderIdx shaderIdx, const std::string& variableName,
                                 vk::ImageAspectFlags aspects, vk::ImageLayout layout){
        auto &attach = shaderParsedAttachmentInfo_.at(shaderIdx).at(variableName);
        attach.attachRef.aspectMask = aspects;
        attach.attachRef.layout = layout;
    }

private:
    vk::Device device_{};
    std::unordered_map<std::string, ShaderIdx> shaderLUT_{};
    std::map<ShaderIdx, FragmentShaderBase> shaders_{};
    enum class AttachmentType {
        //TODO: add support for preserveAttachments
        eUnknown = 0,
        eInput= 1,
        eColor = 2,
        eResolve = 3,
        eDepthStencil = 4
    };
    // Polymorphism through enum, yay
    struct AttachmentParsedInfo {
        AttachmentType attachType{};
        vk::AttachmentReference2 attachRef{};
        // Input attachment only
        typedef uint32_t InAttachIdx;
        InAttachIdx inputAttachIdx{};
        // Color attachment only
        LocIdx location{};
    };
    std::map<ShaderIdx, std::unordered_map<std::string, AttachmentParsedInfo>> shaderParsedAttachmentInfo_{};
    // Very little Vulkan API related info can be parsed from shader code
    // We don't allow name-aliasing here
    static std::unordered_map<std::string, AttachmentParsedInfo> parseAttachments(FragmentShaderBase &shader){
        auto glsl = spirv_cross::CompilerGLSL(shader.irCode_);
        std::unordered_map<std::string, AttachmentParsedInfo> attaches{};
        spirv_cross::ShaderResources resources = glsl.get_shader_resources();
        // We ignore MSAA-related attachments for now.
        for (auto &inputAttach: resources.subpass_inputs){
            AttachmentParsedInfo attachInfo{};
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
            AttachmentParsedInfo attachInfo{};
            attachInfo.attachType = AttachmentType::eColor;
            attachInfo.attachRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
            attachInfo.attachRef.aspectMask = vk::ImageAspectFlagBits::eColor;
            attachInfo.location = glsl.get_decoration(colorAttach.id, spv::DecorationLocation);
            attaches[colorAttach.name] = attachInfo;
        }
        // Depth attachment is implicit.

        return attaches;
    };
};

/* One possible use pattern:

    auto shaderFactory = VertexShaderFactory{device.get()};
    auto testShaderIdx = shaderFactory.registerShader("test");
    shaderFactory.loadShaderModule(testShaderIdx, "shaders/shader.vert.spv");
    std::vector<std::string> testVertAttrs = {"inPosition", "inColor", "inTexCoord"};
    shaderFactory.bindVertexInputAttributeTable<model_info::PCTVertex>(testShaderIdx, vk::VertexInputRate::eVertex, 0,
                                                                       testVertAttrs);
 * */
class VertexShaderFactory {
public:
    using ShaderIdx = shader::ShaderIdx;
    using DescSetIdx = shader::DescSetIdx;
    using BindIdx = shader::BindIdx;

    explicit VertexShaderFactory(vk::Device device) {
        device_ = device;
    }

    ShaderIdx registerShader(const std::string_view shaderName) {
        ShaderIdx index = shaders_.empty() ? 0 : shaders_.rbegin()->first + 1;
        shaders_[index] = {};
        shaders_[index].name_ = shaderName;
        std::string nameStr{shaderName};
        shaderLUT_[nameStr] = index;
        return index;
    }

    [[nodiscard]] VertexShaderBase popShader(ShaderIdx index) {
        if (shaders_.contains(index)) {
            auto result = VertexShaderBase{};
            std::swap(shaders_.at(index), result);
            shaders_.erase(index);
            shaderParsedVertexInputInfo_.erase(index);
            shaderLUT_.erase(result.name_);
            return result;
        }
    }

    void loadShaderModule(ShaderIdx shaderIdx, const std::string_view filePath) {
        auto &shader = shaders_.at(shaderIdx);
        shader.irCode_ = utils::load_shader_byte_code(filePath);
        shader.shaderModule_ = utils::create_shader_module(filePath, device_);
        shader.descriptors_ = shader::parse_descriptors(shader.irCode_, vk::ShaderStageFlagBits::eVertex);
        shaderParsedVertexInputInfo_[shaderIdx] = parseVertexInputAttrs(shader);
    }

    template<class BindType>
    void bindVertexInputAttributeTable(ShaderIdx shaderIdx,
                                       vk::VertexInputRate rate, BindIdx binding,
                                       const std::span<const std::string> attrNames) {
        auto &shader = shaders_.at(shaderIdx);
        auto &inputInfo = shaderParsedVertexInputInfo_.at(shaderIdx);
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
        shader.inputBindTables_[binding].bindingDescription = bindTable;
        shader.inputBindTables_[binding].attributesDescription = tableAttrs;
    }

    void setInputAssemblyState(ShaderIdx shaderIdx, vk::PrimitiveTopology topology) {
        auto &shader = shaders_.at(shaderIdx);
        shader.inputAsmInfo_.flags = {}; // Reserved
        shader.inputAsmInfo_.topology = topology;
        shader.inputAsmInfo_.primitiveRestartEnable = false; // Future
    }

private:
    vk::Device device_{};
    std::unordered_map<std::string, ShaderIdx> shaderLUT_{};
    std::map<ShaderIdx, VertexShaderBase> shaders_{};
    struct VertexInputAttrParsedInfo {
        // We need more info to deal with GLSL's unintuitive rules
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap22.html#fxvertex-attrib-location
        uint32_t startLocation;
        uint32_t multiplier;
        vk::Format format;
        uint32_t fmtBaseCompByteSize;
    };
    std::map<ShaderIdx, std::unordered_map<std::string, VertexInputAttrParsedInfo>> shaderParsedVertexInputInfo_{};

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
                    attrParsedInfo.fmtBaseCompByteSize) = shader::get_resource_format(glsl, vertInput);
            attrs[vertInput.name] = attrParsedInfo;
        }
        return attrs;
    }
};

#endif //VKLEARN_SHADER_MODULES_HPP
