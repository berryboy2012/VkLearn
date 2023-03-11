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
// Push constants seems to be slower than UBO
struct ScenePushConstants {
    glm::mat4 view;
    glm::mat4 proj;
};
struct ModelUBO {
    glm::mat4 model;
};
struct SubpassAttachmentReferences{
    //TODO: add support for preserveAttachments
    std::vector<vk::AttachmentReference2> inputAttachments;
    std::vector<vk::AttachmentReference2> colorAttachments;
    std::vector<vk::AttachmentReference2> resolveAttachments;
    vk::AttachmentReference2 depthStencilAttachment;
};
class VertexShader{
public:
    vk::UniqueShaderModule shaderModule_;
    typedef size_t DescSetIdx;
    std::unordered_map<DescSetIdx, std::vector<vk::DescriptorSetLayoutBinding>> descLayouts_{};
    // Specific to vertex shaders
    std::vector<vk::VertexInputBindingDescription> inputInfos_{};
    std::vector<vk::VertexInputAttributeDescription> attrInfos_{};
    vk::PipelineInputAssemblyStateCreateInfo inputAsmInfo_{};
    // Other less frequent entries
    std::vector<vk::PushConstantRange> pushConstInfos_{};
    VertexShader() = default;
    VertexShader(const VertexShader &) = delete;
    VertexShader& operator= (const VertexShader &) = delete;
    VertexShader(VertexShader &&other) noexcept {
        *this = std::move(other);
    }
    VertexShader& operator= (VertexShader &&other) noexcept {
        if (this != &other) [[likely]]{
            shaderModule_ = std::move(other.shaderModule_);
            descLayouts_ = other.descLayouts_;
            inputInfos_ = other.inputInfos_;
            attrInfos_ = other.attrInfos_;
            inputAsmInfo_ = other.inputAsmInfo_;
            pushConstInfos_ = other.pushConstInfos_;
            device_ = other.device_;
        }
        return *this;
    }
    explicit VertexShader(vk::Device device){
        std::string filePath = "shaders/shader.vert.spv";
        device_ = device;
        shaderModule_ = utils::createShaderModule(filePath, device_);
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
                inPositionInfo.format = utils::glmTypeToFormat<glm::vec3>();
                inPositionInfo.offset = 0;

                vk::VertexInputAttributeDescription inColorInfo{};
                inColorInfo.binding = bindTableInfo.binding;
                inColorInfo.location = 1;
                inColorInfo.format = utils::glmTypeToFormat<glm::vec3>();
                inColorInfo.offset = inPositionInfo.offset + static_cast<uint32_t>(utils::getSizeofVkFormat(inPositionInfo.format));

                vk::VertexInputAttributeDescription inTexCoordInfo{};
                inTexCoordInfo.binding = bindTableInfo.binding;
                inTexCoordInfo.location = 2;
                inTexCoordInfo.format = utils::glmTypeToFormat<glm::vec2>();
                inTexCoordInfo.offset = inColorInfo.offset + static_cast<uint32_t>(utils::getSizeofVkFormat(inColorInfo.format));
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
struct AttachmentInfo{
    // renderpass
    vk::AttachmentDescription2 description{};
    // framebuffer and image
    vk::Format format{};
    vk::ImageCreateFlags flags{};
    vk::ImageUsageFlags usage{};
    vk::ImageTiling tiling{};
    vk::ImageLayout layout{};
    vma::AllocationCreateFlags vmaFlag{};
    // image view
    vk::ImageAspectFlags aspect{};
    // TODO: Corresponding ID in resource manager
    std::string resId{};
};
class FragShader{
public:
    vk::UniqueShaderModule shaderModule_;
    typedef size_t DescSetIdx;
    typedef size_t AttachIdx;
    std::unordered_map<DescSetIdx, std::vector<vk::DescriptorSetLayoutBinding>> descLayouts_{};
    // Mandatory for fragment shaders
    std::unordered_map<AttachIdx, AttachmentInfo> attachmentResourceInfos_{};
    SubpassAttachmentReferences attachmentReferences_{};
    // Other less frequent entries

    FragShader() = default;
    FragShader(const FragShader &) = delete;
    FragShader& operator= (const FragShader &) = delete;
    FragShader(FragShader &&other) noexcept {
        *this = std::move(other);
    }
    FragShader& operator= (FragShader &&other) noexcept {
        if (this != &other) [[likely]]{
            shaderModule_ = std::move(other.shaderModule_);
            descLayouts_ = other.descLayouts_;
            device_ = other.device_;
            attachmentResourceInfos_ = other.attachmentResourceInfos_;
            attachmentReferences_ = other.attachmentReferences_;
        }
        return *this;
    }
    explicit FragShader(vk::Device device){
        std::string filePath = "shaders/shader.frag.spv";
        device_ = device;
        shaderModule_ = utils::createShaderModule(filePath, device_);
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
            // For swapchain images, we don't need image info
            swapchainImageAttachmentInfo.flags = {};
            swapchainImageAttachmentInfo.usage = {};
            swapchainImageAttachmentInfo.format = {};
            swapchainImageAttachmentInfo.tiling = {};
            swapchainImageAttachmentInfo.layout = {};
            swapchainImageAttachmentInfo.vmaFlag = {};
            // Image view info are not needed as well
            swapchainImageAttachmentInfo.aspect = vk::ImageAspectFlagBits::eColor;
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


#endif //VKLEARN_SHADER_MODULES_HPP
