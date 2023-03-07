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
#include "utils.h"
// Push constants seems to be slower than UBO
struct ScenePushConstants {
    glm::mat4 view;
    glm::mat4 proj;
};
struct ModelUBO {
    glm::mat4 model;
};
class VertexShader{
public:
    vk::UniqueShaderModule shaderModule_;
    std::vector<vk::DescriptorSetLayoutBinding> descLayouts_{};
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
        if (this != &other){
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
            vk::DescriptorSetLayoutBinding modelUBOInfo{};
            modelUBOInfo.binding = 0;
            // If you want to specify an array of element objects, descriptorCount is the length of such array.
            modelUBOInfo.descriptorCount = 1;
            modelUBOInfo.descriptorType = vk::DescriptorType::eUniformBuffer;
            modelUBOInfo.pImmutableSamplers = nullptr;
            modelUBOInfo.stageFlags = vk::ShaderStageFlagBits::eVertex;
            descLayouts_.push_back(modelUBOInfo);
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
class FragShader{
public:
    vk::UniqueShaderModule shaderModule_;
    std::vector<vk::DescriptorSetLayoutBinding> descLayouts_{};
    // Other less frequent entries

    FragShader() = default;
    FragShader(const FragShader &) = delete;
    FragShader& operator= (const FragShader &) = delete;
    FragShader(FragShader &&other) noexcept {
        *this = std::move(other);
    }
    FragShader& operator= (FragShader &&other) noexcept {
        if (this != &other){
            shaderModule_ = std::move(other.shaderModule_);
            descLayouts_ = other.descLayouts_;
            device_ = other.device_;
        }
        return *this;
    }
    explicit FragShader(vk::Device device){
        std::string filePath = "shaders/shader.frag.spv";
        device_ = device;
        shaderModule_ = utils::createShaderModule(filePath, device_);
        // filling descriptor set layout info
        {
            vk::DescriptorSetLayoutBinding texSamplerInfo{};
            texSamplerInfo.binding = 1;
            texSamplerInfo.descriptorCount = 1;
            texSamplerInfo.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            texSamplerInfo.pImmutableSamplers = nullptr;
            texSamplerInfo.stageFlags = vk::ShaderStageFlagBits::eFragment;
            descLayouts_.push_back(texSamplerInfo);
        }
    }
private:
    vk::Device device_;
};
#endif //VKLEARN_SHADER_MODULES_HPP
