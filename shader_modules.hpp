//
// Created by berry on 2023/3/3.
//

#ifndef VKLEARN_SHADER_MODULES_HPP
#define VKLEARN_SHADER_MODULES_HPP
#include "vulkan/vulkan.hpp"
#include "utils.hpp"
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

    explicit VertexShader(vk::Device device){
        std::string filePath = "shaders/shader.vert.spv";
        device_ = device;
        shaderModule_ = utils::createShaderModule(filePath, device_);
        // filling descriptor set layout info
        {
            vk::DescriptorSetLayoutBinding modelUBOInfo{};
            modelUBOInfo.binding = 0;
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
                inColorInfo.offset = inPositionInfo.offset + utils::sizeofVkFormat.at(inPositionInfo.format);

                vk::VertexInputAttributeDescription inTexCoordInfo{};
                inTexCoordInfo.binding = bindTableInfo.binding;
                inTexCoordInfo.location = 2;
                inTexCoordInfo.format = utils::glmTypeToFormat<glm::vec2>();
                inTexCoordInfo.offset = inColorInfo.offset + utils::sizeofVkFormat.at(inColorInfo.format);
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
