//
// Created by berry on 2023/3/5.
//
#ifndef VKLEARN_GRAPHICS_PIPELINE_HPP
#define VKLEARN_GRAPHICS_PIPELINE_HPP
struct ViewportInfo{
    vk::Viewport viewport{};
    vk::Rect2D scissor{};
};
ViewportInfo getViewportInfo(uint32_t width, uint32_t height){
    ViewportInfo result{};
    result.viewport.x = 0.0f;
    result.viewport.y = 0.0f;
    result.viewport.width = (float)width;
    result.viewport.height = (float)height;
    result.viewport.minDepth = 0.0f;
    result.viewport.maxDepth = 1.0f;

    result.scissor.offset = {{0, 0}};
    result.scissor.extent = {{width, height}};
    return result;
}
template<class VS, class FS>
class GraphicsPipeline{
public:
    VS vertShader_;
    FS fragShader_;

    // Additional info required by pipeline
    ViewportInfo viewport_;
    std::vector<vk::PipelineColorBlendAttachmentState> colorInfos_{};

    vk::PipelineVertexInputStateCreateInfo vertInputInfo_{};
    vk::PipelineInputAssemblyStateCreateInfo inputAsmInfo_{};
    // Vertex shading is programmable, no info needed
    vk::PipelineTessellationStateCreateInfo tesselInfo_{};
    // Geometry shading is programmable, no info needed
    vk::PipelineViewportStateCreateInfo viewportInfo_{};
    vk::PipelineDepthStencilStateCreateInfo depthInfo_{};
    vk::PipelineMultisampleStateCreateInfo msaaInfo_{};
    vk::PipelineRasterizationStateCreateInfo rastInfo_{};
    // Fragment shading is programmable, no info needed
    vk::PipelineColorBlendStateCreateInfo blendInfo_{};

    // Info needed to build descriptor set
    std::vector<vk::DescriptorSetLayoutBinding> bindings_{};

    static vk::PipelineVertexInputStateCreateInfo setVertexInputStateInfo(
            const std::span<const vk::VertexInputBindingDescription> binds,
            const std::span<const vk::VertexInputAttributeDescription> attrs){
        vk::PipelineVertexInputStateCreateInfo result{};
        result.setVertexBindingDescriptions(binds);
        result.setVertexAttributeDescriptions(attrs);
        return result;
    }
    static vk::PipelineViewportStateCreateInfo setViewportStateInfo(ViewportInfo &view){
        vk::PipelineViewportStateCreateInfo result{};
        result.setViewports(view.viewport);
        result.setScissors(view.scissor);
        return result;
    }
    static vk::PipelineColorBlendStateCreateInfo setColorBlendStateInfo(
            const std::span<const vk::PipelineColorBlendAttachmentState> attaches){
        vk::PipelineColorBlendStateCreateInfo result{};
        result.logicOpEnable = VK_FALSE;
        result.logicOp = vk::LogicOp::eCopy;
        result.setAttachments(attaches);
        result.blendConstants = {{ 0.0f,0.0f,0.0f,0.0f }};
        return result;
    }
    GraphicsPipeline() = default;
    GraphicsPipeline(const GraphicsPipeline<VS, FS> &) = delete;
    GraphicsPipeline& operator= (const GraphicsPipeline<VS, FS> &) = delete;
    // This class does not make much sense for copy and move semantics (but I don't want to write singleton class):
    //  Before creating the Vulkan pipeline object, this object can't change much of its state, but some member variables
    //  are preventing the object from being trivially copyable. In practice one should just initialize another object instead.
    //  After creating the piepline object, moving a pipeline around is even stranger.
    GraphicsPipeline& operator= (GraphicsPipeline<VS, FS> &&other) noexcept {
        if (this != &other){
            device_ = other.device_;
            vertShader_ = std::move(other.vertShader_);
            fragShader_ = std::move(other.fragShader_);
            viewport_ = other.viewport_;
            vertInputInfo_ = setVertexInputStateInfo(vertShader_.inputInfos_, vertShader_.attrInfos_);
            inputAsmInfo_ = vertShader_.inputAsmInfo_;
            tesselInfo_ = other.tesselInfo_;
            viewportInfo_ = setViewportStateInfo(viewport_);
            depthInfo_ = other.depthInfo_;
            msaaInfo_ = other.msaaInfo_;
            rastInfo_ = other.rastInfo_;
            colorInfos_ = other.colorInfos_;
            blendInfo_ = setColorBlendStateInfo(colorInfos_);
            shaderStages_ = other.shaderStages_;
            bindings_ = other.bindings_;
            descLayout_ = std::move(other.descLayout_);
            pipeLayout_ = std::move(other.pipeLayout_);
            pipe_ = std::move(other.pipe_);
        }
        return *this;
    }
    GraphicsPipeline(GraphicsPipeline<VS, FS> &&other) noexcept {
        *this = std::move(other);
    }
    explicit GraphicsPipeline(vk::Device device, const ViewportInfo &viewport){
        device_ = device;
        vertShader_ = VS{device_};
        fragShader_ = FS{device_};
        shaderStages_ = {{
                                 {
                                         vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eVertex,
                                         vertShader_.shaderModule_.get(),
                                         "main"
                                 },
                                 {
                                         vk::PipelineShaderStageCreateFlags(),
                                         vk::ShaderStageFlagBits::eFragment,
                                         fragShader_.shaderModule_.get(),
                                         "main"
                                 }
                         }};
        vertInputInfo_ = setVertexInputStateInfo(vertShader_.inputInfos_, vertShader_.attrInfos_);
        inputAsmInfo_ = vertShader_.inputAsmInfo_;

        // Skip Tessellation State

        viewport_ = viewport;
        viewportInfo_ = setViewportStateInfo(viewport_);

        depthInfo_.depthTestEnable = VK_TRUE;
        depthInfo_.depthWriteEnable = VK_TRUE;
        depthInfo_.depthCompareOp = vk::CompareOp::eGreater;
        depthInfo_.depthBoundsTestEnable = VK_FALSE;
        depthInfo_.stencilTestEnable = VK_FALSE;

        msaaInfo_.sampleShadingEnable = VK_FALSE;
        msaaInfo_.rasterizationSamples = vk::SampleCountFlagBits::e1;

        rastInfo_.depthClampEnable = VK_FALSE;
        rastInfo_.rasterizerDiscardEnable = VK_FALSE;
        rastInfo_.polygonMode = vk::PolygonMode::eFill;
        rastInfo_.lineWidth = 1.0f;
        rastInfo_.cullMode = vk::CullModeFlagBits::eBack;
        rastInfo_.frontFace = vk::FrontFace::eCounterClockwise;
        rastInfo_.depthBiasEnable = VK_FALSE;

        {
            vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask =
                    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                    vk::ColorComponentFlagBits::eA;
            colorBlendAttachment.blendEnable = VK_FALSE;
            colorInfos_.push_back(colorBlendAttachment);
        }

        blendInfo_ = setColorBlendStateInfo(colorInfos_);

        {
            for (auto &bind: vertShader_.descLayouts_) {
                bindings_.push_back(bind);
            }
            for (auto &bind: fragShader_.descLayouts_) {
                bindings_.push_back(bind);
            }
        }

        // Maybe it is not a good time to create vk::PipelineLayout object right now, since descriptor set are not trivial to manage.

        // We cannot create vk::Pipeline object for now. Since renderpass and subpass are higher-level concepts.
    }

    vk::DescriptorSetLayout getDescriptorLayout(){
        if (!static_cast<bool>(descLayout_)){
            descLayout_ = utils::createDescriptorSetLayout(bindings_, device_);
        }
        return descLayout_.get();
    }

    vk::Pipeline createPipeline(vk::RenderPass renderPass, uint32_t subpass){
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setSetLayouts(getDescriptorLayout());
        pipelineLayoutInfo.setPushConstantRanges(vertShader_.pushConstInfos_);

        auto [pLResult, pipelineLayout] = device_.createPipelineLayoutUnique(pipelineLayoutInfo);
        utils::vkEnsure(pLResult);
        pipeLayout_ = std::move(pipelineLayout);

        vk::GraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.setStages(shaderStages_);
        pipelineInfo.pVertexInputState = &vertInputInfo_;
        pipelineInfo.pInputAssemblyState = &inputAsmInfo_;
        pipelineInfo.pViewportState = &viewportInfo_;
        pipelineInfo.pRasterizationState = &rastInfo_;
        pipelineInfo.pMultisampleState = &msaaInfo_;
        pipelineInfo.pDepthStencilState = &depthInfo_;
        pipelineInfo.pColorBlendState = &blendInfo_;
        pipelineInfo.layout = pipeLayout_.get();
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = subpass;
        pipelineInfo.basePipelineHandle = nullptr;

        auto [graphPipeResult, graphicsPipeline] = device_.createGraphicsPipelineUnique(nullptr, pipelineInfo);
        utils::vkEnsure(graphPipeResult);
        pipe_ = std::move(graphicsPipeline);
        return pipe_.get();
    }
private:
    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages_{};
    vk::UniqueDescriptorSetLayout descLayout_{};
    vk::UniquePipelineLayout pipeLayout_{};
    vk::UniquePipeline pipe_{};
    vk::Device device_{};
};
#endif //VKLEARN_GRAPHICS_PIPELINE_HPP
