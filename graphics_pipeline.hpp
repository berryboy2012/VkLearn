//
// Created by berry on 2023/3/5.
//
#ifndef VKLEARN_GRAPHICS_PIPELINE_HPP
#define VKLEARN_GRAPHICS_PIPELINE_HPP
struct SubpassInfo{
    typedef size_t SubpassIdx;
    SubpassIdx index;
    vk::SubpassDescription2 description;
    std::vector<vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2>> dependencies;
};
struct ViewportInfo{
    vk::Viewport viewport{};
    vk::Rect2D scissor{};
};
ViewportInfo get_viewport_info(uint32_t width, uint32_t height){
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
    vk::PipelineLayout pipelineLayout_{};
    vk::Pipeline pipeline_{};

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
    typedef size_t DescSetIdx;
    typedef uint32_t BindIdx;
    //TODO: Add proper resId to connect with handles in the resource part
    std::unordered_map<DescSetIdx, std::unordered_map<std::string, BindIdx>> descResLUT_{};
    std::unordered_map<DescSetIdx, std::vector<vk::DescriptorSetLayoutBinding>> bindings_{};

    // Info about sync between subpasses
    SubpassInfo subpassInfo_{};

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
        if (this != &other) [[likely]]{
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
            descResLUT_ = other.descResLUT_;
            bindings_ = other.bindings_;
            subpassInfo_ = setSubpassInfo();
            descLayout_ = std::move(other.descLayout_);
            pipeLayout_ = std::move(other.pipeLayout_);
            pipelineLayout_ = pipeLayout_.get();
            pipe_ = std::move(other.pipe_);
            pipeline_ = pipe_.get();
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
            for (auto &descResSet: vertShader_.descResLUT_) {
                for (auto &descResBind: descResSet.second){
                    descResLUT_[descResSet.first][descResBind.first] = descResBind.second;
                }
            }
            for (auto &descResSet: fragShader_.descResLUT_) {
                for (auto &descResBind: descResSet.second){
                    descResLUT_[descResSet.first][descResBind.first] = descResBind.second;
                }
            }
        }

        {
            for (auto &bindSet: vertShader_.descLayouts_) {
                for (auto &bind: bindSet.second){
                    bindings_[bindSet.first].push_back(bind);
                }
            }
            for (auto &bindSet: fragShader_.descLayouts_) {
                for (auto &bind: bindSet.second){
                    bindings_[bindSet.first].push_back(bind);
                }
            }
        }

        // filling subpass info, only need to fill the dependencies that this pipeline is waiting for
        subpassInfo_ = setSubpassInfo();

        // Maybe it is not a good time to create vk::PipelineLayout object right now, since descriptor set are not trivial to manage.

        // We cannot create vk::Pipeline object for now. Since renderpass and subpass are higher-level concepts.
    }
    vk::DescriptorSetLayoutBinding queryDescriptorSetLayoutBinding(DescSetIdx set, const std::string &resId){
        auto bindingIndex = descResLUT_.at(set).at(resId);
        for (const auto& binding:bindings_.at(set)){
            if (binding.binding == bindingIndex){
                return binding;
            }
        }
        std::abort();
    }

    vk::DescriptorSetLayout getDescriptorLayout(DescSetIdx index){
        if (!descLayout_.contains(index)){

            descLayout_[index] = utils::create_descriptor_set_layout(bindings_.at(index), device_);
        }
        return descLayout_[index].get();
    }

    vk::PipelineLayout createPipelineLayout(std::span<const DescSetIdx> descriptorSetIndices){
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        std::vector<vk::DescriptorSetLayout> layouts{};
        for (auto& index: descriptorSetIndices){
            layouts.push_back(getDescriptorLayout(index));
        }
        pipelineLayoutInfo.setSetLayouts(layouts);
        pipelineLayoutInfo.setPushConstantRanges(vertShader_.pushConstInfos_);
        pipelineLayoutInfo.flags = {};

        auto [pLResult, pipelineLayout] = device_.createPipelineLayoutUnique(pipelineLayoutInfo);
        utils::vk_ensure(pLResult);
        pipeLayout_ = std::move(pipelineLayout);
        pipelineLayout_ = pipeLayout_.get();
        return pipelineLayout_;
    }

    // Create pipeline layout with all descriptor set layouts.
    vk::PipelineLayout createPipelineLayout(){
        std::vector<DescSetIdx> layoutIndices{};
        for (auto& setBind: bindings_){
            layoutIndices.push_back(setBind.first);
        }
        return createPipelineLayout(layoutIndices);
    }

    vk::Pipeline createPipeline(vk::RenderPass renderPass){
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
        pipelineInfo.subpass = subpassInfo_.index;
        pipelineInfo.basePipelineHandle = nullptr;

        auto [graphPipeResult, graphicsPipeline] = device_.createGraphicsPipelineUnique(nullptr, pipelineInfo);
        utils::vk_ensure(graphPipeResult);
        pipe_ = std::move(graphicsPipeline);
        pipeline_ = pipe_.get();
        return pipeline_;
    }

private:
    SubpassInfo setSubpassInfo() {
        vk::SubpassDescription2 description{};
        description.flags = {};
        description.colorAttachmentCount = fragShader_.attachmentReferences_.colorAttachments.size();
        description.pColorAttachments = fragShader_.attachmentReferences_.colorAttachments.data();
        description.inputAttachmentCount = fragShader_.attachmentReferences_.inputAttachments.size();
        description.pInputAttachments = fragShader_.attachmentReferences_.inputAttachments.data();
        description.pResolveAttachments = fragShader_.attachmentReferences_.resolveAttachments.data();
        description.pDepthStencilAttachment = &fragShader_.attachmentReferences_.depthStencilAttachment;
        SubpassInfo info{};
        info.description = description;
        // Index for this subpass
        info.index = 0;
        // Dependency for this subpass. There are some degree of freedom when writing sync rules.
        // Refer to https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples for inspirations.
        info.dependencies = {};
        {
            // Future-proof when we have multiple renderpass per swapchain frame
            vk::SubpassDependency2 swapchainImageBeforeRender{};
            // ViewLocalBit when no inter-view access needed, ByRegionBit when no inter-fragment access needed.
            swapchainImageBeforeRender.dependencyFlags = vk::DependencyFlagBits::eByRegion;//vk::DependencyFlagBits::eViewLocal|;
            //VUID-VkSubpassDependency2-dependencyFlags-03090/03091
            // VK_SUBPASS_EXTERNAL means anything happened before beginning renderpass scope
            swapchainImageBeforeRender.srcSubpass = VK_SUBPASS_EXTERNAL;
            swapchainImageBeforeRender.dstSubpass = 0;
            // For fine-grained src and dst specification.
            vk::MemoryBarrier2 beforeRenderBarrier{};
            beforeRenderBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            beforeRenderBarrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            beforeRenderBarrier.srcAccessMask = vk::AccessFlagBits2::eNone;
            beforeRenderBarrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2> beforeRenderDep{
                    swapchainImageBeforeRender,
                    beforeRenderBarrier
            };
            // Right now only one renderpass per swapchain frame, no need to include it.
            //info.dependencies.push_back(beforeRenderDep);

            // Can be redundant if the following renderpass properly wait before accessing the swapchain image.
            // In this demonstration, the next renderpass is responsible for drawing UI and gaussian-filtering FX.
            // The swapchain image is used as its input attachment.
            vk::SubpassDependency2 swapchainImageAfterRender{};
            swapchainImageAfterRender.srcSubpass = 0;
            swapchainImageAfterRender.dstSubpass = VK_SUBPASS_EXTERNAL;
            vk::MemoryBarrier2 afterRenderBarrier{};
            afterRenderBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            afterRenderBarrier.dstStageMask = vk::PipelineStageFlagBits2::eAllTransfer|vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            afterRenderBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
            afterRenderBarrier.dstAccessMask = vk::AccessFlagBits2::eNone;
            vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2> afterRenderDep{
                    swapchainImageAfterRender,
                    afterRenderBarrier
            };
            // Right now only one renderpass per swapchain frame, no need to include it.
            //info.dependencies.push_back(afterRenderDep);
        }
        return info;
    }
    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages_{};
    std::unordered_map<DescSetIdx, vk::UniqueDescriptorSetLayout> descLayout_{};
    vk::UniquePipelineLayout pipeLayout_{};
    vk::UniquePipeline pipe_{};
    vk::Device device_{};
};

// Also helps building a pipeline
// The twisted Vulkan API requires that a Pipeline object must be tied to a Renderpass object upon creation
// Thus the creation of vk::Pipeline object has to be postponed to much later
class SubpassFactory{
public:
    explicit SubpassFactory(vk::Device device){
        device_ = device;
    }
/* Things needed to be done in terms of pipeline:
    VkPipelineCreateFlags                            flags;
    // Shader related
    uint32_t                                         stageCount;
    const VkPipelineShaderStageCreateInfo*           pStages;
    // Provided by vertex shader
    const VkPipelineVertexInputStateCreateInfo*      pVertexInputState;
    const VkPipelineInputAssemblyStateCreateInfo*    pInputAssemblyState;
    // No tessel shader for now
    const VkPipelineTessellationStateCreateInfo*     pTessellationState;
    // Renderpass level, as it shares props with framebuffer
    const VkPipelineViewportStateCreateInfo*         pViewportState;
    // Provided by vertex shader
    const VkPipelineRasterizationStateCreateInfo*    pRasterizationState;
    // No MSAA for now
    const VkPipelineMultisampleStateCreateInfo*      pMultisampleState;
    // Fragment shader level
    const VkPipelineDepthStencilStateCreateInfo*     pDepthStencilState;
    const VkPipelineColorBlendStateCreateInfo*       pColorBlendState;
    // Subpass level
    const VkPipelineDynamicStateCreateInfo*          pDynamicState;
    VkPipelineLayout                                 layout;
    // Renderpass level
    VkRenderPass                                     renderPass;
    uint32_t                                         subpass;
    // Don't implement for now
    VkPipeline                                       basePipelineHandle;
    int32_t                                          basePipelineIndex;
 * */


/* For a subpass:
 *
 * */
private:
    vk::Device device_{};
};
#endif //VKLEARN_GRAPHICS_PIPELINE_HPP
