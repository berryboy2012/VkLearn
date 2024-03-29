//
// Created by berry on 2023/3/5.
//

#ifndef VKLEARN_RENDERPASS_HPP
#define VKLEARN_RENDERPASS_HPP

#include "graphics_pipeline.hpp"
#include "resource_management.hpp"

class Renderpass{
    //TODO: support pNext for related Vulkan info structs.
public:
    typedef uint32_t AttachIdx;
    typedef uint32_t SubpassIdx;
    typedef size_t DependIdx;

    vk::RenderPass renderpass_{};
    std::map<AttachIdx, AttachmentInfoBundle> attachDescs_{};
    std::vector<SubpassInfo> subpasses_{};
    std::vector<SubpassAttachmentReferences> subpassRefs_{};

    Renderpass() = default;
    Renderpass& operator= (const Renderpass &) = delete;
    Renderpass(const Renderpass &) = delete;
    Renderpass& operator= (Renderpass && other) noexcept{
        if (this != &other) [[likely]]{
            device_ = other.device_;
            attachDescs_ = other.attachDescs_;
            subpasses_ = other.subpasses_;
            subpassRefs_ = other.subpassRefs_;
            for (size_t i=0;i<subpasses_.size();i++){
                setSubpassInfo(subpasses_[i], subpassRefs_[i]);
            }
            pass_ = std::move(other.pass_);
            renderpass_ = pass_.get();
        }
        return *this;
    }
    Renderpass(Renderpass &&other) noexcept{
        *this = std::move(other);
    }
    explicit Renderpass(vk::Device device){
        device_ = device;
    }
    // Each subpass should register the following infos:
    //  a vk::SubpassDescription
    //  dependencies between other subpasses, should only provide dependencies concerning their resource loads.
    //  any new attachments it introduces, which in turn are registered by corresponding pipelines

    // First register attachments, since attachments' info is referenced by vk::SubpassDescription2
    void registerSubpassAttachment(const AttachmentInfoBundle &attach, AttachIdx index){
        attachDescs_[index] = attach;
    }

    // Register subpass before registering dependencies, as dependencies directly reference indices of subpasses.
    void registerSubpassInfo(const SubpassInfo &subpassDescription){
        subpasses_.push_back(subpassDescription);
        auto& subpassDesc = *subpasses_.rbegin();
        // Deep copy of referenced attachments
        subpassRefs_.push_back({});
        auto& subRefs = *subpassRefs_.rbegin();
        auto inputRefs = std::span<const vk::AttachmentReference2>{
            subpassDesc.description.pInputAttachments, subpassDesc.description.inputAttachmentCount};
        for (const auto& inputRef: inputRefs){
            subRefs.inputAttachments.push_back(inputRef);
        }
        // The Vulkan spec is doing some surprising things
        if (subpassDesc.description.pResolveAttachments != nullptr){
            auto resolveRefs = std::span<const vk::AttachmentReference2>{
                    subpassDesc.description.pResolveAttachments, subpassDesc.description.colorAttachmentCount};
            for (const auto& resolveRef: resolveRefs){
                subRefs.resolveAttachments.push_back(resolveRef);
            }
        } else {
            subRefs.resolveAttachments = {};
        }
        auto colorRefs = std::span<const vk::AttachmentReference2>{
                subpassDesc.description.pColorAttachments, subpassDesc.description.colorAttachmentCount};
        for (const auto& colorRef: colorRefs){
            subRefs.colorAttachments.push_back(colorRef);
        }
        subRefs.depthStencilAttachment = *subpassDesc.description.pDepthStencilAttachment;

        setSubpassInfo(subpassDesc, subRefs);
    }

    [[maybe_unused]] vk::RenderPass createRenderpass(){
        // Renderpass -- span<attachment>
        //        \\____ span<subpass>
        //         \                \____ several span<ref to attachment>: AttachIdx to index in span<attachment>
        //          \____ span<dependency>: SubpassIdx to index in span<subpass>
        //                              \____ ref to subpass: SubpassIdx to index in span<subpass>
        std::vector<vk::AttachmentDescription2> attaches{}; attaches.resize(attachDescs_.size());
        for (const auto& desc: attachDescs_){
            attaches[desc.first] = desc.second.description;
        }
        std::vector<vk::SubpassDescription2> passes{};
        for (const auto& pass: subpasses_){
            passes.push_back(pass.description);
        }
        std::vector<vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2>> depends{};
        for (const auto& pass: subpasses_){
            for (const auto& depend: pass.dependencies){
                depends.push_back(depend);
            }
        }
        auto createInfoDepends = std::vector<vk::SubpassDependency2>{};
        for (const auto& dep: depends){
            createInfoDepends.push_back(dep.get<vk::SubpassDependency2>());
        }
        vk::RenderPassCreateInfo2 renderPassInfo = {};
        renderPassInfo.setAttachments(attaches);
        renderPassInfo.setSubpasses(passes);
        renderPassInfo.dependencyCount = createInfoDepends.size();
        renderPassInfo.pDependencies = createInfoDepends.data();

        auto [result, renderPass] = device_.createRenderPass2Unique(renderPassInfo);
        utils::vk_ensure(result);
        pass_ = std::move(renderPass);
        renderpass_ = pass_.get();
        return renderpass_;
    }

    vk::UniqueFramebuffer createViewlessFramebuffer(vk::Extent2D extent){
        std::vector<vk::FramebufferAttachmentImageInfo> attachInfos{}; attachInfos.resize(attachDescs_.size());
        for (const auto& attach: attachDescs_){
            vk::FramebufferAttachmentImageInfo info{};
            info.usage = attach.second.usage;
            info.flags = attach.second.flags;
            info.height = extent.height;
            info.width = extent.width;
            info.layerCount = 1;
            info.viewFormatCount = 1;
            info.pViewFormats = &attach.second.format;
            attachInfos[attach.first] = info;
        }
        vk::FramebufferAttachmentsCreateInfo attachmentsInfo{};
        attachmentsInfo.setAttachmentImageInfos(attachInfos);
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderpass_;
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;
        framebufferInfo.flags |= vk::FramebufferCreateFlagBits::eImageless;
        framebufferInfo.attachmentCount = attachmentsInfo.attachmentImageInfoCount;
        framebufferInfo.pAttachments = VK_NULL_HANDLE;
        framebufferInfo.pNext = &attachmentsInfo;
        auto [result, fb] = device_.createFramebufferUnique(framebufferInfo);
        utils::vk_ensure(result);
        return std::move(fb);
    }
    
    vk::UniqueFramebuffer createFramebuffer(vk::Extent2D extent, std::span<const vk::ImageView> views){
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderpass_;
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;
        framebufferInfo.setAttachments(views);
        auto [result, fb] = device_.createFramebufferUnique(framebufferInfo);
        utils::vk_ensure(result);
        return std::move(fb);
    }

private:
    static void setSubpassInfo(SubpassInfo &subpassInfo, SubpassAttachmentReferences &attachRefs){
        subpassInfo.description.setInputAttachments(attachRefs.inputAttachments);
        // Only set resolve attachments if there is any, or just set them before color attachments
        subpassInfo.description.setResolveAttachments(attachRefs.resolveAttachments);
        subpassInfo.description.setColorAttachments(attachRefs.colorAttachments);
        subpassInfo.description.pDepthStencilAttachment = &attachRefs.depthStencilAttachment;
    }
    vk::Device device_{};

    vk::UniqueRenderPass pass_{};
};

class RenderpassBase{
/* Lots of things need to be done here:
 * Register subpass
 * Gather attachment references from all subpass
 *     Do not merge the info, may introduce name aliasing problems
 * Populate attachments
 *     Caller should specify most of vk::AttachmentDescription2
 *     Caller don't have control over index
 * Link attachment references to attachments
 *     trivial
 * Link attachments to resources
 *     Kind of hard to decide, first just link by string name, then we'll see how much info should be supplied
 * Add subpass dependencies
 *     trivial
 * Create a corresponding framebuffer
 *     Should use resource name to query some info (for example, attachment linked to swapchain image)
 *     For each attachment, use its resource name to query info needed during creation, but don't store those info, as
 *     resources are mutable
 *
 * Create vk::PipelineLayout, vk::Renderpass, vk::Framebuffer, vk::Pipeline
 *     Where things get really messy
 * */
public:
    explicit RenderpassBase(vk::Extent2D extent, vk::Device device, RenderResourceManager &resourceManager):
    renderExtent_(extent), device_(device), resMgr_(resourceManager){}
    [[nodiscard]] vk::RenderPass getRenderPass(){
        // Renderpass -- span<attachment>
        //        \\____ span<subpass>
        //         \                \____ several span<ref to attachment>: AttachIdx to index in span<attachment>
        //          \____ span<dependency>: SubpassIdx to index in span<subpass>
        //                              \____ ref to subpass: SubpassIdx to index in span<subpass>
        if (static_cast<bool>(renderPass_.get())){
            return renderPass_.get();
        }
        std::vector<vk::AttachmentDescription2> attaches{}; attaches.reserve(attachments_.size());
        std::unordered_map<AttachIdx, size_t> attachIdxLUT{};
        size_t attachVkIdx = 0;
        for (const auto& attach: attachments_){
            attaches.push_back(attach.second.description);
            attachIdxLUT[attach.first] = attachVkIdx;
            attachVkIdx += 1;
        }
        attachVkIdxLUT_ = attachIdxLUT;
        std::vector<vk::SubpassDescription2> passes{}; passes.reserve(subpasses_.size());
        std::unordered_map<SubpassIdx , size_t> subpassIdxLUT{};
        std::unordered_map<SubpassIdx, std::vector<vk::AttachmentReference2>> subpassesColorAttachRefs{};
        std::unordered_map<SubpassIdx, std::vector<vk::AttachmentReference2>> subpassesInputAttachRefs{};
        std::unordered_map<SubpassIdx, std::vector<vk::AttachmentReference2>> subpassesResolveAttachRefs{};
        // Depth attachment only has one element, no need to create temporary vector.
        std::unordered_map<SubpassIdx, vk::AttachmentReference2> subpassesDepthAttachRef{};
        size_t subpassVkIdx = 0;
        for (const auto& subpass: subpasses_){
            vk::SubpassDescription2 subpassInfo{};
            const auto& subpassAttachRefs = attachmentReferences_.at(subpass.first);
            for (const auto& attachRef: subpassAttachRefs){
                switch (attachRef.second.attachType) {
                    case factory::AttachmentType::eInput:
                        subpassesInputAttachRefs[subpass.first].push_back(attachRef.second.attachRef);
                        subpassesInputAttachRefs[subpass.first].rbegin()->attachment = attachIdxLUT[attachRef.second.attachRef.attachment];
                        break;
                    case factory::AttachmentType::eColor:
                        subpassesColorAttachRefs[subpass.first].push_back(attachRef.second.attachRef);
                        subpassesColorAttachRefs[subpass.first].rbegin()->attachment = attachIdxLUT[attachRef.second.attachRef.attachment];
                        break;
                    case factory::AttachmentType::eResolve:
                        subpassesResolveAttachRefs[subpass.first].push_back(attachRef.second.attachRef);
                        subpassesResolveAttachRefs[subpass.first].rbegin()->attachment = attachIdxLUT[attachRef.second.attachRef.attachment];
                        break;
                    case factory::AttachmentType::eDepthStencil:
                        subpassesDepthAttachRef[subpass.first] = attachRef.second.attachRef;
                        subpassesDepthAttachRef[subpass.first].attachment = attachIdxLUT[attachRef.second.attachRef.attachment];
                        break;
                    default:
                        assert(("must specify attachment type", false));
                }
            }
            if (subpassesInputAttachRefs.contains(subpass.first)){
                subpassInfo.inputAttachmentCount = subpassesInputAttachRefs.at(subpass.first).size();
                subpassInfo.pInputAttachments = subpassInfo.inputAttachmentCount > 0 ? subpassesInputAttachRefs.at(subpass.first).data() : nullptr;
            }
            if (subpassesColorAttachRefs.contains(subpass.first)){
                subpassInfo.colorAttachmentCount = subpassesColorAttachRefs.at(subpass.first).size();
                subpassInfo.pColorAttachments = subpassInfo.colorAttachmentCount > 0 ? subpassesColorAttachRefs.at(subpass.first).data() : nullptr;
            }
            if (subpassesResolveAttachRefs.contains(subpass.first)){
                subpassInfo.pResolveAttachments = subpassesResolveAttachRefs.at(subpass.first).empty() ? nullptr : subpassesResolveAttachRefs.at(subpass.first).data();
            }
            if (subpassesDepthAttachRef.contains(subpass.first)){
                subpassInfo.pDepthStencilAttachment = &subpassesDepthAttachRef.at(subpass.first);
            }
            passes.push_back(subpassInfo);
            subpassIdxLUT[subpass.first] = subpassVkIdx;
            subpassVkIdx += 1;
        }
        subpassVkIdxLUT_ = subpassIdxLUT;
        std::vector<vk::StructureChain<vk::SubpassDependency2, vk::MemoryBarrier2>> depends{};
        std::unordered_map<DependIdx , size_t> dependIdxLUT{};
        size_t dependVkIdx = 0;
        for (const auto& depend: dependencies_){
            depends.push_back(depend.second);
            if (depends.rbegin()->get<vk::SubpassDependency2>().srcSubpass != VK_SUBPASS_EXTERNAL){
                depends.rbegin()->get<vk::SubpassDependency2>().srcSubpass = subpassIdxLUT[depend.second.get<vk::SubpassDependency2>().srcSubpass];
            }
            if (depends.rbegin()->get<vk::SubpassDependency2>().dstSubpass != VK_SUBPASS_EXTERNAL){
                depends.rbegin()->get<vk::SubpassDependency2>().dstSubpass = subpassIdxLUT[depend.second.get<vk::SubpassDependency2>().dstSubpass];
            }
            dependIdxLUT[depend.first] = dependVkIdx;
            dependVkIdx += 1;
        }
        dependVkIdxLUT_ = dependIdxLUT;
        auto createInfoDepends = std::vector<vk::SubpassDependency2>{};
        for (const auto& dep: depends){
            createInfoDepends.push_back(dep.get<vk::SubpassDependency2>());
        }
        vk::RenderPassCreateInfo2 renderPassInfo = {};
        renderPassInfo.setAttachments(attaches);
        renderPassInfo.setSubpasses(passes);
        renderPassInfo.dependencyCount = createInfoDepends.size();
        renderPassInfo.pDependencies = createInfoDepends.data();
        auto [result, renderPass] = device_.createRenderPass2Unique(renderPassInfo);
        utils::vk_ensure(result);
        renderPass_ = std::move(renderPass);
        return renderPass_.get();
    }

    [[nodiscard]] vk::Framebuffer getFramebuffer(){
        if(static_cast<bool>(framebuffer_)){
            return framebuffer_.get();
        }
        std::vector<vk::FramebufferAttachmentImageInfo> attachInfos{}; attachInfos.reserve(attachments_.size());
        for (const auto& attach: attachments_){
            vk::FramebufferAttachmentImageInfo info{};
            const auto& resInfo = resMgr_.queryImageInfo(attach.second.resName);
            info.usage = resInfo.usage;
            info.flags = resInfo.flags;
            info.height = resInfo.extent.height;
            info.width = resInfo.extent.width;
            info.layerCount = resInfo.arrayLayers;
            info.viewFormatCount = 1;
            info.pViewFormats = &resInfo.format;
            attachInfos.push_back(info);
        }
        vk::FramebufferAttachmentsCreateInfo attachmentsInfo{};
        attachmentsInfo.setAttachmentImageInfos(attachInfos);
        vk::FramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.renderPass = renderPass_.get();
        framebufferInfo.width = renderExtent_.width;
        framebufferInfo.height = renderExtent_.height;
        framebufferInfo.layers = 1;
        framebufferInfo.flags = vk::FramebufferCreateFlagBits::eImageless;
        framebufferInfo.attachmentCount = attachmentsInfo.attachmentImageInfoCount;
        framebufferInfo.pAttachments = VK_NULL_HANDLE;
        framebufferInfo.pNext = &attachmentsInfo;
        auto [result, fb] = device_.createFramebufferUnique(framebufferInfo);
        utils::vk_ensure(result);
        framebuffer_ = std::move(fb);
        return framebuffer_.get();
    }

protected:
    using SubpassIdx = factory::SubpassIdx;
    using DependIdx = factory::DependIdx;
    using AttachIdx = factory::AttachIdx;
    using VkSubpassDependency2StructChain = factory::VkSubpassDependency2StructChain;
    using AttachmentReferenceInfo = factory::AttachmentReferenceInfo;
    using AttachmentInfo = factory::AttachmentInfo;
    std::string name_{};
    vk::Extent2D renderExtent_{};
    std::map<SubpassIdx, std::string> subpasses_{};
    // I give up, don't change to std::unordered_map<std::string, AttachmentReferenceInfo>, 
    // we need the ordering enforced by std::map to avoid numerous dance caused by Vulkan BS --
    // the order of elements in some api call will constrain the order of elements in subsequent calls
    std::map<SubpassIdx, std::map<std::string, AttachmentReferenceInfo>> attachmentReferences_{};
    std::map<AttachIdx, AttachmentInfo> attachments_{};
    std::map<DependIdx, VkSubpassDependency2StructChain> dependencies_{};
private:
    vk::Device device_{};
    RenderResourceManager &resMgr_;
    vk::UniqueRenderPass renderPass_{};
    vk::UniqueFramebuffer framebuffer_{};
    // Useful when doing Vulkan API call with implicit order requirements
    std::unordered_map<AttachIdx, size_t> attachVkIdxLUT_{};
    std::unordered_map<SubpassIdx , size_t> subpassVkIdxLUT_{};
    std::unordered_map<DependIdx , size_t> dependVkIdxLUT_{};
    friend class RenderpassFactory;
};

class RenderpassFactory{
public:
    using SubpassIdx = factory::SubpassIdx;
    using AttachIdx = factory::AttachIdx;
    using DependIdx = factory::DependIdx;
    using VkSubpassDependency2StructChain = factory::VkSubpassDependency2StructChain;
    using RenderpassIdx = factory::RenderpassIdx;
    RenderpassFactory(
            vk::Device device,
            SubpassFactory &subpassFactory,
            VertexShaderFactory &vertShaderFactory, FragmentShaderFactory &fragShaderFactory, RenderResourceManager &resourceManager):
            device_(device), subpassFactory_(subpassFactory), vertFactory_(vertShaderFactory), fragFactory_(fragShaderFactory), resMgr_(resourceManager)
            {}
    [[nodiscard]] RenderpassIdx registerRenderpass(const std::string &name, vk::Extent2D extent){
        RenderpassIdx index = renderpasses_.empty() ? 0 : renderpasses_.rbegin()->first + 1;
        RenderpassBase renderpass{extent, device_, resMgr_};
        renderpass.name_ = name;
        renderpasses_.try_emplace(index, std::move(renderpass));
        renderpassLUT_[name] = index;
        return index;
    }

    [[nodiscard]] SubpassIdx loadSubpass(RenderpassIdx index, const std::string &subpassName){
        auto& renderpass = renderpasses_.at(index);
        SubpassIdx subpassIdx = renderpass.subpasses_.empty() ? 0 : renderpass.subpasses_.rbegin()->first + 1;
        renderpass.subpasses_[subpassIdx] = subpassName;
        return subpassIdx;
    }

    void unloadSubpass(RenderpassIdx index, SubpassIdx subpassIndex){
        auto& renderpass = renderpasses_.at(index);
        renderpass.subpasses_.erase(subpassIndex);
        renderpass.attachmentReferences_.erase(subpassIndex);
    }

    void gatherAttachmentReferences(RenderpassIdx index){
        auto& renderpass = renderpasses_.at(index);
        for (auto& subpassTranslation: renderpass.subpasses_){
            const auto& subpass = subpassFactory_.propagateSubpass(subpassTranslation.second);
            // Currently only fragment shader has attachments, if there are other possibilities, use propagateEnabledStages()
            for (const auto& attachRef: fragFactory_.propagateShader(subpass.propagateFragShaderIdx()).propagateAttachmentInfo()){
                renderpass.attachmentReferences_[subpassTranslation.first][attachRef.first] = attachRef.second;
            }
        }
    }

    AttachIdx createAttachment(RenderpassIdx index, const vk::AttachmentDescription2 &attachDescription){
        auto& renderpass = renderpasses_.at(index);
        AttachIdx attachIdx = renderpass.attachments_.empty() ? 0 : renderpass.attachments_.rbegin()->first + 1;
        renderpass.attachments_[attachIdx].description = attachDescription;
        return attachIdx;
    }

    void removeAttachment(RenderpassIdx index, AttachIdx attachIndex){
        auto& renderpass = renderpasses_.at(index);
        renderpass.attachments_.erase(attachIndex);
    }

    // Yup, that's what the deep encapsulation of Vulkan API gives you
    void linkAttachmentRefWithAttach(RenderpassIdx index, SubpassIdx subpassIndex, const std::string &shaderVariableName, AttachIdx attachmentIndex){
        auto& renderpass = renderpasses_.at(index);
        renderpass.attachmentReferences_.at(subpassIndex).at(shaderVariableName).attachRef.attachment = attachmentIndex;
    }

    void linkAttachmentWithResource(RenderpassIdx index, AttachIdx attachmentIndex, const std::string &resourceName){
        auto& renderpass = renderpasses_.at(index);
        renderpass.attachments_.at(attachmentIndex).resName = resourceName;
    }

    DependIdx createSubpassDependency(RenderpassIdx index, const VkSubpassDependency2StructChain& subpassDependency){
        auto& renderpass = renderpasses_.at(index);
        DependIdx dependIdx = renderpass.dependencies_.empty() ? 0 : renderpass.dependencies_.rbegin()->first + 1;
        renderpass.dependencies_[dependIdx] = subpassDependency;
        return dependIdx;
    }

    void removeSubpassDependency(RenderpassIdx index, DependIdx dependencyIndex){
        auto& renderpass = renderpasses_.at(index);
        renderpass.dependencies_.erase(dependencyIndex);
    }
    void buildVulkanObjects(RenderpassIdx index){
        // Due to the intertwined nature of Vulkan API, we'd better create vk::Pipeline at renderpass level
        auto& renderpass = renderpasses_.at(index);
        /* The order works as follows:
         * 1. RenderPass
         * 2. Framebuffer
         * 3. For each subpass:
         *   a. ShaderModule
         *   b. Pipeline
         *   c. Descriptor set
         * */
        auto vkRenderPass = renderpass.getRenderPass();
        renderpass.getFramebuffer();
        for (const auto& subpassId: renderpass.subpasses_){
            auto& subpass = subpassFactory_.propagateSubpass(subpassId.second);
            if (!static_cast<bool>(subpass.pipeline_)){
                // ShaderModule
                std::vector<vk::PipelineShaderStageCreateInfo> shaderStages{};
                using ShaderStage = factory::ShaderStage;
                auto stages = subpass.propagateEnabledStages();
                if ((stages&ShaderStage::eVertex) == ShaderStage::eVertex){
                    vk::PipelineShaderStageCreateInfo info{};
                    info.flags = {};
                    info.stage = vk::ShaderStageFlagBits::eVertex;
                    info.module = vertFactory_.getShaderModule(subpass.propagateVertShaderIdx());
                    info.pName = vertFactory_.propagateShader(subpass.propagateVertShaderIdx()).getEntryPointName().data();
                    shaderStages.push_back(info);
                }
                if ((stages&ShaderStage::eFragment) == ShaderStage::eFragment){
                    vk::PipelineShaderStageCreateInfo info{};
                    info.flags = {};
                    info.stage = vk::ShaderStageFlagBits::eFragment;
                    info.module = fragFactory_.getShaderModule(subpass.propagateFragShaderIdx());
                    info.pName = fragFactory_.propagateShader(subpass.propagateFragShaderIdx()).getEntryPointName().data();
                    shaderStages.push_back(info);
                }
                vk::PipelineVertexInputStateCreateInfo vertInputInfo{};
                const auto& vertShader = vertFactory_.propagateShader(subpass.propagateVertShaderIdx());
                auto vertInputData = vertShader.getVertexInputInfoData();
                vertInputInfo = VertexShaderBase::getVertexInputInfo(vertInputData);
                vk::PipelineInputAssemblyStateCreateInfo vertAsmInfo{};
                vertAsmInfo = vertShader.getInputAssemblyInfo();
                auto viewportData = get_viewport_info(renderpass.renderExtent_.width, renderpass.renderExtent_.height);
                vk::PipelineViewportStateCreateInfo viewportInfo{};
                viewportInfo.setViewports(viewportData.viewport);
                viewportInfo.setScissors(viewportData.scissor);
                const auto& fragShader = fragFactory_.propagateShader(subpass.propagateFragShaderIdx());
                vk::PipelineDepthStencilStateCreateInfo depthInfo{};
                depthInfo = fragShader.getDepthStencilInfo();
                vk::PipelineMultisampleStateCreateInfo msaaInfo{};
                msaaInfo = fragShader.getMSAAInfo();
                vk::PipelineRasterizationStateCreateInfo rastInfo{};
                rastInfo = vertShader.getRasterizationInfo();
                vk::PipelineColorBlendStateCreateInfo blendInfo{};
                blendInfo = fragShader.propagateColorBlendInfo();
                // We need the order of attachment references back when creating the renderpass.
                //  As long as we don't use std::unordered_map, things should be fine
                std::vector<vk::PipelineColorBlendAttachmentState> colorInfos{};
                for (const auto& attachRef: renderpass.attachmentReferences_.at(subpassId.first)){
                    if (attachRef.second.attachType == factory::AttachmentType::eColor){
                        colorInfos.push_back(attachRef.second.blendInfo);
                    }
                }
                blendInfo.setAttachments(colorInfos);
                auto pipelineLayout = subpass.getPipelineLayout();
                vk::GraphicsPipelineCreateInfo graphPipeInfo{};
                graphPipeInfo.flags = {};
                graphPipeInfo.setStages(shaderStages);
                graphPipeInfo.pVertexInputState = &vertInputInfo;
                graphPipeInfo.pInputAssemblyState = &vertAsmInfo;
                graphPipeInfo.pTessellationState = nullptr;
                graphPipeInfo.pViewportState = &viewportInfo;
                graphPipeInfo.pRasterizationState = &rastInfo;
                graphPipeInfo.pMultisampleState = &msaaInfo;
                graphPipeInfo.pDepthStencilState = &depthInfo;
                graphPipeInfo.pColorBlendState = &blendInfo;
                graphPipeInfo.pDynamicState = nullptr;
                graphPipeInfo.layout = pipelineLayout;
                graphPipeInfo.renderPass = vkRenderPass;
                graphPipeInfo.subpass = subpassId.first;
                graphPipeInfo.basePipelineHandle = nullptr;
                graphPipeInfo.basePipelineIndex = 0;
                auto [result, graphPipeline] = device_.createGraphicsPipelineUnique(nullptr, graphPipeInfo);
                utils::vk_ensure(result);
                subpass.pipeline_ = std::move(graphPipeline);
            }

        }
    }
private:
    RenderResourceManager &resMgr_;
    SubpassFactory &subpassFactory_;
    VertexShaderFactory &vertFactory_;
    FragmentShaderFactory &fragFactory_;
    std::map<RenderpassIdx, RenderpassBase> renderpasses_{};
    std::unordered_map<std::string, RenderpassIdx> renderpassLUT_{};
    vk::Device device_{};
};
#endif //VKLEARN_RENDERPASS_HPP
