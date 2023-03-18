//
// Created by berry on 2023/3/5.
//

#ifndef VKLEARN_RENDERPASS_HPP
#define VKLEARN_RENDERPASS_HPP

#include "shader_modules.hpp"
#include "graphics_pipeline.hpp"

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
 * Gather attachment references from all subpass
 *     Do not merge the info, may introduce name aliasing problems
 * Populate attachments
 *     Caller should specify most of vk::AttachmentDescription2
 *     Caller don't have control over index
 * Link attachment references to attachments
 *     trivial
 * Link attachments to resources
 *     Kind of hard to decide, first just link by string name, then we'll see how much info should be supplied
 * Create a corresponding framebuffer
 *     Should use resource name to query some info (for example, attachment linked to swapchain image)
 * Add subpass dependencies
 *     trivial
 * Create vk::PipelineLayout, vk::Renderpass, vk::Framebuffer, vk::Pipeline
 * */
private:
    vk::Device device_{};
    vk::UniqueRenderPass renderPass_{};
    vk::UniqueFramebuffer framebuffer_{};
};
#endif //VKLEARN_RENDERPASS_HPP
