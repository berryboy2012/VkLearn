//
// Created by berry on 2023/3/5.
//

#ifndef VKLEARN_RENDERPASS_HPP
#define VKLEARN_RENDERPASS_HPP
struct SubpassAttachmentReferences{
    //TODO: add support for preserveAttachments
    std::vector<vk::AttachmentReference2> inputAttachments;
    std::vector<vk::AttachmentReference2> colorAttachments;
    std::vector<vk::AttachmentReference2> resolveAttachments;
    vk::AttachmentReference2 depthStencilAttachment;
};
class Renderpass{
    //TODO: support pNext for related Vulkan info structs.
public:
    typedef uint32_t AttachIdx;
    typedef uint32_t SubpassIdx;
    typedef size_t DependIdx;

    vk::RenderPass renderpass_{};

    Renderpass() = default;
    Renderpass& operator= (const Renderpass &) = delete;
    Renderpass(const Renderpass &) = delete;
    Renderpass& operator= (Renderpass && other) noexcept{
        if (this != &other){
            device_ = other.device_;
            attachDescs_ = other.attachDescs_;
            subpasses_ = other.subpasses_;
            subpassAttachments_ = other.subpassAttachments_;
            subpassDependencies_ = other.subpassDependencies_;
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
    std::vector<AttachIdx> registerSubpassAttachments(const std::span<const vk::AttachmentDescription2> attaches){
        std::vector<AttachIdx> indices{};
        for(auto& attach: attaches){
            size_t index = attachDescs_.empty() ? 0 : attachDescs_.rbegin()->first+1;
            attachDescs_[index] = attach;
            indices.push_back(index);
        }
        return indices;
    }
    void unregisterSubpassAttachments(const std::span<const AttachIdx> indices){
        for (auto& index: indices){
            if (attachDescs_.contains(index)){
                attachDescs_.erase(index);
            }
        }
    }
    // Register subpass before registering dependencies, as dependencies directly reference indices of subpasses.
    //  When supplying vk::SubpassDescription2,
    //  *(vk::SubpassDescription2::p{}Attachments)::attachment should use the index given when registering attachments.
    //  This helper will take care of the conversion.
    SubpassIdx registerSubpass(const vk::SubpassDescription2* subpassDescription){
        SubpassIdx index = subpasses_.empty() ? 0 : subpasses_.rbegin()->first+1;
        subpasses_[index] = *subpassDescription;
        // Deep copy of referenced attachments
        auto inputAttachView = std::span{subpasses_[index].pInputAttachments, subpasses_[index].inputAttachmentCount};
        for (auto& attach: inputAttachView){
            subpassAttachments_[index].inputAttachments.push_back(attach);
        }
        auto colorAttachView = std::span{subpasses_[index].pColorAttachments, subpasses_[index].colorAttachmentCount};
        for (auto& attach: colorAttachView){
            subpassAttachments_[index].colorAttachments.push_back(attach);
        }
        auto resolveAttachView = std::span{subpasses_[index].pResolveAttachments, subpasses_[index].colorAttachmentCount};
        for (auto& attach: resolveAttachView){
            subpassAttachments_[index].resolveAttachments.push_back(attach);
        }
        subpassAttachments_[index].depthStencilAttachment = *(subpasses_[index].pDepthStencilAttachment);
        return index;
    }
    void unregisterSubpass(SubpassIdx index){
        if(subpasses_.contains(index)){
            subpasses_.erase(index);
            subpassAttachments_.erase(index);
        }
    }
    // Finally register dependencies between subpasses, vk::SubpassDependency2::{src/dst}Subpass should use the index
    //  given when registering subpasses.
    std::vector<DependIdx> registerSubpassDependencies(const std::span<const vk::SubpassDependency2> dependencies){
        std::vector<DependIdx> indices{};
        for (auto& depend: dependencies){
            DependIdx index = subpassDependencies_.empty() ? 0 : subpassDependencies_.rbegin()->first+1;
            subpassDependencies_[index] = depend;
            indices.push_back(index);
        }
        return indices;
    }
    void unregisterSubpassDependencies(const std::span<const DependIdx> indices){
        for (auto& index: indices){
            if (subpassDependencies_.contains(index)){
                subpassDependencies_.erase(index);
            }
        }
    }

    vk::RenderPass createRenderpass(){
        // Renderpass -- span<attachment>
        //        \\____ span<subpass>
        //         \                \____ several span<ref to attachment>
        //          \____ span<dependency>
        //                              \____ ref to subpass
        std::vector<vk::AttachmentDescription2> attaches{};
        std::map<AttachIdx, size_t> attachLUT{};
        for (auto& desc: attachDescs_){
            attachLUT[desc.first] = attaches.size();
            attaches.push_back(desc.second);
        }
        std::vector<vk::SubpassDescription2> passes{};
        std::vector<SubpassAttachmentReferences> passRefs{};
        std::map<SubpassIdx, size_t> passLUT{};
        for (auto& pass: subpasses_){
            passLUT[pass.first] = passes.size();
            passes.push_back(pass.second);
            passRefs.push_back(subpassAttachments_.at(pass.first));
        }
        for (auto& passRef: passRefs){
            for (auto& ref: passRef.inputAttachments){
                ref.attachment = attachLUT[ref.attachment];
            }
            for (auto& ref: passRef.colorAttachments){
                ref.attachment = attachLUT[ref.attachment];
            }
            for (auto& ref: passRef.resolveAttachments){
                ref.attachment = attachLUT[ref.attachment];
            }
            passRef.depthStencilAttachment.attachment = attachLUT[passRef.depthStencilAttachment.attachment];
        }
        std::vector<vk::SubpassDependency2> depends{};
        for (auto& depend: subpassDependencies_){
            depends.push_back(depend.second);
        }
        for (auto& depend: depends){
            depend.srcSubpass = passLUT[depend.srcSubpass];
            depend.dstSubpass = passLUT[depend.dstSubpass];
        }
        vk::RenderPassCreateInfo2 renderPassInfo = {};
        renderPassInfo.setAttachments(attaches);
        renderPassInfo.setSubpasses(passes);
        renderPassInfo.setDependencies(depends);

        auto [result, renderPass] = device_.createRenderPass2Unique(renderPassInfo);
        utils::vkEnsure(result);
        pass_ = std::move(renderPass);
        renderpass_ = pass_.get();
    }


private:
    vk::Device device_{};
    std::map<AttachIdx, vk::AttachmentDescription2> attachDescs_{};

    std::map<SubpassIdx, vk::SubpassDescription2> subpasses_{};
    std::map<SubpassIdx, SubpassAttachmentReferences> subpassAttachments_{};

    std::map<DependIdx, vk::SubpassDependency2> subpassDependencies_{};

    vk::UniqueRenderPass pass_{};
};
#endif //VKLEARN_RENDERPASS_HPP
