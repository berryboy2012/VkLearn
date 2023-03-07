//
// Created by berry on 2023/3/7.
//

#ifndef VKLEARN_COMMANDS_HPP
#define VKLEARN_COMMANDS_HPP
struct QueueStruct{
    vk::Queue queue{};
    uint32_t queueFamilyIdx{};
};
class CommandBufferManager{
    // A thin layer over vk::CommandPool and vk::Queue
public:
    CommandBufferManager() = default;
    CommandBufferManager(const CommandBufferManager &) = delete;
    CommandBufferManager& operator= (const CommandBufferManager &) = delete;
    CommandBufferManager(CommandBufferManager &&other) noexcept {
        *this = std::move(other);
    }
    CommandBufferManager& operator= (CommandBufferManager &&other) = default;
    explicit CommandBufferManager(vk::Device device, QueueStruct queueCGTP){
        // We assume the provided queue can be used for all types of commands.
        //  Including Compute-Graphics-Transfer-Presentation.
        device_ = device;
        queueCGTP_ = queueCGTP.queue;
        queueFamilyIdxCGTP_ = queueCGTP.queueFamilyIdx;
        vk::CommandPoolCreateInfo poolInfo = {};
        poolInfo.queueFamilyIndex = queueFamilyIdxCGTP_;
        // add vk::CommandPoolCreateFlagBits::eResetCommandBuffer to flags to allow resetting individual command buffer.
        vk::Result result{};
        std::tie(result, comPool_) = device_.createCommandPoolUnique(poolInfo).asTuple();
        utils::vkEnsure(result);
    }
    std::vector<vk::CommandBuffer> createCommandBuffers(size_t numBuffers){
        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.commandPool = comPool_.get();
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = (uint32_t)numBuffers;

        auto [buffersResult, commandBuffers] = device_.allocateCommandBuffers(allocInfo);
        utils::vkEnsure(buffersResult);
        std::vector<vk::CommandBuffer> results{};
        for (auto& comBuf: commandBuffers) {
            results.push_back(comBuf);
            buffs_.push_back(comBuf);
        }
        numCombuf_ += numBuffers;
        return results;
    }
    void submitCommandBuffers(const std::span<const vk::SemaphoreSubmitInfo> semaphoresWait,
                              const std::span<const vk::CommandBuffer> commandBuffers,
                              const std::span<const vk::SemaphoreSubmitInfo> semaphoresSignal,
                              const vk::Fence fence = VK_NULL_HANDLE){
        vk::SubmitInfo2 submitInfo{};
        submitInfo.flags = {};
        submitInfo.setWaitSemaphoreInfos(semaphoresWait);
        std::vector<vk::CommandBufferSubmitInfo> bufInfos{};
        for (auto& cmdbuf: commandBuffers){
            vk::CommandBufferSubmitInfo info{};
            info.commandBuffer = cmdbuf;
            // No plans for multi-device support
            info.deviceMask = 0;
            bufInfos.push_back(info);
        }
        submitInfo.setCommandBufferInfos(bufInfos);
        submitInfo.setSignalSemaphoreInfos(semaphoresSignal);
        auto result = queueCGTP_.submit2(submitInfo, fence);
        utils::vkEnsure(result);
    }
    void waitQueueIdle(){
        auto result = queueCGTP_.waitIdle();
        utils::vkEnsure(result);
    }

    void resetPool(){
        comPool_.reset();
    }

    utils::SingleTimeCommandBuffer getSingleTimeCommandBuffer(){
        return std::move(utils::SingleTimeCommandBuffer{comPool_.get(), queueCGTP_, device_});
    }

private:
    vk::Device device_{};
    vk::Queue queueCGTP_{};
    uint32_t queueFamilyIdxCGTP_{};
    vk::UniqueCommandPool comPool_{};
    std::vector<vk::CommandBuffer> buffs_{};
    size_t numCombuf_{};
};
#endif //VKLEARN_COMMANDS_HPP
