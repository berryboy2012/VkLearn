//
// Created by berry on 2023/3/7.
//

#ifndef VKLEARN_COMMANDS_CPP
#define VKLEARN_COMMANDS_CPP

#include "utils.h"
#include <optional>
#include <map>
#include <format>
#include <fstream>
#include <any>
#include <functional>
#include <iostream>
#include <concepts>
#include "commands_management.h"

/*    SingleTimeCommandBuffer    */
SingleTimeCommandBuffer::SingleTimeCommandBuffer(SingleTimeCommandBuffer &&other) noexcept {
    *this = std::move(other);
}

SingleTimeCommandBuffer& SingleTimeCommandBuffer::operator= (SingleTimeCommandBuffer &&other) noexcept {
    if (this != &other) [[likely]]{
        coBuf_ = other.coBuf_;
        other.coBuf_ = VK_NULL_HANDLE;
        isCmdBufRetired_ = other.isCmdBufRetired_;
        other.isCmdBufRetired_ = true;
        device_ = other.device_;
        queue_ = other.queue_;
    }
    return *this;
}

SingleTimeCommandBuffer::SingleTimeCommandBuffer(vk::CommandPool &commandPool, vk::Queue &queue, vk::Device &device):
        queue_(queue),
        device_(device)
{
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    {
        auto [result, buf] = device.allocateCommandBuffers(allocInfo);
        utils::vk_ensure(result);
        coBuf_ = buf[0];
        isCmdBufRetired_ = false;
    }
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    auto result = coBuf_.begin(beginInfo);
    utils::vk_ensure(result);
}

SingleTimeCommandBuffer::~SingleTimeCommandBuffer(){
    if (!isCmdBufRetired_){
        auto result = coBuf_.end();
        utils::vk_ensure(result);
        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &coBuf_;
        auto subResult = queue_.submit(submitInfo);
        utils::vk_ensure(subResult);
        auto waitResult = queue_.waitIdle();
        utils::vk_ensure(waitResult);
    }
}
/*    End of SingleTimeCommandBuffer    */



/*    CommandBufferManager    */
CommandBufferManager::CommandBufferManager(CommandBufferManager &&other) noexcept {
        *this = std::move(other);
}
CommandBufferManager& CommandBufferManager::operator= (CommandBufferManager &&other) noexcept{
    if (this != &other) [[likely]]{
        device_ = other.device_;
        queueCGTP_ = other.queueCGTP_;
        queueFamilyIdxCGTP_ = other.queueFamilyIdxCGTP_;
        comPool_ = std::move(other.comPool_);
        buffs_ = other.buffs_;
        numCombuf_ = other.numCombuf_;
    }
    return *this;
}
CommandBufferManager::CommandBufferManager(vk::Device device, utils::QueueStruct queueCGTP){
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
    utils::vk_ensure(result);
}
std::vector<vk::CommandBuffer> CommandBufferManager::createCommandBuffers(size_t numBuffers){
    vk::CommandBufferAllocateInfo allocInfo = {};
    allocInfo.commandPool = comPool_.get();
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)numBuffers;

    auto [buffersResult, commandBuffers] = device_.allocateCommandBuffers(allocInfo);
    utils::vk_ensure(buffersResult);
    std::vector<vk::CommandBuffer> results{};
    for (auto& comBuf: commandBuffers) {
        results.push_back(comBuf);
        buffs_.push_back(comBuf);
    }
    numCombuf_ += numBuffers;
    return results;
}
void CommandBufferManager::submitCommandBuffers(const std::span<const vk::SemaphoreSubmitInfo> semaphoresWait,
                          const std::span<const vk::CommandBuffer> commandBuffers,
                          const std::span<const vk::SemaphoreSubmitInfo> semaphoresSignal,
                          const vk::Fence fence){
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
    utils::vk_ensure(result);
}
void CommandBufferManager::waitQueueIdle(){
    auto result = queueCGTP_.waitIdle();
    utils::vk_ensure(result);
}

void CommandBufferManager::resetPool(){
    auto result = device_.resetCommandPool(comPool_.get());
    utils::vk_ensure(result);
}

SingleTimeCommandBuffer CommandBufferManager::getSingleTimeCommandBuffer(){
    return std::move(SingleTimeCommandBuffer{comPool_.get(), queueCGTP_, device_});
}
/*    End of CommandBufferManager    */

#endif //VKLEARN_COMMANDS_CPP
