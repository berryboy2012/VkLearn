//
// Created by berry on 2023/3/7.
//

#ifndef VKLEARN_COMMANDS_MANAGEMENT_H
#define VKLEARN_COMMANDS_MANAGEMENT_H

#include <optional>
#include <map>
#include <format>
#include <fstream>
#include <any>
#include <functional>
#include <iostream>
#include <concepts>

#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT_ON_RESULT
#include "vulkan/vulkan.hpp"
#include "utils.h"
struct QueueStruct{
    vk::Queue queue{};
    uint32_t queueFamilyIdx{};
};

class SingleTimeCommandBuffer {
    // Submit the recorded vk::CommandBuffer and wait once dtor is called.
public:
    vk::CommandBuffer coBuf{};
    SingleTimeCommandBuffer() = default;
    SingleTimeCommandBuffer(const SingleTimeCommandBuffer &) = delete;
    SingleTimeCommandBuffer& operator= (const SingleTimeCommandBuffer &) = delete;
    SingleTimeCommandBuffer(SingleTimeCommandBuffer &&other) noexcept ;
    SingleTimeCommandBuffer& operator= (SingleTimeCommandBuffer &&other) noexcept ;
    SingleTimeCommandBuffer(vk::CommandPool &commandPool, vk::Queue &queue, vk::Device &device);
    ~SingleTimeCommandBuffer();
private:
    bool isCmdBufRetired_ = true;
    vk::Device device_;
    vk::Queue queue_;
};

class CommandBufferManager{
    // A thin layer over vk::CommandPool and vk::Queue
public:
    typedef QueueStruct QueueStructType;
    CommandBufferManager() = default;
    CommandBufferManager(const CommandBufferManager &) = delete;
    CommandBufferManager& operator= (const CommandBufferManager &) = delete;
    CommandBufferManager(CommandBufferManager &&other) noexcept;
    CommandBufferManager& operator= (CommandBufferManager &&other) noexcept;
    explicit CommandBufferManager(vk::Device device, QueueStruct queueCGTP);
    std::vector<vk::CommandBuffer> createCommandBuffers(size_t numBuffers);
    void submitCommandBuffers(std::span<const vk::SemaphoreSubmitInfo> semaphoresWait,
                              std::span<const vk::CommandBuffer> commandBuffers,
                              std::span<const vk::SemaphoreSubmitInfo> semaphoresSignal,
                              const vk::Fence fence = VK_NULL_HANDLE);
    void waitQueueIdle();

    void resetPool();

    SingleTimeCommandBuffer getSingleTimeCommandBuffer();

private:
    vk::Device device_{};
    vk::Queue queueCGTP_{};
    uint32_t queueFamilyIdxCGTP_{};
    vk::UniqueCommandPool comPool_{};
    std::vector<vk::CommandBuffer> buffs_{};
    size_t numCombuf_{};
};

#endif //VKLEARN_COMMANDS_MANAGEMENT_H
