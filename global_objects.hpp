//
// Created by berry on 2023/3/10.
//

#ifndef VKLEARN_GLOBAL_OBJECTS_HPP
#define VKLEARN_GLOBAL_OBJECTS_HPP
#include <atomic>
#include <mutex>
#include <semaphore>
#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_ASSERT_ON_RESULT
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
// Yup, that pollutes many names
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
// Required by Vulkan-Hpp (https://github.com/KhronosGroup/Vulkan-Hpp#extensions--per-device-function-pointers)
std::mutex gVulkanMutex{};
#ifndef VK_DYNAMIC_DISPATCHER
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#define VK_DYNAMIC_DISPATCHER
#endif
constexpr size_t INFLIGHT_FRAMES = 2;
struct MainRendererComm{
    std::atomic<vk::ImageView> imageViewHandle{};
    std::binary_semaphore imageViewHandleAvailable{0};
    std::binary_semaphore imageViewHandleConsumed{0};
    std::atomic<bool> swapchainInvalid{false};
};
std::array<MainRendererComm, INFLIGHT_FRAMES> mainRendererComms{};
std::binary_semaphore deviceAvailable{0};
void initialize_main_sync_objs(){
    for (auto& syncObj: mainRendererComms){
        syncObj.imageViewHandle.store({});
        syncObj.imageViewHandleAvailable.try_acquire();
        syncObj.imageViewHandleConsumed.try_acquire();
        syncObj.swapchainInvalid.store(false);
    }
    deviceAvailable.try_acquire();
    deviceAvailable.release();
}
void notify_renderer_exit(){
    for (auto& syncObj: mainRendererComms){
        syncObj.swapchainInvalid.store(true);
    }
}
void wait_vulkan_device_idle(vk::Device device){
    deviceAvailable.acquire();
    utils::vk_ensure(device.waitIdle());
    deviceAvailable.release();
}
#endif //VKLEARN_GLOBAL_OBJECTS_HPP
