//
// Created by berry on 2023/3/8.
//
#include <iostream>
#include <tuple>
#include <optional>
#include <vector>
#include <type_traits>
#include <string>
#include <string_view>
#define WIN32_LEAN_AND_MEAN
#include "tracy/Tracy.hpp"
#include <SDL2/SDL.h>

#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_ASSERT_ON_RESULT
// Yup, that pollutes many names
#define VK_USE_PLATFORM_WIN32_KHR

#include <vulkan/vulkan.hpp>
#include <SDL2/SDL_vulkan.h>
#include <openvr.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "utils.h"
#include "global_objects.hpp"

struct QueueFamilyProps {
    vk::QueueFamilyProperties2 props{};
    bool supportsSurface{};
};
struct PhysicalDeviceInfo {
    vk::SurfaceKHR surface{};// Signify which surface are those info refer to
    std::vector<QueueFamilyProps> queueFamilies{};
    vk::PhysicalDeviceProperties2 props{};
    std::vector<vk::ExtensionProperties> devExts{};
    std::unordered_map<vk::Format, vk::FormatProperties2> formats{};
    vk::SurfaceCapabilities2KHR surfaceCaps{};
    std::vector<vk::SurfaceFormat2KHR> surfaceFmts{};
    std::vector<vk::PresentModeKHR> surfacePres{};
};
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "memory_management.hpp"
#include "render_thread.hpp"

void query_ovr() {
    vr::EVRInitError eError = vr::VRInitError_None;
    auto m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Utility);
    auto OVRVer = m_pHMD->GetRuntimeVersion();
    std::cout << "OpenVR runtime version: " << OVRVer << std::endl;
    if (eError != vr::VRInitError_None) {
        m_pHMD = nullptr;
        char buf[1024];
        sprintf_s(buf, sizeof(buf), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, nullptr);
    } else {
        vr::VR_Shutdown();
        m_pHMD = nullptr;
    }

}

SDL_Window *init_sdl() {
    SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, "0");
    SDL_Init(SDL_INIT_VIDEO);
    //
    SDL_Vulkan_LoadLibrary(nullptr);
    auto window = SDL_CreateWindow(
            "VkLearn",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            800, 500,
            SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI);// | SDL_WINDOW_FULLSCREEN);
    SDL_SetWindowResizable(window, SDL_TRUE);

    return window;
}

void clean_sdl(SDL_Window *&window) {
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

std::vector<std::string> get_required_validation_layers(bool enableValidationLayers) {
    auto validationLayers = std::vector<std::string>{"VK_LAYER_KHRONOS_validation"};
    auto chosenLayers = std::vector<std::string>{};
    if (enableValidationLayers) {
        auto [layerResult, availableLayers] = vk::enumerateInstanceLayerProperties();
        utils::vk_ensure(layerResult);
        for (const auto &requestedLayer: validationLayers) {
            bool layerFound = false;
            for (const auto &availableLayer: availableLayers) {
                if (std::string_view(availableLayer.layerName) == requestedLayer) {
                    layerFound = true;
                    chosenLayers.push_back(requestedLayer);
                    break;
                }
            }
            if (!layerFound) {
                utils::log_and_pause("Validation layer not found");
                std::abort();
            }
        }
    }
    return chosenLayers;
}

std::vector<std::string> get_required_instance_extensions(SDL_Window *window, bool enableValidationLayers) {
    auto instanceExtensions = std::vector<std::string>{};
    {
        // First get extensions required by SDL
        uint32_t extensionCount;
        SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
        auto extensionNames = std::vector<const char *>{extensionCount};
        SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensionNames.data());
        for (auto pExtensionName: extensionNames) {
            instanceExtensions.push_back(std::string(pExtensionName));
        }
    }
    // We require other extensions as well
    if (enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    // We want robust interaction with window subsystem (in the future, as device support is non-existent ATM)
    // NVIDIA users should be getting support with R545.37
    //instanceExtensions.push_back(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
    return instanceExtensions;
}

PhysicalDeviceInfo
query_physical_device_info(const vk::PhysicalDevice chosenPhysicalDevice, vk::UniqueSurfaceKHR &surfaceSDL) {
    PhysicalDeviceInfo physicalDeviceProps{};
    vk::Result vkAPIRes{};
    physicalDeviceProps.props = chosenPhysicalDevice.getProperties2();
    std::tie(vkAPIRes, physicalDeviceProps.devExts) = chosenPhysicalDevice.enumerateDeviceExtensionProperties();
    utils::vk_ensure(vkAPIRes);
    physicalDeviceProps.surface = surfaceSDL.get();
    auto queueFamilies = chosenPhysicalDevice.getQueueFamilyProperties2();
    for (size_t i = 0; i < queueFamilies.size(); ++i) {
        auto [result, surfaceSupported] = chosenPhysicalDevice.getSurfaceSupportKHR(i, physicalDeviceProps.surface);
        utils::vk_ensure(result);
        auto queueFamilyProp = QueueFamilyProps{
                .props = queueFamilies[i],
                .supportsSurface = static_cast<bool>(surfaceSupported)};
        physicalDeviceProps.queueFamilies.emplace_back(queueFamilyProp);
    }
    // Can't iterate over enum for vk::Format
    for (const auto fmt: utils::enum_range<vk::Format>(vk::Format::eUndefined, vk::Format::eAstc12x10SrgbBlock)) {
        auto fmtProps = chosenPhysicalDevice.getFormatProperties2(fmt);
        physicalDeviceProps.formats[fmt] = fmtProps;
    }
    auto result = vk::Result{};
    std::tie(result, physicalDeviceProps.surfaceCaps) = chosenPhysicalDevice.getSurfaceCapabilities2KHR(
            physicalDeviceProps.surface);
    utils::vk_ensure(result);
    std::tie(result, physicalDeviceProps.surfaceFmts) = chosenPhysicalDevice.getSurfaceFormats2KHR(
            physicalDeviceProps.surface);
    utils::vk_ensure(result);
    std::tie(result, physicalDeviceProps.surfacePres) = chosenPhysicalDevice.getSurfacePresentModesKHR(
            physicalDeviceProps.surface);
    utils::vk_ensure(result);
    return physicalDeviceProps;
}

//TODO: add extension query for OpenVR
//Remember, most extensions require respective features to work
std::vector<std::string> get_required_device_extensions() {
    std::vector<std::string> requiredDeviceExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            // We want robust interaction with window subsystem (in the future, as device support is non-existent ATM)
            //VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
            VK_KHR_PRESENT_ID_EXTENSION_NAME,
            VK_KHR_PRESENT_WAIT_EXTENSION_NAME,
            // Vulkan Ray Tracing (https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/#raytracingsetup)
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            // Ray queries
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
            VK_KHR_SPIRV_1_4_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,

            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
            // We want a decent support of dynamic state
            VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME
    };
    return requiredDeviceExtensions;
}

// A jank way to store a linked list of VkPhysDevFeatures, the first element can be retrieved by std::any_cast<vk::PhysicalDeviceFeatures2&>.
// Should use vk::StructureChain instead. The returned list is not copy-able!
std::vector<std::any> get_required_device_features_2(const vk::PhysicalDevice &device) {
    using Feature2 = vk::PhysicalDeviceFeatures2;
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-02829
    using Vulkan11 = vk::PhysicalDeviceVulkan11Features;
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-02830
    using Vulkan12 = vk::PhysicalDeviceVulkan12Features;// Contains VK_KHR_imageless_framebuffer
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-06532
    using Vulkan13 = vk::PhysicalDeviceVulkan13Features;
    using ASFeature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR;
    using RTPFeature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR;
    using VIDSFeature = vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT;
    using RayFeature = vk::PhysicalDeviceRayQueryFeaturesKHR;
    using PresIDFeature = vk::PhysicalDevicePresentIdFeaturesKHR;
    using PresWFeature = vk::PhysicalDevicePresentWaitFeaturesKHR;
    //using SCMntn = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT;
    // Add your feature requirements below

    //Add your feature requirements above
    auto featureList = std::vector<std::any>{};
    featureList.push_back(Feature2{});
    featureList.push_back(ASFeature{});
    featureList.push_back(RTPFeature{});
    featureList.push_back(Vulkan11{});
    featureList.push_back(Vulkan12{});
    featureList.push_back(Vulkan13{});
    featureList.push_back(VIDSFeature{});
    featureList.push_back(RayFeature{});
    featureList.push_back(PresIDFeature{});
    featureList.push_back(PresWFeature{});
    //featureList.push_back(SCMntn{});
    // And below here

    // And above here
    std::any_cast<Feature2 &>(featureList[0]).pNext = (void *) &std::any_cast<ASFeature &>(featureList[1]);
    std::any_cast<ASFeature &>(featureList[1]).pNext = (void *) &std::any_cast<RTPFeature &>(featureList[2]);
    std::any_cast<RTPFeature &>(featureList[2]).pNext = (void *) &std::any_cast<Vulkan11 &>(featureList[3]);
    std::any_cast<Vulkan11 &>(featureList[3]).pNext = (void *) &std::any_cast<Vulkan12 &>(featureList[4]);
    std::any_cast<Vulkan12 &>(featureList[4]).pNext = (void *) &std::any_cast<Vulkan13 &>(featureList[5]);
    std::any_cast<Vulkan13 &>(featureList[5]).pNext = (void *) &std::any_cast<VIDSFeature &>(featureList[6]);
    std::any_cast<VIDSFeature &>(featureList[6]).pNext = (void *) &std::any_cast<RayFeature &>(featureList[7]);
    std::any_cast<RayFeature &>(featureList[7]).pNext = (void *) &std::any_cast<PresIDFeature &>(featureList[8]);
    std::any_cast<PresIDFeature &>(featureList[8]).pNext = (void *) &std::any_cast<PresWFeature &>(featureList[9]);
    //std::any_cast<RayFeature&>(featureList[]).pNext = (void*)&std::any_cast<SCMntn&>(featureList[]);
    // And below here

    // And above here
    device.getFeatures2(&std::any_cast<Feature2 &>(featureList[0]));
    return featureList;
}

vk::UniqueInstance create_vulkan_instance(const std::vector<std::string> &validationLayers,
                                          const std::vector<std::string> &instanceExtensions) {
    auto appInfo = vk::ApplicationInfo(
            "VkLearn",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_3
    );
    auto createInfo = vk::InstanceCreateInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0, nullptr, // enabled layers
            0, nullptr // enabled extensions
    );


    std::vector<const char *> pLayerNames;
    for (const auto &requestedLayer: validationLayers) {
        pLayerNames.push_back(requestedLayer.c_str());
    }
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = pLayerNames.data();

    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    std::vector<const char *> pExtensionNames;
    for (const auto &requestedExtension: instanceExtensions) {
        pExtensionNames.push_back(requestedExtension.c_str());
    }
    createInfo.ppEnabledExtensionNames = pExtensionNames.data();

    auto [result, vkUniqueInstance] = vk::createInstanceUnique(createInfo, nullptr);
    utils::vk_ensure(result);
    return std::move(vkUniqueInstance);
}

vk::UniqueDebugUtilsMessengerEXT create_vulkan_debug_msg(const vk::Instance &vkInstance) {
    using SeverityBits = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using MsgTypeBits = vk::DebugUtilsMessageTypeFlagBitsEXT;

    auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT{
            {},
            SeverityBits::eWarning | SeverityBits::eError, // | SeverityBits::eInfo | SeverityBits::eVerbose,
            MsgTypeBits::eGeneral | MsgTypeBits::ePerformance | MsgTypeBits::eValidation |
            MsgTypeBits::eDeviceAddressBinding,
            &utils::debug_utils_messenger_callback
    };

    auto [debugResult, debugUtilsMessengerUnique] = vkInstance.createDebugUtilsMessengerEXTUnique(createInfo);
    return std::move(debugUtilsMessengerUnique);

}

vk::PhysicalDevice get_physical_device(const vk::Instance &vkInstance) {
    auto [resultDevices, devices] = vkInstance.enumeratePhysicalDevices();
    utils::vk_ensure(resultDevices);
    std::any chosenPhysicalDevice{};
    for (const auto &device: devices) {
        auto devProperty = device.getProperties2();
        std::cout << std::string_view{devProperty.properties.deviceName} << std::endl;
        if (!chosenPhysicalDevice.has_value()) {
            chosenPhysicalDevice = device;
        } else {
            auto chosenDevProps = any_cast<vk::PhysicalDevice>(chosenPhysicalDevice).getProperties2();
            switch (chosenDevProps.properties.deviceType) {
                case vk::PhysicalDeviceType::eDiscreteGpu: break;
                default:
                    if (devProperty.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                        chosenPhysicalDevice = device;
                    }
            }
        }
    }
    return any_cast<vk::PhysicalDevice>(chosenPhysicalDevice);
}

vk::UniqueSurfaceKHR create_surface_sdl(SDL_Window *p_SDLWindow, const vk::Instance &vkInstance) {
    VkSurfaceKHR tmpSurface{};
    if (SDL_Vulkan_CreateSurface(p_SDLWindow, vkInstance, &tmpSurface) != SDL_TRUE) {
        utils::log_and_pause("Surface creation failed");
        std::abort();
    }
    auto surface = vk::UniqueSurfaceKHR(tmpSurface, vkInstance);
    return std::move(surface);
}

vk::Format find_qualified_depth_format(const PhysicalDeviceInfo &physicalDeviceProps) {
    std::array<vk::Format, 3> depthCandidates = {
            vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint};
    std::array<std::tuple<vk::ImageTiling, vk::FormatFeatureFlags>, 1> tilingReqs{{
                                                                                          {vk::ImageTiling::eOptimal,
                                                                                           vk::FormatFeatureFlagBits::eDepthStencilAttachment}
                                                                                          //{eLinear, e...}
                                                                                  }};
    for (const auto &fmt: depthCandidates) {
        const auto &fmtFeats = physicalDeviceProps.formats.at(fmt).formatProperties;
        for (const auto &req: tilingReqs) {
            switch (std::get<0>(req)) {
                case vk::ImageTiling::eOptimal:
                    if ((std::get<1>(req) & fmtFeats.optimalTilingFeatures) == std::get<1>(req)) {
                        return fmt;
                    }
                    break;
                case vk::ImageTiling::eLinear:
                    if ((std::get<1>(req) & fmtFeats.linearTilingFeatures) == std::get<1>(req)) {
                        return fmt;
                    }
                default:
                    utils::log_and_pause("VkImageTiling can only be Optimal or Linear");
                    std::abort();
            }
        }
    }
    utils::log_and_pause("No valid depth format supported by device");
    std::abort();
}

bool want_exit_sdl() {
    SDL_Event sdlEvent;
    static bool bRet = false;

    while ((SDL_PollEvent(&sdlEvent) != 0) & (!bRet)) {
        if (sdlEvent.type == SDL_QUIT) {
            bRet = true;
        } else if (sdlEvent.type == SDL_KEYDOWN) {
            if (sdlEvent.key.keysym.sym == SDLK_ESCAPE
                || sdlEvent.key.keysym.sym == SDLK_q) {
                bRet = true;
            }
            if (sdlEvent.key.keysym.sym == SDLK_RETURN) {
                if (sdlEvent.key.keysym.mod & KMOD_ALT) {
                    // TODO: get full-screen mode working.
                    utils::log_and_pause("Full-screen mode alteration requested.", 0);
                }
            }
        }
    }
    return bRet;
}

std::optional<uint32_t> find_qualified_queue_family(const PhysicalDeviceInfo &physicalDeviceProps, size_t numQueues) {
    using QFlgs = vk::QueueFlagBits;
    auto queueFlags = QFlgs::eTransfer | QFlgs::eGraphics | QFlgs::eCompute;
    for (size_t i = 0; i < physicalDeviceProps.queueFamilies.size(); i++) {
        const auto &queueFamily = physicalDeviceProps.queueFamilies[i];
        if (
                (queueFamily.props.queueFamilyProperties.queueFlags & queueFlags) == queueFlags &&
                queueFamily.supportsSurface &&
                queueFamily.props.queueFamilyProperties.queueCount >= numQueues) {
            return i;
        }
    }
    return {};
}

vk::UniqueDevice
create_vulkan_device(const vk::PhysicalDevice &chosenPhysicalDevice, const PhysicalDeviceInfo &physDevInfo,
                     const std::vector<std::any> &featureList, const std::vector<std::string> &deviceExtensions,
                     uint32_t queueCGTPIdx, size_t numQueues) {
    auto queuePriorities = std::vector<float>(numQueues, 1.0f);
    auto queueCreateInfo = vk::DeviceQueueCreateInfo{};
    queueCreateInfo.queueFamilyIndex = queueCGTPIdx;
    queueCreateInfo.queueCount = numQueues;
    queueCreateInfo.setQueuePriorities(queuePriorities);
    vk::DeviceCreateInfo createInfo{};
    createInfo.setQueueCreateInfos(queueCreateInfo);
    // Add requested features
    auto pEnabledFeatures2 = &std::any_cast<const vk::PhysicalDeviceFeatures2 &>(featureList[0]);
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = (void *) pEnabledFeatures2;

    for (const auto& devExt: deviceExtensions){
        auto found=false;
        for (const auto& availDevExt: physDevInfo.devExts){
            if (std::string_view{availDevExt.extensionName} == devExt){
                found = true;
                break;
            }
        }
        if (!found){
            utils::log_and_pause(std::format("Device extension not available: {}", devExt, 0));
        }
    }

    // Assemble deviceExtension
    auto [vecPtrDeviceExtension, extensionCount] = utils::stringToVecptrU32(deviceExtensions);
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = vecPtrDeviceExtension.data();

    auto [result, logicDevice] = chosenPhysicalDevice.createDeviceUnique(createInfo);
    utils::vk_ensure(result);
    return std::move(logicDevice);
}

vk::Extent2D get_surface_size(SDL_Window *p_SDLWindow, vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface) {
    auto surfaceExtent = vk::Extent2D{};
    int width, height;
    SDL_GetWindowSize(p_SDLWindow, &width, &height);
    utils::log_and_pause(
            std::format(
                    "SDL_GetWindowSize: {}x{}",
                    width, height), 0);
    SDL_GetWindowSizeInPixels(p_SDLWindow, &width, &height);
    utils::log_and_pause(
            std::format(
                    "SDL_GetWindowSizeInPixels: {}x{}",
                    width, height), 0);
    SDL_Vulkan_GetDrawableSize(p_SDLWindow, &width, &height);
    surfaceExtent.width = width;
    surfaceExtent.height = height;
    utils::log_and_pause(
            std::format(
                    "SDL_Vulkan_GetDrawableSize: {}x{}",
                    surfaceExtent.width, surfaceExtent.height), 0);
    // Sometimes SDL's surface size report is unreliable.
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface).value;
    if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        surfaceExtent = surfaceCapabilities.currentExtent;
    }
    return surfaceExtent;
}

vk::SurfaceFormat2KHR find_qualified_surface_format_2(const std::span<const vk::SurfaceFormat2KHR> surfaceFormats2) {
    vk::SurfaceFormat2KHR surfaceFormat2{};
    auto preferredFmt = vk::Format::eB8G8R8A8Srgb;
    auto preferredColor = vk::ColorSpaceKHR::eSrgbNonlinear;
    surfaceFormat2 = surfaceFormats2[0];
    if (surfaceFormats2.size() == 1 && surfaceFormats2[0].surfaceFormat.format == vk::Format::eUndefined) {
        surfaceFormat2.surfaceFormat = {preferredFmt, preferredColor};
    } else {
        for (const auto &availableFormat: surfaceFormats2) {
            if (availableFormat.surfaceFormat.format == preferredFmt &&
                availableFormat.surfaceFormat.colorSpace == preferredColor) {
                surfaceFormat2 = availableFormat;
                break;
            }
        }
    }
    return surfaceFormat2;
}

vk::PresentModeKHR find_qualified_present_mode(const std::span<const vk::PresentModeKHR> surfacePresentModes) {
    vk::PresentModeKHR presentMode{};
    {
        vk::PresentModeKHR preferredMode = vk::PresentModeKHR::eFifoRelaxed;
        presentMode = vk::PresentModeKHR::eFifo;
        for (const auto &availablePresentMode: surfacePresentModes) {
            if (availablePresentMode == preferredMode) {
                presentMode = availablePresentMode;
                break;
            } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                presentMode = availablePresentMode;
            }
        }
    }
    return presentMode;
}

// oldSwapchain will be retired but not destroyed
vk::UniqueSwapchainKHR create_swapchain(vk::Device device, vk::PhysicalDevice physicalDevice,
                                        const vk::Extent2D &surfaceExtent, const vk::SurfaceFormat2KHR &surfaceFormat2,
                                        vk::ImageUsageFlags &imageUsage,
                                        vk::PresentModeKHR presentMode, uint32_t imageCount,
                                        vk::SurfaceKHR surface, vk::SwapchainKHR oldSwapchain = {}) {

    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface).value;
    utils::log_and_pause(
            std::format(
                    "\n\tPhysicalDeviceSurfaceCapabilitiesKHR.currentExtent: {}x{}",
                    surfaceCapabilities.currentExtent.width, surfaceCapabilities.currentExtent.height),0);
    vk::Extent2D extent{};
    {
        if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            extent = surfaceCapabilities.currentExtent;
        } else {
            extent.width = std::clamp(surfaceExtent.width,
                                      surfaceCapabilities.minImageExtent.width,
                                      surfaceCapabilities.maxImageExtent.width);
            extent.height = std::clamp(surfaceExtent.height,
                                       surfaceCapabilities.minImageExtent.height,
                                       surfaceCapabilities.maxImageExtent.height);
        }
    }
    auto currentTransform = surfaceCapabilities.currentTransform;
    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat2.surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat2.surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = imageUsage;
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = true;
    createInfo.oldSwapchain = oldSwapchain;
    auto copyOfCreateInfo = createInfo;
    auto [result, swapchain] = device.createSwapchainKHRUnique(createInfo);
    if (copyOfCreateInfo != createInfo) {
        utils::log_and_pause("WARN: VkSwapchainCreateInfoKHR{} was changed upon submission.", 0);
        imageUsage = createInfo.imageUsage;
    }
    utils::vk_ensure(result, "swapchain creation failed");
    utils::log_and_pause(std::format("Swapchain created: {}x{} Px.", createInfo.imageExtent.width, createInfo.imageExtent.height),0);
    return std::move(swapchain);
}

int main(int argc, char *argv[]) {
    /*The order of initializing global-scope stuffs
     *
     * Record validation layer and instance extension requirements
     *  Query Window Subsystem for other instance extension requirements
     * Create vk::Instance
     *  Set debug facility for vk::Instance
     * Get a surface handle from Window Subsystem
     * Query device information from vk::PhysicalDevice, mostly presentation and vk::Queue related info
     * Record device extension and feature requirements
     *  Query third-party libs for other device extension and feature requirements
     * Create vk::Device
     -------- Things below will happen again as we recover from surface size changes --------
     * Create vk::SwapchainKHR, its vk::Image, and vk::ImageView. Depth image can be created by renderer
     * Create one semaphore for each in-flight frame, renderer will signal the provided semaphore when completed rendering
     * Get one vk::Queue for each vulkan-related threads
     *
     * End of global objects initialization
     * */
    bool bDebug = false;
    // Setup DynamicLoader (process-wide)
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    // Setup SDL2
    auto p_SDLWindow = init_sdl();
    // Instance for Vulkan
    auto validationLayers = get_required_validation_layers(bDebug);
    auto instanceExtensions = get_required_instance_extensions(p_SDLWindow, bDebug);
    auto instance = create_vulkan_instance(validationLayers, instanceExtensions);
    // Setup DynamicLoader (instance-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    std::optional<vk::UniqueDebugUtilsMessengerEXT> vkDebugUtilMsg;
    if (bDebug) {
        vkDebugUtilMsg = create_vulkan_debug_msg(instance.get());
    }

    auto chosenPhysicalDevice = get_physical_device(instance.get());
    // SDL2's surface
    auto surfaceSDL = create_surface_sdl(p_SDLWindow, instance.get());
    auto physicalDeviceProps = query_physical_device_info(chosenPhysicalDevice, surfaceSDL);

    auto featureList = get_required_device_features_2(chosenPhysicalDevice);
    auto deviceExtensions = get_required_device_extensions();
    auto device = vk::UniqueDevice{};
    size_t numQueues = INFLIGHT_FRAMES + 1;// One for global commands, rest for the renderers.
    uint32_t queueFamilyIndex{};
    {
        auto queueFamIdx = find_qualified_queue_family(physicalDeviceProps, numQueues);
        if (!queueFamIdx.has_value()) {
            utils::log_and_pause("Valid queue family not found");
            std::abort();
        }
        queueFamilyIndex = queueFamIdx.value();
    }
    device = create_vulkan_device(chosenPhysicalDevice, physicalDeviceProps, featureList, deviceExtensions, queueFamilyIndex, numQueues);
    // Setup DynamicLoader (device-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());
    auto swapchainFormat = find_qualified_surface_format_2(physicalDeviceProps.surfaceFmts);
    auto swapchainImgUsg = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment;
    auto swapchainPresentMode = find_qualified_present_mode(physicalDeviceProps.surfacePres);
    auto depthFormat = find_qualified_depth_format(physicalDeviceProps);

    bool exitSignal = want_exit_sdl();
    auto swapchain = vk::UniqueSwapchainKHR{};
    uint64_t frameId = 0;
    SDL_StartTextInput();
    // Start rendering
    while (!exitSignal) {
        std::cout<<"MAIN: Init...\n"<<std::endl;
        // -------- Repetitive --------
        uint32_t swapchainImageCount =
                INFLIGHT_FRAMES*2+1;// It has nothing to do with in-flight frames! Driver may not adhere to it.
        // Cannot be smaller than INFLIGHT_FRAMES
        swapchainImageCount = std::clamp(swapchainImageCount,
                                         physicalDeviceProps.surfaceCaps.surfaceCapabilities.minImageCount,
                                         physicalDeviceProps.surfaceCaps.surfaceCapabilities.maxImageCount);
        utils::log_and_pause(std::format("Trying to acquire {} images from swapchain...", swapchainImageCount), 0);
        auto surfaceExtent = get_surface_size(p_SDLWindow, chosenPhysicalDevice, surfaceSDL.get());
        swapchain = create_swapchain(device.get(), chosenPhysicalDevice,
                                     surfaceExtent, swapchainFormat, swapchainImgUsg,
                                     swapchainPresentMode, swapchainImageCount,
                                     surfaceSDL.get(), swapchain.get());
        // Debuggers like Nsight Graphics will access the first queue at the main thread.
        size_t globalQueueIndex = 0;
        auto globalQueue = utils::QueueStruct{.queue = device->getQueue(queueFamilyIndex,
                                                                        globalQueueIndex), .queueFamilyIdx = queueFamilyIndex};
        auto swapchainImagenViews = std::vector<VulkanImageHandle>{};
        {
            auto [result, imgs] = device->getSwapchainImagesKHR(swapchain.get());
            utils::vk_ensure(result);
            for (const auto &img: imgs) {
                VulkanImageHandle imgHandle{};
                imgHandle.device = device.get();
                imgHandle.resource = img;
                imgHandle.resInfo.mipLevels = 1;
                imgHandle.resInfo.arrayLayers = 1;
                imgHandle.resInfo.imageType = vk::ImageType::e2D;
                //imgHandle.resInfo.flags = {};
                imgHandle.resInfo.format = swapchainFormat.surfaceFormat.format;
                imgHandle.resInfo.usage = swapchainImgUsg;
                imgHandle.resInfo.extent = {{.width = surfaceExtent.width, .height = surfaceExtent.height, .depth = 1}};
                imgHandle.resInfo.tiling = vk::ImageTiling::eOptimal;
                imgHandle.resInfo.samples = vk::SampleCountFlagBits::e1;
                imgHandle.resInfo.sharingMode = vk::SharingMode::eExclusive;
                imgHandle.resInfo.initialLayout = vk::ImageLayout::ePresentSrcKHR;
                imgHandle.createView(vk::ImageAspectFlagBits::eColor);
                swapchainImagenViews.push_back(std::move(imgHandle));
            }
        }
        swapchainImageCount = swapchainImagenViews.size();
        utils::log_and_pause(std::format("Acquired {} images from swapchain...", swapchainImageCount), 0);
        auto globalResMgr = VulkanResourceManager{instance.get(), chosenPhysicalDevice, device.get(), globalQueue};

        auto imageAvailableSemaphores = std::vector<vk::UniqueSemaphore>{};
        for (size_t i = 0; i < INFLIGHT_FRAMES; ++i) {
            auto [result, sema] = device->createSemaphoreUnique({});
            utils::vk_ensure(result);
            imageAvailableSemaphores.push_back(std::move(sema));
        }
        auto renderCompleteSemaphores = std::vector<vk::UniqueSemaphore>{};
        for (size_t i = 0; i < INFLIGHT_FRAMES; ++i) {
            auto [result, sema] = device->createSemaphoreUnique({});
            utils::vk_ensure(result);
            renderCompleteSemaphores.push_back(std::move(sema));
        }
        initialize_main_sync_objs();
        auto rendererSwapchainImageIndices = std::array<uint32_t, INFLIGHT_FRAMES>{};
        /* Renderer need the following global handles for initialization:
         * vk::Instance, vk::PhysicalDevice,
         * vk::Device, vk::Queue and its queueFamily index
         * Info for creating depth resources
         * Handle for "isImageAvailable" semaphore and "isRenderComplete"
         * During rendering, it also needs:
         * vk::Image and corresponding vk::ImageView from swapchain
         *
         * */
        std::array<std::thread, INFLIGHT_FRAMES> renderThreads{};
        frameId += 1;
        for (size_t threadIdx = 0; threadIdx < INFLIGHT_FRAMES; ++threadIdx) {
            renderThreads[threadIdx] = std::thread{
                    render_work_thread,
                    threadIdx,
                    instance.get(), chosenPhysicalDevice, physicalDeviceProps, device.get(),
                    queueFamilyIndex,
                    globalResMgr.getManagerHandle(),
                    surfaceExtent, swapchainFormat.surfaceFormat.format, swapchainImgUsg, depthFormat,
                    imageAvailableSemaphores[threadIdx].get(), renderCompleteSemaphores[threadIdx].get()};
        }
        bool resetSwapchain = false;
        int currentRenderer = 0;
        // Begin rendering loop
        std::cout<<"MAIN: Looping...\n"<<std::endl;
        while (!resetSwapchain & !exitSignal) {
            /* During rendering loop, the main thread should orchestrate the following things:
             * Check whether surface size is changed
             * For each loop, only one swapchain image is issued to one thread, and only one image is brought to presentation
             * (thread count should be the same as `INFLIGHT_FRAMES`, thus each thread only has one frame in-flight):
             *  Acquire swapchain image, image view, and check whether its size has changed, as the acquired image index may not
             *  be ready for rendering.
             *  The sync settings when calling vkAcquireNextImageKHR are:
             *   Signal "isImageAvailable" semaphore for this iteration's thread
             *   ~~Signal a "isResourceFree" fence, when this fence is signaled, the resource used when presenting the __acquired image__ can be freed.~~
             *   (Here we just YOLO it)
             *  Set "imageViewHandle" atomic to the obtained swapchain image's view
             *  Notify the thread that the swapchain image view handle is available ("isImageViewHandleAvailable" semaphore)
             *    The thread starts its pacing-insensitive tasks for the frame
             *    The thread waits for thread's own "isImagePresented fence"(vkWaitForPresentKHR uses presentId as "fence"'s identifier)
             *    The thread starts its non-submission tasks
             *    The thread waits the main thread to give the index of a workable swapchain image
             *     Vulkan does not participate in CPU-CPU syncs, thus C++ threading facilities are needed. ("isImageViewHandleAvailable" semaphore)
             *    The thread starts submitting commands, with the following sync settings:
             *     Waits the "isImageAvailable" semaphore at color attachment stage.
             *     Signals the "isRenderingComplete" semaphore given by the main thread once swapchain image's attachment has
             *     been written with final contents.
             *    After submission, the thread can notify the main thread that it had consumed the given image. ("isImageViewHandleConsumed" semaphore)
             *  Wait for the next thread to consume the swapchain image given last time. ("isImageViewHandleConsumed" semaphore):
             *  If timeout, check whether "imageViewHandle" is null, if that's the case, we can safely skip the presentation for the next thread.
             *  Present next thread's rendered image (in terms of timeline, "next" means the oldest among in-flight frames)
             *  with the following sync settings:
             *   Wait the "isRenderingComplete" semaphore
             *   Supply vk::PresentIdKHR with presentId equal to next thread's index to prepare for next thread's "isImagePresented fence"
             *   //Signal "isResourceFree" fence for resource destruction.
             *   //(Impossible to set fence here for now, as device support for VK_EXT_swapchain_maintenance1 is non-existent)
             * Iterate to the next frame
             * */
            exitSignal |= want_exit_sdl();
            if (mainRendererComms[currentRenderer].mainLoopReady.load()) {
                if (!mainRendererComms[currentRenderer].imageViewRendered.try_acquire_for(
                        std::chrono::milliseconds(100))) {
                    std::cout << std::format("Renderer {} is lagging!\n", currentRenderer) << std::endl;
                    exitSignal = want_exit_sdl();
                }
                vk::PresentInfoKHR presentInfo{};
                presentInfo.setWaitSemaphores(renderCompleteSemaphores[currentRenderer].get());
                presentInfo.setSwapchains(swapchain.get());
                presentInfo.setImageIndices(rendererSwapchainImageIndices[currentRenderer]);
                vk::PresentIdKHR presentId{};
                uint64_t presId = frameId;
                frameId += 1;
                presentId.setPresentIds(presId);
                presentId.swapchainCount = 1;
                vk::StructureChain<vk::PresentInfoKHR, vk::PresentIdKHR> presentInfoCombined{
                        presentInfo, presentId
                };
                auto resultPresent = globalQueue.queue.presentKHR(presentInfoCombined.get<vk::PresentInfoKHR>());
                if (resultPresent == vk::Result::eErrorOutOfDateKHR || resultPresent == vk::Result::eSuboptimalKHR) {
                    resetSwapchain = true;
                    utils::log_and_pause("vk::Result::eErrorOutOfDateKHR", 0);
                } else {
                    utils::vk_ensure(resultPresent);
                }
            } else {
                mainRendererComms[currentRenderer].mainLoopReady.store(true);
                utils::log_and_pause(std::format("Main thread is ready for Renderer {}", currentRenderer), 0);
            }

            vk::Result result{};
            uint32_t imageIndex; // This is NOT the same as `frames` !
            std::tie(result, imageIndex) = device->acquireNextImageKHR(
                    swapchain.get(), std::numeric_limits<uint64_t>::max(),
                    imageAvailableSemaphores[currentRenderer].get(), nullptr);
            if (result == vk::Result::eErrorOutOfDateKHR) {
                resetSwapchain = true;
                utils::log_and_pause("vk::Result::eErrorOutOfDateKHR", 0);
                continue;
            } else if (result != vk::Result::eSuboptimalKHR) {
                utils::vk_ensure(result);
            }

            mainRendererComms[currentRenderer].imageViewHandle.store(swapchainImagenViews[imageIndex].view.get());
            mainRendererComms[currentRenderer].imageViewReadyToRender.release();

            rendererSwapchainImageIndices[currentRenderer] = imageIndex;

            currentRenderer = (currentRenderer + 1) % INFLIGHT_FRAMES;

        }// Out of rendering loop, prepare to reset resources
        /* To handle swapchain resize events, we need even more sync:
            * When the main thread is notified through vkAcquireNextImageKHR or vkQueuePresentKHR,
            *  The main thread sets "isSwapchainInvalid" atomic for all rendering threads.
            * Then the main thread waits all rendering threads to terminate.
            *   The rendering thread needs to check "isSwapchainInvalid" atomic before waiting anything signaled by main thread.
            *   This includes "isImageViewHandleAcquired" semaphore and "isImagePresented fence".
            *   The rendering thread now needs to check "isSwapchainInvalid" atomic when waiting every fence and semaphore:
            *    Since we don't need to rush when resetting swapchain, the waiting time can be set to a fairly large value.
            *    We now need a loop and use std::binary_semaphore::try_acquire_for() when waiting for "isImageViewHandleAcquired" semaphore.
            *    When waiting "isImagePresented fence", we have an infinite loop. Inside the loop, we set a time limit when
            *     calling vkWaitForPresentKHR, for each iteration we check the atomic.
            * After all rendering threads are terminated, the main thread can start its reset procedure.
            * */
        notify_renderer_exit();
        for (auto &renderer: renderThreads) {
            renderer.join();
        }
        if (!exitSignal) {
            // We are not going to terminate the program, so rebuild swapchain etc.
            if (resetSwapchain) {
                // Wait until surface size is valid for rendering
                bool isSurfaceZeroSize;
                do {
                    surfaceExtent = get_surface_size(p_SDLWindow, chosenPhysicalDevice, surfaceSDL.get());
                    if (surfaceExtent.width * surfaceExtent.height == 0) {
                        isSurfaceZeroSize = true;
                    } else {
                        isSurfaceZeroSize = false;
                    }
                    exitSignal = want_exit_sdl();
                    // Wait for at most 200ms, we don't care what the event is though.
                    SDL_Event sdlEvent{};
                    SDL_WaitEventTimeout(&sdlEvent, 200);
                } while (!exitSignal & isSurfaceZeroSize);
                wait_vulkan_device_idle(device.get());
                // We only need to explicitly destroy swapchain's ImageViews
                swapchainImagenViews.clear();
            } else {
                //Shouldn't happen for now
            }
        }
        // Workaround for the lack of VK_EXT_swapchain_maintenance1 support
        wait_vulkan_device_idle(device.get());
    }
    // End of rendering, clean up
    wait_vulkan_device_idle(device.get());
    clean_sdl(p_SDLWindow);
    return 0;
}
