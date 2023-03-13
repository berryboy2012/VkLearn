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
    vk::SurfaceKHR surface_{};// Signify which surface are those info refer to
    std::vector<QueueFamilyProps> queueFamilies{};
    vk::PhysicalDeviceProperties2 props{};
    std::unordered_map<vk::Format, vk::FormatProperties2> formats{};
    vk::SurfaceCapabilities2KHR surfaceCaps{};
    std::vector<vk::SurfaceFormat2KHR> surfaceFmts{};
    std::vector<vk::PresentModeKHR> surfacePres{};
};
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include "shader_modules.hpp"
#include "graphics_pipeline.hpp"
#include "bindings_management.hpp"
#include "renderpass.hpp"
#include "commands_management.h"
#include "memory_management.hpp"
#include "resource_management.hpp"
#include "render_thread.hpp"

void queryOVR() {
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

SDL_Window *initSDL() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Vulkan_LoadLibrary(nullptr);
    auto window = SDL_CreateWindow(
            "VkLearn",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            640, 360,
            SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN);
    SDL_SetWindowResizable(window, SDL_TRUE);
    return window;
}

void cleanSDL(SDL_Window *&window) {
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

std::vector<std::string> getRequiredValidationLayers(bool enableValidationLayers) {
    auto validationLayers = std::vector<std::string>{"VK_LAYER_KHRONOS_validation"};
    if (enableValidationLayers) {
        auto [layerResult, availableLayers] = vk::enumerateInstanceLayerProperties();
        for (const auto &requestedLayer: validationLayers) {
            bool layerFound = false;
            for (const auto &availableLayer: availableLayers) {
                if (std::string_view(availableLayer.layerName) == requestedLayer) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                std::abort();
            }
        }
    }
    return validationLayers;
}

std::vector<std::string> getRequiredInstanceExtensions(SDL_Window *window, bool enableValidationLayers) {
    auto instanceExtensions = std::vector<std::string>{};
    {
        // First get extensions required by SDL
        uint32_t extensionCount;
        SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr);
        auto extensionNames = std::vector<const char *>{extensionCount};
        SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensionNames.data());
        for (auto p_extensionName: extensionNames) {
            instanceExtensions.push_back(std::string(p_extensionName));
        }
    }
    // We require other extensions as well
    if (enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    instanceExtensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
    return instanceExtensions;
}

PhysicalDeviceInfo
queryPhysicalDeviceInfo(const vk::PhysicalDevice chosenPhysicalDevice, vk::UniqueSurfaceKHR &surfaceSDL) {
    PhysicalDeviceInfo physicalDeviceProps{};
    physicalDeviceProps.surface_ = surfaceSDL.get();
    auto queueFamilies = chosenPhysicalDevice.getQueueFamilyProperties2();
    for (size_t i = 0; i < queueFamilies.size(); ++i) {
        auto [result, surfaceSupported] = chosenPhysicalDevice.getSurfaceSupportKHR(i, physicalDeviceProps.surface_);
        utils::vkEnsure(result);
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
            physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    std::tie(result, physicalDeviceProps.surfaceFmts) = chosenPhysicalDevice.getSurfaceFormats2KHR(
            physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    std::tie(result, physicalDeviceProps.surfacePres) = chosenPhysicalDevice.getSurfacePresentModesKHR(
            physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    return physicalDeviceProps;
}

//TODO: add extension query for OpenVR
//Remember, most extensions require respective features to work
std::vector<std::string> getRequiredDeviceExtensions() {
    std::vector<std::string> requiredDeviceExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            // We want robust interaction with window subsystem (in the future, as device support is non-existent ATM)
            //VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
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
std::vector<std::any> getRequiredDeviceFeatures2(const vk::PhysicalDevice &device) {
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

vk::UniqueInstance createVulkanInstance(const std::vector<std::string> &validationLayers,
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
    utils::vkEnsure(result);
    return std::move(vkUniqueInstance);
}

vk::UniqueDebugUtilsMessengerEXT createVulkanDebugMsg(const vk::Instance &vkInstance) {
    using SeverityBits = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using MsgTypeBits = vk::DebugUtilsMessageTypeFlagBitsEXT;

    auto createInfo = vk::DebugUtilsMessengerCreateInfoEXT{
            {},
            SeverityBits::eWarning | SeverityBits::eError, // | SeverityBits::eInfo | SeverityBits::eVerbose,
            MsgTypeBits::eGeneral | MsgTypeBits::ePerformance | MsgTypeBits::eValidation |
            MsgTypeBits::eDeviceAddressBinding,
            &utils::debugUtilsMessengerCallback
    };

    auto [debugResult, debugUtilsMessengerUnique] = vkInstance.createDebugUtilsMessengerEXTUnique(createInfo);
    return std::move(debugUtilsMessengerUnique);

}

vk::PhysicalDevice getPhysicalDevice(const vk::Instance &vkInstance) {
    auto [resultDevices, devices] = vkInstance.enumeratePhysicalDevices();
    utils::vkEnsure(resultDevices);
    auto chosenPhysicalDevice = vk::PhysicalDevice{};
    for (const auto &device: devices) {
        auto devProperty = device.getProperties2();
        std::cout << std::string_view{devProperty.properties.deviceName} << std::endl;
        chosenPhysicalDevice = device;
    }
    return chosenPhysicalDevice;
}

vk::UniqueSurfaceKHR createSurfaceSDL(SDL_Window *p_SDLWindow, const vk::Instance &vkInstance) {
    VkSurfaceKHR tmpSurface{};
    if (SDL_Vulkan_CreateSurface(p_SDLWindow, vkInstance, &tmpSurface) != SDL_TRUE) {
        std::abort();
    }
    auto surface = vk::UniqueSurfaceKHR(tmpSurface, vkInstance);
    return std::move(surface);
}

vk::Format findQualifiedDepthFormat(const PhysicalDeviceInfo &physicalDeviceProps) {
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
                    std::abort();//Not implemented
            }
        }
    }
    std::abort();
}

bool wantExitSDL() {
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
        }
    }
    return bRet;
}

std::optional<uint32_t> findQualifiedQueueFamily(const PhysicalDeviceInfo &physicalDeviceProps, size_t numQueues) {
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
createVulkanDevice(const vk::PhysicalDevice &chosenPhysicalDevice,
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

    // Assemble deviceExtension
    auto [vecPtrDeviceExtension, extensionCount] = utils::stringToVecptrU32(deviceExtensions);
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = vecPtrDeviceExtension.data();

    auto [result, logicDevice] = chosenPhysicalDevice.createDeviceUnique(createInfo);
    utils::vkEnsure(result);
    return std::move(logicDevice);
}

vk::Extent2D getSurfaceSize(SDL_Window *p_SDLWindow) {
    auto surfaceExtent = vk::Extent2D{};
    int width, height;
    SDL_Vulkan_GetDrawableSize(p_SDLWindow, &width, &height);
    surfaceExtent.width = width;
    surfaceExtent.height = height;
    return surfaceExtent;
}

vk::SurfaceFormat2KHR findQualifiedSurfaceFormat2(const std::span<const vk::SurfaceFormat2KHR> surfaceFormats2) {
    vk::SurfaceFormat2KHR surfaceFormat2{};
    auto preferredFmt = vk::Format::eB8G8R8A8Unorm;
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

vk::PresentModeKHR findQualifiedPresentMode(const std::span<const vk::PresentModeKHR> surfacePresentModes) {
    vk::PresentModeKHR presentMode{};
    {
        vk::PresentModeKHR preferredMode = vk::PresentModeKHR::eFifo;//Relaxed;
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
vk::UniqueSwapchainKHR createSwapchain(vk::Device device, vk::PhysicalDevice physicalDevice,
                                       const vk::Extent2D &surfaceExtent, const vk::SurfaceFormat2KHR &surfaceFormat2,
                                       vk::PresentModeKHR presentMode, uint32_t imageCount,
                                       vk::SurfaceKHR surface, vk::SwapchainKHR oldSwapchain = {}) {

    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface).value;
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
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = true;
    createInfo.oldSwapchain = oldSwapchain;
    auto [result, swapchain] = device.createSwapchainKHRUnique(createInfo);
    utils::vkEnsure(result, "swapchain creation failed");
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
    bool bDebug = true;
    // Setup DynamicLoader (process-wide)
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    // Setup SDL2
    auto p_SDLWindow = initSDL();
    // Instance for Vulkan
    auto validationLayers = getRequiredValidationLayers(bDebug);
    auto instanceExtensions = getRequiredInstanceExtensions(p_SDLWindow, bDebug);
    auto instance = createVulkanInstance(validationLayers, instanceExtensions);
    // Setup DynamicLoader (instance-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    if (bDebug) {
        auto vkDebugUtilMsg = createVulkanDebugMsg(instance.get());
    }

    auto chosenPhysicalDevice = getPhysicalDevice(instance.get());
    // SDL2's surface
    auto surfaceSDL = createSurfaceSDL(p_SDLWindow, instance.get());
    auto physicalDeviceProps = queryPhysicalDeviceInfo(chosenPhysicalDevice, surfaceSDL);

    auto featureList = getRequiredDeviceFeatures2(chosenPhysicalDevice);
    auto deviceExtensions = getRequiredDeviceExtensions();
    auto device = vk::UniqueDevice{};
    size_t numQueues = INFLIGHT_FRAMES + 1;// One for global commands, rest for the renderers.
    uint32_t queueFamilyIndex{};
    {
        auto queueFamIdx = findQualifiedQueueFamily(physicalDeviceProps, numQueues);
        if (!queueFamIdx.has_value()) {
            std::abort();
        }
        queueFamilyIndex = queueFamIdx.value();
    }
    device = createVulkanDevice(chosenPhysicalDevice, featureList, deviceExtensions, queueFamilyIndex, numQueues);
    // Setup DynamicLoader (device-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());
    auto swapchainFormat = findQualifiedSurfaceFormat2(physicalDeviceProps.surfaceFmts);
    auto swapchainPresentMode = findQualifiedPresentMode(physicalDeviceProps.surfacePres);
    auto depthFormat = findQualifiedDepthFormat(physicalDeviceProps);

    bool exitSignal = wantExitSDL();
    auto swapchain = vk::UniqueSwapchainKHR{};
    uint64_t frameId = 0;
    SDL_StartTextInput();
    // Start rendering
    while (!exitSignal) {
        std::cout<<"MAIN: Init...\n"<<std::endl;
        // -------- Repetitive --------
        uint32_t swapchainImageCount =
                INFLIGHT_FRAMES + 1;// It has nothing to do with in-flight frames! Driver may not adhere to it.
        // Cannot be smaller than INFLIGHT_FRAMES
        swapchainImageCount = std::clamp(swapchainImageCount,
                                         physicalDeviceProps.surfaceCaps.surfaceCapabilities.minImageCount,
                                         physicalDeviceProps.surfaceCaps.surfaceCapabilities.maxImageCount);
        auto surfaceExtent = getSurfaceSize(p_SDLWindow);
        swapchain = createSwapchain(device.get(), chosenPhysicalDevice,
                                    surfaceExtent, swapchainFormat, swapchainPresentMode, swapchainImageCount,
                                    surfaceSDL.get(), swapchain.get());
        size_t globalQueueIndex = INFLIGHT_FRAMES;
        auto globalQueue = utils::QueueStruct{.queue = device->getQueue(queueFamilyIndex,
                                                                        globalQueueIndex), .queueFamilyIdx = queueFamilyIndex};
        auto swapchainImagenViews = std::vector<VulkanImageHandle>{};
        {
            auto [result, imgs] = device->getSwapchainImagesKHR(swapchain.get());
            utils::vkEnsure(result);
            for (const auto &img: imgs) {
                VulkanImageHandle imgHandle{};
                imgHandle.device_ = device.get();
                imgHandle.resource = img;
                imgHandle.resInfo.mipLevels = 1;
                imgHandle.resInfo.arrayLayers = 1;
                imgHandle.resInfo.imageType = vk::ImageType::e2D;
                //imgHandle.resInfo.flags = {};
                imgHandle.resInfo.format = swapchainFormat.surfaceFormat.format;
                imgHandle.resInfo.usage =
                        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
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
        auto globalResMgr = VulkanResourceManager{instance.get(), chosenPhysicalDevice, device.get(), globalQueue};

        auto imageAvailableSemaphores = std::vector<vk::UniqueSemaphore>{};
        for (size_t i = 0; i < INFLIGHT_FRAMES; ++i) {
            auto [result, sema] = device->createSemaphoreUnique({});
            utils::vkEnsure(result);
            imageAvailableSemaphores.push_back(std::move(sema));
        }
        auto renderCompleteSemaphores = std::vector<vk::UniqueSemaphore>{};
        for (size_t i = 0; i < INFLIGHT_FRAMES; ++i) {
            auto [result, sema] = device->createSemaphoreUnique({});
            utils::vkEnsure(result);
            renderCompleteSemaphores.push_back(std::move(sema));
        }
        initializeMainSyncObjs();
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
                    renderWorkThread,
                    threadIdx,
                    instance.get(), chosenPhysicalDevice, physicalDeviceProps, device.get(),
                    queueFamilyIndex,
                    globalResMgr.getManagerHandle(),
                    surfaceExtent, swapchainFormat.surfaceFormat.format, depthFormat,
                    imageAvailableSemaphores[threadIdx].get(), renderCompleteSemaphores[threadIdx].get()};
        }
        bool resetSwapchain = false;
        int currentRenderer = 0;
        // Begin rendering loop
        std::cout<<"MAIN: Looping...\n"<<std::endl;
        while (!resetSwapchain & !exitSignal) {
            exitSignal |= wantExitSDL();

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

            vk::Result result{};
            uint32_t imageIndex; // This is NOT the same as `frames` !
            std::tie(result, imageIndex) = device->acquireNextImageKHR(
                    swapchain.get(), std::numeric_limits<uint64_t>::max(),
                    imageAvailableSemaphores[currentRenderer].get(), nullptr);
            if (result == vk::Result::eErrorOutOfDateKHR) {
                resetSwapchain = true;
                continue;
            } else if (result != vk::Result::eSuboptimalKHR) {
                utils::vkEnsure(result);
            }

            mainRendererComms[currentRenderer].imageViewHandle.store(swapchainImagenViews[imageIndex].view.get());
            mainRendererComms[currentRenderer].imageViewHandleAvailable.release();
            rendererSwapchainImageIndices[currentRenderer] = imageIndex;
            int nextRenderer = (currentRenderer + 1) % int(INFLIGHT_FRAMES);
            bool isFirstLoop = false;
            while (!mainRendererComms[nextRenderer].imageViewHandleConsumed.try_acquire_for(
                    std::chrono::milliseconds(100))) {
                if (mainRendererComms[nextRenderer].imageViewHandle.load() == vk::ImageView{}) {
                    isFirstLoop = true;
                    break;
                } else {
                    std::cout << std::format("Renderer {} is lagging!\n", nextRenderer) << std::endl;
                    exitSignal = wantExitSDL();
                }
            }
            if (!isFirstLoop) {
                vk::PresentInfoKHR presentInfo{};
                presentInfo.setWaitSemaphores(renderCompleteSemaphores[nextRenderer].get());
                presentInfo.setSwapchains(swapchain.get());
                presentInfo.setImageIndices(rendererSwapchainImageIndices[nextRenderer]);
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
                } else {
                    utils::vkEnsure(resultPresent);
                }
            }
            currentRenderer = (currentRenderer + 1) % INFLIGHT_FRAMES;
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
        }// Out of rendering loop, prepare to reset resources
        notifyRendererExit();
        for (auto &renderer: renderThreads) {
            renderer.join();
        }
        // Are we going to terminate the program?
        if (!exitSignal) {
            if (resetSwapchain) {
                // Wait until surface size is valid for rendering
                bool isSurfaceZeroSize;
                do {
                    surfaceExtent = getSurfaceSize(p_SDLWindow);
                    if (surfaceExtent.width * surfaceExtent.height == 0) {
                        isSurfaceZeroSize = true;
                    } else {
                        isSurfaceZeroSize = false;
                    }
                    exitSignal = wantExitSDL();
                } while (!exitSignal & isSurfaceZeroSize);
                utils::vkEnsure(device->waitIdle());
                // We only need to explicitly destroy swapchain's ImageViews
                swapchainImagenViews.clear();
            } else {
                //Shouldn't happen for now
            }
        }
        // Workaround for the lack of VK_EXT_swapchain_maintenance1 support
        utils::vkEnsure(device->waitIdle());
    }
    // End of rendering, clean up
    utils::vkEnsure(device->waitIdle());
    cleanSDL(p_SDLWindow);
    return 0;
}
