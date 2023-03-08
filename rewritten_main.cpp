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

// Required by Vulkan-Hpp (https://github.com/KhronosGroup/Vulkan-Hpp#extensions--per-device-function-pointers)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include "utils.h"

struct QueueFamilyProps{
    vk::QueueFamilyProperties2 props{};
    bool supportsSurface{};
};
struct PhysicalDeviceInfo{
    vk::SurfaceKHR surface_{};// Signify which surface are those info refer to
    std::vector<QueueFamilyProps> queueFamilies{};
    vk::PhysicalDeviceProperties2 props{};
    std::unordered_map<vk::Format, vk::FormatProperties2> formats{};
    vk::SurfaceCapabilities2KHR surfaceCaps{};
    std::vector<vk::SurfaceFormat2KHR> surfaceFmts{};
    std::vector<vk::PresentModeKHR> surfacePres{};
};
std::vector<vk::UniqueImageView> createImageViews(
        const std::span<vk::Image>& images,
        const vk::Format &format, const vk::ImageAspectFlags &imageAspect,
        vk::Device &device);

#include "shader_modules.hpp"
#include "graphics_pipeline.hpp"
#include "bindings_management.hpp"
#include "renderpass.hpp"
#include "commands_management.h"
#include "memory_management.hpp"
void queryOVR(){
    vr::EVRInitError eError = vr::VRInitError_None;
    auto m_pHMD = vr::VR_Init( &eError, vr::VRApplication_Utility );
    auto OVRVer = m_pHMD->GetRuntimeVersion();
    std::cout<<"OpenVR runtime version: "<<OVRVer<<std::endl;
    if ( eError != vr::VRInitError_None )
    {
        m_pHMD = nullptr;
        char buf[1024];
        sprintf_s( buf, sizeof( buf ), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription( eError ) );
        SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, nullptr );
    }
    else{
        vr::VR_Shutdown();
        m_pHMD = nullptr;
    }

}
SDL_Window* initSDL(){
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
void cleanSDL(SDL_Window* &window){
    SDL_DestroyWindow(window);
    window = nullptr;
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
}

std::vector<std::string> getRequiredValidationLayers(bool enableValidationLayers) {
    auto validationLayers = std::vector<std::string>{"VK_LAYER_KHRONOS_validation"};
    if (enableValidationLayers) {
        auto [layerResult, availableLayers] = vk::enumerateInstanceLayerProperties();
        for (const auto& requestedLayer: validationLayers){
            bool layerFound = false;
            for (const auto& availableLayer: availableLayers){
                if (std::string_view(availableLayer.layerName) == requestedLayer){
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound){
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
        auto extensionNames = std::vector<const char*>{extensionCount};
        SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensionNames.data());
        for (auto p_extensionName: extensionNames){
            instanceExtensions.push_back(std::string(p_extensionName));
        }
    }
    // We require other extensions as well
    if (enableValidationLayers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return instanceExtensions;
}

PhysicalDeviceInfo
queryPhysicalDeviceInfo(const vk::PhysicalDevice chosenPhysicalDevice, vk::UniqueSurfaceKHR &surfaceSDL) {
    PhysicalDeviceInfo physicalDeviceProps{};
    physicalDeviceProps.surface_ = surfaceSDL.get();
    auto queueFamilies = chosenPhysicalDevice.getQueueFamilyProperties2();
    for (size_t i=0;i<queueFamilies.size();++i){
        auto [result, surfaceSupported] = chosenPhysicalDevice.getSurfaceSupportKHR(i, physicalDeviceProps.surface_);
        utils::vkEnsure(result);
        auto queueFamilyProp = QueueFamilyProps{
                .props = queueFamilies[i],
                .supportsSurface = static_cast<bool>(surfaceSupported)};
        physicalDeviceProps.queueFamilies.emplace_back(queueFamilyProp);
    }
    for (const auto fmt : utils::enum_range<vk::Format>(vk::Format::eUndefined, vk::Format::eR12X4UnormPack16KHR)){
        auto fmtProps = chosenPhysicalDevice.getFormatProperties2(fmt);
        physicalDeviceProps.formats[fmt] = fmtProps;
    }
    auto result = vk::Result{};
    std::tie(result, physicalDeviceProps.surfaceCaps) = chosenPhysicalDevice.getSurfaceCapabilities2KHR(physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    std::tie(result, physicalDeviceProps.surfaceFmts) = chosenPhysicalDevice.getSurfaceFormats2KHR(physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    std::tie(result, physicalDeviceProps.surfacePres) = chosenPhysicalDevice.getSurfacePresentModesKHR(physicalDeviceProps.surface_);
    utils::vkEnsure(result);
    return physicalDeviceProps;
}

//TODO: add extension query for OpenVR
//Remember, most extensions require respective features to work
std::vector<std::string> getRequiredDeviceExtensions(){
    std::vector< std::string > requiredDeviceExtensions{
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
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
// Should use vk::StructureChain instead. Do not copy the returned list!
std::vector<std::any> getRequiredDeviceFeatures2(const vk::PhysicalDevice &device){
    using Feature2 = vk::PhysicalDeviceFeatures2;
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-02829
    using Vulkan11 = vk::PhysicalDeviceVulkan11Features;
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-02830
    using Vulkan12 = vk::PhysicalDeviceVulkan12Features;
    // https://vulkan.lunarg.com/doc/view/1.3.239.0/windows/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pNext-06532
    using Vulkan13 = vk::PhysicalDeviceVulkan13Features;
    using ASFeature = vk::PhysicalDeviceAccelerationStructureFeaturesKHR;
    using RTPFeature = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR;
    using VIDSFeature = vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT;
    using RayFeature = vk::PhysicalDeviceRayQueryFeaturesKHR;
    // Add your feature requirements below
    auto featureList = std::vector<std::any>{};
    featureList.push_back(Feature2{});
    featureList.push_back(ASFeature{});
    featureList.push_back(RTPFeature{});
    featureList.push_back(Vulkan11{});
    featureList.push_back(Vulkan12{});
    featureList.push_back(Vulkan13{});
    featureList.push_back(VIDSFeature{});
    featureList.push_back(RayFeature{});
    // And here
    std::any_cast<Feature2&>(featureList[0]).pNext = (void*)&std::any_cast<ASFeature&>(featureList[1]);
    std::any_cast<ASFeature&>(featureList[1]).pNext = (void*)&std::any_cast<RTPFeature&>(featureList[2]);
    std::any_cast<RTPFeature&>(featureList[2]).pNext = (void*)&std::any_cast<Vulkan11&>(featureList[3]);
    std::any_cast<Vulkan11&>(featureList[3]).pNext = (void*)&std::any_cast<Vulkan12&>(featureList[4]);
    std::any_cast<Vulkan12&>(featureList[4]).pNext = (void*)&std::any_cast<Vulkan13&>(featureList[5]);
    std::any_cast<Vulkan13&>(featureList[5]).pNext = (void*)&std::any_cast<VIDSFeature&>(featureList[6]);
    std::any_cast<VIDSFeature&>(featureList[6]).pNext = (void*)&std::any_cast<RayFeature&>(featureList[7]);
    // And here
    device.getFeatures2(&std::any_cast<Feature2&>(featureList[0]));
    return featureList;
}

vk::UniqueInstance createVulkanInstance(const std::vector<std::string> &validationLayers, const std::vector<std::string> &instanceExtensions) {
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


    std::vector<const char*> pLayerNames;
    for (const auto& requestedLayer: validationLayers){
        pLayerNames.push_back(requestedLayer.c_str());
    }
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = pLayerNames.data();

    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    std::vector<const char*> pExtensionNames;
    for (const auto& requestedExtension: instanceExtensions){
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
            MsgTypeBits::eGeneral | MsgTypeBits::ePerformance | MsgTypeBits::eValidation | MsgTypeBits::eDeviceAddressBinding,
            &utils::debugUtilsMessengerCallback
    };

    auto [debugResult, debugUtilsMessengerUnique] = vkInstance.createDebugUtilsMessengerEXTUnique(createInfo);
    return std::move(debugUtilsMessengerUnique);

}

vk::PhysicalDevice getPhysicalDevice(const vk::Instance &vkInstance) {
    auto [resultDevices, devices] = vkInstance.enumeratePhysicalDevices();
    utils::vkEnsure(resultDevices);
    auto chosenPhysicalDevice = vk::PhysicalDevice{};
    for (const auto& device: devices){
        auto devProperty = device.getProperties2();
        std::cout<<std::string_view{devProperty.properties.deviceName}<<std::endl;
        chosenPhysicalDevice = device;
    }
    return chosenPhysicalDevice;
}

vk::UniqueSurfaceKHR createSurfaceSDL(SDL_Window *p_SDLWindow, const vk::Instance &vkInstance) {
    VkSurfaceKHR tmpSurface{};
    if (SDL_Vulkan_CreateSurface(p_SDLWindow, vkInstance, &tmpSurface) != SDL_TRUE){
        std::abort();
    }
    auto surface = vk::UniqueSurfaceKHR(tmpSurface, vkInstance);
    return std::move(surface);
}
vk::Format findSupportedFormat(
        const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features,
        const vk::PhysicalDevice &physicalDevice) {
    for (vk::Format format : candidates) {
        auto props = physicalDevice.getFormatProperties2(format);

        if (tiling == vk::ImageTiling::eLinear && (props.formatProperties.linearTilingFeatures & features) == features){
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal && (props.formatProperties.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    std::abort();
}
vk::Format findDepthFormat(const vk::PhysicalDevice &physicalDevice) {
    return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment,
            physicalDevice);
}
bool wantExitSDL(){
    SDL_Event sdlEvent;
    bool bRet = false;

    while ( SDL_PollEvent( &sdlEvent ) != 0 )
    {
        if ( sdlEvent.type == SDL_QUIT )
        {
            bRet = true;
        }
        else if ( sdlEvent.type == SDL_KEYDOWN )
        {
            if ( sdlEvent.key.keysym.sym == SDLK_ESCAPE
                 || sdlEvent.key.keysym.sym == SDLK_q )
            {
                bRet = true;
            }
        }
    }
    return bRet;
}
std::optional<uint32_t> findQualifiedQueueFamily(const PhysicalDeviceInfo &physicalDeviceProps, size_t numQueues){
    using QFlgs = vk::QueueFlagBits;
    auto queueFlags = QFlgs::eTransfer|QFlgs::eGraphics|QFlgs::eCompute;
    for (size_t i=0;i<physicalDeviceProps.queueFamilies.size();i++){
        const auto& queueFamily = physicalDeviceProps.queueFamilies[i];
        if (
                (queueFamily.props.queueFamilyProperties.queueFlags & queueFlags)==queueFlags &&
                queueFamily.supportsSurface &&
                queueFamily.props.queueFamilyProperties.queueCount >= numQueues){
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
    queueCreateInfo.queueFamilyIndex = std::any_cast<size_t>(queueCGTPIdx);
    queueCreateInfo.queueCount = numQueues;
    queueCreateInfo.setQueuePriorities(queuePriorities);
    vk::DeviceCreateInfo createInfo{};
    createInfo.setQueueCreateInfos(queueCreateInfo);
    // Add requested features
    auto pEnabledFeatures2 = &std::any_cast<const vk::PhysicalDeviceFeatures2&>(featureList[0]);
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = (void*)pEnabledFeatures2;

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
vk::SurfaceFormat2KHR findQualifiedSurfaceFormat2(const std::span<const vk::SurfaceFormat2KHR> surfaceFormats2){
    vk::SurfaceFormat2KHR surfaceFormat2{};
    auto preferredFmt = vk::Format::eB8G8R8A8Unorm;
    auto preferredColor = vk::ColorSpaceKHR::eSrgbNonlinear;
    surfaceFormat2 = surfaceFormats2[0];
    if (surfaceFormats2.size() == 1 && surfaceFormats2[0].surfaceFormat.format == vk::Format::eUndefined) {
        surfaceFormat2.surfaceFormat = { preferredFmt, preferredColor };
    }
    else{
        for (const auto& availableFormat : surfaceFormats2) {
            if (availableFormat.surfaceFormat.format == preferredFmt &&
                availableFormat.surfaceFormat.colorSpace == preferredColor) {
                surfaceFormat2 = availableFormat;
                break;
            }
        }
    }
    return surfaceFormat2;
}
vk::PresentModeKHR findQualifiedPresentMode(const std::span<const vk::PresentModeKHR> surfacePresentModes){
    vk::PresentModeKHR presentMode{};
    {
        vk::PresentModeKHR preferredMode = vk::PresentModeKHR::eFifoRelaxed;
        presentMode = vk::PresentModeKHR::eFifo;
        for (const auto& availablePresentMode : surfacePresentModes) {
            if (availablePresentMode == preferredMode){
                presentMode = availablePresentMode;
                break;
            }
            else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                presentMode = availablePresentMode;
            }
        }
    }
    return presentMode;
}
// oldSwapchain will be retired but not destroyed
vk::UniqueSwapchainKHR createSwapchain(vk::Device device, const PhysicalDeviceInfo &devProps,
                                       const vk::Extent2D &surfaceExtent, const vk::SurfaceFormat2KHR &surfaceFormat2, vk::PresentModeKHR presentMode, uint32_t imageCount,
                                       vk::SurfaceKHR surface, vk::SwapchainKHR oldSwapchain = {}){

    auto surfaceCapabilities = devProps.surfaceCaps.surfaceCapabilities;
    auto surfacePresentModes = devProps.surfacePres;
    vk::Extent2D extent{};
    {
        if (devProps.surfaceCaps.surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            extent = devProps.surfaceCaps.surfaceCapabilities.currentExtent;
        }
        else {
            extent.width = std::clamp(surfaceExtent.width,
                                      devProps.surfaceCaps.surfaceCapabilities.minImageExtent.width,
                                      devProps.surfaceCaps.surfaceCapabilities.maxImageExtent.width);
            extent.height = std::clamp(surfaceExtent.height,
                                       devProps.surfaceCaps.surfaceCapabilities.minImageExtent.height,
                                       devProps.surfaceCaps.surfaceCapabilities.maxImageExtent.height);
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
     * Create vk::SwapchainKHR, its vk::Image, and vk::ImageView
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
    auto vkDebugUtilMsg = createVulkanDebugMsg(instance.get());
    auto chosenPhysicalDevice = getPhysicalDevice(instance.get());
    // SDL2's surface
    auto surfaceSDL = createSurfaceSDL(p_SDLWindow, instance.get());
    auto physicalDeviceProps = queryPhysicalDeviceInfo(chosenPhysicalDevice, surfaceSDL);

    auto featureList = getRequiredDeviceFeatures2(chosenPhysicalDevice);
    auto deviceExtensions = getRequiredDeviceExtensions();
    auto device = vk::UniqueDevice{};
    size_t numQueues = 2;// One for global commands, one for the single-thread renderer.
    uint32_t queueFamilyIndex{};
    {
        auto queueFamilyIdx = findQualifiedQueueFamily(physicalDeviceProps, numQueues);
        if (!queueFamilyIdx.has_value()) {
            std::abort();
        }
        queueFamilyIndex = queueFamilyIdx.value();
    }
    device = createVulkanDevice(chosenPhysicalDevice, featureList, deviceExtensions, queueFamilyIndex, numQueues);
    // Setup DynamicLoader (device-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());
    auto swapchainFormat = findQualifiedSurfaceFormat2(physicalDeviceProps.surfaceFmts);
    auto swapchainPresentMode = findQualifiedPresentMode(physicalDeviceProps.surfacePres);
    // -------- Repetitive --------
    uint32_t swapchainImageCount = 3;
    swapchainImageCount = std::clamp(swapchainImageCount,
                                     physicalDeviceProps.surfaceCaps.surfaceCapabilities.minImageCount,
                                     physicalDeviceProps.surfaceCaps.surfaceCapabilities.maxImageCount);
    auto surfaceExtent = getSurfaceSize(p_SDLWindow);
    auto swapchain = createSwapchain(device.get(), physicalDeviceProps,
                                     surfaceExtent, swapchainFormat, swapchainPresentMode, swapchainImageCount,
                                     surfaceSDL.get());
    auto swapchainImages = std::vector<vk::Image>{};
    {
        auto [result, imgs] = device->getSwapchainImagesKHR(swapchain.get());
        utils::vkEnsure(result);
        swapchainImages = imgs;
    }
    swapchainImageCount = swapchainImages.size();

    size_t globalQueueIndex = 0;
    auto globalQueue = utils::QueueStruct{.queue = device->getQueue(queueFamilyIndex, globalQueueIndex), .queueFamilyIdx = queueFamilyIndex};
    size_t INFLIGHT_FRAMES = 2;

{
    /*The order of preparing stuffs:
     * Pipeline
     * Descriptor
     * Renderpass
     * Attachment requested resources and FrameBuffers
     * Descriptor requested resources
     *
     * */
    // Swapchain for SDL2's surface and corresponding imagesPack
//    auto [swapchainSDL, imagesPackSDL] = createSwapChainnImages(chosenPhysicalDevice, surfaceSDL.get(), p_SDLWindow,
//                                                                device.get());
//    auto imageViewsSDL = createImageViews(imagesPackSDL.images, imagesPackSDL.format, vk::ImageAspectFlagBits::eColor,
//                                          device.get());
//    auto viewport = getViewportInfo(imagesPackSDL.extent.width, imagesPackSDL.extent.height);
//    auto graphPipe = GraphicsPipeline<VertexShader, FragShader>{device.get(), viewport};
//    auto descMgr = DescriptorManager{device.get()};
//    for (auto& setBind: graphPipe.bindings_){
//        std::vector<vk::DescriptorSetLayoutBinding> layoutBinds{};
//        for (auto& bind: setBind.second){
//            layoutBinds.push_back(bind);
//        }
//        descMgr.registerDescriptorBindings(layoutBinds);
//    }
//
//    auto resMgr = VulkanResourceManager{instance.get(), chosenPhysicalDevice, device.get(), {.queue = queues[1], .queueFamilyIdx = graphQueueIdx}};
//    auto commandMgrs = std::vector<CommandBufferManager>{};
//    for (size_t i=0;i<INFLIGHT_FRAMES;++i){
//        commandMgrs.emplace_back(CommandBufferManager{device.get(), {.queue = queues[0], .queueFamilyIdx = graphQueueIdx}});
//    }
}
}


