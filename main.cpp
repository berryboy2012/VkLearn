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
std::vector<vk::UniqueImageView> createImageViews(
        const std::span<vk::Image>& images,
        const vk::Format &format, const vk::ImageAspectFlags &imageAspect,
        vk::Device &device);
#include "renderer.hpp"
#include "rt_renderer.hpp"
#include "graphics_pipeline.hpp"
#include "descriptors.hpp"
#include "renderpass.hpp"
#include "commands.hpp"
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
                abort();
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

// Return queueFamilyIndex and queueCount. queueCount will be zero if no suitable queueFamily exists.
std::tuple<uint32_t, uint32_t> findQueueFamilyInfo(
        const vk::PhysicalDevice &device, const vk::QueueFlags &queueFamilyFlag, const std::optional<vk::SurfaceKHR> &surface = std::nullopt) {
    bool needSurface = surface.has_value();
    uint32_t queueFamilyIndex;
    uint32_t queueCount = 0;
    auto queueFamilies = device.getQueueFamilyProperties2();
    uint32_t i = 0;
    for (const auto& queueFamily2 : queueFamilies) {
        const auto& queueFamily = queueFamily2.queueFamilyProperties;
        if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & queueFamilyFlag)==queueFamilyFlag) {
            if (needSurface){
                auto [result, isSupported] = device.getSurfaceSupportKHR(i, surface.value());
                utils::vkEnsure(result, "device.getSurfaceSupportKHR failed");
                if (isSupported == VK_FALSE){
                    continue;
                }
            }
            queueFamilyIndex = i;
            queueCount = queueFamily.queueCount;
            break;
        }
        i++;
    }
    return std::make_tuple(queueFamilyIndex,queueCount);
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
        auto devProperty = device.getProperties();
        std::cout<<std::string_view{devProperty.deviceName}<<std::endl;
        chosenPhysicalDevice = device;
    }
    return chosenPhysicalDevice;
}

std::tuple<vk::UniqueDevice, std::vector<vk::Queue>> createVulkanDevicenQueues(
        const vk::PhysicalDevice &chosenDevice,
        const std::vector<std::string> &deviceExtensions,
        const std::vector<std::any> &featureList2,
        const std::optional<vk::SurfaceKHR> &surface = std::nullopt){
    auto createInfo = vk::DeviceCreateInfo{vk::DeviceCreateFlags()};

    // Assemble queueFamilyInfo
    using QueBit = vk::QueueFlagBits;
    auto [queueFamilyIndex, queueCountMax] = findQueueFamilyInfo(chosenDevice,
                                                                 QueBit::eCompute|QueBit::eGraphics|QueBit::eTransfer, surface);
    auto queueCount = uint32_t{1};
    if (queueCount < queueCountMax/2){
        queueCount = queueCountMax/2;
    }
    auto queuePrioritys = std::vector<float>(queueCount, 1.0f);
    auto queueCreateInfo = vk::DeviceQueueCreateInfo{
            vk::DeviceQueueCreateFlags(),
            queueFamilyIndex,
            queueCount,
            queuePrioritys.data()
    };
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;

    // Add requested features
    auto pEnabledFeatures2 = &std::any_cast<const vk::PhysicalDeviceFeatures2&>(featureList2[0]);
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = (void*)pEnabledFeatures2;

    // Assemble deviceExtension
    auto [vecPtrDeviceExtension, extensionCount] = utils::stringToVecptrU32(deviceExtensions);
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = vecPtrDeviceExtension.data();

    auto [result, logicDevice] = chosenDevice.createDeviceUnique(createInfo);
    utils::vkEnsure(result);
    // TODO: use getQueue2 in the future
    auto queues = std::vector<vk::Queue>();
    queues.reserve(queueCount);
    for (uint32_t idQueue = 0; idQueue < queueCount; ++idQueue){
        queues.push_back(logicDevice->getQueue(queueFamilyIndex, idQueue));
    }
    return std::make_tuple(std::move(logicDevice),queues);
}

vk::UniqueSurfaceKHR createSurfaceSDL(SDL_Window *p_SDLWindow, const vk::Instance &vkInstance) {
    VkSurfaceKHR tmpSurface{};
    if (SDL_Vulkan_CreateSurface(p_SDLWindow, vkInstance, &tmpSurface) != SDL_TRUE){
        std::abort();
    }
    auto surface = vk::UniqueSurfaceKHR(tmpSurface, vkInstance);
    return std::move(surface);
}

// oldSwapchain will be retired but not destroyed
std::tuple<vk::UniqueSwapchainKHR, utils::VkImagesPack>
createSwapChainnImages(const vk::PhysicalDevice &physicalDevice,
                       const vk::SurfaceKHR &surface, SDL_Window* &window, const vk::Device &device, const vk::SwapchainKHR &oldSwapchain = {}) {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities{};
    std::vector<vk::SurfaceFormatKHR> surfaceFormats{};
    std::vector<vk::PresentModeKHR> surfacePresentModes{};
    {
        auto result = vk::Result{};
        std::tie(result, surfaceCapabilities) = physicalDevice.getSurfaceCapabilitiesKHR(surface);
        utils::vkEnsure(result);
        std::tie(result, surfaceFormats) = physicalDevice.getSurfaceFormatsKHR(surface);
        utils::vkEnsure(result);
        std::tie(result, surfacePresentModes) = physicalDevice.getSurfacePresentModesKHR(surface);
        utils::vkEnsure(result);

    }
    vk::Extent2D extent{};
    {
        if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            extent = surfaceCapabilities.currentExtent;
        }
        else {
            int width, height;
            SDL_Vulkan_GetDrawableSize(window, &width, &height);
            vk::Extent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

            actualExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
                                          std::min(surfaceCapabilities.maxImageExtent.width, actualExtent.width));
            actualExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
                                           std::min(surfaceCapabilities.maxImageExtent.height, actualExtent.height));

            extent = actualExtent;
        }
    }
    vk::SurfaceFormatKHR surfaceFormat{};
    {
        std::optional<vk::SurfaceFormatKHR> tmpSurfFormat;
        if (surfaceFormats.size() == 1 && surfaceFormats[0].format == vk::Format::eUndefined) {
            tmpSurfFormat = { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }
        else{
            for (const auto& availableFormat : surfaceFormats) {
                if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
                    availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    tmpSurfFormat = availableFormat;
                    break;
                }
            }
        }
        if (!tmpSurfFormat.has_value()){
            tmpSurfFormat = surfaceFormats[0];
        }
        surfaceFormat = tmpSurfFormat.value();
    }
    vk::PresentModeKHR presentMode{};
    {
        std::optional<vk::PresentModeKHR> tmpPresMod{};
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto& availablePresentMode : surfacePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                tmpPresMod = availablePresentMode;
            }
            else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                bestMode = availablePresentMode;
            }
        }
        if (!tmpPresMod.has_value()){
            tmpPresMod = bestMode;
        }
        presentMode = tmpPresMod.value();
    }

    uint32_t imageCount = surfaceCapabilities.minImageCount + 1;
    if (surfaceCapabilities.maxImageCount > 0 && imageCount > surfaceCapabilities.maxImageCount) {
        imageCount = surfaceCapabilities.maxImageCount;
    }
    auto currentTransform = surfaceCapabilities.currentTransform;
    vk::SwapchainCreateInfoKHR createInfo{
            vk::SwapchainCreateFlagsKHR(),
            surface,
            imageCount,
            surfaceFormat.format,
            surfaceFormat.colorSpace,
            extent,
            1,/*imageArrayLayers*/
            vk::ImageUsageFlagBits::eColorAttachment,
            vk::SharingMode::eExclusive,
            0,/*queueFamilyIndexCount*/
            nullptr,/*pQueueFamilyIndices*/
            currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode,
            VK_TRUE,/*clipped*/
            oldSwapchain,/*oldSwapchain*/
    };
    auto [result, swapChain] = device.createSwapchainKHRUnique(createInfo);
    utils::vkEnsure(result, "swapchain creation failed");
    auto [imageResult, swapChainImages] = device.getSwapchainImagesKHR(swapChain.get());
    utils::vkEnsure(imageResult);
    utils::VkImagesPack imagesPack{
        .images = swapChainImages,
        .format = surfaceFormat.format,
        .extent = extent
    };
    return std::make_tuple(std::move(swapChain),std::move(imagesPack));
}

std::vector<vk::UniqueImageView> createImageViews(
        const std::span<vk::Image>& images,
        const vk::Format &format, const vk::ImageAspectFlags &imageAspect,
        vk::Device &device) {
    auto imageViews = std::vector<vk::UniqueImageView>{};
    imageViews.resize(images.size());
    vk::ImageViewCreateInfo createInfo = {};
    createInfo.viewType = vk::ImageViewType::e2D;
    createInfo.format = format;
    createInfo.components.r = vk::ComponentSwizzle::eIdentity;
    createInfo.components.g = vk::ComponentSwizzle::eIdentity;
    createInfo.components.b = vk::ComponentSwizzle::eIdentity;
    createInfo.components.a = vk::ComponentSwizzle::eIdentity;
    createInfo.subresourceRange.aspectMask = imageAspect;//vk::ImageAspectFlagBits::eColor;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    for (size_t i = 0; i < images.size(); i++) {
        createInfo.image = images[i];
        auto [result, imageView] = device.createImageViewUnique(createInfo);
        utils::vkEnsure(result);
        imageViews[i] = std::move(imageView);
    }
    return std::move(imageViews);
}
vk::Format findSupportedFormat(
        const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features,
        const vk::PhysicalDevice &physicalDevice) {
    for (vk::Format format : candidates) {
        //vk::FormatProperties props;
        //vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        auto props = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features){
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
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

bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}
std::tuple<
        vk::UniqueImage,
        vk::UniqueDeviceMemory,
        vk::UniqueImageView> createDepthResources(const vk::Extent3D &renderExtent,
                                                  const uint32_t &queueFamilyIdx, vk::Device &device, const vk::PhysicalDevice &physicalDevice) {
    auto depthFormat = findDepthFormat(physicalDevice);

    auto [depthImage, depthMemory] = createImagenMemory(
            renderExtent, depthFormat,
            vk::ImageTiling::eOptimal, vk::ImageLayout::eUndefined, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
            queueFamilyIdx, device, physicalDevice);
    auto depthImageView = std::move(createImageViews(std::span(&depthImage.get(), 1), depthFormat, vk::ImageAspectFlagBits::eDepth, device)[0]);
    return std::make_tuple(std::move(depthImage), std::move(depthMemory), std::move(depthImageView));
}
vk::UniqueRenderPass createRenderPassSDL(const vk::Format &swapChainImageFormat, const vk::Format &depthImageFormat, vk::Device &device) {
    vk::AttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentDescription depthAttachment{};
    depthAttachment.format = depthImageFormat;
    depthAttachment.samples = vk::SampleCountFlagBits::e1;
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::AttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthAttachmentOptimal;

    vk::SubpassDescription subpass = {};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.srcAccessMask = vk::AccessFlags{};
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    vk::RenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.attachmentCount = attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    auto [result, renderPass] = device.createRenderPassUnique(renderPassInfo);
    utils::vkEnsure(result);
    return std::move(renderPass);
}
std::vector<vk::UniqueFramebuffer> createFramebuffers(std::vector<vk::UniqueImageView> &imageViews,
                                                      vk::ImageView &depthImageView,
                                                      const vk::Extent2D &extent,
                                                      vk::RenderPass &renderPass,
                                                      vk::Device &device) {
    std::vector<vk::UniqueFramebuffer> framebuffers;
    framebuffers.resize(imageViews.size());
    // https://stackoverflow.com/a/39559418
    vk::FramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.width = extent.width;
    framebufferInfo.height = extent.height;
    framebufferInfo.layers = 1;
    for (size_t i = 0; i < imageViews.size(); i++) {
        std::array<vk::ImageView, 2> attachments = {
                imageViews[i].get(),
                depthImageView
        };
        framebufferInfo.attachmentCount = attachments.size();
        // Actually can be created without supplying an image.
        // Refer to VK_KHR_imageless_framebuffer for more info.(No extensions needed for ver>=1.2)
        framebufferInfo.pAttachments = attachments.data();
        auto [result, frameBuffer] = device.createFramebufferUnique(framebufferInfo);
        utils::vkEnsure(result);
        framebuffers[i] = std::move(frameBuffer);
    }
    return std::move(framebuffers);
}
std::tuple<std::vector<vk::UniqueSemaphore>, std::vector<vk::UniqueSemaphore>, std::vector<vk::UniqueFence>>
createSwapchainSyncObjects(vk::Device &device, const uint32_t &MAX_FRAMES_LAG = 2);
//Create CommandPool on the predefined queue
vk::UniqueCommandPool createCommandPool(vk::PhysicalDevice &chosenDevice, vk::Device &device,
                       const std::optional<vk::SurfaceKHR> &surface = std::nullopt) {
    //QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
    using QueBit = vk::QueueFlagBits;
    auto [queueFamilyIndex, queueCountMax] = findQueueFamilyInfo(chosenDevice,
                                                                 QueBit::eCompute|QueBit::eGraphics|QueBit::eTransfer, surface);
    vk::CommandPoolCreateInfo poolInfo = {};
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    auto [result, commandPool] = device.createCommandPoolUnique(poolInfo);
    utils::vkEnsure(result);
    return std::move(commandPool);
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
int main(int argc, char *argv[]) {
    bool bDebug = true;
    // Setup DynamicLoader (process-wide)
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
    // Setup SDL2
    auto p_SDLWindow = initSDL();
    queryOVR();
    // Instance for Vulkan
    auto validationLayers = getRequiredValidationLayers(bDebug);
    auto instanceExtensions = getRequiredInstanceExtensions(p_SDLWindow, bDebug);
    auto vkUniqueInstance = createVulkanInstance(validationLayers, instanceExtensions);
    // Setup DynamicLoader (instance-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkUniqueInstance.get());
    auto vkDebugUtilMsg = createVulkanDebugMsg(vkUniqueInstance.get());
    auto chosenPhysicalDevice = getPhysicalDevice(vkUniqueInstance.get());
    // SDL2's surface
    auto vkSurfaceSDL = createSurfaceSDL(p_SDLWindow, vkUniqueInstance.get());
    auto featureList = getRequiredDeviceFeatures2(chosenPhysicalDevice);
    auto deviceExtensions = getRequiredDeviceExtensions();
    auto [graphQueueIdx, graphQueueCount] = findQueueFamilyInfo(chosenPhysicalDevice, vk::QueueFlagBits::eGraphics);
    auto [vkUniqueDevice, vkQueues] = createVulkanDevicenQueues(chosenPhysicalDevice, deviceExtensions, featureList, vkSurfaceSDL.get());
    // Setup DynamicLoader (device-wide)
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkUniqueDevice.get());
    auto vkCommandPool = createCommandPool(chosenPhysicalDevice, vkUniqueDevice.get(),vkSurfaceSDL.get());
    // Swapchain for SDL2's surface and corresponding imagesPack
    auto [vkSwapchainSDL, imagesPackSDL] = createSwapChainnImages(chosenPhysicalDevice, vkSurfaceSDL.get(), p_SDLWindow, vkUniqueDevice.get());
    auto imageViewsSDL = createImageViews(imagesPackSDL.images, imagesPackSDL.format, vk::ImageAspectFlagBits::eColor, vkUniqueDevice.get());
    auto depthFormat = findDepthFormat(chosenPhysicalDevice);
    auto [depthImage, depthMemory, depthImageView] = createDepthResources(
            {imagesPackSDL.extent.width, imagesPackSDL.extent.height, 1}, graphQueueIdx, vkUniqueDevice.get(), chosenPhysicalDevice);
    auto renderPassSDL = createRenderPassSDL(imagesPackSDL.format, depthFormat, vkUniqueDevice.get());
    auto framebuffersSDL = createFramebuffers(imageViewsSDL, depthImageView.get(), imagesPackSDL.extent, renderPassSDL.get(), vkUniqueDevice.get());
    setupRender(chosenPhysicalDevice, vkUniqueDevice.get(), imagesPackSDL.extent, graphQueueIdx, renderPassSDL.get(), vkCommandPool.get(), vkQueues[0]);
    //Now for the main loop
    SDL_StartTextInput();
    size_t frames = 0;
    bool resetSwapchain = false;
    auto& graphicsQueue = vkQueues[0];
    auto& presentQueue = vkQueues[0];
    while (!wantExitSDL()){
        if (!resetSwapchain){
            // When last rendering on the current slot in the frames queue is completed, inFlightFence will be signaled.
            auto& inFlightFence = render::inFlightFences[frames];
            // When we have an image in swapchain ready to be drawn upon, imageAvailSemaphore will be signaled.
            auto& imageAvailSemaphore = render::imageAvailableSemaphores[frames];
            // When the image is ready for presentation, renderFinishedSemaphore will be signaled.
            auto& renderFinishedSemaphore = render::renderFinishedSemaphores[frames];
            // Wait for last rendering on this swapchain image to finish
            vk::Result result{};
            result = vkUniqueDevice->waitForFences(inFlightFence, VK_TRUE, std::numeric_limits<uint64_t>::max());
            utils::vkEnsure(result);

            uint32_t imageIndex; // This is NOT the same as `frames` !
            std::tie(result, imageIndex) = vkUniqueDevice->acquireNextImageKHR(
                    vkSwapchainSDL.get(), std::numeric_limits<uint64_t>::max(), imageAvailSemaphore, nullptr);
            if (result == vk::Result::eErrorOutOfDateKHR){
                resetSwapchain = true;
                continue;
            } else if (result != vk::Result::eSuboptimalKHR) {
                utils::vkEnsure(result);
            }
            // Now start rendering the image, first we block the next attempt at rendering in current slot.
            result = vkUniqueDevice->resetFences({inFlightFence});
            utils::vkEnsure(result);

            vk::SubmitInfo2 submitInfo2 = {};
            submitInfo2.flags = vk::SubmitFlags{};
            std::vector<vk::SemaphoreSubmitInfo> waitSemaphores = {};
            // The pipeline will wait imageAvailSemaphore at the ColorAttachmentOutput stage.
            auto imageAvailSemaInfo = vk::SemaphoreSubmitInfo{
                imageAvailSemaphore,
                0,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                0
            };
            waitSemaphores.push_back(imageAvailSemaInfo);
            std::vector<vk::SemaphoreSubmitInfo> signalSemaphores = {};
            // When all commands in this submission are completed, renderFinishedSemaphore will be signaled.
            auto renderFinSemaInfo = vk::SemaphoreSubmitInfo{
                renderFinishedSemaphore,
                0,
                vk::PipelineStageFlagBits2::eAllCommands,
                0
            };
            signalSemaphores.push_back(renderFinSemaInfo);
            std::vector<vk::CommandBufferSubmitInfo> commandBufferBatch = {};
            {

                // Now it's time for renderer to rerecord commandBuffer and update buffers etc.
                updateFrameData(frames, imagesPackSDL.extent);
                bool rerecordCmdBuf = true;
                if(rerecordCmdBuf){
                    result = render::commandBuffers[frames].reset();
                    utils::vkEnsure(result);
                    recordCommandBuffer(
                            framebuffersSDL[imageIndex].get(), renderPassSDL.get(), imagesPackSDL.extent,
                            render::graphPipelineU.get(), render::commandBuffers[frames], frames);
                }
                commandBufferBatch.push_back(render::commandBuffers[frames]);
                // The renderer can also add their semaphores here.

            }
            // Assemble VkQueueSubmitInfo2.
            submitInfo2.waitSemaphoreInfoCount = waitSemaphores.size();
            submitInfo2.pWaitSemaphoreInfos = waitSemaphores.data();
            submitInfo2.commandBufferInfoCount = commandBufferBatch.size();
            submitInfo2.pCommandBufferInfos = commandBufferBatch.data();
            submitInfo2.signalSemaphoreInfoCount = signalSemaphores.size();
            submitInfo2.pSignalSemaphoreInfos = signalSemaphores.data();
            // We can submit multiple batches at once, so an optional fence can be supplied to signal the completion.
            result = graphicsQueue.submit2(submitInfo2, inFlightFence);
            utils::vkEnsure(result);

            // Now we show the rendered image.
            vk::PresentInfoKHR presentInfo = {};
            // We need to wait before the rendering is complete.
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &renderFinishedSemaphore;

            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = &vkSwapchainSDL.get();
            presentInfo.pImageIndices = &imageIndex;

            result = presentQueue.presentKHR(presentInfo);
            if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
                std::cout << "swap chain out of date/suboptimal/window resized - recreating" << std::endl;
                resetSwapchain = true;
            } else {
                utils::vkEnsure(result);
            }
            frames = (frames + 1) % render::MAX_FRAMES_IN_FLIGHT;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
        else{
            auto waitResult = vkUniqueDevice->waitIdle();
            utils::vkEnsure(waitResult);
            {
                int width, height;
                SDL_Vulkan_GetDrawableSize(p_SDLWindow, &width, &height);
                if (width == 0 or height == 0){
                    std::cout<<"Paused.\t";
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            }
            // Destroy objects defined in renderer
            cleanupRender4FB();
            // Destroy vulkan objects related to the swapchain in reverse order
            framebuffersSDL = std::vector<vk::UniqueFramebuffer>{};
            //renderPassSDL = vk::UniqueRenderPass{};
            depthImageView = {};
            depthMemory = {};
            depthImage = {};
            imageViewsSDL = std::vector<vk::UniqueImageView>{};
            //vkCommandPool = {};

            // Recreate those vulkan objects
            //vkCommandPool = createCommandPool(chosenPhysicalDevice, vkUniqueDevice.get(),vkSurfaceSDL.get());
            std::tie(vkSwapchainSDL, imagesPackSDL) = createSwapChainnImages(chosenPhysicalDevice, vkSurfaceSDL.get(), p_SDLWindow, vkUniqueDevice.get(), vkSwapchainSDL.get());
            imageViewsSDL = createImageViews(imagesPackSDL.images, imagesPackSDL.format, vk::ImageAspectFlagBits::eColor, vkUniqueDevice.get());
            std::tie(depthImage, depthMemory, depthImageView) = createDepthResources(
                    {imagesPackSDL.extent.width, imagesPackSDL.extent.height, 1}, graphQueueIdx, vkUniqueDevice.get(), chosenPhysicalDevice);
            //renderPassSDL = createRenderPassSDL(imagesPackSDL.format, vkUniqueDevice.get());
            framebuffersSDL = createFramebuffers(imageViewsSDL, depthImageView.get(), imagesPackSDL.extent, renderPassSDL.get(), vkUniqueDevice.get());
            // Re-initialize renderer
            setupRender(chosenPhysicalDevice, vkUniqueDevice.get(), imagesPackSDL.extent, graphQueueIdx, renderPassSDL.get(), vkCommandPool.get(), vkQueues[0]);
            frames = 0;
            resetSwapchain = false;
        }
    }
    utils::vkEnsure(vkUniqueDevice->waitIdle());
    cleanupRender();
    cleanSDL(p_SDLWindow);
    return 0;
}
