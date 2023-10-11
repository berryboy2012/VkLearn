# VkLearn

A personal project for learning Vulkan1.3, VMA, assimp, meshoptimizer, SDL, Dear ImGui, OpenVR, DLSS/FSR etc.

## Building

Either install the following libs through `vcpkg` manually:
```
SDL2[vulkan]
vulkan-headers
vulkan-memory-allocator
glm
openvr
stb
```
Or rely on the provided `vcpkg.json`.

Add the following to the CMake options:
```shell
-DCMAKE_TOOLCHAIN_FILE=PATH_2_VCPKG\scripts\buildsystems\vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-windows
```

For submodules located at `libs/`, there should be no need to pull `libs/VulkanMemoryAllocator-Hpp/Vulkan-Hpp`.

The rest are trivial stuffs.

## Some design choices:
- Uses `SPIRV-Cross` instead of `SPIRV-Reflect` for shader parameter inspection.
- Right-hand coordinate convention with vanilla Vulkan screen space coordinate rules.
- Aggressive use of vendor-agnostic Vulkan features and extensions.

### Hierarchy of project:

- `renderpass.hpp`: The standalone part of rendering execution.
- `memory_management.hpp`: The standalone part of resource management during rendering.
- `bindings_management.hpp`: The mapping between execution's requirements and data on the machine in terms of resources.
- __TODO__ `timeline_management.hpp`

### [WIP] Directive-style Vulkan object creation

```c++
    /*Vulkan objects needed for rendering-related Vulkan objects creation*/
    vk::Instance inst;
    vk::PhysicalDevice physDev;
    vk::Device renderDev;
    auto renderQueue = utils::QueueStruct{.queue = vk::Queue{}, .queueFamilyIdx = 0};
    vk::Format renderSwapchainFmt; vk::Format renderDepthFmt;
    vk::Extent2D renderExtent;
    vk::ImageUsageFlags renderSwapchainUsg;
    /*Our thin-layered manager objects*/
    auto resMgr = VulkanResourceManager{inst, physDev, renderDev, renderQueue};
    auto descMgr = DescriptorManager{renderDev};
    
    /*Various factories used to manage the creation of corresponding Vulkan objects*/
    auto vertexShaderFactory = VertexShaderFactory{renderDev};
    auto fragmentShaderFactory = FragmentShaderFactory{renderDev};
    auto subpassFactory = SubpassFactory{renderDev, vertexShaderFactory, fragmentShaderFactory};
    auto resourceManager = RenderResourceManager{renderDev, resMgr};
    auto renderpassFactory = RenderpassFactory{renderDev, subpassFactory, vertexShaderFactory, fragmentShaderFactory, resourceManager};

    /*Fill shader info*/
    auto vertShaderIdx = vertexShaderFactory.registerShader("vert");
    vertexShaderFactory.loadShaderModule(vertShaderIdx, "shaders/shader.vert.spv", "main");
    std::vector<std::string> vertAttrs = {"inPosition", "inColor", "inTexCoord"};
    vertexShaderFactory.setVertexInputAttributeTable<model_info::PCTVertex>(vertShaderIdx, vk::VertexInputRate::eVertex, 0,
                                                                            vertAttrs, "foo_ModelGeometry");
    resourceManager.createBuffer("foo_ModelGeometry",
                                 model_info::gVertices.size()*sizeof(model_info::PCTVertex),vk::BufferUsageFlagBits::eVertexBuffer, {});
    resourceManager.copyDataToResource<model_info::PCTVertex>("foo_ModelGeometry", model_info::gVertices);
    vertexShaderFactory.setInputAssemblyState(vertShaderIdx, vk::PrimitiveTopology::eTriangleList);
    vertexShaderFactory.setRasterizationInfo(vertShaderIdx);

    auto fragShaderIdx = fragmentShaderFactory.registerShader("frag");
    fragmentShaderFactory.loadShaderModule(fragShaderIdx, "shaders/shader.frag.spv", "main");
    fragmentShaderFactory.setAttachmentProperties(fragShaderIdx, "outColor", vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal);
    fragmentShaderFactory.setColorAttachmentBlendInfo(fragShaderIdx, "outColor");
    fragmentShaderFactory.setDepthStencilInfo(fragShaderIdx);
    fragmentShaderFactory.setMSAAInfo(fragShaderIdx);
    fragmentShaderFactory.setColorBlendInfo(fragShaderIdx);

    /*Fill subpass info*/
    std::string subpassName  = "first_subpass";
    subpassFactory.registerSubpass(subpassName);
    subpassFactory.loadVertexShader(subpassName, vertShaderIdx);
    subpassFactory.loadFragmentShader(subpassName, fragShaderIdx);
    subpassFactory.gatherSubpassInfo(subpassName);
    subpassFactory.linkDescriptorWithResourceName(subpassName, "modelUBO", "modelMatrix");
    resourceManager.createBuffer("modelMatrix", sizeof(ModelUBO), vk::BufferUsageFlagBits::eUniformBuffer,
                                 vma::AllocationCreateFlagBits::eMapped|vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
    subpassFactory.linkPushConstantWithResourceName(subpassName, "sceneVP", "sceneVPPush");
    resourceManager.createPushConst<ScenePushConstants>("sceneVPPush");
    auto testPConst = ScenePushConstants{.view = glm::identity<glm::mat4>(), .proj = glm::identity<glm::mat4>()};
    auto testPConstView = std::span<const ScenePushConstants>{&testPConst, 1};
    resourceManager.copyDataToResource("sceneVPPush", testPConstView);
    assert(resourceManager.getPushConstData<ScenePushConstants>("sceneVPPush")==testPConst);

    subpassFactory.linkDescriptorWithResourceName(subpassName, "texSampler", "testTexture");
    {
        auto testTexture = TextureObject{"textures/test_512.png", resMgr};
        const auto& testTexInfo = testTexture.getImageInfo();
        resourceManager.createImage("testTexture", testTexInfo.extent, testTexInfo.format, testTexInfo.tiling, testTexInfo.usage, {});
        resourceManager.copyDataToResource("testTexture", testTexture.getImageHostData());
        resourceManager.createViewForImage("testTexture", vk::ImageAspectFlagBits::eColor);
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = 16.0f;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        resourceManager.createSamplerForImage("testTexture", samplerInfo);
    }
    for (const auto& descSetBind: subpassFactory.getPipelineDescriptorBindings(subpassName)){
        descMgr.registerDescriptorBindings(descSetBind.second);
    }

    /*Fill renderpass info*/
    std::string renderpassName = "mainRenderpass";
    auto renderpassIdx = renderpassFactory.registerRenderpass(renderpassName, renderExtent);
    auto subpassIdx = renderpassFactory.loadSubpass(renderpassIdx, subpassName);
    renderpassFactory.gatherAttachmentReferences(renderpassIdx);

    vk::AttachmentDescription2 swapchainAttachDesc{};
    swapchainAttachDesc.format = renderSwapchainFmt;
    swapchainAttachDesc.samples = vk::SampleCountFlagBits::e1;
    swapchainAttachDesc.loadOp = vk::AttachmentLoadOp::eClear;
    swapchainAttachDesc.storeOp = vk::AttachmentStoreOp::eStore;
    swapchainAttachDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    swapchainAttachDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    swapchainAttachDesc.initialLayout = vk::ImageLayout::eUndefined;
    swapchainAttachDesc.finalLayout = vk::ImageLayout::ePresentSrcKHR;
    auto swapchainAttachIdx = renderpassFactory.createAttachment(renderpassIdx, swapchainAttachDesc);

    vk::AttachmentDescription2 depthAttachDesc{};
    depthAttachDesc.format = renderDepthFmt;
    depthAttachDesc.samples = vk::SampleCountFlagBits::e1;
    depthAttachDesc.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachDesc.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachDesc.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachDesc.finalLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    auto depthAttachIdx = renderpassFactory.createAttachment(renderpassIdx, depthAttachDesc);

    renderpassFactory.linkAttachmentRefWithAttach(renderpassIdx, subpassIdx, "outColor", swapchainAttachIdx);
    // the Depth attachment will always have the variable name "__depthShaderVariable"
    renderpassFactory.linkAttachmentRefWithAttach(renderpassIdx, subpassIdx, "__depthShaderVariable", depthAttachIdx);
    {
        vk::ImageCreateInfo swapchainImgInfo{};
        swapchainImgInfo.flags = {};
        swapchainImgInfo.imageType = vk::ImageType::e2D;
        swapchainImgInfo.format = renderSwapchainFmt;
        swapchainImgInfo.extent = vk::Extent3D{renderExtent, 1};
        swapchainImgInfo.mipLevels = 1;
        swapchainImgInfo.arrayLayers = 1;
        swapchainImgInfo.samples = vk::SampleCountFlagBits::e1;
        swapchainImgInfo.tiling = vk::ImageTiling::eOptimal;
        swapchainImgInfo.usage = renderSwapchainUsg;
        swapchainImgInfo.sharingMode = vk::SharingMode::eExclusive;
        auto swapchainViewInfo = factory::image_view_info_builder(
                nullptr,
                swapchainImgInfo.imageType, swapchainImgInfo.format,
                swapchainImgInfo.mipLevels, swapchainImgInfo.arrayLayers,
                vk::ImageAspectFlagBits::eColor);
        resourceManager.createPhantomView("__swapchainView", swapchainImgInfo, vk::ImageLayout::ePresentSrcKHR, swapchainViewInfo);
    }
    {
        resourceManager.createImage("depthImage", vk::Extent3D{renderExtent, 1}, renderDepthFmt, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, {});
        resourceManager.createViewForImage("depthImage", vk::ImageAspectFlagBits::eDepth);
    }

    renderpassFactory.linkAttachmentWithResource(renderpassIdx, swapchainAttachIdx, "__swapchainView");
    renderpassFactory.linkAttachmentWithResource(renderpassIdx, depthAttachIdx, "depthImage");
    /*Create renderpass, framebuffer, pipelines etc.*/
    renderpassFactory.buildVulkanObjects(renderpassIdx);
```

## Current state:
- Uses some features in Vulkan1.3 to make life a bit easier.
- Can display a hard-coded textured mesh with Model-View-Projection matrices in effect.
- Can handle window resize and minimize events.
- Uses right-hand coordinate with correct equivalent implementation of `glm::lookAt` and `glm::perspective`.
- ~~Has a messy structure.~~ Has a somewhat messy structure.
- ~~Tries to split some boilerplate into `main.cpp`, while actual rendering code resides in `renderer.hpp`.~~
- Can load geometries into the GPU and build accelerated structures (AS, usually some form of BVH tree) from them.
- Multi-threaded.

## TODO:
- ~~Vertex staging resource. (Transfer data from/to GPU's local memory)~~
- ~~Index resource. (Tell GPU a correlate vertices with triangles)~~
- ~~Push constants and uniform buffers for MVP matrices. (Send arbitrary structured values into shader)~~
- ~~Descriptor layout and related stuff. (Create generic resource on GPU)~~
- ~~Handle differences in coordinate system between Vulkan and OpenGL, so we can use `glm` without 
 scratching head.~~
- ~~Image, image view and sampler. (Use textures in fragment shader)~~
- ~~Depth buffering. (Tell GPU which one of the overlapping fragments should be drawn)~~
- When should we create static global resources?
  - ~~Before spawning renderer threads: Can't do lazy loading.~~
  - ~~Inside renderer threads: Duplicated resources.~~
  - Yet another thread, dedicated to global resource allocation: Might be hard to manage concurrency.
- Add basic ray-tracing pipeline.
- Add shadow map pipeline.
- ~~Refactoring the code (with the help of VMA), so we don't lose our sanity when using Vulkan. 
 We cannot do this earlier because we haven't implemented a 'working' renderer yet.~~ Almost done.
- ~~Build a thin abstract layer to simplify some resource management tasks. (Mostly things with a pool in their names)~~
- State machine handling interactions and events.
- OpenVR/OpenXR integration.
- FSR/DLSS integration.
- Be able to view arbitrary 3D models.
- Further explorations.

## Hierarchy of Vulkan
Vulkan in its essence is a verbose heterogeneous programming API specialized for graphical workloads. 
There are three essential concepts for heterogeneous programming:
- Execution
- Resource
- Synchronization

Those correspond to the following concepts in Vulkan:

For execution, Vulkan has renderpass:
- Renderpass: A full run of a frame
  - Subpass: A full run of the hardware pipeline.
    - Pipeline: The execution part of the pipeline, during a run of the pipeline, the stages are executed in serial.
      - Vertex shader
      - Geometry shader
      - Fragment shader
      - Descriptor set: Tell pipeline how to bind arguments with custom parameters in the programmable shaders
      - ...
  - Attachment: Descriptor but for the fixed-function stages
  - Framebuffer: Descriptor set but for the fixed-function stages

For other workloads (compute and ray-tracing for example), it might be unnecessary to have an equivalent concept of 
renderpass etc. They may only have concepts for pipeline and things below. 

For resources, Vulkan is pretty hardcore:
- Heap: The whole usable memory space.
  - Memory: A continuous block of raw memory. Only a small number of memory entries can be allocated.
    - Buffer: A handle representing a chunk of a memory.
    - Image: Buffer but with additional format info.
  - Sparse resources: Vulkan's take on virtual memory.

For synchronization, Vulkan gives fine-grained concepts for maximum parallelism. The details can be found at
[the synchronization part of Vulkan specs](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#synchronization).

A rough summary would be as follows (The spec does a pretty poor job at setting up expectations right at first glance):
- The stages inside a subpass are mostly in order, so you don't need to specify the ordering between vertex shader and 
 fragment shader inside a pipeline. But no guarantees about ordering between stages of different subpasses. Those can be
 specified by subpass dependencies. There might be other implicit execution ordering rules in the spec.
- There is no guarantee about the execution order of the commands being sent to devices, provided if you don't use 
 Vulkan's synchronization facilities.
- If synchronization facilities are used, their scopes are built upon the submission order, so you don't need to be 
 insanely verbose when specifying the execution order of your commands. For example, you don't need to keep track of 
 indices of previous commands when setting a semaphore. 

Please note that the summary above is not meant to be accurate, but a mental model to aid understanding when reading the spec.

The facilities to interact between host and device are built around command buffers and queries.
Each thread should have its own `vk::Queue` for command submission. Also, it is generally a good practice for each 
thread to allocate one `vk::CommandPool` per in-flight frame. By doing so, we can just reset the pool when starting a 
new frame.

## Useful resources:
- [Vulkan Tutorial](https://vulkan-tutorial.com), the OG vulkan guide.
- [vulkan-tutorial-hpp](https://github.com/bwasty/vulkan-tutorial-hpp), uses `Vulkan-Hpp` to rewrite the tutorial above.
Has some typos though.
- [Vulkan Guide](https://vkguide.dev/), a more practical Vulkan tutorial.
