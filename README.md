# VkLearn

A personal project for learning Vulkan1.3, VMA, assimp, meshoptimizer, SDL, Dear ImGui, OpenVR, DLSS/FSR etc.

## Building

Use `vcpkg` to install the following libs:
```
SDL2[vulkan]
vulkan-headers
glm
vulkan-memory-allocator-hpp
openvr
stb
```

Add the following to the CMake options:
```shell
-DCMAKE_TOOLCHAIN_FILE=PATH_2_VCPKG\scripts\buildsystems\vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-windows
```

The rest are trivial stuffs.

## Some design choices:
- Uses `SPIRV-Cross` instead of `SPIRV-Reflect` for shader parameter inspection.
- Right-hand coordinate convention with vanilla Vulkan screen space coordinate rules.
- Aggressive use of vendor-agnostic Vulkan features and extensions.

## Current state:
- Uses some features in Vulkan1.3 to make life a bit easier.
- Can display a hard-coded textured mesh with Model-View-Projection matrices in effect.
- Can handle window resize and minimize events.
- Uses right-hand coordinate with correct equivalent implementation of `glm::lookAt` and `glm::perspective`.
- Has a messy structure.
- Tries to split some boilerplate into `main.cpp`, while actual rendering code resides in `renderer.hpp`.
- Can load geometries into the GPU and build accelerated structures (AS, usually some form of BVH tree) from them.

## TODO:
- ~~Vertex staging buffer. (Transfer data from/to GPU's local memory)~~
- ~~Index buffer. (Tell GPU a correlate vertices with triangles)~~
- ~~Push constants and uniform buffers for MVP matrices. (Send arbitrary structured values into shader)~~
- ~~Descriptor layout and related stuff. (Create generic buffer on GPU)~~
- ~~Handle differences in coordinate system between Vulkan and OpenGL, so we can use `glm` without 
 scratching head.~~
- ~~Image, image view and sampler. (Use textures in fragment shader)~~
- ~~Depth buffering. (Tell GPU which one of the overlapping fragments should be drawn)~~
- Add basic ray-tracing pipeline.
- Add shadow map pipeline.
- Refactoring the code (with the help of VMA), so we don't lose our sanity when using Vulkan. 
 We cannot do this earlier because we haven't implemented a 'working' renderer yet.
- Build a thin abstract layer to simplify some resource management tasks. (Mostly things with a pool in their names)
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
  - Attachment: Description set but for the fixed-function stages

For other workloads (compute and ray-tracing for example), it might be unnecessary to have an equivalent concept of 
renderpass etc. They may only have concepts for pipeline and things below. 

For resources, Vulkan is pretty hardcore:
- Heap: The whole usable memory space.
  - Memory: A continuous block of raw memory. Only a small number of memory entries can be allocated.
    - Buffer: A handle representing a chunk of a memory.
    - Image: buffer but with additional format info.
  - Sparse resources: Vulkan's take on virtual memory.

For synchronization, Vulkan gives fine-grained concepts for maximum parallelism. The details can be found at
[the synchronization part of Vulkan specs](https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#synchronization).
The facilities to interact between host and device are built around command buffers and queries.

## Useful resources:
- [Vulkan Tutorial](https://vulkan-tutorial.com), the OG vulkan guide.
- [vulkan-tutorial-hpp](https://github.com/bwasty/vulkan-tutorial-hpp), uses `Vulkan-Hpp` to rewrite the tutorial above.
Has some typos though.
- [Vulkan Guide](https://vkguide.dev/), a more practical Vulkan tutorial.
