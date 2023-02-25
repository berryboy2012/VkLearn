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

## Current state:
- Uses some features in Vulkan1.3 to make life a bit easier.
- Can display a hard-coded textured mesh with Model-View-Projection matrices in effect.
- Can handle window resize and minimize events.
- Uses right-hand coordinate with correct equivalent implementation of `glm::lookAt` and `glm::perspective`.
- Has a messy structure.
- Tries to split some boilerplate into `main.cpp`, while actual rendering code resides in `renderer.hpp`.

## TODO:
- ~~Vertex staging buffer. (Transfer data from/to GPU's local memory)~~
- ~~Index buffer. (Tell GPU a correlate vertices with triangles)~~
- ~~Push constants and uniform buffers for MVP matrices. (Send arbitrary structured values into shader)~~
- ~~Descriptor layout and related stuff. (Create generic buffer on GPU)~~
- ~~Handle differences in coordinate system between Vulkan and OpenGL, so we can use `glm` without 
 scratching head.~~
- ~~Image, image view and sampler. (Use textures in fragment shader)~~
- Depth buffering. (Tell GPU which one of the overlapping fragments should be drawn)
- Refactoring the code (with the help of VMA), so we don't lose our sanity when using Vulkan. 
 We cannot do this earlier because we haven't implemented a 'working' renderer yet.
- OpenVR/OpenXR integration.
- Ray-tracing integration.
- FSR/DLSS integration.
- Be able to view arbitrary 3D models.
- Further explorations.

## Useful resources:
- [Vulkan Tutorial](https://vulkan-tutorial.com), the OG vulkan guide.
- [vulkan-tutorial-hpp](https://github.com/bwasty/vulkan-tutorial-hpp), uses `Vulkan-Hpp` to rewrite the tutorial above.
Has some typos though.
- [Vulkan Guide](https://vkguide.dev/), a more practical Vulkan tutorial.
