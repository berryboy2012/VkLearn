# VkLearn

A personal project for learning Vulkan1.3, VMA, SDL, Dear ImGui, OpenVR, DLSS/FSR etc.

## Building

Use `vcpkg` to install the following libs:
```
SDL2[vulkan]
vulkan-headers
glm
vulkan-memory-allocator-hpp
openvr
```

Add the following to the CMake options:
```shell
-DCMAKE_TOOLCHAIN_FILE=C:\MiscProg\vcpkg\scripts\buildsystems\vcpkg.cmake
-DVCPKG_TARGET_TRIPLET=x64-windows
```

The rest are trivial stuffs.

## TODO:
- Vertex staging buffer. (Transfer data from/to GPU's local memory)
- Index buffer. (Tell GPU a correlate vertices with triangles)
- Descriptor layout and related stuff. (Create generic buffer on GPU)
- Push constants. (Send small constant values into shader)
- Image, image view and sampler. (Use textures in fragment shader)
- Depth buffering. (Tell GPU which one of the overlapping fragments should be drawn)
- Refactoring the code (with the help of VMA), so we don't lose our sanity when using Vulkan.
- OpenVR/OpenXR integration.
- Ray-tracing integration.
- FSR/DLSS integration.
- Be able to view arbitrary 3D models.
- Further explorations.
