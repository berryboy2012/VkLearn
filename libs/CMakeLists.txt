option(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS "Instead of throwing exceptions assert" ON)
option(SPIRV_CROSS_SHARED "Build the C API as a single shared library." OFF)
option(SPIRV_CROSS_STATIC "Build the C and C++ API as static libraries." ON)
option(SPIRV_CROSS_CLI "Build the CLI binary. Requires SPIRV_CROSS_STATIC." OFF)
option(SPIRV_CROSS_ENABLE_TESTS "Enable SPIRV-Cross tests." OFF)

option(SPIRV_CROSS_ENABLE_GLSL "Enable GLSL support." ON)
option(SPIRV_CROSS_ENABLE_HLSL "Enable HLSL target support." OFF)
option(SPIRV_CROSS_ENABLE_MSL "Enable MSL target support." OFF)
option(SPIRV_CROSS_ENABLE_CPP "Enable C++ target support." OFF)
option(SPIRV_CROSS_ENABLE_REFLECT "Enable JSON reflection target support." ON)
option(SPIRV_CROSS_ENABLE_C_API "Enable C API wrapper support in static library." OFF)
option(SPIRV_CROSS_ENABLE_UTIL "Enable util module support." OFF)

add_subdirectory(SPIRV-Cross)
## Find the include path of VMAHpp
find_path(VMAHpp_INCLUDE_DIR NAMES vk_mem_alloc.hpp PATHS "${CMAKE_CURRENT_LIST_DIR}/VulkanMemoryAllocator-Hpp/include")
