add_subdirectory(SPIRV-Cross)
## Find the include path of VMAHpp
find_path(VMAHpp_INCLUDE_DIR NAMES vk_mem_alloc.hpp PATHS "${CMAKE_CURRENT_LIST_DIR}/VulkanMemoryAllocator-Hpp/include")
