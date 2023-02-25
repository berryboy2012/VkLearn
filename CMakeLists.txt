cmake_minimum_required(VERSION 3.24)
project(VkLearn)
if(MSVC)
    add_compile_options(/W4 /permissive- /GR-)
else(MSVC)
    add_compile_options(-Wall)
endif(MSVC)
set(CMAKE_CXX_STANDARD 20)
find_package(SDL2 CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(unofficial-vulkan-memory-allocator-hpp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)

# OpenVR's CMake support is janky
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/AddGLSLShader.cmake)
include(cmake/AddTexture.cmake)
find_package(OpenVR REQUIRED)

add_executable(VkLearn main.cpp utils.hpp renderer.hpp)

add_shader(VkLearn shader.vert)
add_shader(VkLearn shader.frag)

add_texture(VkLearn test_512.png)

target_link_libraries(VkLearn
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)
target_link_libraries(VkLearn PRIVATE OpenVR::OpenVR)
target_link_libraries(VkLearn PRIVATE Vulkan::Headers)
target_link_libraries(VkLearn PRIVATE Vulkan::Vulkan)
target_link_libraries(VkLearn PRIVATE unofficial::VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp)
target_link_libraries(VkLearn PRIVATE glm::glm)
target_include_directories(VkLearn PRIVATE ${Stb_INCLUDE_DIR})
