cmake_minimum_required(VERSION 3.24)
project(VkLearn)
if(MSVC)
    add_compile_options(/W4 /permissive- /GR-)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/GL)
        add_link_options(/LTCG)
    endif()
else(MSVC)
    add_compile_options(-Wall)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif(MSVC)
set(CMAKE_CXX_STANDARD 20)
# Prepare third-party libs
find_package(SDL2 CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
add_subdirectory(libs)
## OpenVR's CMake support is janky
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OpenVR REQUIRED)

set(HEADERS_SRC "${CMAKE_CURRENT_LIST_DIR}/include")
add_executable(VkLearn model_data.hpp shader_modules.hpp graphics_pipeline.hpp renderpass.hpp bindings_management.hpp memory_management.hpp global_objects.hpp main.cpp resource_management.hpp render_thread.hpp)
target_include_directories(VkLearn PRIVATE ${HEADERS_SRC})
target_include_directories(VkLearn PRIVATE ${Stb_INCLUDE_DIR})
target_include_directories(VkLearn PRIVATE ${SPIRV-Cross_SOURCE_DIR})
target_include_directories(VkLearn PRIVATE ${VMAHpp_INCLUDE_DIR})

# Prepare our headers
add_library(VkLearnLibs STATIC ${HEADERS_SRC}/utils.cpp ${HEADERS_SRC}/commands_management.cpp)
target_link_libraries(VkLearnLibs PRIVATE glm::glm)
target_link_libraries(VkLearnLibs PRIVATE spirv-cross-glsl)
target_link_libraries(VkLearnLibs PRIVATE Vulkan::Vulkan)

# Add dependencies with shaders and textures
include(cmake/AddGLSLShader.cmake)
include(cmake/AddTexture.cmake)
add_shader(VkLearn shader.vert)
add_shader(VkLearn shader.frag)
add_shader(VkLearn testAttr.vert)
add_shader(VkLearn testAttr.frag)

add_texture(VkLearn test_512.png)

target_link_libraries(VkLearn
        PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>)
target_link_libraries(VkLearn PRIVATE OpenVR::OpenVR)
target_link_libraries(VkLearn PRIVATE Vulkan::Headers)
target_link_libraries(VkLearn PRIVATE Vulkan::Vulkan)
target_link_libraries(VkLearn PRIVATE glm::glm)
target_link_libraries(VkLearn PRIVATE spirv-cross-glsl)
target_link_libraries(VkLearn PRIVATE VkLearnLibs)
