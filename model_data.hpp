//
// Created by berry on 2023/2/28.
//

#ifndef VKLEARN_MODEL_DATA_HPP
#define VKLEARN_MODEL_DATA_HPP
#include <vector>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
namespace model_info
{/* About memory alignment rules in Vulkan:
 * For things like vertex buffer, we specify the offset of each variable when creating a pipeline, thus no additional
 * padding rules are needed.
 *
 * For other buffers and images, we have to make sure the alignments are right. Those rules can be found at
 * https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout
 *
 * A tl;dr explanation can be found at
 * https://vulkan-tutorial.com/Uniform_buffers/Descriptor_pool_and_sets#page_Alignment-requirements
 "
  Vulkan expects the data in your structure to be aligned in memory in a specific way, for example:
    Scalars have to be aligned by N (= 4 bytes given 32bit floats).
    A vec2 must be aligned by 2N (= 8 bytes)
    A vec3 or vec4 must be aligned by 4N (= 16 bytes)
    A nested structure must be aligned by the base alignment of its members rounded up to a multiple of 16.
    A mat4 matrix must have the same alignment as a vec4.
"
 * */
struct PCTVertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
};

/* Screen coordinate system for vulkan (Zd$\in$[0,1]):
 * Red, Green and Blue correspond to approximate locations of the first three vertices
 *                [-1]
 *                 |
 *                Red
 *                 |
 * [-1]------------0------------[1]>x
 *                 |
 *      Blue       |    Green
 *                 |
 *                [1]
 *                 v
 *                 y
 *
 * For 3D coordinates, we follow the right hand rules.
 * */
const std::vector<PCTVertex> gVertices = {
        //base
        {{0.0f,  -0.5f, 0.0f},  {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},//R
        {{0.5f,  0.5f,  0.0f},  {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},//G
        {{-0.5f, 0.5f,  0.0f},  {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},//B
        //RG-Top
        {{0.0f,  0.0f,  0.5f},  {0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f}},
        //GB-Top
        {{0.0f,  0.0f,  0.5f},  {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
        //BR-Top
        {{0.0f,  0.0f,  0.5f},  {0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},
        //higher
        {{0.0f,  -0.5f, 0.3f},  {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
        {{0.5f,  0.5f,  0.3f},  {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{-0.5f, 0.5f,  0.3f},  {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        //RG-Top
        {{0.0f,  0.0f,  0.8f},  {0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f}},
        //GB-Top
        {{0.0f,  0.0f,  0.8f},  {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
        //BR-Top
        {{0.0f,  0.0f,  0.8f},  {0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}},
        //lower
        {{0.0f,  -0.5f, -0.3f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
        {{0.5f,  0.5f,  -0.3f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        {{-0.5f, 0.5f,  -0.3f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
        //RG-Top
        {{0.0f,  0.0f,  0.2f},  {0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f}},
        //GB-Top
        {{0.0f,  0.0f,  0.2f},  {0.5f, 0.5f, 0.5f}, {0.0f, 0.0f}},
        //BR-Top
        {{0.0f,  0.0f,  0.2f},  {0.5f, 0.5f, 0.5f}, {1.0f, 0.0f}}
};
/*Face culling convention: In OpenGL, the default values are:
 * glCullFace == GL_BACK; glFrontFace == GL_CCW
 * So here we follow the same convention and adjust parameters in VkPipelineRasterizationStateCreateInfo
 * accordingly.
 * */
typedef uint16_t VertIdxType;
const std::vector<VertIdxType> gVertexIdx = {
        0, 1, 3,
        1, 2, 4,
        2, 0, 5,
        2, 1, 0,

        6, 7, 9,
        7, 8, 10,
        8, 6, 11,
        8, 7, 6,

        12, 13, 15,
        13, 14, 16,
        14, 12, 17,
        14, 13, 12
};
}
#endif //VKLEARN_MODEL_DATA_HPP
