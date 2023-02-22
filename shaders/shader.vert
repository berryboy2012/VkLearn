#version 450
layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
} ubo;
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}