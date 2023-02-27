// layout(...) specifier type variable_name;
// Valid qualifiers in layout: set location binding ...
// Valid specifier: in out uniform ...
// Valid type: vec2 mat4 ... StructTypeName {...}
#version 460

layout(push_constant) uniform constants{
    mat4 view;
    mat4 proj;
} sceneVP;

layout(binding = 0) uniform ModelUB {
  mat4 model;
} modelUBO;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    gl_Position = sceneVP.proj * sceneVP.view * modelUBO.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}