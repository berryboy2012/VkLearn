#version 460
struct InTestStruct {
    vec3 inVec;
    mat4 inMat;
};
layout(push_constant) uniform constants{
    mat4 view;
    layout(offset=128) mat4 proj;
} sceneVP;

layout (location = 0) in mat4 inMatrixTest;
layout(location = 4) in vec3 inPosition;
layout(location = 5) in dvec4 inColor;
layout(location = 7) in vec2 inTexCoord;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragTexCoord;
struct Particle {
    vec2 position;
    vec2 velocity;
    vec4 color;
};

layout(binding = 0) uniform ModelUB {
    mat4 model;
} modelUBO;

layout (set = 1, binding = 1) uniform ParameterUBO {
    float deltaTime;
} ubo;

layout (binding = 2) uniform ArrayUBO {
    float deltaTime;
} arrayUBO[2];

layout (binding = 3) uniform UnknownArrayUBO {
    float deltaTime;
} arrayVarUBO[];

layout(std140, binding = 4) readonly buffer ParticleSSBOIn {
    Particle particlesIn[ ];
};

layout(std140, binding = 5) buffer ParticleSSBOOut {
    Particle particlesOut[ ];
};

void main() {

}