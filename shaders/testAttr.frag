#version 460
layout (input_attachment_index = 0, binding = 0) uniform subpassInput inputColor;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput i_albedo;
layout(input_attachment_index = 2, binding = 2) uniform subpassInputMS resolveTex;
layout (location = 0) out vec4 outColor;
void main() {

}