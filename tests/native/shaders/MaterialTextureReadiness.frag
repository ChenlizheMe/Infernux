#version 450

layout(set = 0, binding = 0) uniform sampler2D texSampler;
layout(set = 0, binding = 1) uniform sampler2D detailTex;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(texSampler, vec2(0.5)) * texture(detailTex, vec2(0.5));
}
