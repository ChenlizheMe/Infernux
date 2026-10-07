#version 450

layout(set = 0, binding = 0, std430) readonly buffer RendererParameters
{
    vec4 values[];
} parameters;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = parameters.values[0];
}
