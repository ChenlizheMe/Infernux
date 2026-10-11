#version 450

layout(std430, set = 0, binding = 0) readonly buffer InxBuffer0 {
    uint data[];
} values;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(float(values.data[0]) / 255.0, 0.0, 0.0, 1.0);
}
