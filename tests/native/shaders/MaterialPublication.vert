#version 450
layout(set = 0, binding = 14) uniform MaterialProperties { float vertexGain; } material;
layout(location = 0) out float gain;
void main()
{
    const vec2 positions[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
    gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
    gain = material.vertexGain;
}
