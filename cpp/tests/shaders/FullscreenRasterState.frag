#version 450
layout(push_constant) uniform Parameters { vec4 color; float depth; } params;
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = params.color;
    gl_FragDepth = params.depth;
}
