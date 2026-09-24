#version 450
layout(set = 0, binding = 0) uniform sampler3D volume;
layout(location = 0) out vec4 color;

void main()
{
    float z = gl_FragCoord.x < 1.0 ? 0.25 : 0.75;
    color = texture(volume, vec3(0.25, 0.25, z));
}
