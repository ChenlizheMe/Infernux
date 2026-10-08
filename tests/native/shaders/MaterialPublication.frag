#version 450
layout(set = 0, binding = 0) uniform MaterialProperties { vec4 tint; float fragmentGain; } material;
layout(location = 0) in float gain;
layout(location = 0) out vec4 color;
void main() { color = vec4(material.tint.rgb * material.fragmentGain * gain, material.tint.a); }
