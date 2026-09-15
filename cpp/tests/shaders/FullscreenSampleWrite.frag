#version 450
layout(location = 0) out vec4 color;
void main() {
    color = vec4(gl_SampleID < 2 ? 1.0 : 9.0, 0.0, 0.0, 1.0);
    gl_FragDepth = .2 + .15 * float(gl_SampleID);
}
