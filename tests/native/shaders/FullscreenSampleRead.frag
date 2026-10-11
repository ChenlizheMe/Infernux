#version 450
layout(set = 0, binding = 0) uniform sampler2DMS owners;
layout(set = 0, binding = 1) uniform sampler2DMS depths;
layout(location = 0) out vec4 color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float owner = texelFetch(owners, p, gl_SampleID).r;
    float depth = texelFetch(depths, p, gl_SampleID).r;
    bool correct = abs(owner - (gl_SampleID < 2 ? 1.0 : 9.0)) < .01 &&
                   abs(depth - (.2 + .15 * float(gl_SampleID))) < .001;
    color = !correct ? vec4(0,1,0,1) : owner == 1.0 ? vec4(1,0,0,1) : vec4(0,0,1,1);
}
