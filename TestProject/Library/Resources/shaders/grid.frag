#version 450

@shader_id: Infernux/Grid
@cull: none
@hidden
@property: fadeStart, Float, 15.0
@property: fadeEnd, Float, 80.0

// UBO — camera position is derived from the view matrix at runtime
layout(std140, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 0) out vec4 outColor;

void main() {
    // Extract camera world position: pos = -(R^T * t) using view matrix orthogonality
    vec3 cameraPos = -(transpose(mat3(ubo.view)) * ubo.view[3].xyz);

    vec2 coord = fragWorldPos.xz;

    // ---- Minor grid lines (every 1 unit) ----
    vec2 dMinor = fwidth(coord);
    vec2 gridMinor = abs(fract(coord - 0.5) - 0.5);
    vec2 lineMinor = gridMinor / dMinor;
    float minor = 1.0 - min(min(lineMinor.x, lineMinor.y), 1.0);

    // LOD fade: prevent moiré when lines become sub-pixel
    float lodFadeMinor = 1.0 - clamp(max(dMinor.x, dMinor.y) * 2.0 - 1.0, 0.0, 1.0);
    minor *= lodFadeMinor;

    // ---- Major grid lines (every 10 units) ----
    vec2 coordMajor = coord / 10.0;
    vec2 dMajor = fwidth(coordMajor);
    vec2 gridMajor = abs(fract(coordMajor - 0.5) - 0.5);
    vec2 lineMajor = gridMajor / dMajor;
    float major = 1.0 - min(min(lineMajor.x, lineMajor.y), 1.0);

    float lodFadeMajor = 1.0 - clamp(max(dMajor.x, dMajor.y) * 2.0 - 1.0, 0.0, 1.0);
    major *= lodFadeMajor;

    // ---- Compose line alpha ----
    float lineAlpha = minor * 0.25 + major * 0.4;

    // White line color (Unity style)
    vec3 lineColor = vec3(0.8);

    // ---- Distance fade (XZ plane) ----
    float dist = length(coord - cameraPos.xz);
    float distFade = 1.0 - smoothstep(material.fadeStart, material.fadeEnd, dist);

    float alpha = lineAlpha * distFade;

    if (alpha < 0.005) discard;

    outColor = vec4(lineColor, alpha);
}
