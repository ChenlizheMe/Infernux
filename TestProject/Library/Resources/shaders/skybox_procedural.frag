#version 450

@shader_id: Infernux/Skybox-Procedural
@pass_tag: skybox
@cull: back
@depth_write: false
@depth_test: less_equal
@queue: 32767
@hidden
@property: skyTopColor, Color, [0.20, 0.28, 0.46, 1.0]
@property: skyHorizonColor, Color, [0.50, 0.58, 0.70, 1.0]
@property: groundColor, Color, [0.24, 0.22, 0.22, 1.0]
@property: exposure, Float, 1.35

@import: math

// Input from vertex shader
layout(location = 0) in vec3 fragWorldDir;

// Output
layout(location = 0) out vec4 outColor;

// ============================================================================
// Procedural Sky
// ============================================================================

void main() {
    vec3 dir = normalize(fragWorldDir);

    // ---- Sky gradient ----
    // Y component: +1 = zenith, 0 = horizon, -1 = nadir
    float y = dir.y;

    vec3 skyColor = skyGradient(y,
        material.skyTopColor.rgb,
        material.skyHorizonColor.rgb,
        material.groundColor.rgb);

    // ---- Horizon haze ----
    // Add subtle brightness boost near the horizon
    float horizonGlow = 1.0 - abs(y);
    horizonGlow = pow(horizonGlow, 8.0) * 0.15;
    skyColor += vec3(horizonGlow);

    // ---- Final composition ----
    vec3 color = skyColor;

    // Exposure
    color *= material.exposure;

    // Output linear HDR — tonemapping and gamma correction are handled
    // by the post-process stack (consistent with lit.frag).
    // Applying them here would mix sRGB skybox samples with linear HDR
    // object samples during MSAA resolve, visibly degrading anti-aliasing.

    outColor = vec4(color, 1.0);
}
