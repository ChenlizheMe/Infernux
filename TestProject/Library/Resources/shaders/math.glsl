@shader_id: math

@import: lib/common

// ============================================================================
// math.glsl — Shared math constants and utility functions
//
// Constants (PI, INV_PI, HALF_PI, TWO_PI, EPSILON, FLT_MIN) and saturate()
// are provided by lib/common. This file adds legacy aliases and sky helpers.
// ============================================================================

// Legacy alias — existing code uses saturateVec3(x) instead of saturate(vec3)
vec3 saturateVec3(vec3 x) {
    return clamp(x, vec3(0.0), vec3(1.0));
}

// Shared sky gradient — used by both skybox_procedural.frag and
// sampleAmbientProbe(). Adjust constants here once.
//
// Uses a single continuous blend:
//   ground ──smoothstep──▶ equator ──smoothstep──▶ sky
//   nadir (-1)           horizon (0)              zenith (+1)
//
// SKY_EDGE / GROUND_EDGE control how far the equator band extends.
// Smaller = narrower equator band, sharper transition.
vec3 skyGradient(float y, vec3 sky, vec3 equator, vec3 ground) {
    const float SKY_EDGE    = 0.6;
    const float GROUND_EDGE = 0.28;
    const float EQUATOR_STRENGTH = 0.35;  // how much equator tints the horizon (0=none, 1=full)

    // Base: direct sky↔ground blend through the horizon
    float t = smoothstep(-GROUND_EDGE, SKY_EDGE, y);
    vec3 base = mix(ground, sky, t);

    // Equator: thin additive tint near horizon only
    float horizonMask = 1.0 - smoothstep(0.0, max(SKY_EDGE, GROUND_EDGE), abs(y));
    return mix(base, equator, horizonMask * EQUATOR_STRENGTH);
}
