// ============================================================================
// Infernux — Engine Globals UBO (set 2, binding 0)
//
// Available in ALL shaders (vertex + fragment).  Matches EngineGlobalsUBO
// in C++ (std140 layout, 8 × vec4 = 128 bytes).
// ============================================================================

layout(std140, set = 2, binding = 0) uniform InfGlobals {
    // ─── Time ──────────────────────────────────────────────────────
    // x = elapsed time (s), y = sin(time), z = cos(time), w = deltaTime
    vec4 _Time;
    // x = sin(t/20), y = sin(t/4), z = sin(t*2), w = sin(t*3)
    vec4 _SinTime;
    // x = cos(t/20), y = cos(t/4), z = cos(t*2), w = cos(t*3)
    vec4 _CosTime;

    // ─── Screen ────────────────────────────────────────────────────
    // x = width, y = height, z = 1/width, w = 1/height
    vec4 _ScreenParams;

    // ─── Camera ────────────────────────────────────────────────────
    // xyz = world-space camera position, w = 1.0
    vec4 _WorldSpaceCameraPos;
    // x = near, y = far, z = 1/far, w = near/far
    vec4 _ProjectionParams;
    // Linearization helpers: x = 1 - far/near, y = far/near, z = x/far, w = y/far
    vec4 _ZBufferParams;

    // ─── Frame ─────────────────────────────────────────────────────
    // x = frame count, y = smoothDeltaTime, z = 1/deltaTime, w = unused
    vec4 _FrameParams;
} _Globals;
