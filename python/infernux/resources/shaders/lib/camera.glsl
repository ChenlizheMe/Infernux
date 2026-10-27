ShaderInfo {
    Name "Lib Camera"
}

// ============================================================================
// lib/camera.glsl — Camera and screen-space utility functions
//
// Requires: shader-linker camera contract plus InfGlobals for screen/depth data
// Usage: ShaderInfo Imports: Lib Camera
// ============================================================================

// ---- Screen coordinates ----

// Normalized Vulkan screen UV (0~1, upper-left origin), matching sampled
// render-target textures and the projection matrix's inverted Y axis.
vec2 getScreenUV() {
    return gl_FragCoord.xy * _Globals._ScreenParams.zw;
}

// Integer pixel coordinate
ivec2 getPixelCoord() {
    return ivec2(gl_FragCoord.xy);
}

// ---- Camera ----

// Structured shaders receive a camera source selected by the shader linker.
// Geometry surface fragments, including Unlit, resolve this to the camera-local
// set-1 LightingUBO. Never read _WorldSpaceCameraPos directly
// here: that buffer is shared by Scene and Game graphs and is staged from the
// editor camera.
#ifndef INX_SHADING_CAMERA_POSITION
#define INX_SHADING_CAMERA_POSITION _Globals._WorldSpaceCameraPos.xyz
#endif

// World-space camera position
vec3 getCameraPosition() {
    return INX_SHADING_CAMERA_POSITION;
}

// Normalized view direction from camera to fragment (world-space)
vec3 getViewDirection(vec3 worldPos) {
    return normalize(INX_SHADING_CAMERA_POSITION - worldPos);
}

// Shader-linker camera bindings are per View; explicit-layout shaders supply
// their own globals interface instead.
#ifndef INX_CAMERA_PROJECTION_PARAMS
#define INX_CAMERA_PROJECTION_PARAMS _Globals._ProjectionParams
#define INX_CAMERA_ZBUFFER_PARAMS _Globals._ZBufferParams
#endif

// Camera near plane distance
float getCameraNear() {
    return INX_CAMERA_PROJECTION_PARAMS.x;
}

// Camera far plane distance
float getCameraFar() {
    return INX_CAMERA_PROJECTION_PARAMS.y;
}

// ---- Depth linearization ----


// Linearize depth buffer value to view-space distance (eye depth)
// Perspective device depth: 1 at near, 0 at far
// Returns positive eye-space distance
float linearEyeDepth(float rawDepth) {
    // _ZBufferParams: x = far/near - 1, y = 1, z = x/far, w = 1/far
    return 1.0 / (INX_CAMERA_ZBUFFER_PARAMS.z * rawDepth + INX_CAMERA_ZBUFFER_PARAMS.w);
}

// Linearize depth to [0,1] range (0 = near, 1 = far)
float linear01Depth(float rawDepth) {
    return linearEyeDepth(rawDepth) / INX_CAMERA_PROJECTION_PARAMS.y;
}

// ---- Time helpers ----

// Elapsed time in seconds
float getTime() {
    return _Globals._Time.x;
}

// Delta time (seconds since last frame)
float getDeltaTime() {
    return _Globals._Time.w;
}

// Smooth delta time (exponentially smoothed)
float getSmoothDeltaTime() {
    return _Globals._FrameParams.y;
}
