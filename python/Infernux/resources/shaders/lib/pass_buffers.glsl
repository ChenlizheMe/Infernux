ShaderInfo {
    Name "Lib Pass Buffers"
}

// Source-scoped buffers supplied by the PassResult selected for this effect.
// Shaders importing this library declare Capabilities [Fullscreen, PassBuffers].
// UV (0,0) is the upper-left pixel of this View; UV (1,1) is the lower-right.
// Use the selected buffer's textureSize for pixel offsets at reduced resolution.
// Depth is Vulkan device depth in [0,1]. Read these buffers only in graph
// passes ordered after their opaque producer; graph barriers own visibility.

vec4 samplePassColor(vec2 uv) {
    return texture(_InxPassColor, uv);
}

float samplePassDeviceDepth(vec2 uv) {
    return texture(_InxPassDepth, uv).r;
}

vec3 reconstructPassWorldPosition(float deviceDepth, vec2 uv) {
    vec4 clip = vec4(uv * 2.0 - 1.0, deviceDepth, 1.0);
    vec4 world = ubo.inverseViewProj * clip;
    return world.xyz / max(abs(world.w), 1e-7);
}

vec3 samplePassWorldPosition(vec2 uv) {
    return reconstructPassWorldPosition(samplePassDeviceDepth(uv), uv);
}

vec3 samplePassViewPosition(vec2 uv) {
    return (ubo.view * vec4(samplePassWorldPosition(uv), 1.0)).xyz;
}

float linearizePassEyeDepth(float rawDepth) {
    // Fast path for ordinary perspective cameras. Use the UV-based helper
    // below for orthographic or custom projection matrices.
    return 1.0 / (ubo.zBufferParams.z * rawDepth + ubo.zBufferParams.w);
}

float samplePassLinearEyeDepth(vec2 uv) {
    // Infernux uses a left-handed camera space, so visible geometry normally
    // has positive view Z. abs() also keeps custom/oblique projection
    // overrides on the same positive-distance contract used by lighting.
    return abs(samplePassViewPosition(uv).z);
}

float samplePassLinear01Depth(vec2 uv) {
    return samplePassLinearEyeDepth(uv) / ubo.projectionParams.y;
}

vec4 samplePassNormalEncoded(vec2 uv) {
    return texture(_InxPassNormal, uv);
}

float samplePassNormalCoverage(vec2 uv) {
    return samplePassNormalEncoded(uv).a;
}

vec3 decodePassWorldNormal(vec4 encodedNormal) {
    if (encodedNormal.a <= 0.0)
        return vec3(0.0);
    return normalize(encodedNormal.xyz * 2.0 - 1.0);
}

vec3 samplePassWorldNormal(vec2 uv) {
    return decodePassWorldNormal(samplePassNormalEncoded(uv));
}

vec3 samplePassViewNormal(vec2 uv) {
    vec3 normalWS = samplePassWorldNormal(uv);
    return dot(normalWS, normalWS) > 0.0
        ? normalize(mat3(ubo.view) * normalWS)
        : vec3(0.0);
}

vec2 samplePassMotionUV(vec2 uv) {
    return texture(_InxPassMotion, uv).xy;
}

vec2 samplePassMotionPixels(vec2 uv) {
    return samplePassMotionUV(uv) * vec2(textureSize(_InxPassMotion, 0));
}
