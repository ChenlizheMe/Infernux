// ============================================================================
// gbuffer_output.glsl — GBuffer packing snippet (deferred rendering)
//
// Injected into surface_main.glsl at ${GBUFFER_OUTPUT} when hasGBufferTarget
// is true.  Currently unused (forward-only rendering); the output variables
// outGBufNormal/Material/Emission will be declared by a future deferred
// compile target.
// ============================================================================
    outGBufNormal = vec4(normalize(s.normalWS) * 0.5 + 0.5, 1.0);
    outGBufMaterial = vec4(clamp(s.metallic, 0.0, 1.0), s.smoothness, s.specularHighlights, s.alpha);
    outGBufEmission = vec4(s.emission, 1.0);
