// ============================================================================
// default_gbuffer_evaluate.glsl — Engine GBuffer packing
//
// Used automatically by deferred-capable shading models. Lighting is
// deliberately absent here: this pass
// records material state and the fullscreen deferred pass evaluates lights.
//
// G-Buffer layout:
//   gbuf0.rgb  = base color,                    gbuf0.a = alpha
//   gbuf1.rgb  = normal (encoded),               gbuf1.a = smoothness
//   gbuf2.r    = metallic, gbuf2.g = occlusion,  gbuf2.b = specularHighlights,
//   gbuf2.a    = shadingParam0
//   gbuf3.rgb  = emission,                        gbuf3.a = shadingParam1
//   gbuf4.x = object light-layer mask, gbuf4.y = stable shading-model ID,
//   gbuf4.z = receive shadows (0/1), gbuf4.w = reserved (0)
// ============================================================================

void packGBuffer(in SurfaceData s, out vec4 gbuf0, out vec4 gbuf1,
                 out vec4 gbuf2, out vec4 gbuf3, out uvec4 gbuf4) {
    // Preserve the surface color for the same shading() function used by
    // Forward. Models such as PBR impose their own reflectance limits;
    // Unlit and custom models may intentionally produce HDR base colors.
    gbuf0 = vec4(s.albedo, s.alpha);
    gbuf1 = vec4(normalize(s.normalWS) * 0.5 + 0.5, s.smoothness);
    gbuf2 = vec4(clamp(s.metallic, 0.0, 1.0), clamp(s.occlusion, 0.0, 1.0),
                 clamp(s.specularHighlights, 0.0, 1.0), s.shadingParam0);
    gbuf3 = vec4(s.emission, s.shadingParam1);
    gbuf4 = uvec4(_inx_ObjectLayerMask, ${SHADING_MODEL_ID}, uint(_inx_ReceivesShadows), 0u);
}
