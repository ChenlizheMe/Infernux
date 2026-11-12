// ============================================================================
// surface_main_gbuffer.glsl — Auto-generated main() for GBuffer surface shaders
//
// _alpha_clip_threshold is always available in MaterialProperties UBO.
// ============================================================================

void main() {
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
${SURFACE_CALL}
    s.albedo *= v_LineColor.rgb;
    s.alpha *= v_LineColor.a;
    s.normalWS = ResolveSurfaceNormal(s.normalWS, v_Normal);
    // Unity-style double-sided normal fix (see surface_main.glsl for details)
    if (!gl_FrontFacing)
        s.normalWS = -s.normalWS;
    if (material._alpha_clip_threshold > 0.0 && s.alpha < material._alpha_clip_threshold) discard;
    packGBuffer(s, outGBuf0, outGBuf1, outGBuf2, outGBuf3, outGBuf4);
}
