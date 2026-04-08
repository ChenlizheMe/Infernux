// ============================================================================
// surface_main_gbuffer.glsl — Auto-generated main() for GBuffer surface shaders
//
// _AlphaClipThreshold is always available in MaterialProperties UBO.
// ============================================================================

void main() {
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
    surface(s);
    if (material._AlphaClipThreshold > 0.0 && s.alpha < material._AlphaClipThreshold) discard;
    evaluate(s, outGBuf0, outGBuf1, outGBuf2, outGBuf3);
}
