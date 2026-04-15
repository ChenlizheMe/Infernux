// ============================================================================
// surface_main.glsl — Auto-generated main() for forward surface shaders
//
// Calls surface() to fill SurfaceData, then evaluates lighting via evaluate()
// from the referenced .shadingmodel file.
//
// _AlphaClipThreshold is always available in MaterialProperties UBO.
// When > 0.0, fragments with alpha below the threshold are discarded.
// ============================================================================

void main() {
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
    surface(s);
    if (material._AlphaClipThreshold > 0.0 && s.alpha < material._AlphaClipThreshold) discard;
    vec4 _forwardResult;
    evaluate(s, _forwardResult);
    outColor = _forwardResult;
}
