// Keep the material's real alpha/discard semantics, including procedural
// masks and secondary textures. The shader annotation supplies a default;
// the material's uniform decides whether clipping is active for this draw.
void main() {
    if (material._AlphaClipThreshold <= 0.0) return;
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
${SURFACE_CALL}
    s.alpha *= v_LineColor.a;
    if (s.alpha < material._AlphaClipThreshold) discard;
}
