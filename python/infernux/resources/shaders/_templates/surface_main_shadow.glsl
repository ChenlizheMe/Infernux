// Keep the material's real alpha/discard semantics, including procedural
// masks and secondary textures. AlphaClip controls only the additional
// uniform alpha test; it cannot disable discard authored inside surface().
void main() {
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
${SURFACE_CALL}
    s.alpha *= v_LineColor.a;
    if (material._AlphaClipThreshold > 0.0 && s.alpha < material._AlphaClipThreshold) discard;
}
