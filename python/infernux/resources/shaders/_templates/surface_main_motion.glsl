// Motion keeps the visible material's alpha decision so cutout silhouettes match.
void main() {
    SurfaceData s = InitSurfaceData();
    s.normalWS = normalize(v_Normal);
${SURFACE_CALL}
    s.alpha *= v_LineColor.a;
    if (material._alpha_clip_threshold > 0.0 && s.alpha < material._alpha_clip_threshold) discard;
    outMotion = _inx_MotionVector;
}
