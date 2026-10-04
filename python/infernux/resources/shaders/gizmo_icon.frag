#version 450

ShaderInfo {
    Name "Gizmo Icon"
    Hidden On
    ShadingModel Unlit
    Surface Transparent
    DepthWrite Off
    Blend Alpha
    AlphaClip 0.01
    CastShadows Off
    Properties {
        Color baseColor = [1.0, 1.0, 1.0, 1.0]
        Texture2D texSampler = white
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    // These editor billboards are only a few dozen screen pixels across, yet
    // their source art is 256 px. A regular trilinear lookup blends the 32/64
    // px mips and softens the thin camera/light strokes. Sample roughly one
    // finer mip while retaining filtered edges and stable mip transitions.
    vec4 texColor = texture(texSampler, v_TexCoord, -0.75);
    s.albedo = texColor.rgb * v_Color * material.baseColor.rgb;
    s.alpha = texColor.a * material.baseColor.a;
}
