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
        Color base_color = [1.0, 1.0, 1.0, 1.0]
        Texture2D albedo_map = white
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    // These editor billboards are only a few dozen screen pixels across, yet
    // their source art is 256 px. A regular trilinear lookup blends the 32/64
    // px mips and softens the thin camera/light strokes. Sample roughly one
    // finer mip while retaining filtered edges and stable mip transitions.
    vec4 texColor = texture(albedo_map, v_TexCoord, -0.75);
    s.albedo = texColor.rgb * v_Color * material.base_color.rgb;
    s.alpha = texColor.a * material.base_color.a;
}
