#version 450

ShaderInfo {
    Name "Sprite Lit"
    ShadingModel PBR
    Queue 2000
    Cull None
    AlphaClip 0.5
    Properties {
        Color base_color = [1.0, 1.0, 1.0, 1.0]
        Float metallic = 0.0
        Float smoothness = 0.5
        Float ambient_occlusion = 1.0
        Color emission_color = [0.0, 0.0, 0.0, 0.0] HDR
        Float normal_scale = 1.0
        Float specular_highlights = 1.0
        Texture2D albedo_map = white
        Float4 albedo_map_st = [1.0, 1.0, 0.0, 0.0]
        Texture2D normal_map = normal
        Float4 normal_map_st = [1.0, 1.0, 0.0, 0.0]
        Float4 uvRect = [0.0, 0.0, 1.0, 1.0]
        Float4 displayScale = [1.0, 1.0, 0.0, 0.0]
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();
    // displayScale.xy: fraction of the quad the sprite occupies (aspect fit).
    vec2 dScale = material.displayScale.xy;
    vec2 tc = (v_TexCoord - 0.5) / max(dScale, vec2(1e-6)) + 0.5;
    if (tc.x < 0.0 || tc.x > 1.0 || tc.y < 0.0 || tc.y > 1.0) {
        s.alpha = 0.0;
        return;
    }
    // uvRect: xy = offset, zw = scale (sub-rect in UV space)
    vec2 uv = material.uvRect.xy + tc * material.uvRect.zw;
    vec4 texColor = texture(albedo_map, transformTextureUV(uv, material.albedo_map_st));
    s.albedo     = texColor.rgb * material.base_color.rgb;
    s.metallic   = material.metallic;
    s.smoothness = material.smoothness;
    s.occlusion  = material.ambient_occlusion;
    s.normalWS   = sampleNormal(normal_map, transformTextureUV(v_TexCoord, material.normal_map_st), material.normal_scale);
    s.emission   = material.emission_color.rgb * material.emission_color.a;
    s.alpha      = texColor.a * material.base_color.a;
    s.specular_highlights = material.specular_highlights;
}
