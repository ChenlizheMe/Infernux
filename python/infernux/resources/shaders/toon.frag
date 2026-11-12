#version 450

ShaderInfo {
    Name "Toon"
    ShadingModel Toon
    Queue 2000
    Properties {
        Color base_color = [1.0, 1.0, 1.0, 1.0]
        Float diffuseThreshold = 0.45 Range(0.0, 1.0)
        Float bandSoftness = 0.04 Range(0.0, 0.25)
        Float smoothness = 0.65 Range(0.0, 1.0)
        Float specular_highlights = 0.35 Range(0.0, 1.0)
        Float ambient_occlusion = 1.0 Range(0.0, 1.0)
        Float normal_scale = 1.0 Range(0.0, 2.0)
        Color emission_color = [0.0, 0.0, 0.0, 0.0] HDR
        Texture2D albedo_map = white
        Float4 albedo_map_st = [1.0, 1.0, 0.0, 0.0]
        Texture2D occlusion_map = white
        Float4 occlusion_map_st = [1.0, 1.0, 0.0, 0.0]
        Texture2D normal_map = normal
        Float4 normal_map_st = [1.0, 1.0, 0.0, 0.0]
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    vec4 texColor = sampleAlbedoAlpha(albedo_map, transformTextureUV(v_TexCoord, material.albedo_map_st));
    s.albedo = texColor.rgb * getVertexColor() * material.base_color.rgb;
    s.normalWS = sampleNormal(normal_map, transformTextureUV(v_TexCoord, material.normal_map_st), material.normal_scale);
    s.smoothness = material.smoothness;
    s.occlusion = sampleGrayscale(occlusion_map, transformTextureUV(v_TexCoord, material.occlusion_map_st)) * material.ambient_occlusion;
    s.emission = material.emission_color.rgb * material.emission_color.a;
    s.alpha = texColor.a * material.base_color.a;
    s.specular_highlights = material.specular_highlights;
    s.shadingParam0 = material.diffuseThreshold;
    s.shadingParam1 = material.bandSoftness;
}
