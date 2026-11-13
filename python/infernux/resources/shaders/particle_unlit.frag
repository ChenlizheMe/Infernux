#version 450

ShaderInfo {
    Name "infernux/particle/unlit"
    ShadingModel Unlit
    Queue 3000
    Cull None
    DepthWrite Off
    Blend Alpha
    Properties {
        Color base_color = [1.0, 1.0, 1.0, 1.0]
        Texture2D albedo_map = white
        Float4 albedo_map_st = [1.0, 1.0, 0.0, 0.0]
        Float glow = 0.0 Range(0.0, 8.0)
        Float rainbow = 0.0 Range(0.0, 1.0)
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();
    vec4 texColor = sampleAlbedoAlpha(albedo_map, transformTextureUV(v_TexCoord, material.albedo_map_st));
    vec3 rgb = texColor.rgb * v_Color * material.base_color.rgb;
    if (material.rainbow > 0.001) {
        vec3 k = abs(fract(vec3(v_TexCoord.x) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
        vec3 hue = mix(clamp(k - 1.0, 0.0, 1.0), vec3(1.0), 0.05);
        rgb *= mix(vec3(1.0), hue, clamp(material.rainbow, 0.0, 1.0));
    }
    s.albedo = rgb;
    s.emission = rgb * material.glow;
    s.alpha = texColor.a * material.base_color.a;
}
