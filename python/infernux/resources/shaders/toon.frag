#version 450

ShaderInfo {
    Name "Toon"
    ShadingModel Toon
    Queue 2000
    Properties {
        Color baseColor = [1.0, 1.0, 1.0, 1.0]
        Float diffuseThreshold = 0.45 Range(0.0, 1.0)
        Float bandSoftness = 0.04 Range(0.0, 0.25)
        Float smoothness = 0.65 Range(0.0, 1.0)
        Float specularHighlights = 0.35 Range(0.0, 1.0)
        Float ambientOcclusion = 1.0 Range(0.0, 1.0)
        Float normalScale = 1.0 Range(0.0, 2.0)
        Color emissionColor = [0.0, 0.0, 0.0, 0.0] HDR
        Texture2D texSampler = white
        Float4 texSampler_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D aoMap = white
        Float4 aoMap_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D normalMap = normal
        Float4 normalMap_ST = [1.0, 1.0, 0.0, 0.0]
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    vec4 texColor = sampleAlbedoAlpha(texSampler, transformTextureUV(v_TexCoord, material.texSampler_ST));
    s.albedo = texColor.rgb * getVertexColor() * material.baseColor.rgb;
    s.normalWS = sampleNormal(normalMap, transformTextureUV(v_TexCoord, material.normalMap_ST), material.normalScale);
    s.smoothness = material.smoothness;
    s.occlusion = sampleGrayscale(aoMap, transformTextureUV(v_TexCoord, material.aoMap_ST)) * material.ambientOcclusion;
    s.emission = material.emissionColor.rgb * material.emissionColor.a;
    s.alpha = texColor.a * material.baseColor.a;
    s.specularHighlights = material.specularHighlights;
    s.shadingParam0 = material.diffuseThreshold;
    s.shadingParam1 = material.bandSoftness;
}
