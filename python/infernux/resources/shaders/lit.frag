#version 450

ShaderInfo {
    Name "Lit"
    Capabilities [BindlessTextures]
    ShadingModel PBR
    Queue 2000
    Properties {
        Color baseColor = [1.0, 1.0, 1.0, 1.0]
        Float metallic = 0.0
        Float smoothness = 0.5
        Float ambientOcclusion = 1.0
        Color emissionColor = [0.0, 0.0, 0.0, 0.0] HDR
        Float normalScale = 1.0
        Float specularHighlights = 1.0
        Float4 metallicChannels = [1.0, 0.0, 0.0, 0.0]
        Float4 smoothnessChannels = [1.0, 0.0, 0.0, 0.0]
        Float smoothnessFromRoughness = 0.0
        Float occlusionStrength = 1.0
        Int baseColorUvSet = 0
        Int normalUvSet = 0
        Int metallicUvSet = 0
        Int smoothnessUvSet = 0
        Int occlusionUvSet = 0
        Int emissionUvSet = 0
        Texture2D texSampler = white
        Float4 texSampler_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D metallicMap = white
        Float4 metallicMap_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D smoothnessMap = white
        Float4 smoothnessMap_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D aoMap = white
        Float4 aoMap_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D normalMap = normal
        Float4 normalMap_ST = [1.0, 1.0, 0.0, 0.0]
        Texture2D emissionMap = white
        Float4 emissionMap_ST = [1.0, 1.0, 0.0, 0.0]
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    vec4 texColor = sampleAlbedoAlpha(texSampler, transformTextureUV(getUV(material.baseColorUvSet), material.texSampler_ST));
    s.albedo     = texColor.rgb * getVertexColor() * material.baseColor.rgb;
    s.metallic   = dot(sampleAlbedoAlpha(metallicMap, transformTextureUV(getUV(material.metallicUvSet), material.metallicMap_ST)), material.metallicChannels) * material.metallic;
    float smoothnessSample = dot(sampleAlbedoAlpha(smoothnessMap, transformTextureUV(getUV(material.smoothnessUvSet), material.smoothnessMap_ST)), material.smoothnessChannels);
    s.smoothness = mix(smoothnessSample * material.smoothness,
                      1.0 - smoothnessSample * (1.0 - material.smoothness), material.smoothnessFromRoughness);
    s.occlusion  = mix(1.0, sampleGrayscale(aoMap, transformTextureUV(getUV(material.occlusionUvSet), material.aoMap_ST)), material.occlusionStrength) * material.ambientOcclusion;
    s.normalWS   = sampleNormal(normalMap, transformTextureUV(getUV(material.normalUvSet), material.normalMap_ST), material.normalUvSet, material.normalScale);
    s.emission   = sampleEmission(emissionMap, transformTextureUV(getUV(material.emissionUvSet), material.emissionMap_ST)) * material.emissionColor.rgb * material.emissionColor.a;
    s.alpha      = texColor.a * material.baseColor.a;
    s.specularHighlights = material.specularHighlights;
}
