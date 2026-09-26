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
        Texture2D metallicMap = white
        Texture2D smoothnessMap = white
        Texture2D aoMap = white
        Texture2D normalMap = normal
        Texture2D emissionMap = white
    }
}

void surface(out SurfaceData s) {
    s = InitSurfaceData();

    vec4 texColor = sampleAlbedoAlpha(texSampler, getUV(material.baseColorUvSet));
    s.albedo     = texColor.rgb * getVertexColor() * material.baseColor.rgb;
    s.metallic   = dot(sampleAlbedoAlpha(metallicMap, getUV(material.metallicUvSet)), material.metallicChannels) * material.metallic;
    float smoothnessSample = dot(sampleAlbedoAlpha(smoothnessMap, getUV(material.smoothnessUvSet)), material.smoothnessChannels);
    s.smoothness = mix(smoothnessSample * material.smoothness,
                      1.0 - smoothnessSample * (1.0 - material.smoothness), material.smoothnessFromRoughness);
    s.occlusion  = mix(1.0, sampleGrayscale(aoMap, getUV(material.occlusionUvSet)), material.occlusionStrength) * material.ambientOcclusion;
    s.normalWS   = sampleNormal(normalMap, getUV(material.normalUvSet), material.normalUvSet, material.normalScale);
    s.emission   = sampleEmission(emissionMap, getUV(material.emissionUvSet)) * material.emissionColor.rgb * material.emissionColor.a;
    s.alpha      = texColor.a * material.baseColor.a;
    s.specularHighlights = material.specularHighlights;
}
