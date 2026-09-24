#extension GL_EXT_nonuniform_qualifier : require

layout(set = ${TEXTURE_TABLE_SET}, binding = ${TEXTURE_TABLE_BINDING}) uniform sampler2D _InxBindlessTextures[];

layout(std140, set = ${TEXTURE_INDEX_SET}, binding = ${TEXTURE_INDEX_BINDING}) uniform InxMaterialTextureIndices {
${TEXTURE_MEMBERS}} _InxMaterialTextureIndices;

vec4 inxSampleBindlessTexture(uint textureIndex, vec2 uv) {
    return texture(_InxBindlessTextures[nonuniformEXT(textureIndex)], uv);
}

// Preserve the ordinary material helper API while the property itself is a
// uint resource handle. This keeps built-in surface shaders source-compatible
// across bounded and bindless descriptor implementations.
vec3 sampleAlbedo(uint textureIndex) {
    return inxSampleBindlessTexture(textureIndex, v_TexCoord).rgb;
}

vec4 sampleAlbedoAlpha(uint textureIndex) {
    return inxSampleBindlessTexture(textureIndex, v_TexCoord);
}

vec3 sampleAlbedo(uint textureIndex, vec2 uv) {
    return inxSampleBindlessTexture(textureIndex, uv).rgb;
}

vec4 sampleAlbedoAlpha(uint textureIndex, vec2 uv) {
    return inxSampleBindlessTexture(textureIndex, uv);
}

float sampleGrayscale(uint textureIndex) {
    return inxSampleBindlessTexture(textureIndex, v_TexCoord).r;
}

float sampleGrayscale(uint textureIndex, vec2 uv) {
    return inxSampleBindlessTexture(textureIndex, uv).r;
}

vec3 sampleEmission(uint textureIndex) {
    return inxSampleBindlessTexture(textureIndex, v_TexCoord).rgb;
}

vec3 sampleEmission(uint textureIndex, vec2 uv) {
    return inxSampleBindlessTexture(textureIndex, uv).rgb;
}

vec3 inxSampleBindlessNormal(uint textureIndex, vec2 uv, float scale) {
    vec2 encodedXY = inxSampleBindlessTexture(textureIndex, uv).rg * 2.0 - 1.0;
    vec3 tangentNormal = vec3(encodedXY, sqrt(max(1.0 - dot(encodedXY, encodedXY), 0.0)));
    tangentNormal.xy *= scale;
    vec3 n = normalize(v_Normal);
    vec3 t = normalize(v_Tangent.xyz);
    vec3 b = cross(n, t) * v_Tangent.w;
    return normalize(mat3(t, b, n) * normalize(tangentNormal));
}

vec3 sampleNormal(uint textureIndex, float scale) {
    return inxSampleBindlessNormal(textureIndex, v_TexCoord, scale);
}

vec3 sampleNormal(uint textureIndex, vec2 uv, float scale) {
    return inxSampleBindlessNormal(textureIndex, uv, scale);
}

vec3 sampleNormal(uint textureIndex, vec2 uv, int uvSet, float scale) {
    if (uvSet == 0) {
        return inxSampleBindlessNormal(textureIndex, uv, scale);
    }

    vec3 tangentNormal = inxSampleBindlessTexture(textureIndex, uv).rgb * 2.0 - 1.0;
    tangentNormal.xy *= scale;
    tangentNormal = normalize(tangentNormal);
    vec3 normalWS = normalize(v_Normal);
    vec3 positionDx = dFdx(v_WorldPos);
    vec3 positionDy = dFdy(v_WorldPos);
    vec2 uvDx = dFdx(uv);
    vec2 uvDy = dFdy(uv);
    vec3 dyPerpendicular = cross(positionDy, normalWS);
    vec3 dxPerpendicular = cross(normalWS, positionDx);
    vec3 tangent = dyPerpendicular * uvDx.x + dxPerpendicular * uvDy.x;
    vec3 bitangent = dyPerpendicular * uvDx.y + dxPerpendicular * uvDy.y;
    float basisLengthSq = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if (basisLengthSq <= 1e-12) {
        return inxSampleBindlessNormal(textureIndex, uv, scale);
    }
    float inverseBasisLength = inversesqrt(basisLengthSq);
    mat3 uvBasis = mat3(tangent * inverseBasisLength,
                        bitangent * inverseBasisLength,
                        normalWS);
    return normalize(uvBasis * tangentNormal);
}
