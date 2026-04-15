// ============================================================================
// shadow_vertex_main.glsl — Shadow pass vertex main() template
//
// Transforms vertices into light clip-space using the shadow UBO.
// Outputs varyings needed for alpha-clip shadow support (texCoord, normal).
// ============================================================================

void main() {
    VertexInput v;
    v.position = inPosition;
    v.normal   = inNormal;
    v.tangent  = inTangent;
    v.color    = inColor;
    v.texCoord = inTexCoord;
${VERTEX_CALL}
    mat4 instModel    = instanceModels[gl_InstanceIndex];
    vec4 worldPos     = instModel * vec4(v.position, 1.0);
    mat3 normalMatrix = mat3(instModel);

    v_WorldPos  = worldPos.xyz;
    v_Normal    = normalize(normalMatrix * v.normal);
    v_Tangent   = vec4(normalize(normalMatrix * v.tangent.xyz), v.tangent.w);
    v_Color     = v.color;
    v_TexCoord  = v.texCoord;
    v_ViewDepth = 0.0;
    gl_Position = shadowUBO.proj * shadowUBO.view * worldPos;
}
