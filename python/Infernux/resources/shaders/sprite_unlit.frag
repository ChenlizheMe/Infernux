#version 450

@shader_id: sprite_unlit
@shading_model: unlit
@queue: 3000
@property: baseColor, Color, [1.0, 1.0, 1.0, 1.0]
@property: texSampler, Texture2D, white
@property: uvRect, Float4, [0.0, 0.0, 1.0, 1.0]

void surface(out SurfaceData s) {
    s = InitSurfaceData();
    // uvRect: xy = offset, zw = scale (sub-rect in UV space)
    vec2 uv = material.uvRect.xy + v_TexCoord * material.uvRect.zw;
    vec4 texColor = texture(texSampler, uv);
    s.albedo = texColor.rgb * material.baseColor.rgb;
    s.alpha  = texColor.a * material.baseColor.a;
}
