import infernux.shader as shader


class SpriteUnlit(shader.surface):
    shader_id = "infernux/sprite/unlit"
    shading_model = shader.ref("./unlit.shadingmodel.py")
    cull_mode = "none"
    render_queue = 2000
    alpha_clip = 0.5
    base_color = shader.material.color()
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    uv_rect = shader.material.vec4(default=(0.0, 0.0, 1.0, 1.0))
    display_scale = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        local = (ctx.uv0 - 0.5) / shader.max(self.display_scale.xy, shader.vec2(1e-6)) + 0.5
        uv = self.uv_rect.xy + local * self.uv_rect.zw
        base = shader.sample(self.albedo_map, uv * self.albedo_map_st.xy + self.albedo_map_st.zw)
        inside = local.x >= 0.0 and local.x <= 1.0 and local.y >= 0.0 and local.y <= 1.0
        return shader.SurfaceData(
            albedo=base.rgb * self.base_color.rgb,
            alpha=base.a * self.base_color.a if inside else 0.0,
        )
