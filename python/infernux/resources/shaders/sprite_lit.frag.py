import infernux.shader as shader
from infernux.shader.lib.surface import sample_normal


class SpriteLit(shader.surface):
    shader_id = "infernux/sprite/lit"
    cull_mode = "none"
    render_queue = 2000
    alpha_clip = 0.5
    base_color = shader.material.color()
    metallic = shader.material.float(range=(0.0, 1.0))
    smoothness = shader.material.float(default=0.5, range=(0.0, 1.0))
    ambient_occlusion = shader.material.float(default=1.0, range=(0.0, 1.0))
    emission_color = shader.material.color(default=(0.0, 0.0, 0.0, 0.0), hdr=True)
    normal_scale = shader.material.float(default=1.0, range=(0.0, 2.0))
    specular_highlights = shader.material.float(default=1.0, range=(0.0, 1.0))
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    normal_map = shader.material.texture_2d(default="normal")
    normal_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    uv_rect = shader.material.vec4(default=(0.0, 0.0, 1.0, 1.0))
    display_scale = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        local = (ctx.uv0 - 0.5) / shader.max(self.display_scale.xy, shader.vec2(1e-6)) + 0.5
        uv = self.uv_rect.xy + local * self.uv_rect.zw
        base = shader.sample(self.albedo_map, uv * self.albedo_map_st.xy + self.albedo_map_st.zw)
        normal_uv = ctx.uv0 * self.normal_map_st.xy + self.normal_map_st.zw
        normal = sample_normal(self.normal_map, normal_uv, 0, self.normal_scale, ctx)
        inside = local.x >= 0.0 and local.x <= 1.0 and local.y >= 0.0 and local.y <= 1.0
        return shader.SurfaceData(
            albedo=base.rgb * self.base_color.rgb,
            metallic=self.metallic,
            smoothness=self.smoothness,
            occlusion=self.ambient_occlusion,
            normal_ws=normal,
            emission=self.emission_color.rgb * self.emission_color.a,
            alpha=base.a * self.base_color.a if inside else 0.0,
            specular_highlights=self.specular_highlights,
        )
