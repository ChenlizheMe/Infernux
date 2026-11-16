import infernux.shader as shader


class ParticleUnlit(shader.particle):
    shader_id = "infernux/particle/unlit"
    surface_type = "transparent"
    cull_mode = "none"
    base_color = shader.material.color()
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    glow = shader.material.float(default=0.0, range=(0.0, 8.0))
    rainbow = shader.material.float(default=0.0, range=(0.0, 1.0))

    def main(self, ctx: shader.ParticleContext) -> shader.vec4:
        uv = ctx.uv0 * self.albedo_map_st.xy + self.albedo_map_st.zw
        next_uv = ctx.next_uv * self.albedo_map_st.xy + self.albedo_map_st.zw
        next_dx = shader.ddx(next_uv)
        next_dy = shader.ddy(next_uv)
        base = shader.sample(self.albedo_map, uv)
        if ctx.frame_blend > 0.0:
            base = shader.lerp(base, shader.sample_grad(self.albedo_map, next_uv, next_dx, next_dy), ctx.frame_blend)
        rgb = base.rgb * ctx.color * self.base_color.rgb
        k = shader.abs(shader.frac(shader.vec3(ctx.uv0.x) + shader.vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0)
        hue = shader.lerp(shader.saturate(k - 1.0), shader.vec3(1.0), 0.05)
        rgb = rgb * shader.lerp(shader.vec3(1.0), hue, shader.saturate(self.rainbow))
        return shader.vec4(rgb * (1.0 + self.glow), base.a * self.base_color.a * ctx.alpha)
