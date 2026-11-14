import infernux.shader as shader


class ParticleLit(shader.particle):
    shader_id = "infernux/particle/lit"
    surface_type = "transparent"
    cull_mode = "none"
    base_color = shader.material.color()
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    metallic = shader.material.float(range=(0.0, 1.0))
    smoothness = shader.material.float(default=0.25, range=(0.0, 1.0))
    emission_color = shader.material.color(default=(0.0, 0.0, 0.0, 0.0), hdr=True)

    def main(self, ctx: shader.ParticleContext) -> shader.vec4:
        uv = ctx.uv0 * self.albedo_map_st.xy + self.albedo_map_st.zw
        base = shader.sample(self.albedo_map, uv)
        if ctx.frame_blend > 0.0:
            next_uv = ctx.next_uv * self.albedo_map_st.xy + self.albedo_map_st.zw
            base = shader.lerp(base, shader.sample(self.albedo_map, next_uv), ctx.frame_blend)
        normal = shader.normalize(ctx.normal_ws)
        to_camera = shader.view().camera_position_ws - ctx.position_ws
        if shader.dot(normal, to_camera) < 0.0:
            normal = -normal
        surface = shader.SurfaceData(
            albedo=base.rgb * ctx.color * self.base_color.rgb,
            normal_ws=normal,
            metallic=self.metallic,
            smoothness=self.smoothness,
            emission=self.emission_color.rgb * self.emission_color.a,
        )
        rgb = shader.lighting.particle_pbr(surface, ctx)
        return shader.vec4(rgb, base.a * self.base_color.a * ctx.alpha)
