import infernux.shader as shader
from infernux.shader.lib.surface import sample_normal, texture_uv


class Toon(shader.surface):
    shader_id = "infernux/toon"
    shading_model = shader.ref("infernux/shading/toon")
    base_color = shader.material.color()
    diffuse_threshold = shader.material.float(default=0.45, range=(0.0, 1.0))
    band_softness = shader.material.float(default=0.04, range=(0.0, 0.25))
    smoothness = shader.material.float(default=0.65, range=(0.0, 1.0))
    specular_highlights = shader.material.float(default=0.35, range=(0.0, 1.0))
    ambient_occlusion = shader.material.float(default=1.0, range=(0.0, 1.0))
    normal_scale = shader.material.float(default=1.0, range=(0.0, 2.0))
    emission_color = shader.material.color(default=(0.0, 0.0, 0.0, 0.0), hdr=True)
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    occlusion_map = shader.material.texture_2d()
    occlusion_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    normal_map = shader.material.texture_2d(default="normal")
    normal_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        base = shader.sample(self.albedo_map, texture_uv(ctx, 0, self.albedo_map_st))
        occlusion = shader.sample(self.occlusion_map, texture_uv(ctx, 0, self.occlusion_map_st)).r
        normal_uv = texture_uv(ctx, 0, self.normal_map_st)
        return shader.SurfaceData(
            albedo=base.rgb * ctx.color * self.base_color.rgb,
            normal_ws=sample_normal(self.normal_map, normal_uv, 0, self.normal_scale, ctx),
            smoothness=self.smoothness,
            occlusion=occlusion * self.ambient_occlusion,
            emission=self.emission_color.rgb * self.emission_color.a,
            alpha=base.a * self.base_color.a,
            specular_highlights=self.specular_highlights,
            shading_param0=self.diffuse_threshold,
            shading_param1=self.band_softness,
        )
