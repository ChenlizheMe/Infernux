import infernux.shader as shader
from infernux.shader.lib.surface import sample_normal, texture_uv


class Lit(shader.surface):
    shader_id = "infernux/lit"
    base_color = shader.material.color()
    metallic = shader.material.float(range=(0.0, 1.0))
    smoothness = shader.material.float(default=0.5, range=(0.0, 1.0))
    ambient_occlusion = shader.material.float(default=1.0, range=(0.0, 1.0))
    emission_color = shader.material.color(default=(0.0, 0.0, 0.0, 0.0), hdr=True)
    normal_scale = shader.material.float(default=1.0, range=(0.0, 2.0))
    specular_highlights = shader.material.float(default=1.0, range=(0.0, 1.0))
    metallic_channels = shader.material.vec4(default=(1.0, 0.0, 0.0, 0.0))
    smoothness_channels = shader.material.vec4(default=(1.0, 0.0, 0.0, 0.0))
    smoothness_from_roughness = shader.material.float(range=(0.0, 1.0))
    occlusion_strength = shader.material.float(default=1.0, range=(0.0, 1.0))
    base_color_uv_set = shader.material.int(range=(0, 1))
    normal_uv_set = shader.material.int(range=(0, 1))
    metallic_uv_set = shader.material.int(range=(0, 1))
    smoothness_uv_set = shader.material.int(range=(0, 1))
    occlusion_uv_set = shader.material.int(range=(0, 1))
    emission_uv_set = shader.material.int(range=(0, 1))
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    metallic_map = shader.material.texture_2d()
    metallic_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    smoothness_map = shader.material.texture_2d()
    smoothness_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    occlusion_map = shader.material.texture_2d()
    occlusion_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    normal_map = shader.material.texture_2d(default="normal")
    normal_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))
    emission_map = shader.material.texture_2d()
    emission_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        base = shader.sample(self.albedo_map, texture_uv(ctx, self.base_color_uv_set, self.albedo_map_st))
        metallic = shader.dot(shader.sample(self.metallic_map, texture_uv(ctx, self.metallic_uv_set, self.metallic_map_st)), self.metallic_channels)
        smoothness = shader.dot(shader.sample(self.smoothness_map, texture_uv(ctx, self.smoothness_uv_set, self.smoothness_map_st)), self.smoothness_channels)
        occlusion = shader.sample(self.occlusion_map, texture_uv(ctx, self.occlusion_uv_set, self.occlusion_map_st)).r
        normal_uv = texture_uv(ctx, self.normal_uv_set, self.normal_map_st)
        emission = shader.sample(self.emission_map, texture_uv(ctx, self.emission_uv_set, self.emission_map_st)).rgb
        return shader.SurfaceData(
            albedo=base.rgb * ctx.color * self.base_color.rgb,
            metallic=metallic * self.metallic,
            smoothness=shader.mix(smoothness * self.smoothness, 1.0 - smoothness * (1.0 - self.smoothness), self.smoothness_from_roughness),
            occlusion=shader.mix(1.0, occlusion, self.occlusion_strength) * self.ambient_occlusion,
            normal_ws=sample_normal(self.normal_map, normal_uv, self.normal_uv_set, self.normal_scale, ctx),
            emission=emission * self.emission_color.rgb * self.emission_color.a,
            alpha=base.a * self.base_color.a,
            specular_highlights=self.specular_highlights,
        )
