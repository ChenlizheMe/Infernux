import infernux.shader as shader
from infernux.shader.lib.surface import texture_uv


class Unlit(shader.surface):
    shader_id = "infernux/unlit"
    shading_model = shader.ref("./unlit.shadingmodel.py")
    base_color = shader.material.color()
    albedo_map = shader.material.texture_2d()
    albedo_map_st = shader.material.vec4(default=(1.0, 1.0, 0.0, 0.0))

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        base = shader.sample(self.albedo_map, texture_uv(ctx, 0, self.albedo_map_st))
        return shader.SurfaceData(albedo=base.rgb * ctx.color * self.base_color.rgb,
                                  alpha=base.a * self.base_color.a)
