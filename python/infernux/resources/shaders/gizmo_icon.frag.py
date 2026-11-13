import infernux.shader as shader


class GizmoIcon(shader.surface):
    shader_id = "infernux/editor/gizmo_icon"
    hidden = True
    shading_model = shader.ref("./unlit.shadingmodel.py")
    surface_type = "transparent"
    cull_mode = "none"
    cast_shadows = False
    receive_shadows = False
    base_color = shader.material.color()
    albedo_map = shader.material.texture_2d()

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        sampled = shader.sample_bias(self.albedo_map, ctx.uv0, -0.75)
        return shader.SurfaceData(albedo=sampled.rgb * ctx.color * self.base_color.rgb,
                                  alpha=sampled.a * self.base_color.a)
