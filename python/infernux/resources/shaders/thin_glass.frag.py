import infernux.shader as shader


class ThinGlass(shader.surface):
    shader_id = "infernux/thin_glass"
    surface_type = "transparent"
    cull_mode = "none"
    cast_shadows = False
    ior = shader.material.float(default=1.5, range=(1.0, 3.0))
    thickness = shader.material.float(default=0.01, range=(0.0, 1.0))
    absorption = shader.material.vec3(default=(0.0, 0.0, 0.0))
    coverage = shader.material.float(default=1.0, range=(0.0, 1.0))
    emission_color = shader.material.color(default=(0.0, 0.0, 0.0, 1.0), hdr=True)

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(
            albedo=shader.vec3(0.0), normal_ws=ctx.normal_ws, smoothness=1.0,
            emission=self.emission_color.rgb, alpha=self.coverage,
        )

    def thin_sheet(self, ctx: shader.SurfaceContext) -> shader.ThinSheetData:
        return shader.ThinSheetData(ior=self.ior, thickness=self.thickness, absorption=self.absorption)
