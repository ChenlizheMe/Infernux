import infernux.shader as shader


class PBR(shader.shadingmodel):
    lighting_contract = "standard_pbr"

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return shader.lighting.pbr(surface, ctx)
