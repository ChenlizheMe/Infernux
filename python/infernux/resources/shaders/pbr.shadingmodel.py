import infernux.shader as shader


class PBR(shader.shadingmodel):
    shader_id = "infernux/shading/pbr"
    hidden = True
    lighting_contract = "standard_pbr"

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return shader.lighting.pbr(surface, ctx)
