import infernux.shader as shader


class Unlit(shader.shadingmodel):
    shader_id = "infernux/shading/unlit"
    hidden = True
    lighting_contract = "unlit"

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return surface.albedo + surface.emission
