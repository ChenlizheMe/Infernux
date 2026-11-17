import infernux.shader as shader
from infernux.shader.lib.surface import decode_octahedral_normal
from infernux.shader.lib.thin_sheet import thin_sheet_cosine, thin_sheet_fresnel, thin_sheet_attenuation


class ThinComposite(shader.fullscreen):
    shader_id = "infernux/internal/thin_composite"
    hidden = True
    background = shader.input("thin_background", shader.Color2D)
    layer = shader.input("thin_layer", shader.UInt2D)
    optics = shader.input("thin_optics", shader.UInt2D)
    reflection = shader.input("thin_reflection", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        background = shader.load(self.background, pixel, 0)
        layer = shader.load(self.layer, pixel, 0)
        if layer.x == 0:
            return background
        packed = shader.load(self.optics, pixel, 0)
        first = shader.unpack_half_2x16(packed.x)
        second = shader.unpack_half_2x16(packed.y)
        third = shader.unpack_half_2x16(packed.z)
        fourth = shader.unpack_half_2x16(packed.w)
        coverage = shader.saturate(fourth.y)
        if coverage <= 0.0:
            return background
        normal = decode_octahedral_normal(shader.unpack_half_2x16(layer.y))
        cosine = thin_sheet_cosine(ctx.uv, normal, shader.camera())
        fresnel = thin_sheet_fresnel(cosine, first.x)
        reflected = shader.vec3(1.0)
        transmitted = shader.vec3(0.0)
        if fresnel < 1.0:
            attenuation = thin_sheet_attenuation(cosine, first.x, shader.vec3(first.y, second))
            through = 1.0 - fresnel
            denominator = 1.0 - fresnel * fresnel * attenuation * attenuation
            transmitted = through * through * attenuation / denominator
            reflected = fresnel + through * through * fresnel * attenuation * attenuation / denominator
        radiance = shader.load(self.reflection, pixel, 0)
        emission = shader.vec3(third, fourth.x)
        color = reflected * radiance.rgb + transmitted * background.rgb + emission
        return shader.vec4(shader.mix(background.rgb, color, coverage), background.a)
