import infernux.shader as shader
from infernux.shader.lib.color import linear_to_srgb


class DisplayEncode(shader.fullscreen):
    shader_id = "infernux/internal/display_encode"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")

    def noise(self, pixel: shader.vec2) -> float:
        return shader.frac(52.9829189 * shader.frac(shader.dot(pixel, shader.vec2(0.06711056, 0.00583715))))

    def triangular_dither(self, pixel: shader.vec2, offset: shader.vec2) -> float:
        return self.noise(pixel + offset) - self.noise(pixel.yx + offset.yx + shader.vec2(37.0, 17.0))

    def main(self, fragment: shader.fragment_input) -> shader.vec4:
        view = shader.view()
        source = shader.load(self.source_tex, shader.ivec2(fragment.position.xy), 0)
        if view.stop_nans and (shader.any(shader.isnan(source)) or shader.any(shader.isinf(source))):
            source = shader.vec4(0.0, 0.0, 0.0, 1.0)
        encoded = linear_to_srgb(shader.clamp(source.rgb, 0.0, 1.0))
        if view.dithering:
            pixel = shader.floor(fragment.position.xy)
            noise = shader.vec3(self.triangular_dither(pixel, shader.vec2(0.0, 0.0)),
                                self.triangular_dither(pixel, shader.vec2(19.0, 47.0)),
                                self.triangular_dither(pixel, shader.vec2(73.0, 11.0)))
            encoded = shader.clamp(encoded + noise / 255.0, 0.0, 1.0)
        return shader.vec4(encoded, 1.0 if view.opaque_output else source.a)
