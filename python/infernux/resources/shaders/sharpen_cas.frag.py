import infernux.shader as shader


class SharpenCas(shader.fullscreen):
    shader_id = "infernux/post/sharpen_cas"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="display_linear_ldr")
    intensity = shader.parameter.float(default=0.5)

    def main(self, uv: shader.vec2) -> shader.vec4:
        texel = 1.0 / shader.vec2(shader.dimensions(self.source_tex, 0))
        source = shader.sample(self.source_tex, uv)
        top = shader.sample(self.source_tex, uv - shader.vec2(0.0, texel.y)).rgb
        bottom = shader.sample(self.source_tex, uv + shader.vec2(0.0, texel.y)).rgb
        left = shader.sample(self.source_tex, uv - shader.vec2(texel.x, 0.0)).rgb
        right = shader.sample(self.source_tex, uv + shader.vec2(texel.x, 0.0)).rgb
        minimum = shader.min(source.rgb, shader.min(shader.min(top, bottom), shader.min(left, right)))
        maximum = shader.max(source.rgb, shader.max(shader.max(top, bottom), shader.max(left, right)))
        weight = shader.saturate(shader.min(minimum, 1.0 - maximum) / (maximum - minimum + 0.001))
        average = (top + bottom + left + right) * 0.25
        color = source.rgb + (source.rgb - average) * weight * weight * self.intensity * 2.0
        return shader.vec4(shader.saturate(color), source.a)
