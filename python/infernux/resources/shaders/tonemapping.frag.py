import infernux.shader as shader


class ToneMapping(shader.fullscreen):
    shader_id = "infernux/post/tonemapping"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    mode = shader.parameter.int(default=2, range=(0, 2))
    exposure = shader.parameter.float(default=1.0, range=(0.01, 10.0))

    def aces(self, color: shader.vec3) -> shader.vec3:
        # These rows preserve the original linear RGB input/output transforms.
        input_matrix = shader.mat3(
            shader.vec3(0.59719, 0.35458, 0.04823),
            shader.vec3(0.07600, 0.90834, 0.01566),
            shader.vec3(0.02840, 0.13383, 0.83777),
        )
        output_matrix = shader.mat3(
            shader.vec3(1.60475, -0.53108, -0.07367),
            shader.vec3(-0.10208, 1.10813, -0.00605),
            shader.vec3(-0.00327, -0.07276, 1.07602),
        )
        value = shader.mul(input_matrix, color)
        numerator = value * (value + 0.0245786) - 0.000090537
        denominator = value * (0.983729 * value + 0.4329510) + 0.238081
        value = numerator / shader.max(denominator, shader.vec3(1e-6))
        return shader.saturate(shader.mul(output_matrix, value))

    def main(self, uv: shader.vec2) -> shader.vec4:
        source = shader.sample_level(self.source_tex, uv, 0.0)
        hdr = source.rgb * self.exposure
        if self.mode == 1:
            color = hdr / (hdr + 1.0)
        elif self.mode == 2:
            color = self.aces(hdr)
        else:
            color = shader.saturate(hdr)
        # Display encoding is a later pass; coverage alpha stays linear.
        return shader.vec4(color, source.a)
