import infernux.shader as shader


class ChromaticAberration(shader.fullscreen):
    shader_id = "infernux/post/chromatic_aberration"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    intensity = shader.parameter.float(default=0.1, range=(0.0, 1.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        center = uv - 0.5
        offset = center * shader.length(center) * self.intensity * 0.02
        source = shader.sample_level(self.source_tex, uv, 0.0)
        red = shader.sample_level(self.source_tex, uv - offset, 0.0).r
        blue = shader.sample_level(self.source_tex, uv + offset, 0.0).b
        return shader.vec4(red, source.g, blue, source.a)
