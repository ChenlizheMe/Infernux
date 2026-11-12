import infernux.shader as shader


class BloomUpsample(shader.fullscreen):
    """Nine-tap tent reconstruction blended with the higher-resolution level."""

    source_tex = shader.input("low", shader.Color2D, domain="scene_linear_hdr")
    destination_tex = shader.input("high", shader.Color2D, domain="scene_linear_hdr")
    scatter = shader.parameter.float(default=0.7)

    def main(self, uv: shader.vec2) -> shader.vec4:
        texel = 1.0 / shader.vec2(shader.dimensions(self.source_tex, 0))
        color = shader.sample(self.source_tex, uv + texel * shader.vec2(-1.0, -1.0)).rgb
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(0.0, -1.0)).rgb * 2.0
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(1.0, -1.0)).rgb
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(-1.0, 0.0)).rgb * 2.0
        color += shader.sample(self.source_tex, uv).rgb * 4.0
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(1.0, 0.0)).rgb * 2.0
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(-1.0, 1.0)).rgb
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(0.0, 1.0)).rgb * 2.0
        color += shader.sample(self.source_tex, uv + texel * shader.vec2(1.0, 1.0)).rgb
        high = shader.sample(self.destination_tex, uv).rgb
        return shader.vec4(shader.lerp(high, color * 0.0625, self.scatter), 1.0)
