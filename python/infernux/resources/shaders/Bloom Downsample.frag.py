import infernux.shader as shader


class BloomDownsample(shader.fullscreen):
    """Thirteen bilinear taps preserve energy while reducing bright aliasing."""

    _SourceTex = shader.input("source", shader.Color2D, domain="scene_linear_hdr")

    def main(self, uv: shader.vec2) -> shader.vec4:
        texel = 1.0 / shader.vec2(shader.dimensions(self._SourceTex, 0))
        a = shader.sample(self._SourceTex, uv + texel * shader.vec2(-1.0, -1.0)).rgb
        b = shader.sample(self._SourceTex, uv + texel * shader.vec2(0.0, -1.0)).rgb
        c = shader.sample(self._SourceTex, uv + texel * shader.vec2(1.0, -1.0)).rgb
        d = shader.sample(self._SourceTex, uv + texel * shader.vec2(-1.0, 0.0)).rgb
        e = shader.sample(self._SourceTex, uv).rgb
        f = shader.sample(self._SourceTex, uv + texel * shader.vec2(1.0, 0.0)).rgb
        g = shader.sample(self._SourceTex, uv + texel * shader.vec2(-1.0, 1.0)).rgb
        h = shader.sample(self._SourceTex, uv + texel * shader.vec2(0.0, 1.0)).rgb
        i = shader.sample(self._SourceTex, uv + texel * shader.vec2(1.0, 1.0)).rgb
        j = shader.sample(self._SourceTex, uv + texel * shader.vec2(-0.5, -0.5)).rgb
        k = shader.sample(self._SourceTex, uv + texel * shader.vec2(0.5, -0.5)).rgb
        l = shader.sample(self._SourceTex, uv + texel * shader.vec2(-0.5, 0.5)).rgb
        m = shader.sample(self._SourceTex, uv + texel * shader.vec2(0.5, 0.5)).rgb
        color = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125
        return shader.vec4(color, 1.0)
