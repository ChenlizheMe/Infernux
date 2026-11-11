import infernux.shader as shader


class BloomPrefilter(shader.fullscreen):
    _SourceTex = shader.input("source", shader.Color2D, domain="scene_linear_hdr")
    threshold = shader.parameter.float(default=1.0)
    knee = shader.parameter.float(default=0.5)
    clamp_max = shader.parameter.float(default=65472.0)

    def main(self, uv: shader.vec2) -> shader.vec4:
        color = shader.min(shader.sample(self._SourceTex, uv).rgb, shader.vec3(self.clamp_max))
        brightness = shader.max(color.r, shader.max(color.g, color.b))
        soft_knee = self.threshold * self.knee
        soft = shader.clamp(brightness - self.threshold + soft_knee, 0.0, 2.0 * soft_knee)
        soft = soft * soft / (4.0 * soft_knee + 1e-5)
        weight = shader.max(soft, brightness - self.threshold) / shader.max(brightness, 1e-5)
        return shader.vec4(color * shader.max(weight, 0.0), 1.0)
