import infernux.shader as shader


class BloomComposite(shader.fullscreen):
    _BloomTex = shader.input("bloom", shader.Color2D, domain="scene_linear_hdr")
    _SceneColor = shader.input("scene", shader.Color2D, domain="scene_linear_hdr")
    intensity = shader.parameter.float(default=0.8)
    tint = shader.parameter.color()

    def main(self, pixel: shader.fragment_input) -> shader.vec4:
        bloom = shader.sample(self._BloomTex, pixel.uv).rgb
        scene = shader.load(self._SceneColor, shader.ivec2(pixel.position.xy), 0)
        return shader.vec4(scene.rgb + bloom * self.tint.rgb * self.intensity, scene.a)
