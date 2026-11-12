import infernux.shader as shader


class BloomComposite(shader.fullscreen):
    bloom_tex = shader.input("bloom", shader.Color2D, domain="scene_linear_hdr")
    scene_color = shader.input("scene", shader.Color2D, domain="scene_linear_hdr")
    intensity = shader.parameter.float(default=0.8)
    tint = shader.parameter.color()

    def main(self, pixel: shader.fragment_input) -> shader.vec4:
        bloom = shader.sample(self.bloom_tex, pixel.uv).rgb
        scene = shader.load(self.scene_color, shader.ivec2(pixel.position.xy), 0)
        return shader.vec4(scene.rgb + bloom * self.tint.rgb * self.intensity, scene.a)
