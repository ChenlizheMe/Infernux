import infernux.shader as shader
from .lib.color import white_balance


class WhiteBalance(shader.fullscreen):
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    temperature = shader.parameter.float(default=0.0)
    tint = shader.parameter.float(default=0.0)

    def main(self, uv: shader.vec2) -> shader.vec4:
        source = shader.sample(self.source_tex, uv)
        return shader.vec4(white_balance(source.rgb, self.temperature, self.tint), source.a)
