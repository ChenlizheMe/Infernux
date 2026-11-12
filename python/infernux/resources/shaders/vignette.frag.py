import infernux.shader as shader
from .lib.color import srgb_to_linear


class Vignette(shader.fullscreen):
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    intensity = shader.parameter.float(default=0.35)
    smoothness = shader.parameter.float(default=0.3)
    roundness = shader.parameter.float(default=1.0)
    rounded = shader.parameter.bool(default=False)
    color = shader.parameter.color(default=(0.0, 0.0, 0.0, 1.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        source = shader.sample(self.source_tex, uv)
        size = shader.vec2(shader.dimensions(self.source_tex, 0))
        distance = shader.abs(uv - 0.5) * (self.intensity * 3.0)
        if self.rounded:
            distance.x *= size.x / size.y
        exponent = (1.0 - self.roundness) * 6.0 + self.roundness
        distance = shader.pow(shader.saturate(distance), exponent)
        factor = shader.pow(shader.saturate(1.0 - shader.dot(distance, distance)), self.smoothness * 5.0)
        color = shader.lerp(srgb_to_linear(self.color.rgb), shader.vec3(1.0), factor)
        return shader.vec4(source.rgb * color, source.a)
