import infernux.shader as shader
from .lib.color import luminance, rgb_to_hsv, hsv_to_rgb


class ColorAdjustments(shader.fullscreen):
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    post_exposure = shader.parameter.float(default=0.0)
    contrast = shader.parameter.float(default=0.0)
    saturation = shader.parameter.float(default=0.0)
    hue_shift = shader.parameter.float(default=0.0)

    def main(self, uv: shader.vec2) -> shader.vec4:
        source = shader.sample(self.source_tex, uv)
        color = source.rgb * shader.exp2(self.post_exposure)
        logc = 0.244161 * (shader.log2(shader.max(5.555556 * color + 0.047996, 1e-6)) / shader.log2(10.0)) + 0.386036
        logc = (logc - 0.4135884) * (self.contrast * 0.01 + 1.0) + 0.4135884
        color = shader.max((shader.exp2((logc - 0.386036) / 0.244161 * shader.log2(10.0)) - 0.047996) / 5.555556, 0.0)
        color = shader.max(shader.lerp(shader.vec3(luminance(color)), color, self.saturation * 0.01 + 1.0), 0.0)
        if shader.abs(self.hue_shift) > 0.5:
            hsv = rgb_to_hsv(color)
            hsv.x = shader.frac(hsv.x + self.hue_shift / 360.0)
            color = hsv_to_rgb(hsv)
        return shader.vec4(color, source.a)
