import infernux.shader as shader
from infernux.shader.lib.temporal_aa import taa_color_to_ycbcr, taa_history_sample, taa_ycbcr_to_color


class TemporalAA(shader.fullscreen):
    shader_id = "infernux/post/temporal_aa"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    history_tex = shader.input("history", shader.History2D, domain="scene_linear_hdr")
    motion_tex = shader.input("motion", shader.Color2D)
    feedback = shader.parameter.float(default=0.875, range=(0.0, 0.98))

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.source_tex, 0))
        size = shader.vec2(extent)
        # The projection displacement moves geometry across the raster.
        # The corresponding sample lies on the opposite side of its pixel center.
        sample_offset = -shader._view_jitter_pixels()
        current = shader.vec4(0.0)
        current_weight = 0.0
        moment = shader.vec4(0.0)
        moment_sq = shader.vec4(0.0)
        moment_weight = 0.0
        for y in range(-1, 2):
            for x in range(-1, 2):
                offset = shader.ivec2(x, y)
                tap = shader.clamp(pixel + offset, shader.ivec2(0), extent - 1)
                value = taa_color_to_ycbcr(shader.load(self.source_tex, tap, 0))
                position = shader.vec2(offset) + sample_offset
                radius_sq = shader.dot(position, position)
                near = shader.exp2(-10.0 * radius_sq)
                wide = shader.exp2(-radius_sq)
                current += value * near
                current_weight += near
                moment += value * wide
                moment_sq += value * value * wide
                moment_weight += wide
        current /= current_weight
        if not self.history_tex.valid or self.feedback <= 0.0:
            return taa_ycbcr_to_color(current)

        previous_uv = ctx.uv - shader.load(self.motion_tex, pixel, 0).xy
        if shader.any(previous_uv < 0.0) or shader.any(previous_uv >= 1.0):
            return taa_ycbcr_to_color(current)
        history = taa_color_to_ycbcr(taa_history_sample(self.history_tex, previous_uv, size))
        mean = moment / moment_weight
        deviation = shader.sqrt(shader.max(moment_sq / moment_weight - mean * mean, 0.0))
        variation = deviation.x / (mean.x + 1e-5)
        scale = shader.mix(0.5, 1.0, shader.smoothstep(-0.1, 0.3, variation))
        lo = mean - deviation * scale
        hi = mean + deviation * scale
        delta = shader.min(shader.abs(history.x - lo.x), shader.abs(history.x - hi.x))
        distance = delta / shader.max(1e-5, shader.max(history.x, mean.x))
        weight = (1.0 - self.feedback) * shader.mix(0.2, 1.0, shader.smoothstep(0.0, 2.0, distance))
        return taa_ycbcr_to_color(shader.mix(shader.clamp(history, lo, hi), current, weight))
