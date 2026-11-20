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
        extent = shader.ivec2(shader.dimensions(self.source_tex, 0))
        source_size = shader.vec2(extent)
        source_position = ctx.uv * source_size
        pixel = shader.ivec2(source_position)
        source_to_output = source_size * shader.fwidth(ctx.uv)
        # The projection displacement moves geometry across the raster.
        # The corresponding sample lies on the opposite side of its pixel center.
        sample_offset = ((shader.vec2(pixel) + 0.5 - source_position) / source_to_output
                         - shader._view_jitter_pixels())
        near_scale = -10.0 * source_to_output.x * shader.min(4.0 * source_to_output.x, 1.0)
        wide_scale = -source_to_output.x * source_to_output.x
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
                position = shader.vec2(offset) / source_to_output + sample_offset
                radius_sq = shader.dot(position, position)
                near = shader.exp2(near_scale * radius_sq)
                wide = shader.exp2(wide_scale * radius_sq)
                current += value * near
                current_weight += near
                moment += value * wide
                moment_sq += value * value * wide
                moment_weight += wide
        current /= shader.max(1.0e-5, current_weight)
        if not self.history_tex.valid or self.feedback <= 0.0:
            return taa_ycbcr_to_color(current)

        motion_size = shader.vec2(shader.dimensions(self.motion_tex, 0))
        motion_pixel = shader.ivec2(ctx.uv * motion_size)
        previous_uv = ctx.uv - shader.load(self.motion_tex, motion_pixel, 0).xy
        if shader.any(previous_uv < 0.0) or shader.any(previous_uv >= 1.0):
            return taa_ycbcr_to_color(current)
        history_size = shader.vec2(shader.dimensions(self.history_tex, 0))
        history = taa_color_to_ycbcr(taa_history_sample(self.history_tex, previous_uv, history_size))
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
