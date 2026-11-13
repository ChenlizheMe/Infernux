import infernux.shader as shader


class TemporalAA(shader.fullscreen):
    shader_id = "infernux/post/temporal_aa"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    history_tex = shader.input("history", shader.History2D, domain="scene_linear_hdr")
    motion_tex = shader.input("motion", shader.Color2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    feedback = shader.parameter.float(default=0.9, range=(0.0, 0.98))
    motion_rejection = shader.parameter.float(default=0.08, range=(0.0, 1.0))
    depth_rejection = shader.parameter.float(default=96.0, range=(0.0, 512.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        current = shader.sample_level(self.source_tex, uv, 0.0)
        if not self.history_tex.valid or self.feedback <= 0.0:
            return current
        extent = shader.vec2(shader.dimensions(self.source_tex, 0))
        texel = 1.0 / shader.max(extent, shader.vec2(1.0))
        motion = shader.sample_level(self.motion_tex, uv, 0.0).xy
        history_uv = uv - motion
        if shader.any(history_uv < shader.vec2(0.0)) or shader.any(history_uv > shader.vec2(1.0)):
            return current
        neighborhood_min = current.rgb
        neighborhood_max = current.rgb
        center_depth = shader.sample_level(self.depth_tex, uv, 0.0)
        max_depth_delta = 0.0
        for y in range(-1, 2):
            for x in range(-1, 2):
                sample_uv = shader.clamp(uv + shader.vec2(x, y) * texel, shader.vec2(0.0), shader.vec2(1.0))
                sample_color = shader.sample_level(self.source_tex, sample_uv, 0.0).rgb
                neighborhood_min = shader.min(neighborhood_min, sample_color)
                neighborhood_max = shader.max(neighborhood_max, sample_color)
                sample_depth = shader.sample_level(self.depth_tex, sample_uv, 0.0)
                max_depth_delta = shader.max(max_depth_delta, shader.abs(sample_depth - center_depth))
        history = shader.sample_level(self.history_tex, history_uv, 0.0)
        history.rgb = shader.clamp(history.rgb, neighborhood_min, neighborhood_max)
        motion_weight = shader.exp2(-shader.length(motion * extent) * self.motion_rejection)
        depth_weight = shader.exp2(-max_depth_delta * self.depth_rejection)
        history_weight = shader.clamp(self.feedback * motion_weight * depth_weight, 0.0, self.feedback)
        return shader.lerp(current, history, history_weight)
