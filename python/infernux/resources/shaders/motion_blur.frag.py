import infernux.shader as shader


class MotionBlur(shader.fullscreen):
    shader_id = "infernux/post/motion_blur"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    motion_tex = shader.input("motion", shader.Color2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    intensity = shader.parameter.float(default=1.0, range=(0.0, 2.0))
    max_blur_pixels = shader.parameter.float(default=32.0, range=(0.0, 128.0))
    depth_rejection = shader.parameter.float(default=1.0, range=(0.0, 8.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        center = shader.sample_level(self.source_tex, uv, 0.0)
        extent = shader.vec2(shader.dimensions(self.source_tex, 0))
        motion = shader.sample_level(self.motion_tex, uv, 0.0).xy * self.intensity
        pixels = shader.length(motion * extent)
        if pixels < 0.25 or self.max_blur_pixels <= 0.0 or self.intensity <= 0.0:
            return center
        blur = motion * (shader.min(pixels, self.max_blur_pixels) / shader.max(pixels, 1e-5))
        center_depth = shader.sample_level(self.depth_tex, uv, 0.0)
        accumulated = center
        total_weight = 1.0
        for tap in range(12):
            offset = (shader.f32(tap) + 0.5) / 12.0 - 0.5
            sample_uv = shader.clamp(uv - blur * offset, shader.vec2(0.0), shader.vec2(1.0))
            sample_depth = shader.sample_level(self.depth_tex, sample_uv, 0.0)
            delta = shader.abs(sample_depth - center_depth)
            scale = shader.max(shader.max(shader.abs(center_depth), shader.abs(sample_depth)), 1e-3)
            depth_weight = shader.exp2(-delta * self.depth_rejection * 256.0 / scale)
            weight = shader.max(1.0 - shader.abs(offset) * 1.4, 0.05) * depth_weight
            accumulated += shader.sample_level(self.source_tex, sample_uv, 0.0) * weight
            total_weight += weight
        return accumulated / shader.max(total_weight, 1e-5)
