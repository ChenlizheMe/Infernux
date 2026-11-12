import infernux.shader as shader


class Pixelation(shader.fullscreen):
    shader_id = "infernux/post/pixelation"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="scene_linear_hdr")
    pixel_size = shader.parameter.int(default=8)
    pixel_aspect = shader.parameter.float(default=1.0)
    grid_offset = shader.parameter.vec2(default=(0.0, 0.0))
    intensity = shader.parameter.float(default=1.0)
    sampling_mode = shader.parameter.int(default=1)
    preserve_alpha_coverage = shader.parameter.bool(default=True)

    def sample_pixel(self, pixel: shader.vec2, size: shader.vec2) -> shader.vec4:
        uv = shader.clamp(pixel, shader.vec2(0.5), size - 0.5) / size
        return shader.sample_level(self.source_tex, uv, 0.0)

    def main(self, uv: shader.vec2) -> shader.vec4:
        size = shader.vec2(shader.dimensions(self.source_tex, 0))
        height = shader.clamp(shader.f32(self.pixel_size), 1.0, 256.0)
        cell = shader.vec2(shader.max(1.0, shader.round(height * shader.clamp(self.pixel_aspect, 0.25, 4.0))), height)
        offset = self.grid_offset * cell
        pixel = uv * size
        origin = shader.floor((pixel - offset) / cell) * cell + offset
        amount = shader.saturate(self.intensity)
        center = shader.lerp(pixel, origin + cell * 0.5, amount)
        if self.sampling_mode == 0:
            return self.sample_pixel(center, size)
        width = 2 if self.sampling_mode == 1 else 3
        color = shader.vec3(0.0)
        alpha = 0.0
        maximum_alpha = 0.0
        for y in range(width):
            for x in range(width):
                offset = ((shader.vec2(x, y) + 0.5) / shader.f32(width) - 0.5) * cell * amount
                value = self.sample_pixel(center + offset, size)
                color += value.rgb * value.a
                alpha += value.a
                maximum_alpha = shader.max(maximum_alpha, value.a)
        result_alpha = maximum_alpha if self.preserve_alpha_coverage else alpha / shader.f32(width * width)
        result_color = color / alpha if alpha > 1e-6 else shader.vec3(0.0)
        return shader.vec4(result_color, result_alpha)
