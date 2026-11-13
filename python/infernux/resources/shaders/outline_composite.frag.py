import infernux.shader as shader


class OutlineComposite(shader.fullscreen):
    shader_id = "infernux/editor/outline_composite"
    hidden = True
    mask = shader.input("mask", shader.Color2D)
    outline_color = shader.parameter.color(default=(1.0, 0.214, 0.0, 1.0))
    texel_size = shader.parameter.vec2()
    outline_width = shader.parameter.float(default=3.0, range=(1.0, 10.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        if shader.sample_level(self.mask, uv, 0.0).r > 0.5:
            return shader.vec4(0.0)
        radius = shader.clamp(int(self.outline_width + 0.5), 1, 10)
        radius_sq = float(radius * radius)
        nearest_sq = 999.0
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                distance_sq = float(x * x + y * y)
                if distance_sq <= radius_sq:
                    offset = shader.vec2(float(x), float(y)) * self.texel_size
                    if shader.sample_level(self.mask, uv + offset, 0.0).r > 0.5:
                        nearest_sq = shader.min(nearest_sq, distance_sq)
        if nearest_sq > radius_sq:
            return shader.vec4(0.0)
        alpha = 1.0 - shader.smoothstep(self.outline_width - 1.0, self.outline_width, shader.sqrt(nearest_sq))
        return shader.vec4(self.outline_color.rgb, alpha * self.outline_color.a)
