import infernux.shader as shader


class FilmGrain(shader.fullscreen):
    shader_id = "infernux/post/film_grain"
    hidden = True
    source_tex = shader.input("color", shader.Color2D, domain="display_linear_ldr")
    intensity = shader.parameter.float(default=0.2, range=(0.0, 1.0))
    response = shader.parameter.float(default=0.8, range=(0.0, 1.0))
    size = shader.parameter.float(default=2.4, range=(0.5, 4.0))
    colored = shader.parameter.bool(default=False)

    def grain_sample(self, cell: shader.uvec2, seed: shader.uint) -> float:
        bits = cell.x * shader.u32(1973) ^ cell.y * shader.u32(9277) ^ seed * shader.u32(26699)
        bits ^= bits >> shader.u32(16)
        bits *= shader.u32(0x7FEB352D)
        bits ^= bits >> shader.u32(15)
        bits *= shader.u32(0x846CA68B)
        bits ^= bits >> shader.u32(16)
        return (float(bits & shader.u32(65535)) + float(bits >> shader.u32(16))) / 65535.0 - 1.0

    def soft_grain(self, position: shader.vec2, seed: shader.uint) -> float:
        cell = shader.uvec2(shader.ivec2(shader.floor(position)))
        fraction = shader.frac(position)
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        bottom = shader.lerp(self.grain_sample(cell, seed),
                             self.grain_sample(cell + shader.uvec2(1, 0), seed), fraction.x)
        top = shader.lerp(self.grain_sample(cell + shader.uvec2(0, 1), seed),
                          self.grain_sample(cell + shader.uvec2(1, 1), seed), fraction.x)
        return shader.lerp(bottom, top, fraction.y)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        source = shader.load(self.source_tex, pixel, 0)
        if self.intensity <= 0.0:
            return source
        plate = (shader.vec2(pixel) + 0.5) / shader.max(self.size, 0.5)
        cadence = shader.floor(shader.frame_time() * 24.0)
        frame = shader.u32(shader.frac(cadence / 65536.0) * 65536.0)
        seed = frame * shader.u32(5)
        rotated = shader.vec2(0.8 * plate.x - 0.6 * plate.y, 0.6 * plate.x + 0.8 * plate.y)
        density = 0.74 * self.soft_grain(rotated, seed)
        density += 0.26 * self.soft_grain(plate * 1.8 + shader.vec2(71.0, 13.0), seed + shader.u32(1))
        grain = shader.vec3(density)
        if self.colored:
            grain = shader.lerp(grain, shader.vec3(self.soft_grain(plate, seed + shader.u32(2)),
                                                  self.soft_grain(plate, seed + shader.u32(3)),
                                                  self.soft_grain(plate, seed + shader.u32(4))), 0.30)
        luma = shader.saturate(shader.dot(source.rgb, shader.vec3(0.2126, 0.7152, 0.0722)))
        amplitude = self.intensity * shader.lerp(1.0, 1.0 - shader.sqrt(luma), self.response)
        return shader.vec4(shader.max(source.rgb * (1.0 + grain * amplitude), shader.vec3(0.0)), source.a)
