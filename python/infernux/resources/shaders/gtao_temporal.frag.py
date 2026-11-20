import infernux.shader as shader


class GTAOTemporal(shader.fullscreen):
    shader_id = "infernux/internal/gtao_temporal"
    hidden = True
    visibility = shader.input("visibility", shader.Color2D)
    history = shader.input("history", shader.History2D)
    reprojection = shader.input("reprojection", shader.UInt2D)
    motion = shader.input("motion", shader.Color2D)
    use_reprojection = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        center = shader.load(self.visibility, pixel, 0).x
        if not self.history.valid:
            return shader.vec4(center)

        if self.use_reprojection:
            reprojection = shader.load(self.reprojection, pixel, 0)
            if reprojection.y == 0:
                return shader.vec4(center)
            motion = shader.unpack_half_2x16(reprojection.x)
        else:
            motion = shader.load(self.motion, pixel, 0).xy
        previous_uv = ctx.uv - motion
        if shader.any(previous_uv < 0.0) or shader.any(previous_uv >= 1.0):
            return shader.vec4(center)
        previous = shader.sample_level(self.history, previous_uv, 0.0).x
        size = shader.ivec2(shader.dimensions(self.visibility, 0))
        total = 0.0
        mean = 0.0
        square = 0.0
        for y in range(-1, 2):
            for x in range(-1, 2):
                tap = shader.clamp(pixel + 2 * shader.ivec2(x, y), shader.ivec2(0), size - 1)
                sample_ao = shader.load(self.visibility, tap, 0).x
                weight = shader.exp(-0.75 * shader.float(x * x + y * y))
                total += weight
                mean += sample_ao * weight
                square += sample_ao * sample_ao * weight
        mean /= total
        deviation = 2.5 * shader.sqrt(shader.max(0.0, square / total - mean * mean))
        bound = shader.mix(center, mean, 0.25)
        return shader.vec4(shader.mix(shader.clamp(previous, bound - deviation, bound + deviation), center, 0.125))
