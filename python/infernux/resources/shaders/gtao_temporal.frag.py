import infernux.shader as shader
from infernux.shader.lib.ray_surface_depth import diffuse_previous_uv_delta


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

        size = shader.ivec2(shader.dimensions(self.visibility, 0))
        mask = shader.uint(15)
        if self.use_reprojection:
            reprojection = shader.load(self.reprojection, pixel, 0)
            mask = reprojection.y & shader.uint(15)
            if mask == 0:
                return shader.vec4(center)
            # Match the previous raster footprint used to generate the coverage bits.
            previous_uv = (shader.vec2(pixel) + 0.5) / shader.vec2(size)
            previous_uv = previous_uv + diffuse_previous_uv_delta(reprojection.x) - shader._view_jitter_delta_uv()
        else:
            previous_uv = ctx.uv - shader.load(self.motion, pixel, 0).xy
        if shader.any(previous_uv < shader.vec2(0.0)) or shader.any(previous_uv >= shader.vec2(1.0)):
            return shader.vec4(center)
        if mask == 15:
            previous = shader.sample_level(self.history, previous_uv, 0.0).x
        else:
            position = previous_uv * shader.vec2(size) - 0.5
            origin = shader.ivec2(shader.floor(position))
            fraction = shader.frac(position)
            weighted = 0.0
            coverage = 0.0
            for i in range(4):
                if (mask & (shader.uint(1) << i)) == 0:
                    continue
                offset = shader.ivec2(i & 1, i >> 1)
                weights = 1.0 - shader.abs(fraction - shader.vec2(offset))
                area = weights.x * weights.y
                weighted += shader.load(self.history, origin + offset, 0).x * area
                coverage += area
            previous = weighted / coverage if coverage > 0.0 else center
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
