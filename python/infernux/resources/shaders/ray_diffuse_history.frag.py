import infernux.shader as shader
from infernux.shader.lib.color import luminance
from infernux.shader.lib.ray_sh_filter import sh_clip_history, sh_guide_pixel


class RayDiffuseHistory(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_history"
    hidden = True
    fresh_red = shader.input("fresh_red", shader.Color2D)
    fresh_green = shader.input("fresh_green", shader.Color2D)
    fresh_blue = shader.input("fresh_blue", shader.Color2D)
    fresh_error = shader.input("fresh_error", shader.Color2D)
    old_red = shader.input("old_red", shader.Color2D)
    old_green = shader.input("old_green", shader.Color2D)
    old_blue = shader.input("old_blue", shader.Color2D)
    old_state = shader.input("old_state", shader.History2D)
    primary = shader.input("primary", shader.UInt2D)
    guide = shader.input("guide", shader.UInt2D)
    reprojection = shader.input("reprojection", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.old_state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        source = sh_guide_pixel(cell, extent, full, shader._view_sample_index())
        surface = shader.load(self.primary, source, 0)
        if surface.x == 0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))

        depth = shader.as_float(shader.load(self.guide, source / 2, 0).x)
        state = shader.vec4(0.0, depth, 0.0, 0.0)
        r = shader.load(self.fresh_red, cell, 0)
        g = shader.load(self.fresh_green, cell, 0)
        b = shader.load(self.fresh_blue, cell, 0)
        reproj = shader.load(self.reprojection, source, 0)
        mask = reproj.y
        if not self.old_state.valid or mask == 0:
            return (r, g, b, state)
        motion = shader.unpack_half_2x16(reproj.x)
        previous = (shader.vec2(cell) + 0.5) / shader.vec2(extent) - motion
        if mask != 15:
            raster_motion = motion + shader._view_jitter_delta_uv()
            footprint = shader.frac(shader.vec2(source) - raster_motion * shader.vec2(full))
            correction = shader.vec2(0.0)
            total = 0.0
            for i in range(4):
                if (mask & (shader.uint(1) << i)) == 0:
                    continue
                offset = shader.vec2(i & 1, i >> 1) - footprint
                area = 1.0 - shader.abs(offset)
                weight = area.x * area.y
                correction += offset * weight
                total += weight
            previous += correction / (shader.max(total, 1.0e-5) * shader.vec2(full))

        at = shader.clamp(shader.ivec2(shader.floor(previous * shader.vec2(extent))),
                          shader.ivec2(0), extent - 1)
        old = shader.load(self.old_state, at, 0).xy
        error = shader.abs(old.y - depth)
        if error > 0.03 * depth:
            for i in range(9):
                if i == 4:
                    continue
                neighbor = at + shader.ivec2(i % 3 - 1, int(i / 3) - 1)
                if shader.any(neighbor < shader.ivec2(0)) or shader.any(neighbor >= extent):
                    continue
                nearby = shader.load(self.old_state, neighbor, 0).xy
                delta = shader.abs(nearby.y - depth)
                if delta < error:
                    at = neighbor
                    old = nearby
                    error = delta
                    previous = (shader.vec2(at) + 0.5) / shader.vec2(extent)

        age = shader.min(48.0, 1.0 + old.x)
        sample = shader.clamp(previous, 0.5 / shader.vec2(extent), 1.0 - 0.5 / shader.vec2(extent))
        old_r = shader.sample_level(self.old_red, sample, 0.0)
        old_g = shader.sample_level(self.old_green, sample, 0.0)
        old_b = shader.sample_level(self.old_blue, sample, 0.0)
        fresh_luma = luminance(shader.vec3(r.x, g.x, b.x))
        old_luma = luminance(shader.vec3(old_r.x, old_g.x, old_b.x))
        uncertainty = shader.load(self.fresh_error, cell, 0).x
        level = shader.max(1e-4, shader.max(fresh_luma, old_luma))
        lower = 1.5 * uncertainty + 0.05 * level
        response = shader.smoothstep(lower, lower + 0.2 * level, shader.abs(fresh_luma - old_luma))
        age = shader.mix(age, shader.min(age, 2.0), response)
        r = shader.lerp(old_r, r, 1.0 / age)
        g = shader.lerp(old_g, g, 1.0 / age)
        b = shader.lerp(old_b, b, 1.0 / age)
        return (sh_clip_history(r), sh_clip_history(g), sh_clip_history(b), shader.vec4(age, depth, 0.0, 0.0))
