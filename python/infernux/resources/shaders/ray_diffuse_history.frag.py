import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.ray_sh_filter import sh_guide_pixel
from infernux.shader.lib.ray_surface_depth import previous_depth_plane, ray_previous_eye_depth
from infernux.shader.lib.ray_moments import sh_moment
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseHistory(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_history"
    hidden = True
    fresh_red = shader.input("fresh_red", shader.Color2D)
    fresh_green = shader.input("fresh_green", shader.Color2D)
    fresh_blue = shader.input("fresh_blue", shader.Color2D)
    fresh_state = shader.input("fresh_state", shader.Color2D)
    old_red = shader.input("old_red", shader.Color2D)
    old_green = shader.input("old_green", shader.Color2D)
    old_blue = shader.input("old_blue", shader.Color2D)
    old_state = shader.input("old_state", shader.Color2D)
    primary = shader.input("primary", shader.UInt2D)
    previous_primary = shader.input("previous_primary", shader.UInt2D)
    motion = shader.input("motion", shader.UInt2D)
    geometry = shader.input("world_ray_geometry", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.old_state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        index = shader._view_sample_index()
        source = sh_guide_pixel(cell, extent, full, index)
        p = shader.load(self.primary, source, 0)
        if p.x == 0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        n = decode_octahedral_normal(shader.unpack_half_2x16(p.z))
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        fresh = shader.load(self.fresh_state, cell, 0).xy
        if fresh.x == 0.0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        r = shader.load(self.fresh_red, cell, 0)
        g = shader.load(self.fresh_green, cell, 0)
        b = shader.load(self.fresh_blue, cell, 0)
        basis = shader.vec4(0.0, n)
        initial = shader.vec3(shader.dot(sh_moment(r), basis),
                              shader.dot(sh_moment(g), basis),
                              shader.dot(sh_moment(b), basis))
        error = fresh.y * fresh.y
        cam = shader.camera()
        age = 1.0
        motion = shader.load(self.motion, source, 0)
        if motion.x == p.x and (motion.w >> 31) != 0:
            expected = ray_previous_eye_depth(motion.w)
            prev_raster = uv - shader.unpack_half_2x16(motion.z)
            prev_uv = prev_raster + shader._view_jitter_delta_uv()
            plane = previous_depth_plane(n, prev_raster, expected, cam)
            point = shader.vec2(cell) + (prev_uv - uv) * shader.vec2(extent)
            base = shader.ivec2(shader.floor(point))
            hr = shader.vec4(0.0)
            hg = shader.vec4(0.0)
            hb = shader.vec4(0.0)
            old_total = 0.0
            old_age = 0.0
            strongest = 0.0
            for i in range(4):
                if plane.w == 0.0:
                    break
                at = base + shader.ivec2(i & 1, i >> 1)
                if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                    continue
                previous_site = sh_guide_pixel(at, extent, full, index - shader.uint(1))
                q = shader.load(self.previous_primary, previous_site, 0)
                stats = shader.load(self.old_state, at, 0).xy
                if q.x != p.x or stats.x == 0.0:
                    continue
                qn = decode_octahedral_normal(shader.unpack_half_2x16(q.z))
                zd = linear_eye_depth(shader.as_float(q.y), cam)
                sample_uv = (shader.vec2(previous_site) + shader.vec2(0.5)) / shader.vec2(full)
                device = plane.z - shader.dot(plane.xy, (sample_uv - prev_raster) * shader.vec2(2.0, -2.0))
                predicted = linear_eye_depth(device, cam)
                if predicted <= 0.0 or shader.abs(predicted - zd) > 0.003 + 0.008 * zd or shader.dot(qn, n) < 0.94:
                    continue
                bary = shader.saturate(shader.vec2(1.0) - shader.abs(shader.vec2(at) - point))
                w = bary.x * bary.y
                hr += shader.load(self.old_red, at, 0) * w
                hg += shader.load(self.old_green, at, 0) * w
                hb += shader.load(self.old_blue, at, 0) * w
                if w > strongest:
                    old_age = stats.x
                    strongest = w
                old_total += w
            if old_total <= 0.0 and plane.w != 0.0:
                nearest = shader.ivec2(shader.round(point))
                best = 0.03 * shader.max(expected, 1.0e-6)
                for i in range(9):
                    at = nearest + shader.ivec2(i % 3 - 1, int(i / 3) - 1)
                    if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                        continue
                    previous_site = sh_guide_pixel(at, extent, full, index - shader.uint(1))
                    previous = shader.load(self.previous_primary, previous_site, 0)
                    nearby = shader.load(self.old_state, at, 0).xy
                    if previous.x != p.x or nearby.x == 0.0:
                        continue
                    previous_uv = (shader.vec2(previous_site) + 0.5) / shader.vec2(full)
                    device = plane.z - shader.dot(plane.xy, (previous_uv - prev_raster) * shader.vec2(2.0, -2.0))
                    delta = shader.abs(linear_eye_depth(device, cam) - linear_eye_depth(shader.as_float(previous.y), cam))
                    if delta < best:
                        hr = shader.load(self.old_red, at, 0)
                        hg = shader.load(self.old_green, at, 0)
                        hb = shader.load(self.old_blue, at, 0)
                        old_age = nearby.x
                        old_total = 1.0
                        best = delta
            if old_total > 0.0:
                changed = self.geometry[1].y != shader._previous_ray_geometry_serial() or shader._ray_lighting_changed()
                travel = shader.abs(prev_uv - uv) * shader.vec2(full)
                moving = shader.max(travel.x, travel.y) > 0.25
                retention = shader.exp2(-shader.max(shader.frame_delta_time(), 0.0) / 0.023)
                budget = shader.min(1.0, retention / shader.max(1.0 - retention, 1.0 / 64.0)) if changed or moving else 47.0
                keep = shader.min(old_age, budget) * shader.saturate(old_total)
                old = shader.vec3(shader.dot(sh_moment(hr), basis), shader.dot(sh_moment(hg), basis), shader.dot(sh_moment(hb), basis)) / old_total
                difference = initial - old
                if changed or moving:
                    mismatch = shader.dot(difference, difference)
                    keep *= shader.min(1.0, (4.0 * error + 1e-9) / (mismatch + 1e-9))
                age += keep
                r = (r + hr * (keep / old_total)) / age
                g = (g + hg * (keep / old_total)) / age
                b = (b + hb * (keep / old_total)) / age
        return (r, g, b, shader.vec4(shader.round(age), shader.as_float(p.y), 0.0, 0.0))
