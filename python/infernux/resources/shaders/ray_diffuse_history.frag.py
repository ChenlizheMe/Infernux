import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.ray_surface_depth import previous_depth_plane, ray_previous_eye_depth
from infernux.shader.lib.ray_moments import unpack_moment, pack_direction
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseHistory(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_history"
    hidden = True
    fresh_dc = shader.input("fresh_dc", shader.UInt2D)
    fresh_axes = shader.input("fresh_axes", shader.UInt2D)
    old_dc = shader.input("old_dc", shader.UInt2D)
    old_axes = shader.input("old_axes", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    previous_primary = shader.input("previous_primary", shader.UInt2D)
    motion = shader.input("motion", shader.UInt2D)
    geometry = shader.input("world_ray_geometry", shader.Buffer[shader.uvec4])
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.old_dc, 0))
        observed = shader.ivec2(shader.dimensions(self.fresh_dc, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        source = ray_grid_pixel(cell, extent, full)
        p = shader.load(self.primary, source, 0)
        if p.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        n = decode_octahedral_normal(shader.unpack_half_2x16(p.z))
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        anchor = ray_grid_pixel(source, full, observed)
        center = shader.vec2(anchor) + shader.vec2(source - ray_grid_pixel(anchor, observed, full)) * shader.vec2(observed) / shader.vec2(full)
        cam = shader.camera()
        pos = world_position(uv, shader.as_float(p.y), cam)
        band = 0.008 + 0.003 * linear_eye_depth(shader.as_float(p.y), cam)
        projection_w = (cam.view_projection_matrix @ shader.vec4(pos, 1.0)).w
        focal = shader.abs(shader.vec2(cam.projection_matrix[0][0], cam.projection_matrix[1][1]))
        pitch = 2.0 * projection_w / (focal * shader.vec2(observed))
        r = shader.vec4(0.0)
        g = shader.vec4(0.0)
        b = shader.vec4(0.0)
        total = 0.0
        second_moment = 0.0
        squared_weights = 0.0
        basis = shader.vec4(0.0, n)
        visibility_extent = shader.ivec2(shader.dimensions(self.ambient_visibility, 0))
        visibility = 1.0
        if self.ssao_strength > 0.0:
            visibility = shader.load(self.ambient_visibility, shader.ivec2(uv * shader.vec2(visibility_extent)), 0).x
        for i in range(25):
            d = shader.ivec2(i % 5 - 2, int(i / 5) - 2)
            at = anchor + d
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= observed):
                continue
            site = ray_grid_pixel(at, observed, full)
            q = shader.load(self.primary, site, 0)
            dc = shader.load(self.fresh_dc, at, 0)
            if q.x == 0 or dc.w == 0:
                continue
            qn = decode_octahedral_normal(shader.unpack_half_2x16(q.z))
            qpos = world_position((shader.vec2(site) + shader.vec2(0.5)) / shader.vec2(full), shader.as_float(q.y), cam)
            offset = shader.vec2(at) - center
            w = shader.exp(-0.378 * shader.dot(offset, offset))
            w *= shader.saturate((shader.dot(qn, n) - 0.72) / 0.22) * shader.saturate(1.0 - shader.abs(shader.dot(qpos - pos, n)) / band)
            footprint = offset * pitch
            reach2 = 2.25 * shader.dot(footprint, footprint)
            distance2 = shader.dot(qpos - pos, qpos - pos)
            w *= shader.min(1.0, (reach2 + 1e-14) / shader.max(distance2, 1e-14))
            if self.ssao_strength > 0.0:
                vcell = ray_grid_pixel(site, full, visibility_extent)
                other_visibility = shader.load(self.ambient_visibility, vcell, 0).x
                w /= 1.0 + 12.0 * shader.min(1.0, contact_density(self.ssao_strength)) * shader.abs(other_visibility - visibility)
            axes = shader.load(self.fresh_axes, at, 0)
            color = shader.as_float(dc.xyz)
            mr = unpack_moment(color.x, axes.x)
            mg = unpack_moment(color.y, axes.y)
            mb = unpack_moment(color.z, axes.z)
            measured = shader.vec3(shader.dot(mr, basis), shader.dot(mg, basis), shader.dot(mb, basis))
            second_moment += w * shader.dot(measured, measured)
            squared_weights += w*w
            r += mr*w
            g += mg*w
            b += mb*w
            total += w
        if total <= 0.0:
            return (shader.uvec4(0), shader.uvec4(0))
        r /= total
        g /= total
        b /= total
        initial = shader.vec3(shader.dot(r, basis), shader.dot(g, basis), shader.dot(b, basis))
        spread = shader.max(second_moment / total - shader.dot(initial, initial), 0.0)
        error = spread * squared_weights / shader.max(total*total-squared_weights, 0.00001)
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
            for i in range(4):
                if plane.w == 0.0:
                    break
                at = base + shader.ivec2(i & 1, i >> 1)
                if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                    continue
                previous_site = ray_grid_pixel(at, extent, full)
                q = shader.load(self.previous_primary, previous_site, 0)
                dc = shader.load(self.old_dc, at, 0)
                if q.x != p.x or dc.w == 0:
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
                axes = shader.load(self.old_axes, at, 0)
                color = shader.as_float(dc.xyz)
                hr += unpack_moment(color.x, axes.x) * w
                hg += unpack_moment(color.y, axes.y) * w
                hb += unpack_moment(color.z, axes.z) * w
                old_age += shader.float(dc.w) * w
                old_total += w
            if old_total > 0.0:
                changed = self.geometry[1].y != shader._previous_ray_geometry_serial() or shader._ray_lighting_changed()
                travel = shader.abs(prev_uv - uv) * shader.vec2(full)
                moving = shader.max(travel.x, travel.y) > 0.25
                retention = shader.exp2(-shader.max(shader.frame_delta_time(), 0.0) / 0.023)
                budget = shader.min(1.0, retention / shader.max(1.0 - retention, 1.0 / 64.0)) if changed or moving else 7.0
                keep = shader.min(old_age / old_total, budget) * shader.saturate(old_total)
                old = shader.vec3(shader.dot(hr,basis),shader.dot(hg,basis),shader.dot(hb,basis)) / old_total
                difference = initial - old
                # Recheck disagreement after the scene stops moving as well.
                # Otherwise the last state of an occluder can linger on the receiver.
                mismatch = shader.dot(difference,difference)
                keep *= shader.min(1.0, (4.0 * error + 1e-9) / (mismatch + 1e-9))
                age += keep
                r = (r + hr * (keep / old_total)) / age
                g = (g + hg * (keep / old_total)) / age
                b = (b + hb * (keep / old_total)) / age
        return (shader.uvec4(shader.as_uint(shader.vec3(r.x, g.x, b.x)), shader.uint(shader.round(age))),
                shader.uvec4(pack_direction(r), pack_direction(g), pack_direction(b), shader.as_uint(error / age)))
