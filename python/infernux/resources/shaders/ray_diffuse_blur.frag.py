import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.ray_moments import pack_direction, unpack_moment
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseBlur(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_blur"
    hidden = True
    coefficients = shader.input("coefficients", shader.UInt2D)
    directions = shader.input("directions", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    visibility = shader.input("visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.coefficients, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        site = ray_grid_pixel(cell, extent, full)
        receiver = shader.load(self.primary, site, 0)
        center = shader.load(self.coefficients, cell, 0)
        if receiver.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        # Low-confidence pixels borrow a wider neighborhood. The filtered
        # state recurs across frames, so a converged pixel only needs a small kernel.
        stability = shader.saturate((shader.float(center.w) - 1.0) / 6.0)
        stride = 3 - shader.int(shader.round(2.0 * stability))
        camera = shader.camera()
        uv = (shader.vec2(site) + shader.vec2(0.5)) / shader.vec2(full)
        z = shader.as_float(receiver.y)
        pos = world_position(uv, z, camera)
        n = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        vis_extent = shader.ivec2(shader.dimensions(self.visibility, 0))
        vis = 1.0
        if self.ssao_strength > 0.0:
            vis = shader.load(self.visibility, ray_grid_pixel(site, full, vis_extent), 0).x
        band = 0.007 + 0.004 * linear_eye_depth(z, camera)
        projection_w = (camera.view_projection_matrix @ shader.vec4(pos, 1.0)).w
        focal = shader.abs(shader.vec2(camera.projection_matrix[0][0], camera.projection_matrix[1][1]))
        pitch = 2.0 * projection_w / (focal * shader.vec2(extent))
        center_axes = shader.load(self.directions, cell, 0)
        level_c = shader.as_float(center.xyz)
        irradiance = shader.vec3(shader.dot(n, unpack_moment(level_c.x, center_axes.x).yzw),
                                 shader.dot(n, unpack_moment(level_c.y, center_axes.y).yzw),
                                 shader.dot(n, unpack_moment(level_c.z, center_axes.z).yzw))
        center_error = shader.as_float(center_axes.w)
        error_sum = 0.0
        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        total = 0.0
        for i in range(9):
            offset = shader.ivec2(i % 3 - 1, int(i / 3) - 1)
            d = offset * stride
            kernel_distance2 = shader.float(offset.x * offset.x + offset.y * offset.y)
            weight = shader.exp2(-stability * kernel_distance2)
            at = cell + d
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            pt = ray_grid_pixel(at, extent, full)
            guide = shader.load(self.primary, pt, 0)
            dc = shader.load(self.coefficients, at, 0)
            if guide.x == 0 or dc.w == 0:
                continue
            normal = decode_octahedral_normal(shader.unpack_half_2x16(guide.z))
            q = world_position((shader.vec2(pt) + shader.vec2(0.5)) / shader.vec2(full), shader.as_float(guide.y), camera)
            separation = shader.max(shader.abs(shader.dot(n, q - pos)), shader.abs(shader.dot(normal, q - pos)))
            w = shader.saturate((shader.dot(n, normal) - 0.75) * 4.0) * shader.saturate(1.0 - separation / band)
            w *= weight
            if self.ssao_strength > 0.0:
                local = shader.load(self.visibility, ray_grid_pixel(pt, full, vis_extent), 0).x
                w /= 1.0 + 18.0 * shader.min(1.0, contact_density(self.ssao_strength)) * shader.abs(vis - local)
            axes = shader.load(self.directions, at, 0)
            level = shader.as_float(dc.xyz)
            here = shader.vec3(shader.dot(n, unpack_moment(level.x, axes.x).yzw),
                               shader.dot(n, unpack_moment(level.y, axes.y).yzw),
                               shader.dot(n, unpack_moment(level.z, axes.z).yzw))
            delta = here - irradiance
            footprint = shader.vec2(d) * pitch
            reach2 = 2.25 * shader.dot(footprint, footprint)
            distance2 = shader.dot(q-pos,q-pos)
            w *= shader.min(1.0, (reach2 + 1e-14) / shader.max(distance2, 1e-14))
            sigma2 = center_error + shader.as_float(axes.w)
            sigma2 += 0.0004 * shader.max(shader.dot(irradiance,irradiance),shader.dot(here,here)) + 1e-16
            innovation2 = shader.dot(delta,delta)
            agreement = shader.min(1.0, 4.0 * sigma2 / shader.max(innovation2, 1e-16))
            w *= shader.mix(1.0, agreement, stability)
            error_sum += w * shader.as_float(axes.w)
            red += unpack_moment(level.x, axes.x) * w
            green += unpack_moment(level.y, axes.y) * w
            blue += unpack_moment(level.z, axes.z) * w
            total += w
        if total <= 0.0:
            return (shader.uvec4(0), shader.uvec4(0))
        level = shader.vec3(red.x, green.x, blue.x) / total
        return (shader.uvec4(shader.as_uint(level), shader.max(center.w, shader.uint(1))),
                shader.uvec4(pack_direction(red), pack_direction(green), pack_direction(blue), shader.as_uint(error_sum / total)))
