import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_signal import (
    pack_ray_gradient, pack_ray_signal, ray_grid_pixel, unpack_ray_gradient, unpack_ray_signal,
)
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseFilter(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_filter"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)
    stride = shader.parameter.int(default=1)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        packet = shader.load(self.signal, pixel, 0)
        center = unpack_ray_signal(packet)
        if center.a <= 0.0 or (self.stride == 8 and center.a >= 8.0):
            return packet

        extent = shader.ivec2(shader.dimensions(self.signal, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        reduced = shader.any(extent != full)
        source = ray_grid_pixel(pixel, extent, full) if reduced else pixel
        witness = shader.load(self.primary, source, 0)
        camera = shader.camera()
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        depth = shader.as_float(witness.y)
        visibility_size = shader.ivec2(shader.dimensions(self.ambient_visibility, 0))
        center_visibility = 1.0
        if self.ssao_strength > 0.0:
            center_visibility = shader.load(self.ambient_visibility, shader.ivec2(uv * shader.vec2(visibility_size)), 0).x
        position = world_position(uv, depth, camera)
        plane_normal = decode_octahedral_normal(shader.unpack_half_2x16(witness.z))
        normal = decode_octahedral_normal(shader.load(self.frame, source, 0).xy)
        plane_band = 0.001 + 0.002 * linear_eye_depth(depth, camera)
        clip_w = (camera.view_projection_matrix @ shader.vec4(position, 1.0)).w
        focal = shader.abs(shader.vec2(camera.projection_matrix[0][0], camera.projection_matrix[1][1]))
        footprint = 2.0 * shader.float(self.stride) * clip_w / (shader.vec2(extent) * focal)
        luma = shader.vec3(0.2126, 0.7152, 0.0722)
        center_luma = shader.dot(center.rgb, luma)
        variance = shader.as_float(packet.w) / shader.max(center.a - 1.0, 1.0)
        # Separable binomial weights over a 3x3 atrous stencil.
        radiance = center.rgb * 4.0
        total = 4.0
        # The first two scales measure current-frame irradiance. Wider taps use the reconstructed moments.
        raw_samples = self.stride == 1
        local_means = self.stride == 2
        squared_sum = 4.0 * shader.dot(center.rgb, center.rgb)
        squared_weights = 16.0
        gradient_sum = 4.0 * unpack_ray_gradient(packet.z)
        mean_error_sum = 4.0 * shader.as_float(packet.w)
        # Widen support after disocclusion or relighting, then taper the outer taps as samples accumulate.
        wide_weight = shader.saturate((8.0 - center.a) / 7.0) if self.stride == 8 else 1.0
        for tap in range(9):
            if tap == 4:
                continue
            offset = shader.ivec2(tap % 3 - 1, int(tap / 3) - 1)
            neighbor = pixel + self.stride * offset
            if shader.any(neighbor < shader.ivec2(0)) or shader.any(neighbor >= extent):
                continue
            sample_source = ray_grid_pixel(neighbor, extent, full) if reduced else neighbor
            sample_witness = shader.load(self.primary, sample_source, 0)
            if sample_witness.x != witness.x:
                continue
            sample = shader.load(self.signal, neighbor, 0)
            estimate = unpack_ray_signal(sample)
            if estimate.a <= 0.0:
                continue
            sample_uv = (shader.vec2(sample_source) + shader.vec2(0.5)) / shader.vec2(full)
            sample_position = world_position(sample_uv, shader.as_float(sample_witness.y), camera)
            displacement = sample_position - position
            plane_distance = shader.abs(shader.dot(displacement, plane_normal))
            sample_normal = decode_octahedral_normal(shader.load(self.frame, sample_source, 0).xy)
            normal_weight = shader.saturate((shader.dot(normal, sample_normal) - 0.94) / 0.06)
            weight = (2.0 if offset.x == 0 or offset.y == 0 else 1.0) * normal_weight
            weight *= shader.saturate(1.0 - plane_distance / plane_band)
            if self.stride >= 4:
                # At a grazing angle two screen neighbors can lie far apart on the receiver.
                step = shader.vec2(offset) * footprint
                support2 = 4.0 * shader.dot(step, step)
                weight *= shader.min(1.0, support2 / shader.max(shader.dot(displacement, displacement), 1e-16))
            if self.ssao_strength > 0.0:
                cell = shader.ivec2(sample_uv * shader.vec2(visibility_size))
                visibility = shader.load(self.ambient_visibility, cell, 0).x
                weight /= 1.0 + 12.0 * self.ssao_strength * shader.abs(center_visibility - visibility)
            if raw_samples or local_means:
                gradient_sum += weight * unpack_ray_gradient(sample.z)
            if raw_samples:
                squared_sum += weight * shader.dot(estimate.rgb, estimate.rgb)
                squared_weights += weight * weight
            elif local_means:
                mean_error_sum += weight * shader.as_float(sample.w)
            elif center.a > 1.0 and estimate.a > 1.0:
                sample_luma = shader.dot(estimate.rgb, luma)
                sample_variance = shader.as_float(sample.w) / (estimate.a - 1.0)
                sigma = 4.0 * shader.sqrt(variance + sample_variance)
                sigma += 0.001 + 0.02 * shader.max(center_luma, sample_luma)
                weight *= sigma / (sigma + shader.abs(sample_luma - center_luma))
            weight *= wide_weight
            radiance += weight * estimate.rgb
            total += weight
        mean = radiance / total
        if raw_samples:
            inverse_samples = squared_weights / (total * total)
            spread = shader.max(squared_sum / total - shader.dot(mean, mean), 0.0)
            mean_error = spread * inverse_samples / shader.max(1.0 - inverse_samples, 0.0001)
            guides = shader.uvec2(pack_ray_gradient(gradient_sum / total), shader.as_uint(mean_error))
            return pack_ray_signal(shader.vec4(mean, 1.0), guides)
        if local_means:
            # Overlapping neighborhoods share samples. The weighted mean of their variances bounds
            # the variance of the combined estimate without assuming independent observations.
            guides = shader.uvec2(pack_ray_gradient(gradient_sum / total), shader.as_uint(mean_error_sum / total))
            return pack_ray_signal(shader.vec4(mean, 1.0), guides)
        return pack_ray_signal(shader.vec4(mean, center.a), packet.zw)
