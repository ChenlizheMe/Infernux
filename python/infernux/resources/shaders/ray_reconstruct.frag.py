import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_signal import (
    pack_ray_signal, ray_grid_pixel, specular_hit_distance, specular_luminance_variance,
    specular_witness, unpack_ray_signal,
)
from infernux.shader.lib.surface import decode_octahedral_normal


class RayReconstruct(shader.fullscreen):
    shader_id = "infernux/internal/ray_reconstruct"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    motion = shader.input("motion", shader.UInt2D)
    previous_signal = shader.input("previous_signal", shader.UInt2D)
    previous_primary = shader.input("previous_primary", shader.UInt2D)
    geometry = shader.input("world_ray_geometry", shader.Buffer[shader.uvec4])
    specular = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        fresh = shader.load(self.signal, pixel, 0)
        current = unpack_ray_signal(fresh)
        if current.a <= 0.0:
            return fresh
        distance = shader.as_float(fresh.w)
        fresh = shader.uvec4(fresh.xyz, specular_witness(distance, 0.0) if self.specular else shader.uint(0))

        extent = shader.ivec2(shader.dimensions(self.signal, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        reduced = shader.any(extent != full)
        source = ray_grid_pixel(pixel, extent, full) if reduced else pixel
        primary = shader.load(self.primary, source, 0)
        motion = shader.load(self.motion, source, 0)
        if (motion.w >> 31) == 0 or motion.x != primary.x:
            return fresh
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        previous_uv = uv - shader.unpack_half_2x16(motion.z)
        # The accumulator lives on the stable pixel grid. Rounding the jittered position each
        # frame transports radiance into neighboring pixels even when the camera is still.
        history_uv = previous_uv + shader._view_jitter_delta_uv()
        if shader.any(history_uv < shader.vec2(0.0)) or shader.any(history_uv >= shader.vec2(1.0)):
            return fresh
        previous_pixel = shader.ivec2(history_uv * shader.vec2(extent))
        previous_source = ray_grid_pixel(previous_pixel, extent, full) if reduced else previous_pixel
        previous_primary = shader.load(self.previous_primary, previous_source, 0)
        if primary.x != previous_primary.x:
            return fresh
        normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
        previous_normal = decode_octahedral_normal(shader.unpack_half_2x16(previous_primary.z))
        normal_threshold = 0.985 if self.specular else 0.94
        if shader.dot(normal, previous_normal) < normal_threshold:
            return fresh

        camera = shader.camera()
        position = world_position(uv, shader.as_float(primary.y), camera)
        previous_clip = camera.previous_view_projection_matrix @ shader.vec4(position, 1.0)
        if previous_clip.w <= 0.0 or previous_clip.z <= 0.0 or previous_clip.z > previous_clip.w:
            return fresh
        expected_depth = linear_eye_depth(previous_clip.z / previous_clip.w, camera)
        previous_depth = linear_eye_depth(shader.as_float(previous_primary.y), camera)
        # Rigid motion within the previous depth band can reuse history. Depth motion,
        # newly revealed surfaces, and projections crossing the frustum restart the estimate.
        if shader.abs(expected_depth - previous_depth) > 0.01 + 0.015 * expected_depth:
            return fresh

        previous_packet = shader.load(self.previous_signal, previous_pixel, 0)
        previous = unpack_ray_signal(previous_packet)
        if previous.a <= 0.0:
            return fresh

        visibility = self.geometry[1]
        accepted = shader._previous_ray_geometry_serial()
        limit = 64.0
        if self.specular:
            # Irradiance is independent of receiver roughness. Only a changed reflection lobe
            # invalidates the history of its material-weighted signal.
            roughness = shader.as_float(primary.w)
            previous_roughness = shader.as_float(previous_primary.w)
            if shader.abs(roughness - previous_roughness) > 0.015 + 0.08 * roughness:
                return fresh
            # The virtual image behind a planar mirror projects along the incident camera ray.
            # Check camera parallax at that distance as well as motion at the visible receiver.
            eye = camera.inverse_view_projection_matrix @ shader.vec4(0.0, 0.0, 1.0, 0.0)
            incident = shader.normalize(position * eye.w - eye.xyz)
            travel = distance if distance > 0.0 else camera.projection_params.y
            virtual_clip = camera.previous_view_projection_matrix @ shader.vec4(position + travel * incident, 1.0)
            if virtual_clip.w <= 0.0:
                return fresh
            virtual_uv = virtual_clip.xy / virtual_clip.w * shader.vec2(0.5, -0.5) + shader.vec2(0.5)
            receiver_uv = previous_clip.xy / previous_clip.w * shader.vec2(0.5, -0.5) + shader.vec2(0.5)
            # Both predictions are in the previous raster, so a jitter change or camera rotation
            # about a fixed eye cannot masquerade as motion of the receiver or reflected ray.
            drift = shader.max(shader.abs(virtual_uv - previous_uv), shader.abs(receiver_uv - previous_uv)) * shader.vec2(full)
            still = visibility.y == accepted and shader.max(drift.x, drift.y) < 0.0625
            # With a stationary ray distribution, different random endpoints sample the same integral.
            # While moving, only a matching secondary surface and distance may carry mirror detail.
            if still:
                limit = 96.0
            else:
                if fresh.z != previous_packet.z:
                    return fresh
                previous_distance = specular_hit_distance(previous_packet.w)
                if shader.abs(distance - previous_distance) > 0.02 + 0.075 * distance:
                    return fresh
                # Reject history when reflected detail moved farther than its projected GGX lobe.
                focal_pixels = 0.5 * shader.float(full.y) * shader.abs(camera.projection_matrix[1][1])
                cone = focal_pixels * travel * roughness * roughness / virtual_clip.w
                parallax = shader.abs(virtual_uv - previous_uv) * shader.vec2(full)
                if shader.max(parallax.x, parallax.y) > 0.5 + cone:
                    return fresh
                if visibility.y != accepted:
                    if visibility.y < accepted:
                        return fresh
                    surface_id = fresh.z
                    if surface_id != 0 and surface_id != shader.uint(0xFFFFFFFF):
                        stamps = self.geometry[visibility.x + (surface_id >> 2)]
                        if stamps[surface_id & 3] > accepted:
                            return fresh
                limit = 8.0
        elif visibility.y != accepted:
            # This estimate integrates many directions. After a visibility change its
            # old paths need fresh evaluation, including paths that missed geometry.
            return fresh
        elif shader.any(shader.abs(history_uv - uv) * shader.vec2(full) > shader.vec2(0.5)):
            limit = 32.0

        count = shader.min(previous.a + 1.0, limit)
        estimate = shader.mix(previous.rgb, current.rgb, 1.0 / count)
        delta = shader.dot(current.rgb - previous.rgb, shader.vec3(0.2126, 0.7152, 0.0722))
        prior_variance = (specular_luminance_variance(previous_packet.w) if self.specular
                          else shader.as_float(previous_packet.w))
        variance = (1.0 - 1.0 / count) * (prior_variance + delta * delta / count)
        witness = specular_witness(distance, variance) if self.specular else shader.as_uint(variance)
        return pack_ray_signal(shader.vec4(estimate, count), shader.uvec2(fresh.z, witness))
