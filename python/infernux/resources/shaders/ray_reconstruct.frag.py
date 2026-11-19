import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_diffuse_history import diffuse_history
from infernux.shader.lib.ray_signal import (
    pack_ray_signal, ray_grid_pixel, specular_color_variance, specular_hit_distance,
    specular_witness, unpack_ray_signal,
)
from infernux.shader.lib.ray_surface_depth import ray_previous_eye_depth
from infernux.shader.lib.surface import decode_octahedral_normal


class RayReconstruct(shader.fullscreen):
    shader_id = "infernux/internal/ray_reconstruct"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)
    local_estimate = shader.input("local_estimate", shader.UInt2D)
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
        # The incoming fourth word holds ray distance for specular, or uncertainty for diffuse.
        signal_aux = shader.as_float(fresh.w)
        current_error = 0.0
        if self.specular:
            # Two estimates of this receiver already exist before and after spatial reuse.
            # Their disagreement measures color noise even on a newly exposed surface.
            local = unpack_ray_signal(shader.load(self.local_estimate, pixel, 0))
            innovation = current.rgb - local.rgb
            current_error = shader.dot(innovation * innovation, shader.vec3(0.2126, 0.7152, 0.0722))
        fresh = shader.uvec4(fresh.xyz, specular_witness(signal_aux, current_error) if self.specular else shader.uint(0))

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
        camera = shader.camera()
        expected_depth = ray_previous_eye_depth(motion.w)
        previous_primary = shader.uvec4(0)
        previous_packet = shader.uvec4(0)
        if reduced and not self.specular:
            previous_packet = diffuse_history(self.previous_signal, self.previous_primary, primary, history_uv,
                                              previous_uv, expected_depth, extent, full)
        else:
            previous_pixel = shader.ivec2(history_uv * shader.vec2(extent))
            previous_primary = shader.load(self.previous_primary, previous_pixel, 0)
            if primary.x != previous_primary.x:
                return fresh
            normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
            previous_normal = decode_octahedral_normal(shader.unpack_half_2x16(previous_primary.z))
            normal_threshold = 0.985 if self.specular else 0.94
            if shader.dot(normal, previous_normal) < normal_threshold:
                return fresh
            previous_depth = linear_eye_depth(shader.as_float(previous_primary.y), camera)
            if shader.abs(expected_depth - previous_depth) > 0.01 + 0.015 * expected_depth:
                return fresh
            previous_packet = shader.load(self.previous_signal, previous_pixel, 0)
        previous = unpack_ray_signal(previous_packet)
        if previous.a <= 0.0:
            return fresh

        visibility = self.geometry[1]
        accepted = shader._previous_ray_geometry_serial()
        scene_changed = visibility.y != accepted or shader._ray_lighting_changed()
        limit = 64.0
        if self.specular:
            distance = signal_aux
            # Irradiance is independent of receiver roughness. Only a changed reflection lobe
            # invalidates the history of its material-weighted signal.
            roughness = shader.as_float(primary.w)
            previous_roughness = shader.as_float(previous_primary.w)
            if shader.abs(roughness - previous_roughness) > 0.015 + 0.08 * roughness:
                return fresh
            # The virtual image behind a planar mirror projects along the incident camera ray.
            # Check camera parallax at that distance as well as motion at the visible receiver.
            position = world_position(uv, shader.as_float(primary.y), camera)
            previous_clip = camera.previous_view_projection_matrix @ shader.vec4(position, 1.0)
            if previous_clip.w <= 0.0:
                return fresh
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
        elif shader._ray_proposals_changed():
            # Measure the RGB change against the energy at this receiver, including equal-luminance color edits.
            change = previous.rgb - current.rgb
            energy = shader.max(shader.dot(previous.rgb, previous.rgb), shader.dot(current.rgb, current.rgb))
            agreement = shader.saturate(1.0 - 4.0 * shader.dot(change, change) / shader.max(energy, 1e-12))
            limit = 1.0 + 3.0 * agreement * agreement
        elif scene_changed:
            # The diffuse prefilter stores the squared RGB error of its directional mean.
            # Let new measurements dominate when they contradict the old lighting.
            innovation = previous.rgb - current.rgb
            uncertainty = signal_aux
            confidence = uncertainty / shader.max(uncertainty + shader.dot(innovation, innovation), 1e-12)
            limit = 1.0 + 7.0 * confidence
        elif shader.any(shader.abs(history_uv - uv) * shader.vec2(full) > shader.vec2(0.5)):
            limit = 6.0

        count = shader.min(previous.a + 1.0, limit)
        difference = current.rgb - previous.rgb
        weights = shader.vec3(0.2126, 0.7152, 0.0722)
        delta = shader.dot(difference, weights)
        if not self.specular and scene_changed:
            # A coherent change in the replayed path identifies a moving shadow on a stationary receiver.
            magnitude = shader.max(shader.dot(current.rgb, weights), shader.dot(previous.rgb, weights))
            matched = shader.max(0.0, shader.sign(delta) * shader.as_float(fresh.z))
            relative = matched / shader.max(magnitude, 1e-8)
            blend = shader.saturate(6.0 * relative * relative)
            count = shader.min(count, 1.0 / shader.max(blend, 1.0 / 64.0))
        estimate = shader.mix(previous.rgb, current.rgb, 1.0 / count)
        # The reflection filter measures squared RGB distance. Equal-luminance
        # changes between colored lights contribute to its uncertainty as well.
        deviation2 = shader.dot(difference * difference, weights) if self.specular else delta * delta
        prior_variance = (specular_color_variance(previous_packet.w) if self.specular
                          else shader.as_float(previous_packet.w))
        variance = (1.0 - 1.0 / count) * (prior_variance + deviation2 / count)
        witness = specular_witness(signal_aux, variance) if self.specular else shader.as_uint(variance)
        return pack_ray_signal(shader.vec4(estimate, count), shader.uvec2(fresh.z, witness))
