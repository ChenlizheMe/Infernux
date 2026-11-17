import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_signal import pack_ray_signal, ray_grid_pixel, unpack_ray_signal
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

        extent = shader.ivec2(shader.dimensions(self.signal, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        reduced = shader.any(extent != full)
        source = ray_grid_pixel(pixel, extent, full) if reduced else pixel
        primary = shader.load(self.primary, source, 0)
        motion = shader.load(self.motion, source, 0)
        if motion.w == 0 or motion.x != primary.x:
            return fresh
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        previous_uv = uv - shader.unpack_half_2x16(motion.z)
        if shader.any(previous_uv < shader.vec2(0.0)) or shader.any(previous_uv >= shader.vec2(1.0)):
            return fresh
        previous_pixel = shader.min(shader.ivec2(previous_uv * shader.vec2(extent)), extent - shader.ivec2(1))
        previous_source = ray_grid_pixel(previous_pixel, extent, full) if reduced else previous_pixel
        previous_primary = shader.load(self.previous_primary, previous_source, 0)
        if primary.x != previous_primary.x or primary.w != previous_primary.w:
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
        if previous.a <= 0.0 or fresh.z != previous_packet.z:
            return fresh

        # Stamps travel with the immutable TLAS. Comparing to the submitted
        # view catches changes even when it never rendered intermediate pages.
        visibility = self.geometry[1]
        accepted = shader._previous_ray_geometry_serial()
        if visibility.y != accepted:
            if visibility.y < accepted:
                return fresh
            surface_id = fresh.z
            if surface_id != 0 and surface_id != shader.uint(0xFFFFFFFF):
                stamps = self.geometry[visibility.x + (surface_id >> 2)]
                if stamps[surface_id & 3] > accepted:
                    return fresh
        distance = shader.as_float(fresh.w)
        previous_distance = shader.as_float(previous_packet.w)
        relative_tolerance = 0.075 if self.specular else 0.25
        if shader.abs(distance - previous_distance) > 0.02 + relative_tolerance * distance:
            return fresh

        count = shader.min(previous.a + 1.0, 8.0)
        estimate = shader.mix(previous.rgb, current.rgb, 1.0 / count)
        return pack_ray_signal(shader.vec4(estimate, count), fresh.zw)
