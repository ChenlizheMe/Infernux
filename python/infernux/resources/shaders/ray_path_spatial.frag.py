import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.deferred_surface import deferred_shading_context, load_deferred_surface
from infernux.shader.lib.ray_path import path_pixel, related_receiver
from infernux.shader.lib.ray_signal import pack_ray_signal, unpack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


class RayPathSpatial(shader.fullscreen):
    shader_id = "infernux/internal/ray_path_spatial"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    tail_cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    paths = shader.input("paths", shader.UInt2D)
    estimates = shader.input("estimates", shader.UInt2D)
    specular = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        cell = shader.ivec2(ctx.position.xy)
        packet = shader.load(self.estimates, cell, 0)
        path = shader.load(self.paths, cell, 0)
        count = path.w & shader.uint(7)
        if not self.specular:
            # Irradiance reconstruction uses the signed change along the replayed ray.
            packet = shader.uvec4(packet.xy, path.w & shader.uint(0xFFFE0000), packet.w)
        if count == 0:
            return packet
        extent = shader.ivec2(shader.dimensions(self.paths, 0))
        neighbor = shader.ray._path_neighbor(shader.uvec2(cell))
        if shader.any(neighbor < shader.ivec2(0)) or shader.any(neighbor >= extent):
            return packet
        other = shader.load(self.paths, neighbor, 0)
        neighbor_count = other.w & shader.uint(7)
        if neighbor_count == 0:
            return packet
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        pixel = path_pixel(cell, extent, full)
        primary = shader.load(self.primary, pixel, 0)
        adjacent = shader.load(self.primary, path_pixel(neighbor, extent, full), 0)
        depth = shader.as_float(primary.y)
        if not related_receiver(primary, adjacent, linear_eye_depth(depth, shader.camera())):
            return packet
        center = shader.vec2(pixel) + shader.vec2(0.5)
        metadata = self.surface_records[primary.x * 3]
        surface = load_deferred_surface(self.albedo_tex, self.normal_tex, self.emission_tex, self.surface_tex, pixel)
        context = deferred_shading_context(center / shader.vec2(full), center, depth, surface.normal_ws, metadata)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
        sample = shader.ray._replay_path(surface, context, normal, self.surface_records, metadata.y,
                                         self.specular, True, self.tail_cache, other.xy)
        local = unpack_ray_signal(packet)
        represented = shader.float(shader.min(neighbor_count, shader.uint(4)))
        fraction = represented / (shader.float(count) + represented)
        radiance = shader.mix(local.rgb, shader.as_float(other.z) * sample.radiance.rgb, fraction)
        return pack_ray_signal(shader.vec4(radiance, local.w), packet.zw)
