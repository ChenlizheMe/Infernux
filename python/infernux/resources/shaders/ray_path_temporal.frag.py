import infernux.shader as shader
from infernux.shader.lib.deferred_surface import deferred_shading_context, load_deferred_surface
from infernux.shader.lib.ray_path import path_pixel, previous_path
from infernux.shader.lib.ray_signal import pack_ray_gradient, pack_ray_signal, unpack_ray_gradient, unpack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


class RayPathTemporal(shader.fullscreen):
    shader_id = "infernux/internal/ray_path_temporal"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    tail_cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    fresh = shader.input("fresh", shader.UInt2D)
    motion = shader.input("motion", shader.UInt2D)
    previous_surface = shader.input("previous_surface", shader.UInt2D)
    previous_paths = shader.input("previous_paths", shader.UInt2D)
    specular = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        extent = shader.ivec2(shader.dimensions(self.previous_paths, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        pixel = path_pixel(shader.ivec2(ctx.position.xy), extent, full)
        primary = shader.load(self.primary, pixel, 0)
        if primary.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        center = shader.vec2(pixel) + shader.vec2(0.5)
        uv = center / shader.vec2(full)
        metadata = self.surface_records[primary.x * 3]
        surface = load_deferred_surface(self.albedo_tex, self.normal_tex, self.emission_tex, self.surface_tex, pixel)
        context = deferred_shading_context(uv, center, shader.as_float(primary.y), surface.normal_ws, metadata)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
        history = shader.uvec4(0)
        if self.specular or not shader._ray_proposals_changed():
            history = previous_path(primary, shader.load(self.motion, pixel, 0), self.previous_surface,
                                    self.previous_paths, uv, extent, full)
        cell = shader.ivec2(ctx.position.xy)
        fresh = shader.load(self.fresh, cell, 0)
        result = shader.ray._path_temporal(surface, context, normal, self.surface_records, metadata.y,
                                           self.specular, self.tail_cache, history, shader.uvec2(cell),
                                           unpack_ray_signal(fresh), fresh.zw)
        packet = pack_ray_signal(result.signal.radiance, result.signal.witness)
        reservoir = result.reservoir
        if not self.specular:
            change = shader.vec3(0.0)
            if (history.w & shader.uint(7)) != 0:
                # Compare the same path at the same precision on both sides of the frame boundary.
                replayed = unpack_ray_gradient(pack_ray_gradient(result.replayed_radiance))
                change = shader.as_float(history.z) * (replayed - unpack_ray_gradient(history.w >> 3))
            packet = shader.uvec4(packet.xy, pack_ray_gradient(change), packet.w)
            selected = pack_ray_gradient(result.selected_radiance)
            reservoir = shader.uvec4(reservoir.xyz, (selected << 3) | reservoir.w)
        return (packet, reservoir)
