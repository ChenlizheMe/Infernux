import infernux.shader as shader
from infernux.shader.lib.deferred_surface import deferred_shading_context, load_deferred_surface
from infernux.shader.lib.ray_path import path_pixel, previous_path
from infernux.shader.lib.ray_signal import pack_ray_signal
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
        result = shader.ray._path_temporal(surface, context, normal, self.surface_records, metadata.y,
                                           self.specular, self.tail_cache, history, shader.uvec2(ctx.position.xy))
        return (pack_ray_signal(result.signal.radiance, result.signal.witness), result.reservoir)
