import infernux.shader as shader
from infernux.shader.lib.deferred_surface import deferred_shading_context, load_deferred_surface
from infernux.shader.lib.ray_signal import pack_ray_signal, ray_grid_pixel
from infernux.shader.lib.surface import decode_octahedral_normal


class RayIndirect(shader.fullscreen):
    shader_id = "infernux/internal/ray_indirect"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    tail_cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    specular = shader.parameter.bool(default=False)
    temporal = shader.parameter.bool(default=True)
    resolution_divisor = shader.parameter.int(default=1)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        full_pixel = ctx.position.xy
        uv = ctx.uv
        if self.resolution_divisor > 1:
            full = shader.ivec2(shader.dimensions(self.primary, 0))
            extent = shader.max(shader.ivec2(full / self.resolution_divisor), shader.ivec2(1))
            pixel = ray_grid_pixel(pixel, extent, full)
            full_pixel = shader.vec2(pixel) + shader.vec2(0.5)
            uv = full_pixel / shader.vec2(full)
        primary = shader.load(self.primary, pixel, 0)
        if primary.x == 0:
            return shader.uvec4(0)
        depth = shader.as_float(primary.y)
        metadata = self.surface_records[primary.x * 3]
        surface = load_deferred_surface(self.albedo_tex, self.normal_tex, self.emission_tex, self.surface_tex, pixel)
        context = deferred_shading_context(uv, full_pixel, depth, surface.normal_ws, metadata)
        geometric_normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
        sample = shader.ray._indirect_signal(
            surface, context, geometric_normal, self.surface_records, metadata.y, self.specular, self.temporal,
            self.tail_cache,
        )
        return pack_ray_signal(sample.radiance, sample.witness)
