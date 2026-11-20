import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.ray_sh_filter import sh_guide_pixel
from infernux.shader.lib.ray_signal import pack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseObservation(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_observation"
    hidden = True
    primary = shader.input("primary", shader.UInt2D)
    screen_radiance = shader.input("screen_radiance", shader.Color2D)
    screen_depth = shader.input("screen_depth", shader.Color2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    tail_cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    divisor = shader.parameter.int(default=4)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        index = shader._view_sample_index() if self.temporal else shader.uint(0)
        half_grid = extent * 2
        site = sh_guide_pixel(cell * 2, half_grid, full, index)
        receiver = shader.load(self.primary, site, 0)
        if receiver.x == 0:
            return shader.uvec4(0)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        # The exact triangle traversal needs the point on the phase-selected raster surface.
        uv = (shader.vec2(site) + 0.5) / shader.vec2(full)
        position = world_position(uv, shader.as_float(receiver.y), shader.camera())
        sample = shader.ray._diffuse_sample(position, normal, shader.uvec2(cell), self.temporal, self.primary,
                                            self.screen_radiance, self.screen_depth, self.surface_records, self.tail_cache)
        return pack_ray_signal(sample.radiance, sample.witness)
