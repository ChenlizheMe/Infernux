import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.ray_moments import pack_direction
from infernux.shader.lib.ray_signal import pack_ray_signal, ray_grid_pixel
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

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        site = ray_grid_pixel(cell, extent, full)
        receiver = shader.load(self.primary, site, 0)
        if receiver.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        uv = (shader.vec2(site) + shader.vec2(0.5)) / shader.vec2(full)
        position = world_position(uv, shader.as_float(receiver.y), shader.camera())
        samples = shader.ray._diffuse_samples(position, normal, shader.uvec2(cell), self.temporal, self.primary,
                                              self.screen_radiance, self.screen_depth, self.surface_records, self.tail_cache)
        body_axis = pack_direction(shader.vec4(1.0, samples[1].xyz))
        light_axis = pack_direction(shader.vec4(1.0, samples[3].xyz))
        return (pack_ray_signal(samples[0], shader.uvec2(body_axis, 0)),
                pack_ray_signal(samples[2], shader.uvec2(light_axis, shader.as_uint(samples[1].w))))
