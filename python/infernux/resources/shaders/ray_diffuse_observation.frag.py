import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.ray_moments import pack_direction
from infernux.shader.lib.ray_signal import ray_grid_pixel
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
        moments = shader.ray._diffuse_moments(position, normal, shader.uvec2(cell), self.temporal, self.primary,
                                              self.screen_radiance, self.screen_depth, self.surface_records, self.tail_cache)
        energy = shader.vec3(moments[0].x, moments[1].x, moments[2].x)
        return (shader.uvec4(shader.as_uint(energy), shader.uint(moments[3].x)),
                shader.uvec4(pack_direction(moments[0]), pack_direction(moments[1]), pack_direction(moments[2]),
                             shader.as_uint(moments[3].y)))
