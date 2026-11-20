import infernux.shader as shader
from infernux.shader.lib.noise import hash_2d
from infernux.shader.lib.ray_moments import diffuse_irradiance, sample_direction, unpack_moment
from infernux.shader.lib.ray_signal import ray_grid_pixel, unpack_ray_signal
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_resolve_offset
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseResolve(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_resolve"
    hidden = True
    coefficients = shader.input("coefficients", shader.UInt2D)
    directions = shader.input("directions", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)
    raw = shader.parameter.bool(default=False)
    observations = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.coefficients, 0))
        if self.raw:
            pixel = ray_grid_pixel(pixel, extent, full)
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        depth = shader.as_float(receiver.y)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        cell = ray_grid_pixel(pixel, full, extent)
        if self.observations:
            body_packet = shader.load(self.coefficients, cell, 0)
            light_packet = shader.load(self.directions, cell, 0)
            body = unpack_ray_signal(body_packet)
            light = unpack_ray_signal(light_packet).rgb
            body_axis = shader.vec4(1.0, sample_direction(body_packet))
            light_axis = shader.vec4(1.0, sample_direction(light_packet))
            return shader.vec4(diffuse_irradiance(body.r * body_axis + light.r * light_axis,
                                                   body.g * body_axis + light.g * light_axis,
                                                   body.b * body_axis + light.b * light_axis,
                                                   normal, shading), body.a)

        seed = shader.vec2(pixel) + shader.float(shader._view_sample_index() & shader.uint(255))
        pattern = shader.uint(3.0 * hash_2d(seed) + 0.5)
        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        total = 0.0
        for tap in range(6):
            if tap == 3 and total >= 0.001:
                break
            at = cell + sh_resolve_offset(pattern, tap)
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            source = ray_grid_pixel(at, extent, full)
            guide = shader.load(self.primary, source, 0)
            dc = shader.load(self.coefficients, at, 0)
            if guide.x == 0 or dc.w == 0:
                continue
            weight = sh_bilateral(depth, shader.as_float(guide.y), 1.0, 0.0, 0.0)
            axes = shader.load(self.directions, at, 0)
            color = shader.as_float(dc.xyz)
            red += weight * unpack_moment(color.r, axes.r)
            green += weight * unpack_moment(color.g, axes.g)
            blue += weight * unpack_moment(color.b, axes.b)
            total += weight
        if total <= 0.0:
            return shader.vec4(0.0)
        return shader.vec4(diffuse_irradiance(red, green, blue, normal, shading) / total, 1.0)
