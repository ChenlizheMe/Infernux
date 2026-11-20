import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.noise import hash_vec2
from infernux.shader.lib.ray_signal import ray_grid_pixel, unpack_ray_signal
from infernux.shader.lib.ray_sh_filter import sh_bilateral
from infernux.shader.lib.ray_moments import pack_direction, sample_direction
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseGather(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_gather"
    hidden = True
    body_samples = shader.input("body_samples", shader.UInt2D)
    light_samples = shader.input("light_samples", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)
    divisor = shader.parameter.int(default=2)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        cell = shader.ivec2(ctx.position.xy)
        observed = shader.ivec2(shader.dimensions(self.body_samples, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        source = ray_grid_pixel(cell, extent, full)
        p = shader.load(self.primary, source, 0)
        if p.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        n = decode_octahedral_normal(shader.unpack_half_2x16(p.z))
        trace_scale = shader.vec2(observed) / shader.vec2(full)
        anchor = (shader.vec2(source) + 0.5) * trace_scale
        footprint = 32.0 * trace_scale
        view_normal = (shader.camera().view_matrix @ shader.vec4(n, 0.0)).xyz
        facing = shader.max(0.3, view_normal.z)
        depth = shader.as_float(p.y)
        r = shader.vec4(0.0)
        g = shader.vec4(0.0)
        b = shader.vec4(0.0)
        light_r = shader.vec4(0.0)
        light_g = shader.vec4(0.0)
        light_b = shader.vec4(0.0)
        total = 0.0
        placement_total = 0.0
        second_moment = shader.vec2(0.0)
        squared_weights = shader.vec2(0.0)
        visibility_extent = shader.ivec2(shader.dimensions(self.ambient_visibility, 0))
        ao_strength = shader.min(1.0, contact_density(self.ssao_strength))
        visibility = 0.0
        if ao_strength > 0.0:
            visibility = ao_strength * shader.load(self.ambient_visibility, ray_grid_pixel(source, full, visibility_extent), 0).x
        phase = shader.float(shader._view_sample_index() & shader.uint(255))
        rotation = shader.frac(hash_vec2(shader.vec2(cell)) + phase * shader.vec2(0.754877666, 0.569840296))
        for i in range(8):
            radius = shader.sqrt((shader.float(i) + rotation.y) * 0.125)
            angle = rotation.x * 6.283185307 + shader.float(i) * 2.39996323
            point = anchor + footprint * radius * shader.vec2(shader.cos(angle), shader.sin(angle))
            if shader.any(point < shader.vec2(0.0)) or shader.any(point >= shader.vec2(observed)):
                continue
            at = shader.ivec2(point)
            site = ray_grid_pixel(at, observed, full)
            q = shader.load(self.primary, site, 0)
            body_packet = shader.load(self.body_samples, at, 0)
            body = unpack_ray_signal(body_packet)
            if body.a <= 0.0:
                continue
            neighbor_visibility = 0.0
            if ao_strength > 0.0:
                neighbor_visibility = ao_strength * shader.load(self.ambient_visibility, ray_grid_pixel(site, full, visibility_extent), 0).x
            w = sh_bilateral(depth, shader.as_float(q.y), facing, visibility, neighbor_visibility)
            light_packet = shader.load(self.light_samples, at, 0)
            light = unpack_ray_signal(light_packet).rgb
            body_axis = sample_direction(body_packet)
            light_axis = sample_direction(light_packet)
            body_cosine = shader.max(0.0, shader.dot(n, body_axis))
            light_cosine = shader.max(0.0, shader.dot(n, light_axis))
            body_weight = w if body_cosine > 0.0 else 0.0
            light_weight = w if light_cosine > 0.0 else 0.0
            body_basis = shader.vec4(1.0, body_axis) * body_weight
            light_basis = shader.vec4(1.0, light_axis) * light_weight
            r += body.r * body_basis
            g += body.g * body_basis
            b += body.b * body_basis
            light_r += light.r * light_basis
            light_g += light.g * light_basis
            light_b += light.b * light_basis
            body_irradiance = body.rgb * body_cosine
            light_irradiance = light * light_cosine
            lane_weights = shader.vec2(body_weight, w)
            second_moment += lane_weights * shader.vec2(shader.dot(body_irradiance, body_irradiance),
                                                       shader.dot(light_irradiance, light_irradiance))
            squared_weights += lane_weights * lane_weights
            total += body_weight
            placement_total += w
        if total <= 1.0e-5:
            return (shader.uvec4(0), shader.uvec4(0))
        r /= total
        g /= total
        b /= total
        # Dark or below-hemisphere light proposals still count in the MIS mean.
        light_r /= placement_total
        light_g /= placement_total
        light_b /= placement_total
        body_mean = shader.vec3(shader.dot(r.yzw, n), shader.dot(g.yzw, n), shader.dot(b.yzw, n))
        light_mean = shader.vec3(shader.dot(light_r.yzw, n), shader.dot(light_g.yzw, n), shader.dot(light_b.yzw, n))
        totals = shader.vec2(total, placement_total)
        mean_square = shader.vec2(shader.dot(body_mean, body_mean), shader.dot(light_mean, light_mean))
        spread = shader.max(second_moment / totals - mean_square, shader.vec2(0.0))
        error = spread * squared_weights / shader.max(totals * totals - squared_weights, shader.vec2(1.0e-5))
        r += light_r
        g += light_g
        b += light_b
        return (shader.uvec4(shader.as_uint(shader.vec3(r.x, g.x, b.x)), 1),
                shader.uvec4(pack_direction(r), pack_direction(g), pack_direction(b), shader.as_uint(error.x + error.y)))
