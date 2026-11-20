import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.noise import hash_vec2
from infernux.shader.lib.ray_signal import ray_grid_pixel, unpack_ray_signal
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_guide_pixel
from infernux.shader.lib.ray_moments import sample_direction, sh_coefficients
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseGather(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_gather"
    hidden = True
    observations = shader.input("observations", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)
    divisor = shader.parameter.int(default=2)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        observed = shader.ivec2(shader.dimensions(self.observations, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        index = shader._view_sample_index()
        source = sh_guide_pixel(cell, extent, full, index)
        p = shader.load(self.primary, source, 0)
        if p.x == 0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        n = decode_octahedral_normal(shader.unpack_half_2x16(p.z))
        trace_scale = shader.vec2(observed) / shader.vec2(full)
        anchor = (shader.vec2(cell) + 0.5) * shader.vec2(observed) / shader.vec2(extent)
        footprint = 32.0 * trace_scale
        view_normal = (shader.camera().view_matrix @ shader.vec4(n, 0.0)).xyz
        facing = shader.max(0.3, view_normal.z)
        depth = shader.as_float(p.y)
        r = shader.vec4(0.0)
        g = shader.vec4(0.0)
        b = shader.vec4(0.0)
        total = 0.0
        second_moment = 0.0
        squared_weights = 0.0
        visibility_extent = shader.ivec2(shader.dimensions(self.ambient_visibility, 0))
        ao_strength = shader.min(1.0, contact_density(self.ssao_strength))
        visibility = 0.0
        if ao_strength > 0.0:
            visibility = ao_strength * shader.load(self.ambient_visibility, ray_grid_pixel(source, full, visibility_extent), 0).x
        phase = shader.float(index & shader.uint(255))
        rotation = shader.frac(hash_vec2(shader.vec2(cell)) + phase * shader.vec2(0.754877666, 0.569840296))
        for i in range(8):
            radius = shader.sqrt((shader.float(i) + rotation.y) * 0.125)
            angle = rotation.x * 6.283185307 + shader.float(i) * 2.39996323
            point = anchor + footprint * radius * shader.vec2(shader.cos(angle), shader.sin(angle))
            if shader.any(point < shader.vec2(0.0)) or shader.any(point >= shader.vec2(observed)):
                continue
            at = shader.ivec2(point)
            site = sh_guide_pixel(at * 2, observed * 2, full, index)
            q = shader.load(self.primary, site, 0)
            body_packet = shader.load(self.observations, at, 0)
            body = unpack_ray_signal(body_packet)
            if body.a <= 0.0:
                continue
            neighbor_visibility = 0.0
            if ao_strength > 0.0:
                neighbor_visibility = ao_strength * shader.load(self.ambient_visibility, ray_grid_pixel(site, full, visibility_extent), 0).x
            w = sh_bilateral(depth, shader.as_float(q.y), facing, visibility, neighbor_visibility)
            axis = sample_direction(body_packet)
            cosine = shader.dot(n, axis)
            if cosine <= 0.0:
                continue
            basis = shader.vec4(1.0, axis) * w
            r += body.r * basis
            g += body.g * basis
            b += body.b * basis
            irradiance = body.rgb * cosine
            second_moment += w * shader.dot(irradiance, irradiance)
            squared_weights += w * w
            total += w
        if total <= 1.0e-5:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        r /= total
        g /= total
        b /= total
        mean = shader.vec3(shader.dot(r.yzw, n), shader.dot(g.yzw, n), shader.dot(b.yzw, n))
        variance = shader.max(0.0, second_moment / total - shader.dot(mean, mean))
        error = variance * squared_weights / shader.max(total * total - squared_weights, 1.0e-5)
        sigma = shader.min(65504.0, shader.sqrt(error))
        return (sh_coefficients(r), sh_coefficients(g), sh_coefficients(b), shader.vec4(1.0, sigma, 0.0, 0.0))
