import infernux.shader as shader
from infernux.shader.lib.noise import hash_vec2
from infernux.shader.lib.ray_signal import unpack_ray_signal
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_guide_normal, sh_guide_pixel
from infernux.shader.lib.ray_moments import sample_direction, sh_coefficients


class RayDiffuseGather(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_gather"
    hidden = True
    observations = shader.input("observations", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    guide = shader.input("guide", shader.UInt2D)
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    use_visibility = shader.parameter.bool(default=False)
    divisor = shader.parameter.int(default=2)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        observed = shader.ivec2(shader.dimensions(self.observations, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        index = shader._view_sample_index()
        source = sh_guide_pixel(cell, extent, full, index)
        geometry = shader.load(self.guide, source / 2, 0)
        depth = shader.as_float(geometry.x)
        if depth <= 0.0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        view_normal = sh_guide_normal(geometry.y)
        n = (shader.transpose(shader.camera().view_matrix) @ shader.vec4(view_normal, 0.0)).xyz
        trace_scale = shader.vec2(observed) / shader.vec2(full)
        anchor = (shader.vec2(cell) + 0.5) * shader.vec2(observed) / shader.vec2(extent)
        footprint = 32.0 * trace_scale
        facing = shader.max(0.3, view_normal.z)
        r = shader.vec4(0.0)
        g = shader.vec4(0.0)
        b = shader.vec4(0.0)
        total = 0.0
        visibility = shader.load(self.ambient_visibility, source, 0).x if self.use_visibility else 0.0
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
            other_depth = shader.as_float(shader.load(self.guide, site / 2, 0).x)
            body_packet = shader.load(self.observations, at, 0)
            body = unpack_ray_signal(body_packet)
            if body.a <= 0.0:
                continue
            neighbor_visibility = shader.load(self.ambient_visibility, site, 0).x if self.use_visibility else 0.0
            w = sh_bilateral(depth, other_depth, facing, visibility, neighbor_visibility)
            axis = sample_direction(body_packet)
            cosine = shader.dot(n, axis)
            if cosine <= 0.0:
                continue
            basis = shader.vec4(1.0, axis) * w
            r += body.r * basis
            g += body.g * basis
            b += body.b * basis
            total += w
        norm = 1.0 / shader.max(total, 1.0e-5)
        return (sh_coefficients(r * norm), sh_coefficients(g * norm), sh_coefficients(b * norm))
