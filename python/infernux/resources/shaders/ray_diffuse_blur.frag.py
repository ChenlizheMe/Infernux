import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.ray_moments import pack_direction, unpack_moment
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_normal_weight
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseBlur(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_blur"
    hidden = True
    coefficients = shader.input("coefficients", shader.UInt2D)
    directions = shader.input("directions", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    visibility = shader.input("visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.coefficients, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        site = ray_grid_pixel(cell, extent, full)
        receiver = shader.load(self.primary, site, 0)
        center = shader.load(self.coefficients, cell, 0)
        if receiver.x == 0:
            return (shader.uvec4(0), shader.uvec4(0))
        mature = center.w > 33
        step = 1 if mature else 3
        depth = shader.as_float(receiver.y)
        n = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        view_normal = (shader.camera().view_matrix @ shader.vec4(n, 0.0)).xyz
        facing = shader.max(0.3, view_normal.z)
        vis_extent = shader.ivec2(shader.dimensions(self.visibility, 0))
        ao_strength = shader.min(1.0, contact_density(self.ssao_strength))
        visibility = 0.0
        if ao_strength > 0.0:
            visibility = ao_strength * shader.load(self.visibility, ray_grid_pixel(site, full, vis_extent), 0).x

        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        total = 0.0
        for i in range(9):
            offset = shader.ivec2(i % 3 - 1, int(i / 3) - 1)
            at = cell + offset * step
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            pt = ray_grid_pixel(at, extent, full)
            guide = shader.load(self.primary, pt, 0)
            dc = shader.load(self.coefficients, at, 0)
            if guide.x == 0 or dc.w == 0:
                continue
            normal = decode_octahedral_normal(shader.unpack_half_2x16(guide.z))
            other_vis = 0.0
            if ao_strength > 0.0:
                other_vis = ao_strength * shader.load(self.visibility, ray_grid_pixel(pt, full, vis_extent), 0).x
            weight = sh_bilateral(depth, shader.as_float(guide.y), facing, visibility, other_vis)
            weight *= sh_normal_weight(n, normal)
            if mature:
                weight *= shader.exp2(-shader.float(offset.x * offset.x + offset.y * offset.y))
            axes = shader.load(self.directions, at, 0)
            level = shader.as_float(dc.xyz)
            red += unpack_moment(level.x, axes.x) * weight
            green += unpack_moment(level.y, axes.y) * weight
            blue += unpack_moment(level.z, axes.z) * weight
            total += weight
        if total <= 0.0:
            return (shader.uvec4(0), shader.uvec4(0))
        age = shader.max(center.w, shader.uint(1))
        center_axes = shader.load(self.directions, cell, 0)
        center_level = shader.as_float(center.xyz)
        # A spatial update may contribute one sample to the recurrent estimate.
        # Averaging the full accumulated signal again spreads light across every frame.
        spatial_fraction = 1.0 / shader.float(age)
        red = shader.mix(unpack_moment(center_level.r, center_axes.r), red / total, spatial_fraction)
        green = shader.mix(unpack_moment(center_level.g, center_axes.g), green / total, spatial_fraction)
        blue = shader.mix(unpack_moment(center_level.b, center_axes.b), blue / total, spatial_fraction)
        level = shader.vec3(red.x, green.x, blue.x)
        return (shader.uvec4(shader.as_uint(level), age),
                shader.uvec4(pack_direction(red), pack_direction(green), pack_direction(blue), 0))
