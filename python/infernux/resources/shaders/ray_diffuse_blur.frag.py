import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import contact_density
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_guide_pixel, sh_normal_weight
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseBlur(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_blur"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    state = shader.input("state", shader.Color2D)
    primary = shader.input("primary", shader.UInt2D)
    visibility = shader.input("visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        index = shader._view_sample_index()
        site = sh_guide_pixel(cell, extent, full, index)
        receiver = shader.load(self.primary, site, 0)
        age = shader.load(self.state, cell, 0).x
        if receiver.x == 0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        mature = age > 33.0
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
            pt = sh_guide_pixel(at, extent, full, index)
            guide = shader.load(self.primary, pt, 0)
            if guide.x == 0 or shader.load(self.state, at, 0).x == 0.0:
                continue
            normal = decode_octahedral_normal(shader.unpack_half_2x16(guide.z))
            other_vis = 0.0
            if ao_strength > 0.0:
                other_vis = ao_strength * shader.load(self.visibility, ray_grid_pixel(pt, full, vis_extent), 0).x
            weight = sh_bilateral(depth, shader.as_float(guide.y), facing, visibility, other_vis)
            weight *= sh_normal_weight(n, normal)
            if mature:
                weight *= shader.exp2(-shader.float(offset.x * offset.x + offset.y * offset.y))
            red += shader.load(self.red_sh, at, 0) * weight
            green += shader.load(self.green_sh, at, 0) * weight
            blue += shader.load(self.blue_sh, at, 0) * weight
            total += weight
        norm = 1.0 / shader.max(total, 1.0e-20)
        return (red * norm, green * norm, blue * norm)
