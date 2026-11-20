import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_facing, sh_guide_normal, sh_guide_pixel, sh_normal_weight


class RayDiffuseBlur(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_blur"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    state = shader.input("state", shader.Color2D)
    primary = shader.input("primary", shader.UInt2D)
    guide = shader.input("guide", shader.UInt2D)
    visibility = shader.input("visibility", shader.Color2D)
    use_visibility = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        extent = shader.ivec2(shader.dimensions(self.state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        index = shader._view_sample_index()
        site = sh_guide_pixel(cell, extent, full, index)
        geometry = shader.load(self.guide, site / 2, 0)
        depth = shader.as_float(geometry.x)
        age = shader.load(self.state, cell, 0).x
        if depth <= 0.0:
            return (shader.vec4(0.0), shader.vec4(0.0), shader.vec4(0.0))
        mature = age > 33.0
        step = 1 if mature else 3
        n = sh_guide_normal(geometry.y)
        facing = sh_facing(n)
        visibility = shader.load(self.visibility, site, 0).x if self.use_visibility else 0.0

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
            guide = shader.load(self.guide, pt / 2, 0)
            if shader.load(self.state, at, 0).y <= 0.0:
                continue
            normal = sh_guide_normal(guide.y)
            other_vis = shader.load(self.visibility, pt, 0).x if self.use_visibility else 0.0
            weight = sh_bilateral(depth, shader.as_float(guide.x), facing, visibility, other_vis)
            weight *= sh_normal_weight(n, normal)
            if mature:
                weight *= shader.exp2(-shader.float(offset.x * offset.x + offset.y * offset.y))
            red += shader.load(self.red_sh, at, 0) * weight
            green += shader.load(self.green_sh, at, 0) * weight
            blue += shader.load(self.blue_sh, at, 0) * weight
            total += weight
        norm = 1.0 / shader.max(total, 1.0e-10)
        return (red * norm, green * norm, blue * norm)
