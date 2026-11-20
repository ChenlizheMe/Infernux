import infernux.shader as shader
from infernux.shader.lib.ray_moments import sh_diffuse_irradiance
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_blue_noise, sh_guide_cell, sh_resolve_offset
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseResolve(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_resolve"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    guide = shader.input("guide", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    blue_noise = shader.input("blue_noise", shader.Color2D)
    frame = shader.input("frame", shader.Color2D)
    divisor = shader.parameter.int(default=2)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.red_sh, 0))
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        depth = shader.as_float(receiver.y)
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        cell = shader.min(shader.ivec2(pixel / self.divisor), extent - 1)
        index = shader._view_sample_index()
        pattern = shader.uint(3.0 * sh_blue_noise(self.blue_noise, pixel, index).x + 0.5)
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
            guide_cell = sh_guide_cell(at, extent, full)
            guide_depth = shader.as_float(shader.load(self.guide, guide_cell, 0).x)
            weight = sh_bilateral(depth, guide_depth, 1.0, 0.0, 0.0)
            red += weight * shader.load(self.red_sh, at, 0)
            green += weight * shader.load(self.green_sh, at, 0)
            blue += weight * shader.load(self.blue_sh, at, 0)
            total += weight
        norm = 1.0 / shader.max(total, 1.0e-4)
        return shader.vec4(sh_diffuse_irradiance(red * norm, green * norm, blue * norm, shading), 1.0)
