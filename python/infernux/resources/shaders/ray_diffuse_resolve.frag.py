import infernux.shader as shader
from infernux.shader.lib.noise import hash_2d
from infernux.shader.lib.ray_moments import sh_diffuse_irradiance
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.ray_sh_filter import sh_bilateral, sh_guide_pixel, sh_resolve_offset
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseResolve(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_resolve"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    state = shader.input("state", shader.Color2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.state, 0))
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        depth = shader.as_float(receiver.y)
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        cell = ray_grid_pixel(pixel, full, extent)
        index = shader._view_sample_index()
        seed = shader.vec2(pixel) + shader.float(index & shader.uint(255))
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
            source = sh_guide_pixel(at, extent, full, index)
            guide = shader.load(self.primary, source, 0)
            if shader.load(self.state, at, 0).y <= 0.0:
                continue
            weight = sh_bilateral(depth, shader.as_float(guide.y), 1.0, 0.0, 0.0)
            red += weight * shader.load(self.red_sh, at, 0)
            green += weight * shader.load(self.green_sh, at, 0)
            blue += weight * shader.load(self.blue_sh, at, 0)
            total += weight
        if total <= 0.0:
            return shader.vec4(0.0)
        norm = 1.0 / shader.max(total, 1.0e-4)
        return shader.vec4(sh_diffuse_irradiance(red * norm, green * norm, blue * norm, shading), 1.0)
