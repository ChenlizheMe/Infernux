import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_clip_history, sh_guide_pixel
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseFill(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_fill"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    state = shader.input("state", shader.Color2D)
    old_state = shader.input("old_state", shader.History2D)
    primary = shader.input("primary", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        state = shader.load(self.state, cell, 0)
        if state.x > 0.0 or state.y <= 0.0 or not self.old_state.valid:
            return (shader.load(self.red_sh, cell, 0), shader.load(self.green_sh, cell, 0),
                    shader.load(self.blue_sh, cell, 0), state)
        extent = shader.ivec2(shader.dimensions(self.state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        site = sh_guide_pixel(cell, extent, full, shader._view_sample_index())
        surface = shader.load(self.primary, site, 0)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(surface.z))
        depth = shader.as_float(surface.y)
        view_normal = (shader.camera().view_matrix @ shader.vec4(normal, 0.0)).xyz
        facing = shader.max(0.3, view_normal.z)

        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        weighted_samples = 0.0
        weights = 0.0
        for i in range(25):
            at = cell + 2 * shader.ivec2(i % 5 - 2, int(i / 5) - 2)
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            neighbor = shader.load(self.state, at, 0).xy
            if neighbor.y <= 0.0:
                continue
            similarity = shader.exp2(-shader.min(30.0, 10.0 * facing * shader.abs(1.0 - neighbor.y / depth)))
            weight = similarity * shader.max(neighbor.x, 1.0)
            red += shader.load(self.red_sh, at, 0) * weight
            green += shader.load(self.green_sh, at, 0) * weight
            blue += shader.load(self.blue_sh, at, 0) * weight
            weighted_samples += weight
            weights += similarity
        age = shader.min(weighted_samples / shader.max(weights, 1.0e-20), 24.0)
        norm = 1.0 / shader.max(weighted_samples, 1.0e-20)
        return (sh_clip_history(red * norm), sh_clip_history(green * norm), sh_clip_history(blue * norm),
                shader.vec4(age, state.y, 0.0, 0.0))
