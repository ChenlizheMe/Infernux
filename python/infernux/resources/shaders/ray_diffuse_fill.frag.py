import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_clip_history, sh_facing, sh_guide_cell, sh_guide_normal


class RayDiffuseFill(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_fill"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    state = shader.input("state", shader.Color2D)
    old_state = shader.input("old_state", shader.History2D)
    primary = shader.input("primary", shader.UInt2D)
    guide = shader.input("guide", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4, shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        state = shader.load(self.state, cell, 0)
        if state.x > 0.0 or state.y <= 0.0 or not self.old_state.valid:
            return (shader.load(self.red_sh, cell, 0), shader.load(self.green_sh, cell, 0),
                    shader.load(self.blue_sh, cell, 0), state)
        extent = shader.ivec2(shader.dimensions(self.state, 0))
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        geometry = shader.load(self.guide, sh_guide_cell(cell, extent, full), 0)
        depth = shader.as_float(geometry.x)
        facing = sh_facing(sh_guide_normal(geometry.y))

        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        weighted_samples = 0.0
        weights = 0.0
        for i in range(25):
            at = cell + 2 * shader.ivec2(i % 5 - 2, int(i / 5) - 2)
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            neighbor_age = shader.load(self.state, at, 0).x
            neighbor_depth = shader.as_float(shader.load(self.guide, sh_guide_cell(at, extent, full), 0).x)
            similarity = shader.exp2(-shader.min(30.0, 10.0 * facing * shader.abs(1.0 - neighbor_depth / depth)))
            weight = similarity * shader.max(neighbor_age, 1.0)
            red += shader.load(self.red_sh, at, 0) * weight
            green += shader.load(self.green_sh, at, 0) * weight
            blue += shader.load(self.blue_sh, at, 0) * weight
            weighted_samples += weight
            weights += similarity
        age = shader.min(weighted_samples / shader.max(weights, 1.0e-20), 24.0)
        norm = 1.0 / shader.max(weighted_samples, 1.0e-20)
        return (sh_clip_history(red * norm), sh_clip_history(green * norm), sh_clip_history(blue * norm),
                shader.vec4(age, state.y, 0.0, 0.0))
