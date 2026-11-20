import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_guide_pixel


class RayDiffuseScreenDepth(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_screen_depth"
    hidden = True
    depth = shader.input("depth", shader.Depth2D)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.depth, 0))
        extent = shader.max(shader.ivec2(full / 2), shader.ivec2(1))
        index = shader._view_sample_index() if self.temporal else shader.uint(0)
        pixel = sh_guide_pixel(cell, extent, full, index)
        return shader.vec4(shader.load(self.depth, pixel, 0), 0.0, 0.0, 1.0)
