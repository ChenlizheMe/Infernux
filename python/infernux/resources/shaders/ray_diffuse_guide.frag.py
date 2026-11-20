import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_guide_phase
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseGuide(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_guide"
    hidden = True
    depth = shader.input("depth", shader.Depth2D)
    frame = shader.input("surface_frame", shader.Color2D)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.depth, 0))
        index = shader._view_sample_index() if self.temporal else shader.uint(0)
        phase = sh_guide_phase(index)
        pixel = shader.min(cell * 2 + phase, full - 1)
        z = shader.as_uint(shader.load(self.depth, pixel, 0))
        normal = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        view = (shader.camera().view_matrix @ shader.vec4(normal, 0.0)).xyz
        bits = shader.uvec3(shader.ivec3(shader.round(shader.clamp(view, -1.0, 1.0) * 127.0))) & shader.uvec3(255)
        word = bits.x | (bits.y << 8) | (bits.z << 16)
        return shader.uvec4(z, word, 0, 0)
