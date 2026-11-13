import infernux.shader as shader
from infernux.shader.lib.color import srgb_to_linear


class Gizmo(shader.frag):
    shader_id = "infernux/editor/gizmo"
    hidden = True
    cast_shadows = False

    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        return shader.vec4(srgb_to_linear(ctx.color), 1.0)
