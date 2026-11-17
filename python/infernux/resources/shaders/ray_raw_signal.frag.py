import infernux.shader as shader
from infernux.shader.lib.ray_signal import unpack_ray_signal


class RayRawSignal(shader.fullscreen):
    shader_id = "infernux/internal/ray_raw_signal"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        return unpack_ray_signal(shader.load(self.signal, shader.ivec2(ctx.position.xy), 0))
