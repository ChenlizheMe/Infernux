import infernux.shader as shader


class RayHitDistance(shader.fullscreen):
    shader_id = "infernux/internal/ray_hit_distance"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        packet = shader.load(self.signal, shader.ivec2(ctx.position.xy), 0)
        return shader.vec4(shader.as_float(packet.w), 0.0, 0.0, 1.0)
