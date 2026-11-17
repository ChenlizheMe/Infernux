import infernux.shader as shader
from infernux.shader.lib.surface import decode_octahedral_normal


class RayPrimarySurface(shader.fullscreen):
    shader_id = "infernux/internal/ray_primary_surface"
    hidden = True
    gbuffer_surface = shader.input("gbuffer_surface", shader.UInt2D)
    gbuffer_depth = shader.input("gbuffer_depth", shader.Depth2D)
    replay_surface = shader.input("replay_surface", shader.UInt2D)
    replay_depth = shader.input("replay_depth", shader.Depth2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        surface = shader.load(self.gbuffer_surface, pixel, 0).x
        replay = shader.load(self.replay_surface, pixel, 0)
        depth = shader.load(self.gbuffer_depth, pixel, 0)
        replay_depth = shader.load(self.replay_depth, pixel, 0)
        if surface == 0 or replay.x != surface or replay_depth != depth:
            return shader.vec4(0.0)
        octahedral = shader.vec2(
            shader.f16_to_f32(replay.y & shader.uint(65535)), shader.f16_to_f32(replay.y >> 16),
        )
        return shader.vec4(decode_octahedral_normal(octahedral), 1.0)
