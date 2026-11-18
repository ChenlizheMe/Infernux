import infernux.shader as shader


class RayPrimarySurface(shader.fullscreen):
    shader_id = "infernux/internal/ray_primary_surface"
    hidden = True
    gbuffer_surface = shader.input("gbuffer_surface", shader.UInt2D)
    gbuffer_depth = shader.input("gbuffer_depth", shader.Depth2D)
    gbuffer_normal = shader.input("gbuffer_normal", shader.Color2D)
    replay_surface = shader.input("replay_surface", shader.UInt2D)
    replay_depth = shader.input("replay_depth", shader.Depth2D)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        surface = shader.load(self.gbuffer_surface, pixel, 0).x
        replay = shader.load(self.replay_surface, pixel, 0)
        depth = shader.load(self.gbuffer_depth, pixel, 0)
        replay_depth = shader.load(self.replay_depth, pixel, 0)
        if surface == 0 or replay.x != surface or replay_depth != depth:
            return shader.uvec4(0)
        frame = shader.load(self.gbuffer_normal, pixel, 0)
        # Visibility above uses hardware depth. Outgoing rays and their history
        # use the depth of the original triangle at this camera sample.
        return shader.uvec4(surface, replay.w & shader.uint(0x7FFFFFFF), replay.y, shader.as_uint(frame.z))
