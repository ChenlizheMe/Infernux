import infernux.shader as shader


class RayTailEstimate(shader.fullscreen):
    shader_id = "infernux/internal/ray_tail_estimate"
    hidden = True
    requests = shader.input("ray_tail_requests", shader.Buffer[shader.uvec4])
    cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        if shader.uint(ctx.position.x) >= self.requests[0].y:
            return shader.vec4(0.0, 0.0, 0.0, -1.0)
        return shader.ray._tail_estimate(self.requests, self.cache, self.surface_records,
                                          shader.uint(ctx.position.x))
