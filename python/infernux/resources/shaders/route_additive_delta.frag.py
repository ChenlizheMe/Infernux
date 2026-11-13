import infernux.shader as shader


class RouteAdditiveDelta(shader.fullscreen):
    shader_id = "infernux/internal/route_additive_delta"
    hidden = True
    original_tex = shader.input("original", shader.Color2D)
    processed_tex = shader.input("processed", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        original = shader.load(self.original_tex, pixel, 0).rgb
        processed = shader.load(self.processed_tex, pixel, 0).rgb
        return shader.vec4(shader.max(processed - original, shader.vec3(0.0)), 0.0)
