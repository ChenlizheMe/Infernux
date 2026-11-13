import infernux.shader as shader


class RouteCoverageSplit(shader.fullscreen):
    shader_id = "infernux/internal/route_coverage_split"
    hidden = True
    original_tex = shader.input("original", shader.Color2D)
    processed_tex = shader.input("processed", shader.Color2D)
    outside = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        geometry = shader.load(self.original_tex, pixel, 0).a > 0.0
        # Split whole premultiplied pixels without multiplying MSAA coverage again.
        if geometry != self.outside:
            return shader.load(self.processed_tex, pixel, 0)
        return shader.vec4(0.0)
