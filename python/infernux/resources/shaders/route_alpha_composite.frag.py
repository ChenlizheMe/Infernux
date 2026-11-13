import infernux.shader as shader


class RouteAlphaComposite(shader.fullscreen):
    shader_id = "infernux/internal/route_alpha_composite"
    hidden = True
    base_tex = shader.input("base", shader.Color2D)
    layer_tex = shader.input("layer", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        base = shader.load(self.base_tex, pixel, 0)
        layer = shader.load(self.layer_tex, pixel, 0)
        # Route targets already contain premultiplied RGB, including coverage.
        return layer + base * (1.0 - layer.a)
