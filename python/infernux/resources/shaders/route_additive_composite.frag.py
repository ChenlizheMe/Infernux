import infernux.shader as shader


class RouteAdditiveComposite(shader.fullscreen):
    shader_id = "infernux/internal/route_additive_composite"
    hidden = True
    base_tex = shader.input("base", shader.Color2D)
    additive_tex = shader.input("additive", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        base = shader.load(self.base_tex, pixel, 0)
        additive = shader.load(self.additive_tex, pixel, 0).rgb
        return shader.vec4(base.rgb + additive, base.a)
