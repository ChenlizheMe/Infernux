import infernux.shader as shader


class PreviewDisplay(shader.fullscreen):
    shader_id = "infernux/internal/preview_display"
    hidden = True
    source_tex = shader.input("color", shader.Color2D)
    sphere_radius = shader.parameter.float(default=0.0)
    hdr = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        # Producers render at twice the display size. Filter mapped, linear,
        # alpha-weighted samples; the sRGB output attachment performs encoding.
        origin = shader.ivec2(ctx.position.xy) * 2
        source_size = shader.f32(shader.dimensions(self.source_tex, 0).x)
        premultiplied = shader.vec3(0.0)
        alpha_sum = 0.0
        for y in range(2):
            for x in range(2):
                pixel = origin + shader.ivec2(x, y)
                source = shader.load(self.source_tex, pixel, 0)
                alpha = shader.saturate(source.a)
                color = shader.max(source.rgb, shader.vec3(0.0))
                if self.sphere_radius > 0.0:
                    distance = shader.length((shader.vec2(pixel) + 0.5) / source_size - 0.5)
                    feather = 1.5 / source_size
                    alpha *= shader.saturate((self.sphere_radius + feather - distance) / (2.0 * feather))
                    # Sphere previews draw onto transparent black. Undo the
                    # matte before mapping; the GUI consumes straight alpha.
                    if alpha > 0.0 and alpha < 0.999:
                        color /= alpha
                if self.hdr:
                    color = color / (1.0 + color)
                premultiplied += color * alpha
                alpha_sum += alpha
        if alpha_sum > 0.0:
            return shader.vec4(premultiplied / alpha_sum, alpha_sum * 0.25)
        return shader.vec4(0.0)
