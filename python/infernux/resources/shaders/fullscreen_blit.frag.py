"""Resample a graph image with its declared sampler."""

import infernux.shader as shader


class FullscreenBlit(shader.fullscreen):
    source_tex = shader.input("source", shader.Color2D)

    def main(self, uv: shader.vec2) -> shader.vec4:
        return shader.sample(self.source_tex, uv)
