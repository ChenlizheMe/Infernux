"""Preserve texels exactly when the graph source and destination have equal extents."""

import infernux.shader as shader


class FullscreenCopy(shader.fullscreen):
    source_tex = shader.input("source", shader.Color2D)

    def main(self, fragment: shader.fragment_input) -> shader.vec4:
        return shader.load(self.source_tex, shader.ivec2(fragment.position.xy), 0)
