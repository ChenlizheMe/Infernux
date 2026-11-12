"""Production pixelation for full images and isolated render-queue routes."""

from __future__ import annotations

from enum import IntEnum

from infernux.components.fields import serialized_field
from infernux.renderstack.fullscreen_effect import EffectColorComposition, FullScreenEffect


class PixelationSampling(IntEnum):
    """Quality and alpha reconstruction mode used inside each pixel cell."""

    CENTER = 0
    BALANCED = 1
    HIGH_QUALITY = 2


class PixelationEffect(FullScreenEffect):
    """Screen-aligned, resolution-independent pixelation.

    The balanced and high-quality modes reconstruct color from alpha-weighted
    samples and optionally retain the strongest coverage in each cell. This
    keeps isolated objects stable instead of allowing thin silhouettes to
    disappear when the center of a large pixel lands on transparent space.
    """

    name = "Pixelation"
    injection_point = "before_post_process"
    default_order = 300
    menu_path = "Stylized/Pixelation"
    color_composition = EffectColorComposition.REPLACE

    pixel_size: int = serialized_field(
        default=8,
        range=(1, 256),
        slider=False,
        tooltip="Height of one output pixel cell in screen pixels",
        header="Pixel Grid",
    )
    pixel_aspect: float = serialized_field(
        default=1.0,
        range=(0.25, 4.0),
        slider=False,
        drag_speed=0.01,
        tooltip="Cell width divided by cell height",
    )
    grid_offset_x: float = serialized_field(
        default=0.0,
        range=(-1.0, 1.0),
        slider=False,
        drag_speed=0.01,
        tooltip="Horizontal grid offset measured in pixel cells",
    )
    grid_offset_y: float = serialized_field(
        default=0.0,
        range=(-1.0, 1.0),
        slider=False,
        drag_speed=0.01,
        tooltip="Vertical grid offset measured in pixel cells",
    )
    intensity: float = serialized_field(
        default=1.0,
        range=(0.0, 1.0),
        slider=False,
        tooltip="Move sampling toward each pixel-cell center without layering the original image",
        header="Reconstruction",
    )
    sampling: PixelationSampling = serialized_field(
        default=PixelationSampling.BALANCED,
        enum_labels=["Center (Fast)", "Alpha-aware 4 Tap", "Alpha-aware 9 Tap"],
        tooltip="Higher modes preserve color and silhouettes inside large cells",
    )
    preserve_alpha_coverage: bool = serialized_field(
        default=True,
        tooltip="Keep thin isolated geometry visible across an occupied pixel cell",
    )

    def get_shader_list(self):
        return ["infernux/post/pixelation"]

    def setup_passes(self, graph, bus) -> None:
        from infernux.rendergraph.graph import Format

        self.apply_single_source_effect(
            graph,
            bus,
            output_name="_pixelation_out",
            pass_name="Pixelation_Apply",
            shader_name="infernux/post/pixelation",
            format=Format.RGBA16_SFLOAT,
            params={
                "pixel_size": self.pixel_size,
                "pixel_aspect": self.pixel_aspect,
                "grid_offset": (self.grid_offset_x, self.grid_offset_y),
                "intensity": self.intensity,
                "sampling_mode": int(PixelationSampling(self.sampling)),
                "preserve_alpha_coverage": self.preserve_alpha_coverage,
            },
        )


__all__ = ["PixelationEffect", "PixelationSampling"]
