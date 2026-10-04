"""Type stubs for infernux.renderstack.color_adjustments_effect."""

from __future__ import annotations

from typing import ClassVar, List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class ColorAdjustmentsEffect(FullScreenEffect):
    """URP-aligned Color Adjustments post-processing effect.

    Post-exposure, contrast, saturation, hue shift — operates in HDR space.

    Attributes:
        post_exposure: Exposure offset in EV stops (default 0.0).
        contrast: Contrast adjustment (-100 to 100).
        saturation: Saturation adjustment (-100 to 100).
        hue_shift: Hue rotation in degrees (-180 to 180).
    """

    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]

    post_exposure: float
    contrast: float
    saturation: float
    hue_shift: float

    def get_shader_list(self) -> List[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
