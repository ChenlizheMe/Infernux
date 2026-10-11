"""Type stubs for infernux.renderstack.chromatic_aberration_effect."""

from __future__ import annotations

from typing import ClassVar, List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class ChromaticAberrationEffect(FullScreenEffect):
    """URP-aligned Chromatic Aberration post-processing effect.

    Simulates lens imperfection where different wavelengths refract
    at different angles, producing RGB channel separation from screen center.

    Attributes:
        intensity: Channel separation strength (0 = off, 1 = strong).
    """

    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]

    intensity: float

    def get_shader_list(self) -> List[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
