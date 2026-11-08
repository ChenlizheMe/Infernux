"""Type stubs for infernux.renderstack.film_grain_effect."""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar, List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class FilmGrainSize(IntEnum):
    Small = 0
    Medium = 1
    Large = 2
    Custom = 3


class FilmGrainEffect(FullScreenEffect):
    """URP-aligned Film Grain post-processing effect.

    Adds cinematic noise overlay.  Operates in LDR space (after tone mapping).

    Attributes:
        intensity: Grain strength (0 = off, 1 = heavy).
        response: Luminance response (0 = uniform, 1 = highlights only).
    """

    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]

    intensity: float
    response: float
    preset: FilmGrainSize
    size: float
    colored: bool

    def get_shader_list(self) -> List[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
