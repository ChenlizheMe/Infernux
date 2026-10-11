"""Type stubs for infernux.renderstack.white_balance_effect."""

from __future__ import annotations

from typing import ClassVar, List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class WhiteBalanceEffect(FullScreenEffect):
    """URP-aligned White Balance post-processing effect.

    Color temperature and tint adjustment using Bradford chromatic adaptation.

    Attributes:
        temperature: Warm/cool shift (-100 to 100, 0 = neutral).
        tint: Green/magenta shift (-100 to 100, 0 = neutral).
    """

    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]

    temperature: float
    tint: float

    def get_shader_list(self) -> List[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
