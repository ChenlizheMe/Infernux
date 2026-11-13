"""Type stubs for infernux.renderstack.tonemapping_effect — HDR-to-LDR tone mapping."""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar, List, TYPE_CHECKING

from infernux.renderstack.fullscreen_effect import FullScreenEffect

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class ToneMappingMode(IntEnum):
    """Tone mapping operator selection."""
    None_ = 0
    """Clamp to the display-linear range without applying a tone curve."""
    Reinhard = 1
    """Reinhard operator (simple, soft rolloff)."""
    ACES = 2
    """ACES Filmic (default — matches Unity/Unreal look)."""


class ToneMappingEffect(FullScreenEffect):
    """HDR-to-LDR tone mapping post-processing effect.

    Should be the last effect in the post-process chain so that bloom
    and other HDR effects can operate on the full dynamic range.

    Attributes:
        mode: Tone mapping operator (ACES is recommended).
        exposure: Pre-tonemap exposure multiplier (default 1.0).
    """

    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]

    mode: ToneMappingMode
    exposure: float

    def get_shader_list(self) -> List[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
