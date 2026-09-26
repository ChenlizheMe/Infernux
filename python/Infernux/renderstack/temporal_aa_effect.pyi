"""Temporal anti-aliasing backed by native per-view history."""
from typing import ClassVar
from Infernux.rendergraph.graph import RenderGraph
from Infernux.renderstack.fullscreen_effect import FullScreenEffect
from Infernux.renderstack.resource_bus import ResourceBus


class TemporalAAEffect(FullScreenEffect):
    name: ClassVar[str]
    injection_point: ClassVar[str]
    default_order: ClassVar[int]
    menu_path: ClassVar[str]
    requires: ClassVar[set[str]]
    modifies: ClassVar[set[str]]
    feedback: float
    motion_rejection: float
    depth_rejection: float

    def get_shader_list(self) -> list[str]: ...
    def setup_passes(self, graph: RenderGraph, bus: ResourceBus) -> None: ...
