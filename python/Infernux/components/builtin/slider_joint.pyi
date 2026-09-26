from __future__ import annotations

from typing import Any, Optional
from Infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class SliderJoint(BuiltinComponent):
    """A one-axis prismatic joint connected to another body or the world."""
    _cpp_type_name: str
    anchor: Any
    axis: Any
    connected_body: Optional[Rigidbody]
    use_limits: bool
    minimum_distance: float
    maximum_distance: float
    enable_collision: bool
    @property
    def current_position(self) -> float: ...
    def on_draw_gizmos_selected(self) -> None: ...
