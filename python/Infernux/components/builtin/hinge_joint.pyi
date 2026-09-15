from __future__ import annotations

from typing import Any, Optional
from Infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class HingeJoint(BuiltinComponent):
    """A single-axis Rigidbody hinge connected to another body or the world."""
    _cpp_type_name: str
    anchor: Any
    axis: Any
    connected_body: Optional[Rigidbody]
    use_limits: bool
    minimum_angle: float
    maximum_angle: float
    enable_collision: bool
    @property
    def current_angle(self) -> float: ...
    def on_draw_gizmos_selected(self) -> None: ...
