from typing import Optional
from infernux.lib import Vector3
from infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class DistanceJoint(BuiltinComponent):
    anchor: Vector3
    connected_body: Optional[Rigidbody]
    enable_collision: bool
    connected_anchor: Vector3
    auto_distance: bool
    minimum_distance: float
    maximum_distance: float
    stiffness: float
    damping: float
    @property
    def current_distance(self) -> float: ...
    def set_limits(self, minimum: float, maximum: float) -> None: ...
