from typing import Optional
from infernux.lib import Vector3
from infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class FixedJoint(BuiltinComponent):
    anchor: Vector3
    connected_body: Optional[Rigidbody]
    enable_collision: bool
