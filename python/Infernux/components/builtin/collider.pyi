from __future__ import annotations

from enum import IntEnum
from typing import Any

from Infernux.components.builtin_component import BuiltinComponent
from Infernux.core.asset_ref import PhysicMaterialRef

class PhysicsMaterialCombine(IntEnum):
    Average = 0
    Minimum = 1
    Multiply = 2
    Maximum = 3

class Collider(BuiltinComponent):
    """Base class for all collider components."""

    _cpp_type_name: str
    _component_category_: str
    _always_show: bool

    # ---- CppProperty fields as properties ----

    @property
    def center(self) -> Any:
        """The center of the collider in local space."""
        ...
    @center.setter
    def center(self, value: Any) -> None: ...

    @property
    def is_trigger(self) -> bool:
        """Whether the collider is a trigger (non-physical)."""
        ...
    @is_trigger.setter
    def is_trigger(self, value: bool) -> None: ...

    physic_material: PhysicMaterialRef
    def raycast(self, origin: Any, direction: Any, max_distance: float = 1000.0) -> Any | None: ...
    def closest_point(self, point: Any) -> Any: ...
