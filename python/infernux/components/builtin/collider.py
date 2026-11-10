"""
Collider — Abstract base class for all collider BuiltinComponent wrappers.

Mirrors Unity's ``Collider`` base class. Concrete subclasses
(``BoxCollider``, ``SphereCollider``, ``CapsuleCollider``, ``CylinderCollider``)
inherit
the shared ``center``, ``is_trigger`` properties and category.

Example::

    from infernux.components.builtin import BoxCollider

    class MyScript(InxComponent):
        def start(self):
            col = self.game_object.get_component(BoxCollider)
            if isinstance(col, Collider):
                print("It's a collider!")
"""

from __future__ import annotations

from enum import IntEnum

from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.components.fields import FieldType
from infernux.core.asset_ref import PhysicMaterialRef


class PhysicsMaterialCombine(IntEnum):
    """How two collider material values are combined for a contact."""

    Average = 0
    Minimum = 1
    Multiply = 2
    Maximum = 3


def _wrap_physic_material_state(state) -> PhysicMaterialRef:
    guid, native, instance = state
    ref = PhysicMaterialRef(guid=str(guid or ""))
    ref._is_instance = instance
    if native is not None:
        from infernux.core.physic_material import PhysicMaterial

        ref._cached = PhysicMaterial(native)
    return ref


def _set_native_physic_material(cpp, value) -> None:
    instance = False
    if isinstance(value, PhysicMaterialRef):
        if value.guid:
            cpp.physic_material_guid = str(value.guid)
            return
        if value._cached is not None:
            instance = value._is_instance and not value.guid
            value = value._cached
        else:
            value = None
    from infernux.core.physic_material import PhysicMaterial
    if isinstance(value, PhysicMaterial):
        value = value.native
    from infernux.lib import InxPhysicMaterial as NativePhysicMaterial
    if value is None or isinstance(value, NativePhysicMaterial):
        cpp._set_physic_material_binding(value, instance)
        return
    raise TypeError("physic_material must be PhysicMaterial, PhysicMaterialRef, or None")


def _native_collider_properties(type_name: str):
    """Project one concrete collider's shared native declaration."""
    return (
        CppProperty.from_native(type_name, "center"),
        CppProperty.from_native(type_name, "is_trigger"),
        CppProperty.from_native(
            type_name,
            "physic_material",
            native_getter=lambda cpp: (cpp.physic_material_guid, cpp.physic_material, cpp._material_is_instance),
            get_converter=_wrap_physic_material_state,
            native_setter=_set_native_physic_material,
        ),
    )


class Collider(BuiltinComponent):
    """Abstract Python base for all collider wrappers (mirrors Unity's Collider).

    Subclasses must still set ``_cpp_type_name`` to their concrete C++ type
    (e.g. ``"BoxCollider"``).  This class itself is **not** registered in
    ``_builtin_registry`` because ``_cpp_type_name`` is left empty.
    """

    # Not a concrete component — don't register
    _cpp_type_name = ""

    _component_category_ = "Physics"

    @property
    def attached_rigidbody(self):
        """The Rigidbody owning this shape, including a Rigidbody on a parent."""
        from infernux.lib import _wrap_native_builtin_component
        return _wrap_native_builtin_component(self._require_cpp_component().attached_rigidbody)

    # Always-draw flag inherited by subclasses
    _always_show = False

    # ---- Shared properties (common to all collider types) ----
    center = CppProperty(
        "center",
        FieldType.VEC3,
        default=None,
        tooltip="Center offset in local space",
    )
    is_trigger = CppProperty(
        "is_trigger",
        FieldType.BOOL,
        default=False,
        tooltip="Is this collider a trigger volume?",
    )
    physic_material = CppProperty(
        "physic_material",
        FieldType.ASSET,
        default=PhysicMaterialRef(),
        asset_type="PhysicMaterial",
        tooltip="Shared physics surface material.",
        native_getter=lambda cpp: (cpp.physic_material_guid, cpp.physic_material, cpp._material_is_instance),
        get_converter=_wrap_physic_material_state,
        native_setter=_set_native_physic_material,
    )

    @property
    def material(self):
        """Private surface material, copied from physic_material on first read.

        physic_material is the shared asset entry (Unity sharedMaterial).
        material lets this collider change its surface without changing others.
        Its values remain inside the ordinary Scene/Prefab component document.
        """
        from infernux.core.physic_material import PhysicMaterial
        return PhysicMaterial(self._require_cpp_component().material)

    @material.setter
    def material(self, value):
        _set_native_physic_material(self._require_cpp_component(), value)

    def raycast(self, origin, direction, max_distance: float = 1000.0):
        """Cast a world-space ray against this collider only."""
        return self._require_cpp_component().raycast(origin, direction, max_distance)

    def closest_point(self, point):
        """Return the closest world-space point on this collider."""
        return self._require_cpp_component().closest_point(point)
