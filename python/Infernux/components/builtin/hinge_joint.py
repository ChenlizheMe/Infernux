"""Single-axis physics hinge backed by the engine's Jolt world."""

from __future__ import annotations

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
from Infernux.components.fields import FieldType


def _to_native_rigidbody(value):
    if value is None:
        return None
    getter = getattr(value, "_require_cpp_component", None)
    return getter() if getter is not None else value


def _from_native_rigidbody(value):
    if value is None:
        return None
    from .rigidbody import Rigidbody

    return Rigidbody._get_or_create_wrapper(value, value.game_object)


class HingeJoint(BuiltinComponent):
    """Constrain a Rigidbody to rotate around one local axis."""

    _cpp_type_name = "HingeJoint"
    _component_category_ = "Physics"

    anchor = CppProperty.from_native("HingeJoint", "anchor")
    axis = CppProperty.from_native("HingeJoint", "axis")
    connected_body = CppProperty.from_native(
        "HingeJoint", "connected_body",
        get_converter=_from_native_rigidbody,
        set_converter=_to_native_rigidbody,
    )
    connected_body.metadata.component_type = "Rigidbody"
    use_limits = CppProperty.from_native("HingeJoint", "use_limits")
    minimum_angle = CppProperty.from_native("HingeJoint", "minimum_angle")
    maximum_angle = CppProperty.from_native("HingeJoint", "maximum_angle")
    enable_collision = CppProperty.from_native("HingeJoint", "enable_collision")
    current_angle = CppProperty(
        "current_angle", FieldType.FLOAT, default=0.0, readonly=True,
        tooltip="Current hinge angle in degrees",
    )

    def on_draw_gizmos_selected(self):
        from Infernux.gizmos import Gizmos

        transform = self.transform
        if transform is None:
            return
        old_matrix = Gizmos.matrix
        old_color = Gizmos.color
        Gizmos.matrix = transform.local_to_world_matrix()
        Gizmos.color = (1.0, 0.72, 0.2)
        anchor = self.anchor
        axis = self.axis
        Gizmos.draw_wire_sphere((anchor.x, anchor.y, anchor.z), 0.08)
        Gizmos.draw_line(
            (anchor.x - axis.x * 0.5, anchor.y - axis.y * 0.5, anchor.z - axis.z * 0.5),
            (anchor.x + axis.x * 0.5, anchor.y + axis.y * 0.5, anchor.z + axis.z * 0.5),
        )
        Gizmos.color = old_color
        Gizmos.matrix = old_matrix
