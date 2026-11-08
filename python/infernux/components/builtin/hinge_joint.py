"""Single-axis physics hinge backed by the engine's Jolt world."""

from __future__ import annotations

from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.components.fields import FieldType
from ._joint_fields import joint_fields


class HingeJoint(BuiltinComponent):
    """Constrain a Rigidbody to rotate around one local axis."""

    _cpp_type_name = "HingeJoint"
    _component_category_ = "Physics"

    anchor, connected_body, enable_collision = joint_fields("HingeJoint")
    axis = CppProperty.from_native("HingeJoint", "axis")
    use_limits = CppProperty.from_native("HingeJoint", "use_limits")
    minimum_angle = CppProperty.from_native("HingeJoint", "minimum_angle")
    maximum_angle = CppProperty.from_native("HingeJoint", "maximum_angle")
    motor_enabled = CppProperty.from_native("HingeJoint", "motor_enabled")
    motor_target_angle = CppProperty.from_native("HingeJoint", "motor_target_angle")
    motor_stiffness = CppProperty.from_native("HingeJoint", "motor_stiffness")
    motor_damping = CppProperty.from_native("HingeJoint", "motor_damping")
    motor_max_torque = CppProperty.from_native("HingeJoint", "motor_max_torque")
    joint_damping = CppProperty.from_native("HingeJoint", "joint_damping")
    friction_torque = CppProperty.from_native("HingeJoint", "friction_torque")
    armature = CppProperty.from_native("HingeJoint", "armature")
    current_angle = CppProperty(
        "current_angle", FieldType.FLOAT, default=0.0, readonly=True,
        tooltip="Current hinge angle in degrees",
    )
    current_angular_velocity = CppProperty(
        "current_angular_velocity", FieldType.FLOAT, default=0.0, readonly=True,
        tooltip="Relative angular velocity about the hinge axis in degrees per second",
    )

    def on_draw_gizmos_selected(self):
        from infernux.gizmos import Gizmos

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
