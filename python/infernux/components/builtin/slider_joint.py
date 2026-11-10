"""Single-axis prismatic joint backed by the engine's Jolt world."""

from __future__ import annotations

from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.components.fields import FieldType
from ._joint_fields import joint_fields


class SliderJoint(BuiltinComponent):
    """Allow one-axis translation while locking rotation and other axes."""

    _cpp_type_name = "SliderJoint"
    _component_category_ = "Physics"

    anchor, connected_body, enable_collision = joint_fields("SliderJoint")
    axis = CppProperty.from_native("SliderJoint", "axis")
    use_limits = CppProperty.from_native("SliderJoint", "use_limits")
    minimum_distance = CppProperty.from_native("SliderJoint", "minimum_distance")
    maximum_distance = CppProperty.from_native("SliderJoint", "maximum_distance")
    current_position = CppProperty(
        "current_position", FieldType.FLOAT, default=0.0, readonly=True,
        tooltip="Current position along the slider axis in metres",
    )

    def on_draw_gizmos_selected(self):
        from infernux.gizmos import Gizmos

        transform = self.transform
        if transform is None:
            return
        old_matrix = Gizmos.matrix
        old_color = Gizmos.color
        Gizmos.matrix = transform.local_to_world_matrix()
        Gizmos.color = (0.25, 0.78, 1.0)
        anchor = self.anchor
        axis = self.axis
        minimum = self.minimum_distance if self.use_limits else -1.0
        maximum = self.maximum_distance if self.use_limits else 1.0
        Gizmos.draw_wire_sphere((anchor.x, anchor.y, anchor.z), 0.08)
        Gizmos.draw_line(
            (anchor.x + axis.x * minimum, anchor.y + axis.y * minimum, anchor.z + axis.z * minimum),
            (anchor.x + axis.x * maximum, anchor.y + axis.y * maximum, anchor.z + axis.z * maximum),
        )
        Gizmos.color = old_color
        Gizmos.matrix = old_matrix
