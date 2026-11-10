"""Native distance range or spring between two attachment points."""
from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.components.fields import FieldType
from ._joint_fields import joint_fields


class DistanceJoint(BuiltinComponent):
    _cpp_type_name = "DistanceJoint"
    _component_category_ = "Physics"
    anchor, connected_body, enable_collision = joint_fields("DistanceJoint")
    connected_anchor = CppProperty.from_native("DistanceJoint", "connected_anchor")
    auto_distance = CppProperty.from_native("DistanceJoint", "auto_distance")
    minimum_distance = CppProperty.from_native("DistanceJoint", "minimum_distance")
    maximum_distance = CppProperty.from_native("DistanceJoint", "maximum_distance")
    stiffness = CppProperty.from_native("DistanceJoint", "stiffness")
    damping = CppProperty.from_native("DistanceJoint", "damping")
    current_distance = CppProperty("current_distance", FieldType.FLOAT, default=0.0, readonly=True,
                                   tooltip="Current attachment-point separation in metres")

    def set_limits(self, minimum: float, maximum: float) -> None:
        """Set both non-negative metre limits in one operation."""
        self._require_cpp_component().set_limits(minimum, maximum)
