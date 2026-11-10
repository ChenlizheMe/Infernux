"""A native Jolt weld that preserves two bodies' relative pose."""
from infernux.components.builtin_component import BuiltinComponent
from ._joint_fields import joint_fields


class FixedJoint(BuiltinComponent):
    _cpp_type_name = "FixedJoint"
    _component_category_ = "Physics"
    anchor, connected_body, enable_collision = joint_fields("FixedJoint")
