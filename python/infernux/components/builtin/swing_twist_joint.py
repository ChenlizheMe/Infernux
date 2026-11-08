"""Native Jolt cone/pyramid swing and twist joint; all public angles use degrees."""
from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.lib import JointMotorMode, JointSwingType
from ._joint_fields import joint_fields


class SwingTwistJoint(BuiltinComponent):
    _cpp_type_name = "SwingTwistJoint"
    _component_category_ = "Physics"
    anchor, connected_body, enable_collision = joint_fields("SwingTwistJoint")
    twist_axis = CppProperty.from_native("SwingTwistJoint", "twist_axis")
    plane_axis = CppProperty.from_native("SwingTwistJoint", "plane_axis")
    swing_type = CppProperty.from_native("SwingTwistJoint", "swing_type")
    normal_half_cone_angle = CppProperty.from_native("SwingTwistJoint", "normal_half_cone_angle")
    plane_half_cone_angle = CppProperty.from_native("SwingTwistJoint", "plane_half_cone_angle")
    minimum_twist_angle = CppProperty.from_native("SwingTwistJoint", "minimum_twist_angle")
    maximum_twist_angle = CppProperty.from_native("SwingTwistJoint", "maximum_twist_angle")
    friction_torque = CppProperty.from_native("SwingTwistJoint", "friction_torque")
    swing_motor_mode = CppProperty.from_native("SwingTwistJoint", "swing_motor_mode")
    twist_motor_mode = CppProperty.from_native("SwingTwistJoint", "twist_motor_mode")
    swing_motor_stiffness = CppProperty.from_native("SwingTwistJoint", "swing_motor_stiffness")
    swing_motor_damping = CppProperty.from_native("SwingTwistJoint", "swing_motor_damping")
    swing_motor_max_torque = CppProperty.from_native("SwingTwistJoint", "swing_motor_max_torque")
    twist_motor_stiffness = CppProperty.from_native("SwingTwistJoint", "twist_motor_stiffness")
    twist_motor_damping = CppProperty.from_native("SwingTwistJoint", "twist_motor_damping")
    twist_motor_max_torque = CppProperty.from_native("SwingTwistJoint", "twist_motor_max_torque")
    target_orientation = CppProperty.from_native("SwingTwistJoint", "target_orientation")
    target_angular_velocity = CppProperty.from_native("SwingTwistJoint", "target_angular_velocity")

    @property
    def current_orientation(self):
        return self._require_cpp_component().current_orientation

    def set_frame(self, twist_axis, plane_axis):
        """Set local X (twist) and Z (plane) together; axes are orthonormalized."""
        self._require_cpp_component().set_frame(twist_axis, plane_axis)

    def set_twist_limits(self, minimum: float, maximum: float):
        self._require_cpp_component().set_twist_limits(minimum, maximum)
