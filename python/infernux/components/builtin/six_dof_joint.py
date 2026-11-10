"""Six native Jolt motion axes. Linear distances use metres, angles use degrees."""
from infernux.components.builtin_component import BuiltinComponent, CppProperty
from infernux.lib import JointMotorMode, JointMotion
from ._joint_fields import joint_fields


class SixDOFJoint(BuiltinComponent):
    """Linear X/Y/Z and angular X twist / Y-Z swing in the bound frame.

    Angular bounds form Jolt's swing pyramid, not an Euler-angle box for
    combined rotations. Distances use metres; authored angles use degrees.
    """
    _cpp_type_name = "SixDOFJoint"
    _component_category_ = "Physics"
    anchor, connected_body, enable_collision = joint_fields("SixDOFJoint")
    axis = CppProperty.from_native("SixDOFJoint", "axis")
    secondary_axis = CppProperty.from_native("SixDOFJoint", "secondary_axis")
    x_motion = CppProperty.from_native("SixDOFJoint", "x_motion")
    y_motion = CppProperty.from_native("SixDOFJoint", "y_motion")
    z_motion = CppProperty.from_native("SixDOFJoint", "z_motion")
    angular_x_motion = CppProperty.from_native("SixDOFJoint", "angular_x_motion")
    angular_y_motion = CppProperty.from_native("SixDOFJoint", "angular_y_motion")
    angular_z_motion = CppProperty.from_native("SixDOFJoint", "angular_z_motion")
    linear_limit_minimum = CppProperty.from_native("SixDOFJoint", "linear_limit_minimum")
    linear_limit_maximum = CppProperty.from_native("SixDOFJoint", "linear_limit_maximum")
    angular_limit_minimum = CppProperty.from_native("SixDOFJoint", "angular_limit_minimum")
    angular_limit_maximum = CppProperty.from_native("SixDOFJoint", "angular_limit_maximum")
    linear_limit_stiffness = CppProperty.from_native("SixDOFJoint", "linear_limit_stiffness")
    linear_limit_damping = CppProperty.from_native("SixDOFJoint", "linear_limit_damping")
    linear_friction_force = CppProperty.from_native("SixDOFJoint", "linear_friction_force")
    angular_friction_torque = CppProperty.from_native("SixDOFJoint", "angular_friction_torque")
    linear_motor_stiffness = CppProperty.from_native("SixDOFJoint", "linear_motor_stiffness")
    linear_motor_damping = CppProperty.from_native("SixDOFJoint", "linear_motor_damping")
    linear_motor_max_force = CppProperty.from_native("SixDOFJoint", "linear_motor_max_force")
    angular_motor_stiffness = CppProperty.from_native("SixDOFJoint", "angular_motor_stiffness")
    angular_motor_damping = CppProperty.from_native("SixDOFJoint", "angular_motor_damping")
    angular_motor_max_torque = CppProperty.from_native("SixDOFJoint", "angular_motor_max_torque")
    x_motor_mode = CppProperty.from_native("SixDOFJoint", "x_motor_mode")
    y_motor_mode = CppProperty.from_native("SixDOFJoint", "y_motor_mode")
    z_motor_mode = CppProperty.from_native("SixDOFJoint", "z_motor_mode")
    angular_x_motor_mode = CppProperty.from_native("SixDOFJoint", "angular_x_motor_mode")
    angular_y_motor_mode = CppProperty.from_native("SixDOFJoint", "angular_y_motor_mode")
    angular_z_motor_mode = CppProperty.from_native("SixDOFJoint", "angular_z_motor_mode")
    target_position = CppProperty.from_native("SixDOFJoint", "target_position")
    target_orientation = CppProperty.from_native("SixDOFJoint", "target_orientation")
    target_velocity = CppProperty.from_native("SixDOFJoint", "target_velocity")
    target_angular_velocity = CppProperty.from_native("SixDOFJoint", "target_angular_velocity")

    @property
    def current_position(self):
        return self._require_cpp_component().current_position

    @property
    def current_orientation(self):
        return self._require_cpp_component().current_orientation

    def set_frame(self, axis, secondary_axis):
        self._require_cpp_component().set_frame(axis, secondary_axis)

    def set_linear_limits(self, minimum, maximum):
        self._require_cpp_component().set_linear_limits(minimum, maximum)

    def set_angular_limits(self, minimum, maximum):
        self._require_cpp_component().set_angular_limits(minimum, maximum)
