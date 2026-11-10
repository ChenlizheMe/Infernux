from typing import Optional
from infernux.lib import Vector3, vec4f, quatf, JointMotorMode, JointMotion
from infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class SixDOFJoint(BuiltinComponent):
    anchor: Vector3
    connected_body: Optional[Rigidbody]
    enable_collision: bool
    axis: Vector3
    secondary_axis: Vector3
    x_motion: JointMotion
    y_motion: JointMotion
    z_motion: JointMotion
    angular_x_motion: JointMotion
    angular_y_motion: JointMotion
    angular_z_motion: JointMotion
    linear_limit_minimum: Vector3
    linear_limit_maximum: Vector3
    angular_limit_minimum: Vector3
    angular_limit_maximum: Vector3
    linear_limit_stiffness: Vector3
    linear_limit_damping: Vector3
    linear_friction_force: Vector3
    angular_friction_torque: Vector3
    linear_motor_stiffness: Vector3
    linear_motor_damping: Vector3
    linear_motor_max_force: Vector3
    angular_motor_stiffness: Vector3
    angular_motor_damping: Vector3
    angular_motor_max_torque: Vector3
    x_motor_mode: JointMotorMode
    y_motor_mode: JointMotorMode
    z_motor_mode: JointMotorMode
    angular_x_motor_mode: JointMotorMode
    angular_y_motor_mode: JointMotorMode
    angular_z_motor_mode: JointMotorMode
    target_position: Vector3
    target_orientation: vec4f
    target_velocity: Vector3
    target_angular_velocity: Vector3
    @property
    def current_position(self) -> Vector3: ...
    @property
    def current_orientation(self) -> quatf: ...
    def set_frame(self, axis: Vector3, secondary_axis: Vector3) -> None: ...
    def set_linear_limits(self, minimum: Vector3, maximum: Vector3) -> None: ...
    def set_angular_limits(self, minimum: Vector3, maximum: Vector3) -> None: ...
