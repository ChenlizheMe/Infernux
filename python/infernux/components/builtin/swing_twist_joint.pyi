from typing import Optional
from infernux.lib import Vector3, vec4f, quatf, JointMotorMode, JointSwingType
from infernux.components.builtin_component import BuiltinComponent
from .rigidbody import Rigidbody

class SwingTwistJoint(BuiltinComponent):
    anchor: Vector3
    connected_body: Optional[Rigidbody]
    enable_collision: bool
    twist_axis: Vector3
    plane_axis: Vector3
    swing_type: JointSwingType
    normal_half_cone_angle: float
    plane_half_cone_angle: float
    minimum_twist_angle: float
    maximum_twist_angle: float
    friction_torque: float
    swing_motor_mode: JointMotorMode
    twist_motor_mode: JointMotorMode
    swing_motor_stiffness: float
    swing_motor_damping: float
    swing_motor_max_torque: float
    twist_motor_stiffness: float
    twist_motor_damping: float
    twist_motor_max_torque: float
    target_orientation: vec4f
    target_angular_velocity: Vector3
    @property
    def current_orientation(self) -> quatf: ...
    def set_frame(self, twist_axis: Vector3, plane_axis: Vector3) -> None: ...
    def set_twist_limits(self, minimum: float, maximum: float) -> None: ...
