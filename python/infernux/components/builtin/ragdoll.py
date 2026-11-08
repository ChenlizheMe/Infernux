"""Root control of one Jolt ragdoll using real Rigidbody and Joint components."""
from infernux.components.builtin_component import BuiltinComponent, CppProperty


class Ragdoll(BuiltinComponent):
    _cpp_type_name = "Ragdoll"
    _component_category_ = "Physics"

    @property
    def active(self) -> bool:
        """Whether any configured member is dynamic. Assignment sets all members."""
        return self._require_cpp_component().active

    @active.setter
    def active(self, value: bool) -> None:
        """Take over the displayed pose, or return to animation at the physical root."""
        self._require_cpp_component().active = value

    # Explicit enable starts Hinge/SwingTwist motors once. Later manual motor
    # changes remain authoritative; false only stops following target updates.
    drive_animation = CppProperty.from_native(
        "Ragdoll", "drive_animation", related_components=lambda component: component._animation_drive_joints(),
    )

    def _animation_drive_joints(self):
        from .hinge_joint import HingeJoint
        from .swing_twist_joint import SwingTwistJoint
        joints = []
        for body in self.bodies:
            hinge = body.game_object.get_component(HingeJoint)
            swing = body.game_object.get_component(SwingTwistJoint)
            if hinge is not None:
                joints.append((hinge, ("motor_enabled", "motor_target_angle")))
            elif swing is not None:
                joints.append((swing, ("swing_motor_mode", "twist_motor_mode", "target_orientation")))
        return joints

    def set_passive(self) -> None:
        """Stop animation following and turn off actual joint motors."""
        self._require_cpp_component().set_passive()

    @property
    def is_bound(self) -> bool:
        return self._require_cpp_component().is_bound

    @property
    def bodies(self):
        from infernux.lib import _wrap_native_builtin_component
        return [_wrap_native_builtin_component(body) for body in self._require_cpp_component().bodies]

    @property
    def body_count(self) -> int:
        return self._require_cpp_component().body_count

    @property
    def constraint_count(self) -> int:
        return self._require_cpp_component().constraint_count

    def rebuild(self) -> bool:
        return self._require_cpp_component().rebuild()

    @property
    def skeleton(self):
        from infernux.lib import _wrap_native_builtin_component
        native = self._require_cpp_component().skeleton
        return _wrap_native_builtin_component(native) if native is not None else None

    @property
    def total_mass(self) -> float:
        return self._require_cpp_component().total_mass

    @total_mass.setter
    def total_mass(self, value: float) -> None:
        self._require_cpp_component().total_mass = value

    def generate_physics(self) -> int:
        """Fit ordinary bone physics once; existing components keep their values."""
        return self._require_cpp_component().generate_physics()

    def drive_pose(self, world_rotations) -> None:
        """Drive existing Hinge/SwingTwist motors to body world rotations."""
        self._require_cpp_component().drive_pose(world_rotations)
