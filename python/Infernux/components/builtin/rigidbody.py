"""
Rigidbody — Python BuiltinComponent wrapper for C++ Rigidbody.

Mirrors Unity's ``Rigidbody`` component. When attached alongside a Collider,
the Collider's body becomes dynamic (affected by gravity, forces, etc.).

Example::

    from Infernux.components.builtin import Rigidbody
    from Infernux.math import Vector3

    class MyScript(InxComponent):
        def start(self):
            rb = self.game_object.get_component(Rigidbody)
            rb.mass = 2.0
            rb.use_gravity = True

        def fixed_update(self, dt):
            rb = self.game_object.get_component(Rigidbody)
            rb.add_force(Vector3(0, 10, 0))
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

from Infernux.components.builtin_component import BuiltinComponent, CppProperty
from Infernux.components.fields import FieldType
from Infernux.math.coerce import coerce_quat, coerce_vec3
from Infernux.lib import ForceMode as _ForceMode


class RigidbodyConstraints(IntFlag):
    """Unity-style rigidbody constraint bitmask."""

    None_ = 0
    FreezePositionX = 2
    FreezePositionY = 4
    FreezePositionZ = 8
    FreezePosition = FreezePositionX | FreezePositionY | FreezePositionZ
    FreezeRotationX = 16
    FreezeRotationY = 32
    FreezeRotationZ = 64
    FreezeRotation = FreezeRotationX | FreezeRotationY | FreezeRotationZ
    FreezeAll = FreezePosition | FreezeRotation


class CollisionDetectionMode(IntEnum):
    """Collision algorithms supported by the Jolt backend.

    Backend mapping:
    Dynamic ``Continuous`` uses Jolt LinearCast sweep CCD against static
    geometry. ``ContinuousDynamic`` additionally resolves fast dynamic pairs
    and is substantially more expensive.
    Kinematic ``Continuous`` uses Jolt's discrete quality because Jolt does not
    expose Unity's speculative mode as an equivalent per-body setting.
    """

    Discrete = 0
    Continuous = 1
    ContinuousDynamic = 2


class RigidbodyInterpolation(IntEnum):
    """Unity-style rigidbody interpolation mode."""

    None_ = 0
    Interpolate = 1


class Rigidbody(BuiltinComponent):
    """Python wrapper for the C++ Rigidbody component."""

    _cpp_type_name = "Rigidbody"

    _component_category_ = "Physics"

    # ---- Serialized properties (displayed in Inspector) ----

    mass = CppProperty.from_native("Rigidbody", "mass")
    drag = CppProperty.from_native("Rigidbody", "drag")
    angular_drag = CppProperty.from_native("Rigidbody", "angular_drag")
    use_gravity = CppProperty.from_native("Rigidbody", "use_gravity")
    is_kinematic = CppProperty.from_native("Rigidbody", "is_kinematic")
    collision_detection_mode = CppProperty.from_native("Rigidbody", "collision_detection_mode")
    collision_detection_mode.metadata.enum_type = CollisionDetectionMode
    interpolation = CppProperty.from_native("Rigidbody", "interpolation")
    interpolation.metadata.enum_type = RigidbodyInterpolation

    # ---- Per-axis freeze constraints (displayed as checkboxes) ----

    freeze_position_x = CppProperty(
        "freeze_position_x", FieldType.BOOL, default=False,
        tooltip="Freeze movement on the X axis",
    )
    freeze_position_y = CppProperty(
        "freeze_position_y", FieldType.BOOL, default=False,
        tooltip="Freeze movement on the Y axis",
    )
    freeze_position_z = CppProperty(
        "freeze_position_z", FieldType.BOOL, default=False,
        tooltip="Freeze movement on the Z axis",
    )
    freeze_rotation_x = CppProperty(
        "freeze_rotation_x", FieldType.BOOL, default=False,
        tooltip="Freeze rotation on the X axis",
    )
    freeze_rotation_y = CppProperty(
        "freeze_rotation_y", FieldType.BOOL, default=False,
        tooltip="Freeze rotation on the Y axis",
    )
    freeze_rotation_z = CppProperty(
        "freeze_rotation_z", FieldType.BOOL, default=False,
        tooltip="Freeze rotation on the Z axis",
    )

    _FREEZE_FIELDS = frozenset({
        'freeze_position_x', 'freeze_position_y', 'freeze_position_z',
        'freeze_rotation_x', 'freeze_rotation_y', 'freeze_rotation_z',
    })

    # ------------------------------------------------------------------
    # Custom inspector: freeze checkboxes in compact two-row layout
    # ------------------------------------------------------------------

    def render_inspector(self, ctx) -> None:
        from Infernux.engine.ui.inspector_components import (
            render_builtin_via_setters, _record_builtin_property,
        )

        # Standard properties (skip freeze fields — rendered below)
        render_builtin_via_setters(ctx, self, type(self),
                                   skip_fields=self._FREEZE_FIELDS)

        # Freeze Transform section
        ctx.separator()
        ctx.label("Freeze Transform")
        self._render_freeze_row(ctx, "Position", "freeze_position")
        self._render_freeze_row(ctx, "Rotation", "freeze_rotation")

    def _render_freeze_row(self, ctx, label: str, prefix: str) -> None:
        from Infernux.engine.ui.inspector_components import _record_builtin_property

        ctx.align_text_to_frame_padding()
        ctx.label(label)
        for i, axis in enumerate('xyz'):
            if i == 0:
                ctx.same_line(90)
            else:
                ctx.same_line(0, 8)
            attr = f"{prefix}_{axis}"
            current = getattr(self, attr)
            new_val = ctx.checkbox(f"{axis.upper()}##{attr}", current)
            if new_val != current:
                _record_builtin_property(self, attr, current, new_val,
                                         f"Set {attr}")

    # ---- Convenience property: freeze_rotation ----

    @property
    def freeze_rotation(self) -> bool:
        """Shortcut to freeze all rotation axes."""
        cpp = self._require_cpp_component()
        return cpp.freeze_rotation

    @freeze_rotation.setter
    def freeze_rotation(self, value: bool):
        self._require_cpp_component().freeze_rotation = value

    @property
    def constraints(self) -> int:
        """Constraints bitmask — use RigidbodyConstraints helpers for readability."""
        cpp = self._require_cpp_component()
        return cpp.constraints

    @constraints.setter
    def constraints(self, value: int):
        self._require_cpp_component().constraints = int(value)

    max_angular_velocity = CppProperty.from_native("Rigidbody", "max_angular_velocity")
    max_linear_velocity = CppProperty.from_native("Rigidbody", "max_linear_velocity")

    @property
    def constraints_flags(self) -> RigidbodyConstraints:
        """Typed view of the constraint bitmask."""
        return RigidbodyConstraints(int(self.constraints))

    @constraints_flags.setter
    def constraints_flags(self, value):
        self.constraints = int(value)

    def has_constraint(self, constraint: RigidbodyConstraints) -> bool:
        """Return True when the given constraint flag is enabled."""
        return bool(self.constraints_flags & RigidbodyConstraints(constraint))

    def add_constraint(self, constraint: RigidbodyConstraints):
        """Enable one or more constraint flags."""
        self.constraints = int(self.constraints_flags | RigidbodyConstraints(constraint))

    def remove_constraint(self, constraint: RigidbodyConstraints):
        """Disable one or more constraint flags."""
        self.constraints = int(self.constraints_flags & ~RigidbodyConstraints(constraint))

    # ---- Runtime-only properties (not serialized via CppProperty, accessed via methods) ----

    @property
    def velocity(self):
        """Linear velocity in world space (Vector3)."""
        cpp = self._require_cpp_component()
        return cpp.velocity

    @velocity.setter
    def velocity(self, value):
        self._require_cpp_component().velocity = coerce_vec3(value)

    @property
    def angular_velocity(self):
        """Angular velocity in world space (Vector3)."""
        cpp = self._require_cpp_component()
        return cpp.angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        self._require_cpp_component().angular_velocity = coerce_vec3(value)

    # ---- Read-only world info ----

    def get_point_velocity(self, point):
        """World-space velocity at ``point``, including the body's angular motion.

        Uses the physical center of mass, not the interpolated display transform.
        """
        return self._require_cpp_component().get_point_velocity(coerce_vec3(point))

    def get_point_velocities(self, points, output):
        """Write point velocities to caller-owned NumPy storage and return it.

        Both arrays must be C-contiguous float32 ``(N, 3)``. Output must be
        writable; exact in-place use is supported, partial overlap is not.
        Reads body state once for the whole batch, without per-point Python calls.
        """
        return self._require_cpp_component().get_point_velocities(points, output)

    @property
    def world_center_of_mass(self):
        """World-space center of mass (read-only)."""
        cpp = self._require_cpp_component()
        return cpp.world_center_of_mass

    @property
    def position(self):
        """World-space position of the rigidbody."""
        cpp = self._require_cpp_component()
        return cpp.position

    @position.setter
    def position(self, value):
        self._require_cpp_component().position = coerce_vec3(value)

    @property
    def rotation(self):
        """World-space rotation quaternion (x, y, z, w)."""
        cpp = self._require_cpp_component()
        return cpp.rotation

    @rotation.setter
    def rotation(self, value):
        self._require_cpp_component().rotation = coerce_quat(value)

    # ---- Force / Torque API ----

    def add_force(self, force, mode=None):
        """Add a force to the rigidbody.

        Args:
            force: Force vector (Vector3 or tuple).
            mode: ForceMode enum value (default: ForceMode.Force).
        """
        cpp = self._require_cpp_component()
        if mode is None:
            mode = _ForceMode.Force
        cpp.add_force(coerce_vec3(force), mode)

    def add_torque(self, torque, mode=None):
        """Add a torque to the rigidbody.

        Args:
            torque: Torque vector (Vector3 or tuple).
            mode: ForceMode enum value (default: ForceMode.Force).
        """
        cpp = self._require_cpp_component()
        if mode is None:
            mode = _ForceMode.Force
        cpp.add_torque(coerce_vec3(torque), mode)

    def add_force_at_position(self, force, position, mode=None):
        """Add a force at a world-space position.

        Args:
            force: Force vector (Vector3 or tuple).
            position: World-space point where force is applied.
            mode: ForceMode enum value (default: ForceMode.Force).
        """
        cpp = self._require_cpp_component()
        if mode is None:
            mode = _ForceMode.Force
        cpp.add_force_at_position(coerce_vec3(force), coerce_vec3(position), mode)

    # ---- Kinematic movement ----

    def move_position(self, position):
        """Move a kinematic body to target position (Unity: Rigidbody.MovePosition).

        Call from fixed_update for smooth interpolation. Raises when the
        Rigidbody is not kinematic or has no active Collider body.

        Args:
            position: Target world-space position (Vector3 or tuple).
        """
        cpp = self._require_cpp_component()
        cpp.move_position(coerce_vec3(position))

    def move_rotation(self, rotation):
        """Rotate a kinematic body to target rotation (Unity: Rigidbody.MoveRotation).

        Call from fixed_update for smooth interpolation. Raises when the
        Rigidbody is not kinematic or has no active Collider body.

        Args:
            rotation: Target rotation as (x, y, z, w) quaternion tuple.
        """
        cpp = self._require_cpp_component()
        cpp.move_rotation(rotation)

    # ---- Sleep API ----

    def is_sleeping(self) -> bool:
        """Is the rigidbody sleeping?"""
        cpp = self._require_cpp_component()
        return cpp.is_sleeping()

    def wake_up(self):
        """Wake the rigidbody up."""
        self._require_cpp_component().wake_up()

    def sleep(self):
        """Put the rigidbody to sleep."""
        self._require_cpp_component().sleep()
