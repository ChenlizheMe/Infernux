from __future__ import annotations

from typing import Any, ClassVar

from infernux.lib import Vector2, Vector3, vec4f, quatf


class _VecMeta(type):
    def __getattr__(cls, name: str) -> Any: ...
    def __instancecheck__(cls, instance: object) -> bool: ...
    def __subclasscheck__(cls, subclass: type) -> bool: ...


class vector3(metaclass=_VecMeta):
    """A representation of 3D vectors and points."""

    _cpp_type: type
    _constant_names: frozenset[str]

    def __new__(cls, x: float = ..., y: float = ..., z: float = ...) -> Vector3: ...

    # Constants (class-level, no parentheses)
    zero: Vector3
    """Shorthand for writing vector3(0, 0, 0)."""
    one: Vector3
    """Shorthand for writing vector3(1, 1, 1)."""
    up: Vector3
    """Shorthand for writing vector3(0, 1, 0)."""
    down: Vector3
    """Shorthand for writing vector3(0, -1, 0)."""
    left: Vector3
    """Shorthand for writing vector3(-1, 0, 0)."""
    right: Vector3
    """Shorthand for writing vector3(1, 0, 0)."""
    forward: Vector3
    """Shorthand for writing vector3(0, 0, 1)."""
    back: Vector3
    """Shorthand for writing vector3(0, 0, -1)."""
    positive_infinity: Vector3
    """Shorthand for writing vector3(inf, inf, inf)."""
    negative_infinity: Vector3
    """Shorthand for writing vector3(-inf, -inf, -inf)."""

    # Static methods (delegated to Vector3)
    @staticmethod
    def angle(a: Vector3, b: Vector3) -> float:
        """Return the angle in degrees between two vectors."""
        ...
    @staticmethod
    def clamp_magnitude(v: Vector3, max_length: float) -> Vector3:
        """Return a copy of the vector with its magnitude clamped."""
        ...
    @staticmethod
    def cross(a: Vector3, b: Vector3) -> Vector3:
        """Return the cross product of two vectors."""
        ...
    @staticmethod
    def distance(a: Vector3, b: Vector3) -> float:
        """Return the distance between two points."""
        ...
    @staticmethod
    def dot(a: Vector3, b: Vector3) -> float:
        """Return the dot product of two vectors."""
        ...
    @staticmethod
    def lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
        """Linearly interpolate between two vectors."""
        ...
    @staticmethod
    def lerp_unclamped(a: Vector3, b: Vector3, t: float) -> Vector3:
        """Linearly interpolate between two vectors without clamping t."""
        ...
    @staticmethod
    def max(a: Vector3, b: Vector3) -> Vector3:
        """Return a vector made from the largest components of two vectors."""
        ...
    @staticmethod
    def min(a: Vector3, b: Vector3) -> Vector3:
        """Return a vector made from the smallest components of two vectors."""
        ...
    @staticmethod
    def move_towards(current: Vector3, target: Vector3, max_delta: float) -> Vector3:
        """Move current towards target by at most max_delta."""
        ...
    @staticmethod
    def normalize(v: Vector3) -> Vector3:
        """Return the vector with a magnitude of 1."""
        ...
    @staticmethod
    def ortho_normalize(v1: Vector3, v2: Vector3, v3: Vector3) -> tuple[Vector3, Vector3, Vector3]:
        """Make vectors normalized and orthogonal to each other."""
        ...
    @staticmethod
    def project(v: Vector3, on_normal: Vector3) -> Vector3:
        """Project a vector onto another vector."""
        ...
    @staticmethod
    def project_on_plane(v: Vector3, plane_normal: Vector3) -> Vector3:
        """Project a vector onto a plane defined by its normal."""
        ...
    @staticmethod
    def reflect(in_dir: Vector3, normal: Vector3) -> Vector3:
        """Reflect a vector off the plane defined by a normal."""
        ...
    @staticmethod
    def rotate_towards(current: Vector3, target: Vector3, max_radians: float, max_mag: float) -> Vector3:
        """Rotate current towards target, limited by max angle and magnitude."""
        ...
    @staticmethod
    def scale(a: Vector3, b: Vector3) -> Vector3:
        """Multiply two vectors component-wise."""
        ...
    @staticmethod
    def signed_angle(from_v: Vector3, to_v: Vector3, axis: Vector3) -> float:
        """Return the signed angle in degrees between two vectors around an axis."""
        ...
    @staticmethod
    def slerp(a: Vector3, b: Vector3, t: float) -> Vector3:
        """Interpolate direction and length, clamping t to [0, 1].

        Zero endpoints use Cartesian lerp. Opposite directions use a stable
        perpendicular plane chosen from the least aligned coordinate axis.
        """
        ...
    @staticmethod
    def slerp_unclamped(a: Vector3, b: Vector3, t: float) -> Vector3:
        """Use the same direction/length path as slerp without clamping t."""
        ...
    @staticmethod
    def smooth_damp(
        current: Vector3,
        target: Vector3,
        current_velocity: Vector3,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[Vector3, Vector3]:
        """Return the updated position and velocity."""
        ...
    # Class lookup forwards native descriptors; constructed Vector3 instances
    # expose their native read-only float properties.
    magnitude: ClassVar[property]
    sqr_magnitude: ClassVar[property]


class vector2(metaclass=_VecMeta):
    """A representation of 2D vectors and points."""

    _cpp_type: type
    _constant_names: frozenset[str]

    def __new__(cls, x: float = ..., y: float = ...) -> Vector2: ...

    # Constants
    zero: Vector2
    """Shorthand for writing vector2(0, 0)."""
    one: Vector2
    """Shorthand for writing vector2(1, 1)."""
    up: Vector2
    """Shorthand for writing vector2(0, 1)."""
    down: Vector2
    """Shorthand for writing vector2(0, -1)."""
    left: Vector2
    """Shorthand for writing vector2(-1, 0)."""
    right: Vector2
    """Shorthand for writing vector2(1, 0)."""
    positive_infinity: Vector2
    """Shorthand for writing vector2(inf, inf)."""
    negative_infinity: Vector2
    """Shorthand for writing vector2(-inf, -inf)."""

    # Static methods
    @staticmethod
    def angle(a: Vector2, b: Vector2) -> float:
        """Return the unsigned angle in degrees between two vectors."""
        ...
    @staticmethod
    def clamp_magnitude(v: Vector2, max_length: float) -> Vector2:
        """Return a copy of the vector with its magnitude clamped."""
        ...
    @staticmethod
    def cross(a: Vector2, b: Vector2) -> float:
        """Return the 2D cross product (z-component of 3D cross)."""
        ...
    @staticmethod
    def distance(a: Vector2, b: Vector2) -> float:
        """Return the distance between two points."""
        ...
    @staticmethod
    def dot(a: Vector2, b: Vector2) -> float:
        """Return the dot product of two vectors."""
        ...
    @staticmethod
    def lerp(a: Vector2, b: Vector2, t: float) -> Vector2:
        """Linearly interpolate between two vectors."""
        ...
    @staticmethod
    def lerp_unclamped(a: Vector2, b: Vector2, t: float) -> Vector2:
        """Linearly interpolate between two vectors without clamping t."""
        ...
    @staticmethod
    def max(a: Vector2, b: Vector2) -> Vector2:
        """Return a vector made from the largest components of two vectors."""
        ...
    @staticmethod
    def min(a: Vector2, b: Vector2) -> Vector2:
        """Return a vector made from the smallest components of two vectors."""
        ...
    @staticmethod
    def move_towards(current: Vector2, target: Vector2, max_delta: float) -> Vector2:
        """Move current towards target by at most max_delta."""
        ...
    @staticmethod
    def normalize(v: Vector2) -> Vector2:
        """Return the vector with a magnitude of 1."""
        ...
    @staticmethod
    def perpendicular(v: Vector2) -> Vector2:
        """Return the 2D vector perpendicular to this vector."""
        ...
    @staticmethod
    def reflect(direction: Vector2, normal: Vector2) -> Vector2:
        """Reflect a vector off the surface defined by a normal."""
        ...
    @staticmethod
    def scale(a: Vector2, b: Vector2) -> Vector2:
        """Multiply two vectors component-wise."""
        ...
    @staticmethod
    def signed_angle(a: Vector2, b: Vector2) -> float:
        """Return the signed angle in degrees between two vectors."""
        ...
    @staticmethod
    def smooth_damp(
        current: Vector2,
        target: Vector2,
        current_velocity: Vector2,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[Vector2, Vector2]:
        """Return the updated position and velocity."""
        ...
    magnitude: ClassVar[property]
    sqr_magnitude: ClassVar[property]


class vector4(metaclass=_VecMeta):
    """A representation of 4D vectors."""

    _cpp_type: type
    _constant_names: frozenset[str]

    def __new__(cls, x: float = ..., y: float = ..., z: float = ..., w: float = ...) -> vec4f: ...

    # Constants
    zero: vec4f
    """Shorthand for writing vector4(0, 0, 0, 0)."""
    one: vec4f
    """Shorthand for writing vector4(1, 1, 1, 1)."""
    positive_infinity: vec4f
    """A vector with all components set to positive infinity."""
    negative_infinity: vec4f
    """A vector with all components set to negative infinity."""

    # Static methods
    @staticmethod
    def distance(a: vec4f, b: vec4f) -> float:
        """Return the distance between two points."""
        ...
    @staticmethod
    def dot(a: vec4f, b: vec4f) -> float:
        """Return the dot product of two vectors."""
        ...
    @staticmethod
    def lerp(a: vec4f, b: vec4f, t: float) -> vec4f:
        """Linearly interpolate between two vectors."""
        ...
    @staticmethod
    def lerp_unclamped(a: vec4f, b: vec4f, t: float) -> vec4f:
        """Linearly interpolate between two vectors without clamping t."""
        ...
    @staticmethod
    def max(a: vec4f, b: vec4f) -> vec4f:
        """Return a vector made from the largest components of two vectors."""
        ...
    @staticmethod
    def min(a: vec4f, b: vec4f) -> vec4f:
        """Return a vector made from the smallest components of two vectors."""
        ...
    @staticmethod
    def move_towards(current: vec4f, target: vec4f, max_delta: float) -> vec4f:
        """Move current towards target by at most max_delta."""
        ...
    @staticmethod
    def normalize(v: vec4f) -> vec4f:
        """Return the vector with a magnitude of 1."""
        ...
    @staticmethod
    def project(a: vec4f, b: vec4f) -> vec4f:
        """Project vector a onto vector b."""
        ...
    @staticmethod
    def scale(a: vec4f, b: vec4f) -> vec4f:
        """Multiply two vectors component-wise."""
        ...
    @staticmethod
    def smooth_damp(
        current: vec4f,
        target: vec4f,
        current_velocity: vec4f,
        smooth_time: float,
        max_speed: float,
        delta_time: float,
    ) -> tuple[vec4f, vec4f]:
        """Return the updated position and velocity."""
        ...
    magnitude: ClassVar[property]
    sqr_magnitude: ClassVar[property]


class quaternion(metaclass=_VecMeta):
    """A representation of rotations using a quaternion."""

    _cpp_type: type
    _constant_names: frozenset[str]

    def __new__(cls, x: float = ..., y: float = ..., z: float = ..., w: float = ...) -> quatf: ...

    # Constants
    identity: quatf
    """The identity rotation (no rotation)."""

    # Static constructors
    @staticmethod
    def euler(x: float, y: float, z: float) -> quatf:
        """Create a rotation from Euler angles in degrees."""
        ...
    @staticmethod
    def angle_axis(angle: float, axis: Vector3) -> quatf:
        """Create a rotation of angle degrees around axis."""
        ...
    @staticmethod
    def look_rotation(forward: Vector3, up: Vector3 = ...) -> quatf:
        """Create a rotation looking in the forward direction."""
        ...

    # Static methods
    @staticmethod
    def dot(a: quatf, b: quatf) -> float:
        """Return the dot product of two quaternions."""
        ...
    @staticmethod
    def angle(a: quatf, b: quatf) -> float:
        """Return the angle in degrees between two rotations."""
        ...
    @staticmethod
    def slerp(a: quatf, b: quatf, t: float) -> quatf:
        """Spherically interpolate between two rotations."""
        ...
    @staticmethod
    def lerp(a: quatf, b: quatf, t: float) -> quatf:
        """Linearly interpolate between two quaternions (normalized)."""
        ...
    @staticmethod
    def inverse(q: quatf) -> quatf:
        """Return the inverse of a rotation."""
        ...
    @staticmethod
    def rotate_towards(from_: quatf, to: quatf, max_degrees_delta: float) -> quatf:
        """Rotate from towards to by at most max_degrees_delta degrees."""
        ...


__all__ = [
    "vector2",
    "vector3",
    "vector4",
    "quaternion",
]
