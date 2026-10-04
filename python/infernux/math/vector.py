"""
Unity-compatible vector2 / vector3 / vector4 wrapper classes.

These wrap the C++ ``Vector2`` / ``Vector3`` / ``vec4f`` types and add Unity-style
class-level properties that do NOT require parentheses — matching Unity's C#
calling convention while using Python-friendly lowercase names.

Usage::

    from infernux.math import vector3, vector2, vector4

    direction = vector3.forward          # (0, 0, 1) — no parentheses!
    origin    = vector3.zero             # (0, 0, 0)
    v         = vector3(1, 2, 3)
    print(v.magnitude)                   # instance property
    n         = v.normalized             # instance property, returns new vector3
    print(vector3.dot(v, direction))     # static method
    print(vector3.distance(v, origin))   # static method
    isinstance(v, vector3)              # True!

Naming convention
-----------------
- Type names: ``vector2``, ``vector3``, ``vector4`` (lowercase, Pythonic)
- Static methods: ``snake_case`` (``dot``, ``cross``, ``clamp_magnitude``, …)
- Constants: read-only class properties (``vector3.forward``, not ``.forward()``)
- Instance properties: ``magnitude``, ``normalized``, ``sqr_magnitude``
"""

from infernux.lib import Vector2 as _Vector2, Vector3 as _Vector3, vec4f, quatf


# ============================================================================
# Metaclass — turns def_static() methods into read-only class properties
# ============================================================================

class _VecMeta(type):
    """
    Metaclass for vector wrapper classes.

    - Intercepts class-level attribute access so that ``vector3.forward``
      (no parentheses) returns the constant value.
    - Forwards all other attribute lookups (e.g. ``vector3.dot``) to the
      underlying C++ pybind11 type.
    - Implements ``isinstance()`` / ``issubclass()`` checks against the
      underlying C++ type so that ``isinstance(Vector3(...), vector3)`` is True.
    """

    def __getattr__(cls, name: str):
        cpp_type = cls.__dict__.get("_cpp_type")
        if cpp_type is not None and hasattr(cpp_type, name):
            attr = getattr(cpp_type, name)
            # If this is one of the known constant names, call the static
            # method and return the value directly (property-style).
            if callable(attr) and name in cls._constant_names:
                return attr()
            return attr
        raise AttributeError(f"type '{cls.__name__}' has no attribute '{name}'")

    def __instancecheck__(cls, instance):
        """``isinstance(v, vector3)`` returns True for Vector3 instances."""
        return isinstance(instance, cls._cpp_type)

    def __subclasscheck__(cls, subclass):
        if subclass is cls._cpp_type:
            return True
        return super().__subclasscheck__(subclass)


# ============================================================================
# vector3
# ============================================================================

class vector3(metaclass=_VecMeta):
    """
    Unity-compatible Vector3 type (lowercase, Pythonic).

    Constructing ``vector3(x, y, z)`` returns a ``Vector3`` instance — they
    are fully interchangeable at runtime.

    Static properties (class-level, no parentheses)::

        vector3.zero                 # (0, 0, 0)
        vector3.one                  # (1, 1, 1)
        vector3.up                   # (0, 1, 0)
        vector3.down                 # (0, -1, 0)
        vector3.left                 # (-1, 0, 0)
        vector3.right                # (1, 0, 0)
        vector3.forward              # (0, 0, 1)
        vector3.back                 # (0, 0, -1)
        vector3.positive_infinity
        vector3.negative_infinity

    Instance properties (no parentheses)::

        v.x, v.y, v.z               # component access (read/write)
        v.magnitude                  # length (read-only)
        v.sqr_magnitude              # squared length (read-only)
        v.normalized                 # unit-length copy (read-only)

    Instance methods::

        v.set(x, y, z)              # set components in-place

    Static methods::

        vector3.angle(a, b)
        vector3.clamp_magnitude(v, max_length)
        vector3.cross(a, b)
        vector3.distance(a, b)
        vector3.dot(a, b)
        vector3.lerp(a, b, t)
        vector3.lerp_unclamped(a, b, t)
        vector3.max(a, b)
        vector3.min(a, b)
        vector3.move_towards(current, target, max_delta)
        vector3.normalize(v)            # returns normalised copy
        vector3.ortho_normalize(v1, v2, v3)
        vector3.project(v, on_normal)
        vector3.project_on_plane(v, plane_normal)
        vector3.reflect(in_dir, normal)
        vector3.rotate_towards(current, target, max_radians, max_mag)
        vector3.scale(a, b)
        vector3.signed_angle(from_v, to_v)
        vector3.slerp(a, b, t)
        vector3.slerp_unclamped(a, b, t)
        vector3.smooth_damp(...)

    Operators::

        v + w, v - w, v * s, v / s, -v, v == w, v != w
    """

    _cpp_type = _Vector3
    _constant_names = frozenset({
        "zero", "one", "up", "down", "left", "right",
        "forward", "back", "negative_infinity", "positive_infinity",
    })

    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        return _Vector3(x, y, z)


# ============================================================================
# vector2
# ============================================================================

class vector2(metaclass=_VecMeta):
    """
    Unity-compatible Vector2 type (lowercase, Pythonic).

    Static properties::

        vector2.zero, vector2.one, vector2.up, vector2.down,
        vector2.left, vector2.right,
        vector2.negative_infinity, vector2.positive_infinity

    Instance properties::

        v.x, v.y, v.magnitude, v.sqr_magnitude, v.normalized

    Static methods::

        vector2.angle, vector2.clamp_magnitude, vector2.cross,
        vector2.distance, vector2.dot, vector2.lerp, vector2.lerp_unclamped,
        vector2.max, vector2.min, vector2.move_towards, vector2.normalize,
        vector2.perpendicular, vector2.reflect, vector2.scale,
        vector2.signed_angle, vector2.smooth_damp
    """

    _cpp_type = _Vector2
    _constant_names = frozenset({
        "zero", "one", "up", "down", "left", "right",
        "negative_infinity", "positive_infinity",
    })

    def __new__(cls, x: float = 0.0, y: float = 0.0):
        return _Vector2(x, y)


# ============================================================================
# vector4
# ============================================================================

class vector4(metaclass=_VecMeta):
    """
    Unity-compatible Vector4 type (lowercase, Pythonic).

    Static properties::

        vector4.zero, vector4.one,
        vector4.negative_infinity, vector4.positive_infinity

    Instance properties::

        v.x, v.y, v.z, v.w, v.magnitude, v.sqr_magnitude, v.normalized

    Static methods::

        vector4.distance, vector4.dot, vector4.lerp, vector4.lerp_unclamped,
        vector4.max, vector4.min, vector4.move_towards, vector4.normalize,
        vector4.project, vector4.scale, vector4.smooth_damp
    """

    _cpp_type = vec4f
    _constant_names = frozenset({
        "zero", "one", "negative_infinity", "positive_infinity",
    })

    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0):
        return vec4f(x, y, z, w)


# ============================================================================
# quaternion
# ============================================================================

class quaternion(metaclass=_VecMeta):
    """
    Unity-compatible Quaternion type (lowercase, Pythonic).

    Constructing ``quaternion(x, y, z, w)`` returns a ``quatf`` instance.

    Static constructors::

        quaternion.identity              # (0, 0, 0, 1)
        quaternion.euler(x, y, z)        # from Euler angles (degrees)
        quaternion.angle_axis(angle, axis)
        quaternion.look_rotation(forward, up)

    Instance properties::

        q.x, q.y, q.z, q.w              # component access
        q.euler_angles                   # as Vector3 (degrees)
        q.normalized                     # normalized copy

    Static methods::

        quaternion.dot(a, b)
        quaternion.angle(a, b)
        quaternion.slerp(a, b, t)
        quaternion.lerp(a, b, t)
        quaternion.inverse(q)
        quaternion.rotate_towards(from, to, max_degrees_delta)
    """

    _cpp_type = quatf
    _constant_names = frozenset({
        "identity",
    })

    def __new__(cls, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        return quatf(x, y, z, w)


__all__ = [
    "vector2", "vector3", "vector4", "quaternion",
]
