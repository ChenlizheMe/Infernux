"""Shared vector coercion utilities."""

from __future__ import annotations

from infernux.lib import Vector3 as _Vector3, quatf as _Quat

def coerce_vec3(value) -> _Vector3:
    """Convert a tuple/list/Vector3 to a Vector3, passing through if already one."""
    if isinstance(value, _Vector3):
        return value
    return _Vector3(value[0], value[1], value[2])


def coerce_quat(value) -> _Quat:
    """Convert an (x, y, z, w) sequence to quatf; None means identity."""
    if value is None:
        return _Quat(0.0, 0.0, 0.0, 1.0)
    if isinstance(value, _Quat):
        return value
    return _Quat(value[0], value[1], value[2], value[3])


def quat_rotate(q, v):
    """Rotate a 3D vector *v* by quaternion *q* (x, y, z, w)."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )
