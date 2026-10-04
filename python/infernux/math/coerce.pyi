"""Shared vector coercion utilities.

Example::

    from infernux.math.coerce import coerce_vec3, quat_rotate
    v = coerce_vec3((1, 2, 3))      # tuple → Vector3
"""

from __future__ import annotations

from typing import Sequence, Union

from infernux.lib import Vector3

def coerce_vec3(value: Union[Vector3, Sequence[float]]) -> Vector3:
    """Convert a tuple / list / Vector3 to a :class:`Vector3`.

    Passes through unchanged if *value* is already a ``Vector3``.

    Args:
        value: A ``Vector3`` or any indexable with at least 3 float elements.
    """
    ...

def quat_rotate(
    q: object,
    v: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a 3-D vector *v* by quaternion *q* ``(x, y, z, w)``.

    Uses the Hamilton product shortcut (no matrix conversion).

    Args:
        q: Quaternion-like object with ``.x``, ``.y``, ``.z``, ``.w`` attrs.
        v: 3-element sequence ``(x, y, z)`` to rotate.

    Returns:
        Rotated ``(x, y, z)`` tuple.
    """
    ...
