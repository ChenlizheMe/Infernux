"""Transform dependency tracking used by the UI submission path.

The native implementation is authoritative on desktop/editor builds.  Some
already-published Web players contain an older ``_Infernux`` module without
the private helper; those players use the deliberately small Python tracker
below until their native host is rebuilt.  This is a platform ABI boundary,
not a second editor/runtime transform system.
"""

from __future__ import annotations

import math
import os
import sys


def _web_runtime() -> bool:
    return os.environ.get("INFERNUX_WEB_RUNTIME") == "1" or sys.platform == "emscripten"


def _scalar(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _vec_signature(value):
    if value is None:
        return None
    return tuple(_scalar(getattr(value, axis, 0.0)) for axis in ("x", "y", "z"))


def _transform_signature(game_object):
    transform = game_object.transform
    matrix = getattr(transform, "local_to_world_matrix", None)
    if callable(matrix):
        try:
            return tuple(_scalar(value) for value in matrix())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return (
        _vec_signature(getattr(transform, "position", None)),
        _vec_signature(getattr(transform, "rotation", None)),
        _vec_signature(getattr(transform, "local_scale", None)),
    )


class _WebUITransformDependencies:
    """Small deterministic replacement for the missing legacy Web symbol."""

    def __init__(self, screen_objects, world_objects):
        self._objects = tuple(screen_objects) + tuple(world_objects)
        self._world_count = len(tuple(world_objects))
        self._signatures = tuple(_transform_signature(obj) for obj in self._objects)
        self._revision = 0
        self.changed_entries = tuple(range(len(self._objects)))

    def poll(self) -> int:
        current = tuple(_transform_signature(obj) for obj in self._objects)
        changed = tuple(index for index, (before, after) in enumerate(zip(self._signatures, current))
                       if before != after)
        if len(current) != len(self._signatures):
            changed = tuple(sorted(set(changed) | set(range(len(self._signatures), len(current)))))
        if changed:
            self._revision += 1
            self._signatures = current
        self.changed_entries = changed
        return self._revision

    def project_world_ray(self, ray_origin, ray_direction, layer_mask=0xFFFFFFFF):
        """Project world rays onto each element's local XY plane.

        The native helper performs the same operation in a batched call.  The
        compatibility implementation intentionally handles only the authored
        world-UI plane and returns NaN for a parallel/missing intersection.
        """
        results = []
        for game_object in self._objects[-self._world_count:] if self._world_count else ():
            transform = game_object.transform
            try:
                origin = transform.inverse_transform_point(ray_origin)
                direction = transform.inverse_transform_direction(ray_direction)
                dz = _scalar(getattr(direction, "z", 0.0))
                if abs(dz) <= 1e-8:
                    results.append((math.nan, math.nan, math.inf))
                    continue
                t = -_scalar(getattr(origin, "z", 0.0)) / dz
                if t < 0.0:
                    results.append((math.nan, math.nan, math.inf))
                    continue
                local_hit = transform.inverse_transform_point(
                    type(ray_origin)(
                        _scalar(getattr(ray_origin, "x", 0.0))
                        + _scalar(getattr(ray_direction, "x", 0.0)) * t,
                        _scalar(getattr(ray_origin, "y", 0.0))
                        + _scalar(getattr(ray_direction, "y", 0.0)) * t,
                        _scalar(getattr(ray_origin, "z", 0.0))
                        + _scalar(getattr(ray_direction, "z", 0.0)) * t,
                    )
                )
                results.append((
                    _scalar(getattr(local_hit, "x", 0.0)),
                    _scalar(getattr(local_hit, "y", 0.0)),
                    float(t),
                ))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                results.append((math.nan, math.nan, math.inf))
        return results


def create_ui_transform_dependencies(screen_objects, world_objects):
    """Create the native tracker, or the explicit stale-Web ABI tracker."""
    try:
        from Infernux.lib._Infernux import _UITransformDependencies
    except (ImportError, AttributeError):
        if not _web_runtime():
            raise
        return _WebUITransformDependencies(screen_objects, world_objects)
    return _UITransformDependencies(screen_objects, world_objects)


__all__ = ["create_ui_transform_dependencies"]
