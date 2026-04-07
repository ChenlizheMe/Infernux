"""SceneViewMathMixin — extracted from SceneViewPanel."""
from __future__ import annotations

"""
Unity-style Scene View panel with 3D viewport and camera controls.
"""

import math
import os
from Infernux.lib import InxGUIContext, TextureLoader, InputManager
from Infernux.engine.i18n import t
from .editor_panel import EditorPanel
from .closable_panel import ClosablePanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from .viewport_utils import ViewportInfo, capture_viewport_info
from . import imgui_keys as _keys
import Infernux.resources as _resources

# Gizmo handle IDs — must match C++ EditorTools constants
from Infernux.debug import Debug
from Infernux.lib._Infernux import (
    GIZMO_X_AXIS_ID,
    GIZMO_Y_AXIS_ID,
    GIZMO_Z_AXIS_ID,
    GIZMO_XY_PLANE_ID,
    GIZMO_XZ_PLANE_ID,
    GIZMO_YZ_PLANE_ID,
)


class SceneViewMathMixin:
    """SceneViewMathMixin method group for SceneViewPanel."""

    @staticmethod
    def _dot3(a, b) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross3(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _sub3(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _scale3(v, scalar: float):
        return (v[0] * scalar, v[1] * scalar, v[2] * scalar)

    @staticmethod
    def _add3(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    @staticmethod
    def _closest_param_on_axis(ray_o, ray_d, axis_o, axis_d):
        """Closest-point-between-two-lines: parameter *s* along the axis line.

        Given ray P = ray_o + t*ray_d  and  axis Q = axis_o + s*axis_d,
        returns the s that minimises distance between the two lines.
        """
        w = (ray_o[0] - axis_o[0], ray_o[1] - axis_o[1], ray_o[2] - axis_o[2])
        a = ray_d[0]*ray_d[0] + ray_d[1]*ray_d[1] + ray_d[2]*ray_d[2]
        b = ray_d[0]*axis_d[0] + ray_d[1]*axis_d[1] + ray_d[2]*axis_d[2]
        c = axis_d[0]*axis_d[0] + axis_d[1]*axis_d[1] + axis_d[2]*axis_d[2]
        d = ray_d[0]*w[0] + ray_d[1]*w[1] + ray_d[2]*w[2]
        e = axis_d[0]*w[0] + axis_d[1]*w[1] + axis_d[2]*w[2]
        denom = a * c - b * b
        if abs(denom) < 1e-10:
            return -e / c if abs(c) > 1e-10 else 0.0
        return (a * e - b * d) / denom

