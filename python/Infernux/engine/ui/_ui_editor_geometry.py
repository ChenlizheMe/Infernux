"""UIEditorGeometryMixin — extracted from UIEditorPanel."""
from __future__ import annotations

"""UI Editor panel — Figma-style 2D canvas editor for screen-space UI layout.

Displays the selected UICanvas at its reference resolution and lets users
visually position UI elements via drag.  Max zoom is 100% (1:1 pixels).

Docked alongside Scene / Game views.
"""

import configparser
import math
import os
from contextlib import nullcontext as _nullcontext
from time import perf_counter as _pc

from typing import Optional
from Infernux.lib import InxGUIContext
from Infernux.engine.i18n import t
from Infernux.engine.project_context import get_project_root
from Infernux.ui.enums import TextResizeMode
from Infernux.ui.inx_ui_screen_component import clear_rect_cache
from Infernux.ui.ui_texture_cache import get_shared_cache as _get_tex_cache
from Infernux.ui.ui_render_dispatch import dispatch as _ui_dispatch
from Infernux.ui.ui_canvas_utils import collect_canvases_with_go
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .editor_icons import EditorIcons
from .theme import Theme, ImGuiCol, ImGuiStyleVar, ImGuiMouseCursor
from .ui_editor_shortcuts import UIEditorInput
from Infernux.debug import Debug
from .imgui_keys import (
    KEY_LEFT_ARROW, KEY_RIGHT_ARROW, KEY_UP_ARROW, KEY_DOWN_ARROW,
)


class UIEditorGeometryMixin:
    """UIEditorGeometryMixin method group for UIEditorPanel."""

    def _is_descendant_of(self, obj_id, ancestor_go):
        """Check if obj_id is the ancestor or one of its descendants."""
        if ancestor_go.id == obj_id:
            return True
        for child in ancestor_go.get_children():
            if self._is_descendant_of(obj_id, child):
                return True
        return False

    def _canvas_to_screen(self, cx, cy, origin_x, origin_y):
        """Canvas-space (pixels in reference resolution) → screen-space (window coords)."""
        return (origin_x + cx * self._zoom + self._pan_x,
                origin_y + cy * self._zoom + self._pan_y)

    def _screen_to_canvas(self, sx, sy, origin_x, origin_y):
        """Screen-space → canvas-space."""
        return ((sx - origin_x - self._pan_x) / self._zoom,
                (sy - origin_y - self._pan_y) / self._zoom)

    @staticmethod
    def _distance_point_to_segment(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 1e-6:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))
        cx = ax + abx * t
        cy = ay + aby * t
        return math.hypot(px - cx, py - cy)

    @staticmethod
    def _point_in_quad(px, py, corners):
        signs = []
        for idx in range(4):
            ax, ay = corners[idx]
            bx, by = corners[(idx + 1) % 4]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            signs.append(cross)
        has_pos = any(value > 0.0 for value in signs)
        has_neg = any(value < 0.0 for value in signs)
        return not (has_pos and has_neg)

    def _get_oriented_box_screen(self, elem, ref_w, ref_h, origin_x, origin_y):
        rx, ry, rw, rh = elem.get_rect(ref_w, ref_h)
        cx = rx + rw * 0.5
        cy = ry + rh * 0.5
        rot = math.radians(float(getattr(elem, 'rotation', 0.0)))
        cos_a = math.cos(rot)
        sin_a = math.sin(rot)

        local = [
            (-rw * 0.5, -rh * 0.5),
            (rw * 0.5, -rh * 0.5),
            (rw * 0.5, rh * 0.5),
            (-rw * 0.5, rh * 0.5),
        ]
        corners = []
        for lx, ly in local:
            px = cx + lx * cos_a - ly * sin_a
            py = cy + lx * sin_a + ly * cos_a
            sx, sy = self._canvas_to_screen(px, py, origin_x, origin_y)
            corners.append((round(sx), round(sy)))

        center_sx, center_sy = self._canvas_to_screen(cx, cy, origin_x, origin_y)
        top_mid = ((corners[0][0] + corners[1][0]) * 0.5, (corners[0][1] + corners[1][1]) * 0.5)
        dir_x = top_mid[0] - center_sx
        dir_y = top_mid[1] - center_sy
        dir_len = math.hypot(dir_x, dir_y)
        if dir_len <= 1e-6:
            dir_x, dir_y, dir_len = 0.0, -1.0, 1.0
        dir_x /= dir_len
        dir_y /= dir_len
        rotate_handle = (top_mid[0] + dir_x * Theme.UI_EDITOR_ROTATE_DISTANCE, top_mid[1] + dir_y * Theme.UI_EDITOR_ROTATE_DISTANCE)
        edge_mids = [
            ((corners[0][0] + corners[1][0]) * 0.5, (corners[0][1] + corners[1][1]) * 0.5),
            ((corners[1][0] + corners[2][0]) * 0.5, (corners[1][1] + corners[2][1]) * 0.5),
            ((corners[2][0] + corners[3][0]) * 0.5, (corners[2][1] + corners[3][1]) * 0.5),
            ((corners[3][0] + corners[0][0]) * 0.5, (corners[3][1] + corners[0][1]) * 0.5),
        ]
        return {
            'corners': corners,
            'center': (center_sx, center_sy),
            'rotate_handle': rotate_handle,
            'top_mid': top_mid,
            'edge_mids': edge_mids,
        }

