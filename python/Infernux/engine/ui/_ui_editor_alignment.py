"""UIEditorAlignmentMixin — extracted from UIEditorPanel."""
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


class UIEditorAlignmentMixin:
    """UIEditorAlignmentMixin method group for UIEditorPanel."""

    def _draw_alignment_guides(self, ctx: InxGUIContext, area_min_x: float, area_min_y: float):
        """Draw active alignment guides in screen space."""
        if not self._active_alignment_guides:
            return
        for orient, pos, span0, span1 in self._active_alignment_guides:
            if orient == "v":
                sx0, sy0 = self._canvas_to_screen(pos, span0, area_min_x, area_min_y)
                sx1, sy1 = self._canvas_to_screen(pos, span1, area_min_x, area_min_y)
            else:
                sx0, sy0 = self._canvas_to_screen(span0, pos, area_min_x, area_min_y)
                sx1, sy1 = self._canvas_to_screen(span1, pos, area_min_x, area_min_y)
            ctx.draw_line(sx0, sy0, sx1, sy1,
                          *Theme.UI_EDITOR_ALIGN_GUIDE,
                          Theme.UI_EDITOR_ALIGN_GUIDE_W)

    def _get_parent_alignment_rect(self, elem, ref_w: float, ref_h: float):
        """Return the parent alignment rect for the selected element."""
        px, py, pw, ph = elem._get_parent_world_rect(ref_w, ref_h)
        return (float(px), float(py), float(pw), float(ph))

    def _collect_alignment_candidates(self, canvas, selected, ref_w: float, ref_h: float):
        """Collect sibling/parent alignment candidates in canvas space."""
        candidates_x = []
        candidates_y = []
        px, py, pw, ph = self._get_parent_alignment_rect(selected, ref_w, ref_h)
        parent_rect = (px, py, pw, ph)

        def _append_rect(rect):
            rx, ry, rw, rh = rect
            candidates_x.extend([("left", rx, ry, ry + rh), ("center", rx + rw * 0.5, ry, ry + rh), ("right", rx + rw, ry, ry + rh)])
            candidates_y.extend([("top", ry, rx, rx + rw), ("center", ry + rh * 0.5, rx, rx + rw), ("bottom", ry + rh, rx, rx + rw)])

        _append_rect(parent_rect)

        sel_go = selected.game_object
        parent_go = sel_go.get_parent() if sel_go is not None else None
        for elem in canvas.iter_ui_elements():
            if elem is selected:
                continue
            elem_go = elem.game_object
            if parent_go is not None:
                if elem_go is None or elem_go.get_parent() is not parent_go:
                    continue
            elif elem_go is not None and elem_go.get_parent() is not canvas.game_object:
                continue
            _append_rect(elem.get_visual_rect(ref_w, ref_h))

        return candidates_x, candidates_y

    def _apply_alignment_snapping(self, canvas, selected, vis_x: float, vis_y: float,
                                  ref_w: float, ref_h: float):
        """Snap dragged element to parent/sibling guides (Figma-style).

        Any edge/center of the selected element can snap to any edge/center
        of a sibling or parent, and ALL matching guides at the snapped
        position are shown.
        """
        if selected is None:
            self._active_alignment_guides = []
            return vis_x, vis_y

        cur_w = float(selected.get_visual_rect(ref_w, ref_h)[2])
        cur_h = float(selected.get_visual_rect(ref_w, ref_h)[3])
        snap_tol = Theme.UI_EDITOR_ALIGN_SNAP_PX / max(self._zoom, 1e-6)
        sel_x_points = [vis_x, vis_x + cur_w * 0.5, vis_x + cur_w]
        sel_y_points = [vis_y, vis_y + cur_h * 0.5, vis_y + cur_h]
        candidates_x, candidates_y = self._collect_alignment_candidates(canvas, selected, ref_w, ref_h)
        guides = []

        # --- X axis: find smallest |delta| across ALL sel×cand pairs ---
        best_dx = None
        for sel_pos in sel_x_points:
            for _ck, cand_pos, _s0, _s1 in candidates_x:
                delta = cand_pos - sel_pos
                if abs(delta) <= snap_tol and (best_dx is None or abs(delta) < abs(best_dx)):
                    best_dx = delta

        if best_dx is not None:
            vis_x += best_dx
            # Re-derive sel points after snap
            snapped_x_points = [vis_x, vis_x + cur_w * 0.5, vis_x + cur_w]
            # Collect ALL guides at snapped positions (tolerance ≈ 0 after snap)
            _eps = 0.5
            seen_x = set()
            for sp in snapped_x_points:
                for _ck, cand_pos, span0, span1 in candidates_x:
                    if abs(cand_pos - sp) < _eps and cand_pos not in seen_x:
                        seen_x.add(cand_pos)
                        top = min(vis_y, span0)
                        bottom = max(vis_y + cur_h, span1)
                        guides.append(("v", cand_pos, top, bottom))

        # --- Y axis: same logic ---
        best_dy = None
        for sel_pos in sel_y_points:
            for _ck, cand_pos, _s0, _s1 in candidates_y:
                delta = cand_pos - sel_pos
                if abs(delta) <= snap_tol and (best_dy is None or abs(delta) < abs(best_dy)):
                    best_dy = delta

        if best_dy is not None:
            vis_y += best_dy
            snapped_y_points = [vis_y, vis_y + cur_h * 0.5, vis_y + cur_h]
            _eps = 0.5
            seen_y = set()
            for sp in snapped_y_points:
                for _ck, cand_pos, span0, span1 in candidates_y:
                    if abs(cand_pos - sp) < _eps and cand_pos not in seen_y:
                        seen_y.add(cand_pos)
                        left = min(vis_x, span0)
                        right = max(vis_x + cur_w, span1)
                        guides.append(("h", cand_pos, left, right))

        self._active_alignment_guides = guides
        return vis_x, vis_y

    def _apply_resize_alignment_snapping(self, canvas, elem, new_w, new_h,
                                         w_sign, h_sign, fixed_idx, ref_w, ref_h):
        """Snap moving edges of a resize operation to alignment guides.

        Only active for non-rotated elements (rotation makes edge snapping
        ambiguous).  Returns adjusted (new_w, new_h).
        """
        rot = float(elem.rotation) % 360.0
        if abs(rot) > 0.5:
            self._active_alignment_guides = []
            return new_w, new_h

        if w_sign == 0 and h_sign == 0:
            self._active_alignment_guides = []
            return new_w, new_h

        candidates_x, candidates_y = self._collect_alignment_candidates(
            canvas, elem, ref_w, ref_h,
        )
        snap_tol = Theme.UI_EDITOR_ALIGN_SNAP_PX / max(self._zoom, 1e-6)

        # Compute the rect that would result from the current resize
        fixed_cx, fixed_cy = self._resize_start_corners[fixed_idx]
        off_x, off_y = elem._rotated_corner_offset(new_w, new_h, fixed_idx)
        rect_x = fixed_cx - off_x
        rect_y = fixed_cy - off_y
        # For non-rotated: visual rect == content rect
        rect_left = rect_x
        rect_right = rect_x + new_w
        rect_top = rect_y
        rect_bottom = rect_y + new_h
        rect_cx = rect_x + new_w * 0.5
        rect_cy = rect_y + new_h * 0.5

        guides = []

        # --- Horizontal (X) snapping: check moving left/right edges ---
        if w_sign != 0:
            if w_sign > 0:
                check_points = [rect_right, rect_cx]
            else:
                check_points = [rect_left, rect_cx]

            best_dx = None
            for edge_pos in check_points:
                for _ck, cand_pos, _s0, _s1 in candidates_x:
                    delta = cand_pos - edge_pos
                    if abs(delta) <= snap_tol and (best_dx is None or abs(delta) < abs(best_dx)):
                        best_dx = delta

            if best_dx is not None:
                new_w += best_dx * w_sign
                new_w = max(new_w, Theme.UI_EDITOR_MIN_ELEM_SIZE)
                # Recompute rect and collect ALL matching guides
                off_xa, off_ya = elem._rotated_corner_offset(new_w, new_h, fixed_idx)
                snapped_left = fixed_cx - off_xa
                snapped_right = snapped_left + new_w
                snapped_cx = snapped_left + new_w * 0.5
                snapped_top = fixed_cy - off_ya
                snapped_bot = snapped_top + new_h
                snap_pts = [snapped_left, snapped_cx, snapped_right]
                _eps = 0.5
                seen_x = set()
                for sp in snap_pts:
                    for _ck, cand_pos, span0, span1 in candidates_x:
                        if abs(cand_pos - sp) < _eps and cand_pos not in seen_x:
                            seen_x.add(cand_pos)
                            top = min(snapped_top, span0)
                            bottom = max(snapped_bot, span1)
                            guides.append(("v", cand_pos, top, bottom))

        # --- Vertical (Y) snapping: check moving top/bottom edges ---
        if h_sign != 0:
            off_x2, off_y2 = elem._rotated_corner_offset(new_w, new_h, fixed_idx)
            rect_x2 = fixed_cx - off_x2
            rect_y2 = fixed_cy - off_y2
            rect_top2 = rect_y2
            rect_bottom2 = rect_y2 + new_h
            rect_cy2 = rect_y2 + new_h * 0.5
            rect_left2 = rect_x2
            rect_right2 = rect_x2 + new_w

            if h_sign > 0:
                check_points = [rect_bottom2, rect_cy2]
            else:
                check_points = [rect_top2, rect_cy2]

            best_dy = None
            for edge_pos in check_points:
                for _ck, cand_pos, _s0, _s1 in candidates_y:
                    delta = cand_pos - edge_pos
                    if abs(delta) <= snap_tol and (best_dy is None or abs(delta) < abs(best_dy)):
                        best_dy = delta

            if best_dy is not None:
                new_h += best_dy * h_sign
                new_h = max(new_h, Theme.UI_EDITOR_MIN_ELEM_SIZE)
                # Recompute rect and collect ALL matching guides
                off_x3, off_y3 = elem._rotated_corner_offset(new_w, new_h, fixed_idx)
                snapped_left2 = fixed_cx - off_x3
                snapped_right2 = snapped_left2 + new_w
                snapped_top2 = fixed_cy - off_y3
                snapped_bot2 = snapped_top2 + new_h
                snapped_cy2 = snapped_top2 + new_h * 0.5
                snap_pts = [snapped_top2, snapped_cy2, snapped_bot2]
                _eps = 0.5
                seen_y = set()
                for sp in snap_pts:
                    for _ck, cand_pos, span0, span1 in candidates_y:
                        if abs(cand_pos - sp) < _eps and cand_pos not in seen_y:
                            seen_y.add(cand_pos)
                            left = min(snapped_left2, span0)
                            right = max(snapped_right2, span1)
                            guides.append(("h", cand_pos, left, right))

        self._active_alignment_guides = guides
        return new_w, new_h

