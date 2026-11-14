"""UIEditorAlignmentMixin — extracted from UIEditorPanel."""
from __future__ import annotations



from infernux.lib import InxGUIContext
from .theme import Theme


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

    @staticmethod
    def _resize_visual_bounds(elem, width, height, fixed_idx, fixed_x, fixed_y):
        geometry = elem._rect_geometry_for_size(width, height)
        corner_x, corner_y = geometry.corners[fixed_idx]
        x, y, w, h = geometry.visual_rect
        return fixed_x - corner_x + x, fixed_y - corner_y + y, w, h

    def _snap_resize_x(self, elem, w_sign, new_w, new_h, fixed_idx,
                       fixed_cx, fixed_cy, candidates_x, snap_tol):
        scale_x, _ = elem.get_layout_scale()
        response = w_sign * scale_x
        if response == 0.0:
            return new_w, []
        left, top, width, height = self._resize_visual_bounds(
            elem, new_w, new_h, fixed_idx, fixed_cx, fixed_cy)
        points = ((left + width if response > 0 else left, 1.0), (left + width * .5, .5))
        best = None
        for point, fraction in points:
            for _kind, target, _begin, _end in candidates_x:
                delta = target - point
                if abs(delta) <= snap_tol and (best is None or abs(delta) < abs(best[0])):
                    best = delta, fraction
        if best is None:
            return new_w, []
        new_w = max(new_w + best[0] / (response * best[1]), Theme.UI_EDITOR_MIN_ELEM_SIZE)
        left, top, width, height = self._resize_visual_bounds(
            elem, new_w, new_h, fixed_idx, fixed_cx, fixed_cy)
        guides, seen = [], set()
        for point in (left, left + width * .5, left + width):
            for _kind, target, begin, end in candidates_x:
                if abs(target - point) < .5 and target not in seen:
                    seen.add(target)
                    guides.append(("v", target, min(top, begin), max(top + height, end)))
        return new_w, guides

    def _snap_resize_y(self, elem, h_sign, new_w, new_h, fixed_idx,
                       fixed_cx, fixed_cy, candidates_y, snap_tol):
        _, scale_y = elem.get_layout_scale()
        response = h_sign * scale_y
        if response == 0.0:
            return new_h, []
        left, top, width, height = self._resize_visual_bounds(
            elem, new_w, new_h, fixed_idx, fixed_cx, fixed_cy)
        points = ((top + height if response > 0 else top, 1.0), (top + height * .5, .5))
        best = None
        for point, fraction in points:
            for _kind, target, _begin, _end in candidates_y:
                delta = target - point
                if abs(delta) <= snap_tol and (best is None or abs(delta) < abs(best[0])):
                    best = delta, fraction
        if best is None:
            return new_h, []
        new_h = max(new_h + best[0] / (response * best[1]), Theme.UI_EDITOR_MIN_ELEM_SIZE)
        left, top, width, height = self._resize_visual_bounds(
            elem, new_w, new_h, fixed_idx, fixed_cx, fixed_cy)
        guides, seen = [], set()
        for point in (top, top + height * .5, top + height):
            for _kind, target, begin, end in candidates_y:
                if abs(target - point) < .5 and target not in seen:
                    seen.add(target)
                    guides.append(("h", target, min(left, begin), max(left + width, end)))
        return new_h, guides

    def _apply_resize_alignment_snapping(self, canvas, elem, new_w, new_h,
                                         w_sign, h_sign, fixed_idx, ref_w, ref_h):
        """Snap moving edges of a resize operation to alignment guides.

        Only active for non-rotated elements (rotation makes edge snapping
        ambiguous).  Returns adjusted (new_w, new_h).
        """
        rot = float(elem.get_layout_rotation()) % 360.0
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

        fixed_cx, fixed_cy = self._resize_start_corners[fixed_idx]

        guides = []

        new_w, x_guides = self._snap_resize_x(
            elem, w_sign, new_w, new_h, fixed_idx, fixed_cx, fixed_cy, candidates_x, snap_tol)
        guides.extend(x_guides)

        new_h, y_guides = self._snap_resize_y(
            elem, h_sign, new_w, new_h, fixed_idx, fixed_cx, fixed_cy, candidates_y, snap_tol)
        guides.extend(y_guides)

        self._active_alignment_guides = guides
        return new_w, new_h

