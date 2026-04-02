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
from .imgui_keys import (
    KEY_LEFT_ARROW, KEY_RIGHT_ARROW, KEY_UP_ARROW, KEY_DOWN_ARROW,
)


@editor_panel("UI Editor", type_id="ui_editor", title_key="panel.ui_editor")
class UIEditorPanel(EditorPanel):
    """Figma-style 2D UI editor panel."""

    WINDOW_TYPE_ID = "ui_editor"
    WINDOW_DISPLAY_NAME = "UI Editor"

    def __init__(self, title: str = "UI Editor"):
        super().__init__(title, window_id="ui_editor")

        # ── Canvas navigation ──
        self._zoom: float = 1.0
        self._pan_x: float = 0.0       # Pan offset in screen pixels
        self._pan_y: float = 0.0
        self._is_panning: bool = False

        # ── Selection state ──
        self._selected_element_comp = None   # Currently selected screen-space UI component
        self._dragging: bool = False
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
        self._drag_elem_start_x: float = 0.0
        self._drag_elem_start_y: float = 0.0

        # ── Resize handle state ──
        self._resizing: bool = False
        self._resize_handle_idx: int = -1     # Which handle is being dragged
        self._resize_start_mx: float = 0.0    # Mouse pos at resize start (screen)
        self._resize_start_my: float = 0.0
        self._resize_start_rect = (0.0, 0.0, 0.0, 0.0)  # (x, y, w, h) at start
        self._resize_start_rotation: float = 0.0
        self._resize_start_corners = [(0.0, 0.0)] * 4
        self._rotating: bool = False
        self._rotate_start_angle: float = 0.0
        self._rotate_start_rotation: float = 0.0
        self._rotate_center_sx: float = 0.0
        self._rotate_center_sy: float = 0.0

        # ── Undo snapshots for continuous interactions ──
        self._undo_pre_drag: tuple = (0.0, 0.0)        # (elem.x, elem.y) before drag
        self._undo_pre_resize: tuple = (0.0, 0.0, 0.0, 0.0)  # (x, y, w, h) before resize
        self._undo_pre_resize_mode = None               # resize_mode before resize (TextUI only)
        self._undo_pre_rotate: float = 0.0              # elem.rotation before rotate

        # ── External references ──
        self._engine = None                  # Engine instance (for game texture)
        self._on_selection_changed = None    # Callback(go_or_None)
        self._hierarchy_panel = None
        self._on_request_ui_mode = None      # Callback(bool) to toggle hierarchy UI mode

        # ── Focus tracking ──
        self._was_focused: bool = False
        self._settings_loaded: bool = False
        self._active_alignment_guides: list[tuple[str, float, float, float]] = []

        # ── Click-through cycling state (Unity/ScenePanel-style) ──
        self._pick_cycle_candidates: list = []
        self._pick_cycle_index: int = -1
        self._pick_cycle_last_canvas_pos: tuple[float, float] = (-1.0, -1.0)

        # ── Multi-canvas layout ──
        self._canvas_panel_positions: dict = {}   # {go_id: [wx, wy]} workspace coords
        self._focused_canvas_id: int = 0          # GO id of the focused canvas
        self._dragging_canvas: bool = False
        self._drag_canvas_id: int = 0
        self._drag_canvas_start_mx: float = 0.0
        self._drag_canvas_start_my: float = 0.0
        self._drag_canvas_start_wx: float = 0.0
        self._drag_canvas_start_wy: float = 0.0



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_on_selection_changed(self, callback):
        """Set callback when a UI element is selected.  Receives the GameObject."""
        self._on_selection_changed = callback

    def set_hierarchy_panel(self, panel):
        self._hierarchy_panel = panel

    def set_engine(self, engine):
        """Set engine instance (needed for Game background mode)."""
        self._engine = engine

    def set_on_request_ui_mode(self, callback):
        """callback(enter: bool) — ask hierarchy to enter/exit UI mode."""
        self._on_request_ui_mode = callback

    def notify_hierarchy_selection(self, go):
        """Sync UI editor selection when hierarchy selection changes.

        If *go* has an InxUIScreenComponent, select it; if it's a Canvas,
        focus that canvas; otherwise clear element selection but keep canvas
        focus if the object is inside a canvas tree.
        """
        if go is None:
            self._clear_interaction_state()
            return
        self._focus_canvas_for_object(go)
        # Reset interaction state but keep focused canvas
        self._dragging = False
        self._resizing = False
        self._resize_handle_idx = -1
        self._rotating = False
        self._active_alignment_guides = []
        self._selected_element_comp = None
        from Infernux.ui import UICanvas
        from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
        for comp in go.get_py_components():
            if isinstance(comp, UICanvas):
                self._focused_canvas_id = go.id
                return
            if isinstance(comp, InxUIScreenComponent):
                self._selected_element_comp = comp
                return

    def _clear_interaction_state(self):
        """Reset all selection / drag / resize state."""
        self._selected_element_comp = None
        self._dragging = False
        self._resizing = False
        self._resize_handle_idx = -1
        self._rotating = False
        self._active_alignment_guides = []

    def _settings_ini_path(self) -> Optional[str]:
        root = get_project_root()
        if not root:
            return None
        return os.path.join(root, "ProjectSettings", "GameView.ini")

    def _load_view_settings(self):
        if self._settings_loaded:
            return
        self._settings_loaded = True

        path = self._settings_ini_path()
        if not path:
            return
        if not os.path.isfile(path):
            self._save_view_settings()
            return

        cp = configparser.ConfigParser()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                cp.read_string(f.read())
        except (OSError, configparser.Error):
            return

        if "UIEditor" not in cp:
            self._save_view_settings()
            return

        section = cp["UIEditor"]
        self._zoom = max(Theme.UI_EDITOR_MIN_ZOOM, min(Theme.UI_EDITOR_MAX_ZOOM, section.getfloat("zoom", fallback=1.0)))
        self._pan_x = section.getfloat("pan_x", fallback=0.0)
        self._pan_y = section.getfloat("pan_y", fallback=0.0)
        # bg_mode removed — always solid background

        # Canvas panel positions: stored as "canvas_pos_<id> = x,y"
        self._canvas_panel_positions.clear()
        for key, value in section.items():
            if key.startswith("canvas_pos_"):
                try:
                    go_id = int(key[len("canvas_pos_"):])
                    parts = value.split(",")
                    if len(parts) == 2:
                        self._canvas_panel_positions[go_id] = [float(parts[0]), float(parts[1])]
                except (ValueError, IndexError):
                    pass

    def _save_view_settings(self):
        path = self._settings_ini_path()
        if not path:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cp = configparser.ConfigParser()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    cp.read_string(f.read())
            except (OSError, configparser.Error):
                cp = configparser.ConfigParser()

        cp["UIEditor"] = {
            "zoom": f"{self._zoom:.6f}",
            "pan_x": f"{self._pan_x:.3f}",
            "pan_y": f"{self._pan_y:.3f}",
        }
        # Persist canvas panel positions
        for go_id, pos in self._canvas_panel_positions.items():
            cp["UIEditor"][f"canvas_pos_{go_id}"] = f"{pos[0]:.1f},{pos[1]:.1f}"
        with open(path, "w", encoding="utf-8") as f:
            cp.write(f)

    # ------------------------------------------------------------------
    # Helpers — canvas / element discovery
    # ------------------------------------------------------------------

    def _get_all_canvases(self):
        """Return list of (GameObject, UICanvas) for every Canvas in the scene."""
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            return []
        return collect_canvases_with_go(scene)

    def _get_active_canvas(self):
        """Return (go, UICanvas) for the first canvas, or (None, None)."""
        canvases = self._get_all_canvases()
        if not canvases:
            return None, None
        # If hierarchy has a selected object, prefer canvas that is ancestor
        if self._hierarchy_panel:
            sel_id = getattr(self._hierarchy_panel, '_selected_object_id', 0)
            if sel_id:
                for go, canvas in canvases:
                    if self._is_descendant_of(sel_id, go):
                        return go, canvas
        return canvases[0]

    def _get_focused_canvas(self, all_canvases=None):
        """Return (go, UICanvas) the user is currently focused on, or (None, None)."""
        if all_canvases is None:
            all_canvases = self._get_all_canvases()
        if not all_canvases:
            return None, None
        for go, canvas in all_canvases:
            if go.id == self._focused_canvas_id:
                return go, canvas
        # Fallback to first canvas
        go, canvas = all_canvases[0]
        self._focused_canvas_id = go.id
        return go, canvas

    def _find_canvas_for_object(self, go, all_canvases=None):
        """Return the owning canvas pair for *go*, or (None, None)."""
        if go is None:
            return None, None
        if all_canvases is None:
            all_canvases = self._get_all_canvases()
        for canvas_go, canvas in all_canvases:
            if self._is_descendant_of(go.id, canvas_go):
                return canvas_go, canvas
        return None, None

    def _focus_canvas_for_object(self, go, all_canvases=None):
        """Set the focused canvas to the one that owns *go*."""
        canvas_go, canvas = self._find_canvas_for_object(go, all_canvases)
        if canvas_go is not None:
            self._focused_canvas_id = canvas_go.id
        return canvas_go, canvas

    # ------------------------------------------------------------------
    # Multi-canvas layout helpers
    # ------------------------------------------------------------------

    def _ensure_canvas_layout(self, all_canvases):
        """Ensure every canvas has a workspace position.  Auto-layout new ones."""
        changed = False
        for go, canvas in all_canvases:
            if go.id not in self._canvas_panel_positions:
                pos = self._auto_layout_position(canvas, all_canvases)
                self._canvas_panel_positions[go.id] = list(pos)
                changed = True
        # Prune stale IDs
        active_ids = {go.id for go, _ in all_canvases}
        for stale in [k for k in self._canvas_panel_positions if k not in active_ids]:
            del self._canvas_panel_positions[stale]
            changed = True
        if changed:
            self._save_view_settings()

    def _auto_layout_position(self, canvas, all_canvases):
        """Find a position to the right of all existing canvases."""
        SPACING = Theme.UI_EDITOR_CANVAS_SPACING
        if not self._canvas_panel_positions:
            return (0.0, 0.0)
        max_right = 0.0
        for go, cv in all_canvases:
            if go.id in self._canvas_panel_positions:
                wx = self._canvas_panel_positions[go.id][0]
                right = wx + float(cv.reference_width)
                if right > max_right:
                    max_right = right
        return (max_right + SPACING, 0.0)

    def _canvas_workspace_rect(self, go_id, canvas):
        """Return (x, y, w, h) of a canvas in workspace coords (includes header)."""
        pos = self._canvas_panel_positions.get(go_id, [0.0, 0.0])
        ref_w = float(canvas.reference_width)
        ref_h = float(canvas.reference_height)
        header_h = Theme.UI_EDITOR_CANVAS_HEADER_H / max(self._zoom, 0.01)
        return (pos[0], pos[1] - header_h, ref_w, ref_h + header_h)

    @staticmethod
    def _rects_overlap(ax, ay, aw, ah, bx, by, bw, bh):
        return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

    def _clamp_canvas_no_overlap(self, drag_id, new_wx, new_wy, all_canvases):
        """Push back canvas position to prevent overlap with other canvases."""
        drag_canvas = None
        for go, cv in all_canvases:
            if go.id == drag_id:
                drag_canvas = cv
                break
        if drag_canvas is None:
            return new_wx, new_wy

        ref_w = float(drag_canvas.reference_width)
        ref_h = float(drag_canvas.reference_height)
        header_h = Theme.UI_EDITOR_CANVAS_HEADER_H / max(self._zoom, 0.01)
        drag_rect = (new_wx, new_wy - header_h, ref_w, ref_h + header_h)

        for go, cv in all_canvases:
            if go.id == drag_id:
                continue
            other = self._canvas_workspace_rect(go.id, cv)
            if self._rects_overlap(*drag_rect, *other):
                # Push to nearest non-overlapping position
                # Try four directions and pick shortest displacement
                candidates = []
                # Push right of other
                rx = other[0] + other[2]
                candidates.append((rx, new_wy, abs(rx - new_wx)))
                # Push left of other
                lx = other[0] - ref_w
                candidates.append((lx, new_wy, abs(lx - new_wx)))
                # Push below other
                by_ = other[1] + other[3] + header_h
                candidates.append((new_wx, by_, abs(by_ - new_wy)))
                # Push above other
                ty = other[1] - ref_h - header_h
                candidates.append((new_wx, ty, abs(ty - new_wy)))
                candidates.sort(key=lambda c: c[2])
                new_wx, new_wy = candidates[0][0], candidates[0][1]
                drag_rect = (new_wx, new_wy - header_h, ref_w, ref_h + header_h)
        return new_wx, new_wy

    def _get_focused_canvas_origin(self, area_min_x, area_min_y, all_canvases=None):
        """Get screen-space origin for the focused canvas."""
        if all_canvases is None:
            all_canvases = self._get_all_canvases()
        foc_go, foc_cv = self._get_focused_canvas(all_canvases)
        if foc_go is None:
            return area_min_x, area_min_y
        pos = self._canvas_panel_positions.get(foc_go.id, [0.0, 0.0])
        return (area_min_x + pos[0] * self._zoom, area_min_y + pos[1] * self._zoom)

    def _is_descendant_of(self, obj_id, ancestor_go):
        """Check if obj_id is the ancestor or one of its descendants."""
        if ancestor_go.id == obj_id:
            return True
        for child in ancestor_go.get_children():
            if self._is_descendant_of(obj_id, child):
                return True
        return False

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

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

    def _sync_text_layout(self, ctx: InxGUIContext, text_comp):
        text = getattr(text_comp, "text", "")
        font_size = max(1.0, float(getattr(text_comp, "font_size", 24.0)))
        wrap_width = float(text_comp.get_editor_wrap_width()) if hasattr(text_comp, "get_editor_wrap_width") else (
            float(text_comp.get_wrap_width()) if hasattr(text_comp, "get_wrap_width") else 0.0
        )
        font_path = str(getattr(text_comp, "font_path", "") or "")
        line_height = float(getattr(text_comp, "line_height", 1.2))
        letter_spacing = float(getattr(text_comp, "letter_spacing", 0.0))
        pad_x, pad_y = getattr(text_comp, "get_auto_size_padding", lambda: (0.0, 0.0))()
        if wrap_width > 0.0:
            measured_w, measured_h = ctx.calc_text_size_wrapped(
                text, font_size, wrap_width, font_path, line_height, letter_spacing
            )
        else:
            measured_w, measured_h = ctx.calc_text_size(text, font_size, font_path, line_height, letter_spacing)

        _TOL = 0.5  # Skip writes when measured size is close enough (sub-pixel)

        _, canvas = self._get_focused_canvas()
        if canvas is None:
            if getattr(text_comp, "is_auto_width", lambda: False)():
                target_w = max(1.0, float(measured_w) + float(pad_x))
                if abs(float(text_comp.width) - target_w) > _TOL:
                    text_comp.width = target_w
            elif getattr(text_comp, "is_auto_height", lambda: False)():
                target_h = max(1.0, float(measured_h) + float(pad_y))
                if abs(float(text_comp.height) - target_h) > _TOL:
                    text_comp.height = target_h
            return

        # While the user is actively dragging or resizing, only adjust the
        # auto-sized dimension (width or height) without touching x/y.
        # set_size_preserve_corner rewrites x/y which fights the drag.
        _interacting = (self._dragging or self._resizing)

        if getattr(text_comp, "is_auto_width", lambda: False)():
            target_w = max(1.0, float(measured_w) + float(pad_x))
            if abs(float(text_comp.width) - target_w) > _TOL:
                if _interacting:
                    text_comp.width = target_w
                else:
                    text_comp.set_size_preserve_corner(
                        target_w,
                        float(text_comp.height),
                        float(canvas.reference_width),
                        float(canvas.reference_height),
                        "top_left",
                    )
        elif getattr(text_comp, "is_auto_height", lambda: False)():
            target_h = max(1.0, float(measured_h) + float(pad_y))
            if abs(float(text_comp.height) - target_h) > _TOL:
                if _interacting:
                    text_comp.height = target_h
                else:
                    text_comp.set_size_preserve_corner(
                        float(text_comp.width),
                        target_h,
                        float(canvas.reference_width),
                        float(canvas.reference_height),
                        "top_left",
                    )

    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (Theme.UI_EDITOR_INIT_WINDOW_W, Theme.UI_EDITOR_INIT_WINDOW_H)

    def _window_flags(self) -> int:
        return Theme.WINDOW_FLAGS_VIEWPORT | Theme.WINDOW_FLAGS_NO_SCROLL

    def _pre_render(self, ctx):
        self._load_view_settings()

    def _on_visible_pre(self, ctx):
        # Keep UI mode active the entire time the panel is visible
        if self._on_request_ui_mode:
            self._on_request_ui_mode(True)
        self._was_focused = ctx.is_window_focused(0)

    def _on_not_visible(self, ctx):
        if self._on_request_ui_mode:
            self._on_request_ui_mode(False)
        self._was_focused = False

    def on_render_content(self, ctx: InxGUIContext):
        all_canvases = self._get_all_canvases()
        if not all_canvases:
            self._render_no_canvas(ctx)
        else:
            self._ensure_canvas_layout(all_canvases)
            focused_go, focused_cv = self._get_focused_canvas(all_canvases)
            self._render_toolbar(ctx, focused_go, focused_cv)
            self._render_multi_canvas_area(ctx, all_canvases)

    # ── No Canvas placeholder ────────────────────────────────────────

    def _render_no_canvas(self, ctx: InxGUIContext):
        # Canvas is gone — clear any stale selection / interaction state
        self._clear_interaction_state()
        ctx.label("")
        ctx.label("  " + t("ui_editor.no_canvas"))
        ctx.label("")
        ctx.label("  " + t("ui_editor.create_canvas_hint").split("\n")[0])
        ctx.label("  " + t("ui_editor.create_canvas_hint").split("\n")[1])
        ctx.label("")
        ctx.button(t("ui_editor.create_canvas"), self._create_canvas,
                   width=Theme.UI_EDITOR_CREATE_BTN_W, height=Theme.UI_EDITOR_CREATE_BTN_H)

    # ── Toolbar ──────────────────────────────────────────────────────

    def _render_toolbar(self, ctx: InxGUIContext, canvas_go, canvas):
        """Top toolbar with creation buttons, zoom control, and background toggle."""
        _GAP = Theme.UI_EDITOR_TOOLBAR_GAP
        _SEC = Theme.UI_EDITOR_TOOLBAR_SECTION_GAP

        ctx.label(f"Canvas: {canvas_go.name}")
        ctx.same_line(0, _SEC)

        native = self._engine.get_native_engine() if self._engine else None
        _ICO_SZ = Theme.EDITOR_ICON_SIZE

        # ── Create Canvas button ──
        ctx.button("Canvas", self._create_canvas, width=72, height=_ICO_SZ + 8)
        if ctx.is_item_hovered():
            ctx.set_tooltip(t("ui_editor.tooltip_canvas"))

        ctx.same_line(0, _GAP)
        tid_text = EditorIcons.get(native, Theme.ICON_IMG_UI_TEXT)
        if tid_text:
            if ctx.image_button("##add_text", tid_text, _ICO_SZ, _ICO_SZ):
                self._create_text_element(canvas_go)
        else:
            ctx.button("T", lambda: self._create_text_element(canvas_go), width=_ICO_SZ + 8, height=_ICO_SZ + 8)
        if ctx.is_item_hovered():
            ctx.set_tooltip(t("ui_editor.tooltip_text"))

        ctx.same_line(0, _GAP)
        tid_img = EditorIcons.get(native, Theme.ICON_IMG_UI_IMAGE)
        if tid_img:
            if ctx.image_button("##add_image", tid_img, _ICO_SZ, _ICO_SZ):
                self._create_image_element(canvas_go)
        else:
            ctx.button("Img", lambda: self._create_image_element(canvas_go), width=_ICO_SZ + 8, height=_ICO_SZ + 8)
        if ctx.is_item_hovered():
            ctx.set_tooltip(t("ui_editor.tooltip_image"))

        ctx.same_line(0, _GAP)
        tid_btn = EditorIcons.get(native, Theme.ICON_IMG_UI_BUTTON)
        if tid_btn:
            if ctx.image_button("##add_button", tid_btn, _ICO_SZ, _ICO_SZ):
                self._create_button_element(canvas_go)
        else:
            ctx.button("Btn", lambda: self._create_button_element(canvas_go), width=_ICO_SZ + 8, height=_ICO_SZ + 8)
        if ctx.is_item_hovered():
            ctx.set_tooltip(t("ui_editor.tooltip_button"))

        ctx.same_line(0, _SEC // 2)
        zoom_pct = int(self._zoom * 100)
        ctx.label(t("ui_editor.zoom").format(pct=zoom_pct))
        ctx.same_line(0, _SEC // 2)
        ctx.button(t("ui_editor.fit"), lambda: self._fit_zoom(ctx, canvas), width=56)

        ctx.separator()

    # ── Canvas area (multi-canvas) ───────────────────────────────────

    def _render_multi_canvas_area(self, ctx: InxGUIContext, all_canvases):
        """Main area: multi-canvas workspace with zoomable panels."""

        # Validate selection against all canvases
        if self._selected_element_comp is not None:
            try:
                sel_go = self._selected_element_comp.game_object
                if sel_go is None:
                    self._clear_interaction_state()
                else:
                    found = any(self._is_descendant_of(sel_go.id, cgo)
                                for cgo, _ in all_canvases)
                    if not found:
                        self._clear_interaction_state()
            except Exception:
                self._clear_interaction_state()

        # Content region (below toolbar)
        region_w = ctx.get_content_region_avail_width()
        region_h = ctx.get_content_region_avail_height()
        if region_w < 1 or region_h < 1:
            return

        ctx.invisible_button("##ui_canvas_area", region_w, region_h)
        area_hovered = ctx.is_item_hovered()

        area_min_x = ctx.get_item_rect_min_x()
        area_min_y = ctx.get_item_rect_min_y()
        area_max_x = area_min_x + region_w
        area_max_y = area_min_y + region_h

        # ── Input snapshot ──
        inp = UIEditorInput(ctx, area_hovered)

        # ── Zoom (mouse wheel) ──
        if abs(inp.wheel_delta) > 0.01:
            old_zoom = self._zoom
            self._zoom = max(Theme.UI_EDITOR_MIN_ZOOM,
                             min(Theme.UI_EDITOR_MAX_ZOOM,
                                 self._zoom * (1.0 + inp.wheel_delta * Theme.UI_EDITOR_ZOOM_STEP)))
            factor = self._zoom / old_zoom
            self._pan_x = inp.mouse_x - area_min_x - factor * (inp.mouse_x - area_min_x - self._pan_x)
            self._pan_y = inp.mouse_y - area_min_y - factor * (inp.mouse_y - area_min_y - self._pan_y)
            self._save_view_settings()

        # ── Canvas panel drag ──
        if self._dragging_canvas:
            if inp.lmb_down:
                dx = (inp.mouse_x - self._drag_canvas_start_mx) / self._zoom
                dy = (inp.mouse_y - self._drag_canvas_start_my) / self._zoom
                new_wx = self._drag_canvas_start_wx + dx
                new_wy = self._drag_canvas_start_wy + dy
                new_wx, new_wy = self._clamp_canvas_no_overlap(
                    self._drag_canvas_id, new_wx, new_wy, all_canvases)
                self._canvas_panel_positions[self._drag_canvas_id] = [new_wx, new_wy]
            else:
                self._dragging_canvas = False
                self._drag_canvas_id = 0
                self._save_view_settings()

        # ── Workspace pan (Space+LMB or MMB) ──
        if inp.wants_pan and not self._dragging_canvas:
            if not self._is_panning:
                self._is_panning = True
            drag_btn = inp.pan_drag_button
            dx = ctx.get_mouse_drag_delta_x(drag_btn)
            dy = ctx.get_mouse_drag_delta_y(drag_btn)
            self._pan_x += dx
            self._pan_y += dy
            ctx.reset_mouse_drag_delta(drag_btn)
            self._save_view_settings()
        else:
            self._is_panning = False

        # ── Focused canvas for element interactions ──
        focused_go, focused_canvas = self._get_focused_canvas(all_canvases)
        foc_origin_x, foc_origin_y = self._get_focused_canvas_origin(
            area_min_x, area_min_y, all_canvases)

        clear_rect_cache(_pc())

        # ── Process ongoing element interactions BEFORE drawing (zero-lag) ──
        if focused_canvas is not None:
            foc_ref_w = float(focused_canvas.reference_width)
            foc_ref_h = float(focused_canvas.reference_height)
        else:
            foc_ref_w = foc_ref_h = 1.0

        if self._resizing:
            if inp.lmb_down:
                self._apply_resize_suppressed(inp)
            else:
                self._record_resize_undo()
                self._resizing = False
                self._resize_handle_idx = -1

        if self._rotating:
            if inp.lmb_down:
                self._apply_rotation_drag_suppressed(inp)
            else:
                self._record_rotate_undo()
                self._rotating = False

        if self._dragging:
            if inp.lmb_down:
                dx = (inp.mouse_x - self._drag_start_x) / self._zoom
                dy = (inp.mouse_y - self._drag_start_y) / self._zoom
                if inp.ctrl_down:
                    if abs(dx) >= abs(dy):
                        dy = 0.0
                    else:
                        dx = 0.0
                snap = self._drag_snap_step()
                new_vis_x = round((self._drag_elem_start_x + dx) / snap) * snap
                new_vis_y = round((self._drag_elem_start_y + dy) / snap) * snap
                if focused_canvas is not None:
                    new_vis_x, new_vis_y = self._apply_alignment_snapping(
                        focused_canvas, self._selected_element_comp,
                        new_vis_x, new_vis_y, foc_ref_w, foc_ref_h,
                    )
                self._apply_drag_suppressed(new_vis_x, new_vis_y, foc_ref_w, foc_ref_h)
            else:
                self._record_drag_undo()
                self._dragging = False
                self._active_alignment_guides = []
        elif not self._resizing and not self._rotating:
            self._active_alignment_guides = []

        # ══════════════════════════════════════════════════════════════
        #  Draw all canvases
        # ══════════════════════════════════════════════════════════════
        from Infernux.ui import UIText
        _tex_cache = _get_tex_cache()
        _get_tid = lambda tp: _tex_cache.get(self._engine, tp)

        ctx.push_draw_list_clip_rect(area_min_x, area_min_y, area_max_x, area_max_y, True)

        hovered_canvas_id = 0
        hovered_elem = None
        hovered_all: list = []
        HEADER_H = Theme.UI_EDITOR_CANVAS_HEADER_H
        _PICK_TOL = 3.0 / self._zoom  # 3 screen-px tolerance in canvas space

        for canvas_go, canvas in all_canvases:
            go_id = canvas_go.id
            panel_pos = self._canvas_panel_positions.get(go_id, [0.0, 0.0])
            origin_x = area_min_x + panel_pos[0] * self._zoom
            origin_y = area_min_y + panel_pos[1] * self._zoom

            ref_w = float(canvas.reference_width)
            ref_h = float(canvas.reference_height)

            # Active / enabled state
            go_active = canvas_go.active_in_hierarchy
            canvas_enabled = getattr(canvas, 'enabled', True)
            is_active = go_active and canvas_enabled
            alpha_mult = 1.0 if is_active else Theme.UI_EDITOR_CANVAS_INACTIVE_ALPHA

            # Canvas rect in screen space
            c_tl_x, c_tl_y = self._canvas_to_screen(0, 0, origin_x, origin_y)
            c_br_x, c_br_y = self._canvas_to_screen(ref_w, ref_h, origin_x, origin_y)
            c_tl_x = round(c_tl_x)
            c_tl_y = round(c_tl_y)
            c_br_x = round(c_br_x)
            c_br_y = round(c_br_y)

            # Header rect (above canvas)
            h_tl_y = c_tl_y - HEADER_H

            # Visible clamped rects
            v_tl_x = max(c_tl_x, area_min_x)
            v_tl_y = max(c_tl_y, area_min_y)
            v_br_x = min(c_br_x, area_max_x)
            v_br_y = min(c_br_y, area_max_y)
            canvas_visible = (v_br_x > v_tl_x and v_br_y > v_tl_y)

            v_h_tl_x = max(c_tl_x, area_min_x)
            v_h_tl_y = max(h_tl_y, area_min_y)
            v_h_br_x = min(c_br_x, area_max_x)
            v_h_br_y = min(c_tl_y, area_max_y)
            header_visible = (v_h_br_x > v_h_tl_x and v_h_br_y > v_h_tl_y)

            # ── Draw header ──
            is_focused = (go_id == self._focused_canvas_id)
            if header_visible:
                hdr_bg = Theme.UI_EDITOR_CANVAS_HEADER_BG_FOC if is_focused else Theme.UI_EDITOR_CANVAS_HEADER_BG
                ctx.draw_filled_rect(v_h_tl_x, v_h_tl_y, v_h_br_x, v_h_br_y,
                                     hdr_bg[0], hdr_bg[1], hdr_bg[2],
                                     hdr_bg[3] * alpha_mult, 0.0)
                label = f"{canvas_go.name}  {int(ref_w)}\u00d7{int(ref_h)}"
                if not is_active:
                    label += "  (inactive)"
                tc = Theme.UI_EDITOR_CANVAS_HEADER_TEXT
                ctx.draw_text(c_tl_x + 6, h_tl_y + 3, label,
                              tc[0], tc[1], tc[2], tc[3] * alpha_mult, 0.0)

            # ── Draw canvas background ──
            if canvas_visible:
                bg = Theme.UI_EDITOR_CANVAS_BG
                ctx.draw_filled_rect(v_tl_x, v_tl_y, v_br_x, v_br_y,
                                     bg[0], bg[1], bg[2], bg[3] * alpha_mult, 0.0)

            # ── Hit-test: is mouse over this canvas? ──
            if area_hovered and not self._dragging_canvas:
                if (c_tl_x <= inp.mouse_x <= c_br_x
                        and h_tl_y <= inp.mouse_y <= c_br_y):
                    hovered_canvas_id = go_id
                    # Only pick elements on the focused canvas (or any if none focused)
                    if (is_active and c_tl_y <= inp.mouse_y <= c_br_y
                            and (is_focused or not self._focused_canvas_id)):
                        cmx, cmy = self._screen_to_canvas(
                            inp.mouse_x, inp.mouse_y, origin_x, origin_y)
                        if 0.0 <= cmx <= ref_w and 0.0 <= cmy <= ref_h:
                            _all = canvas.raycast_all(cmx, cmy, _PICK_TOL)
                            if _all:
                                hovered_all = _all
                                hovered_elem = _all[0]
                # Focused canvas: also pick elements outside canvas bounds
                if is_focused and is_active and not hovered_elem:
                    cmx, cmy = self._screen_to_canvas(
                        inp.mouse_x, inp.mouse_y, origin_x, origin_y)
                    _all = canvas.raycast_all(cmx, cmy, _PICK_TOL)
                    if _all:
                        hovered_all = _all
                        hovered_elem = _all[0]
                        if not hovered_canvas_id:
                            hovered_canvas_id = go_id

            # ── Skip element rendering for inactive canvases ──
            if not is_active:
                continue

            # ── Draw UI elements ──
            elements = list(canvas.iter_ui_elements())

            for elem in elements:
                if isinstance(elem, UIText):
                    self._sync_text_layout(ctx, elem)
            clear_rect_cache(_pc())

            for elem in elements:
                elem_go = elem.game_object
                if elem_go is not None and not elem_go.active_in_hierarchy:
                    continue
                if not getattr(elem, 'enabled', True):
                    continue

                ex, ey, ew, eh = elem.get_visual_rect(ref_w, ref_h)
                base_x, base_y, base_w, base_h = elem.get_rect(ref_w, ref_h)
                s_x, s_y = self._canvas_to_screen(ex, ey, origin_x, origin_y)
                s_w = ew * self._zoom
                s_h = eh * self._zoom
                base_sx, base_sy = self._canvas_to_screen(base_x, base_y, origin_x, origin_y)
                base_sw = base_w * self._zoom
                base_sh = base_h * self._zoom

                s_x = round(s_x)
                s_y = round(s_y)
                s_w = round(s_w)
                s_h = round(s_h)

                is_hovered = (elem is hovered_elem)
                is_selected = (elem is self._selected_element_comp)

                cx0 = max(s_x, area_min_x)
                cy0 = max(s_y, area_min_y)
                cx1 = min(s_x + s_w, area_max_x)
                cy1 = min(s_y + s_h, area_max_y)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                if not is_selected and is_hovered:
                    ctx.draw_filled_rect(cx0, cy0, cx1, cy1,
                                         *Theme.UI_EDITOR_ELEMENT_HOVER, 0.0)
                    ctx.draw_rect(cx0, cy0, cx1, cy1,
                                  *Theme.UI_EDITOR_ELEMENT_SELECT[:3], 0.6, 1.0, 0.0)

                if not _ui_dispatch(
                    elem, "editor",
                    ctx=ctx,
                    base_sx=base_sx, base_sy=base_sy,
                    base_sw=base_sw, base_sh=base_sh,
                    zoom=self._zoom,
                    get_tex_id=_get_tid,
                ):
                    tx = max(s_x + 2, area_min_x)
                    ty = max(s_y + 2, area_min_y)
                    if tx < area_max_x and ty < area_max_y:
                        ctx.draw_text(tx, ty,
                                      elem.type_name, *Theme.UI_EDITOR_FALLBACK_TEXT, 0.0)

        # ── Fallback hit-test: focused canvas elements outside canvas bounds ──
        if not hovered_elem and self._focused_canvas_id and area_hovered and not self._dragging_canvas:
            for cgo, cv in all_canvases:
                if cgo.id != self._focused_canvas_id:
                    continue
                if not cgo.active_in_hierarchy or not getattr(cv, 'enabled', True):
                    break
                pp = self._canvas_panel_positions.get(cgo.id, [0.0, 0.0])
                foc_ox = area_min_x + pp[0] * self._zoom
                foc_oy = area_min_y + pp[1] * self._zoom
                cmx, cmy = self._screen_to_canvas(inp.mouse_x, inp.mouse_y, foc_ox, foc_oy)
                _all = cv.raycast_all(cmx, cmy, _PICK_TOL)
                if _all:
                    hovered_all = _all
                    hovered_elem = _all[0]
                    if not hovered_canvas_id:
                        hovered_canvas_id = cgo.id
                break

        # ══════════════════════════════════════════════════════════════
        #  Selection overlay (focused canvas only)
        # ══════════════════════════════════════════════════════════════
        self._handle_positions = []
        self._draw_alignment_guides(ctx, foc_origin_x, foc_origin_y)
        if self._selected_element_comp is not None and focused_canvas is not None:
            sel = self._selected_element_comp
            self._selection_geometry = self._get_oriented_box_screen(
                sel, foc_ref_w, foc_ref_h, foc_origin_x, foc_origin_y)
            corners = self._selection_geometry['corners']
            _HS = Theme.UI_EDITOR_HANDLE_SIZE
            self._handle_positions = [(cx - _HS, cy - _HS) for cx, cy in corners]

            for idx in range(4):
                ax, ay = corners[idx]
                bx, by = corners[(idx + 1) % 4]
                ctx.draw_line(ax, ay, bx, by,
                              *Theme.UI_EDITOR_ELEMENT_SELECT, Theme.UI_EDITOR_SELECT_LINE_W)

            top_mid_x, top_mid_y = self._selection_geometry['top_mid']
            rotate_x, rotate_y = self._selection_geometry['rotate_handle']
            ctx.draw_line(top_mid_x, top_mid_y, rotate_x, rotate_y,
                          *Theme.UI_EDITOR_ELEMENT_SELECT, Theme.UI_EDITOR_ROTATE_LINE_W)
            ctx.draw_filled_circle(rotate_x, rotate_y, Theme.UI_EDITOR_ROTATE_RADIUS,
                                   *Theme.UI_EDITOR_HANDLE_COLOR, 0)
            ctx.draw_circle(rotate_x, rotate_y, Theme.UI_EDITOR_ROTATE_RADIUS,
                            *Theme.UI_EDITOR_ELEMENT_SELECT, 1.0, 0)

            hs2 = _HS * 2
            for px, py in self._handle_positions:
                h_x1 = px + hs2
                h_y1 = py + hs2
                if (h_x1 > area_min_x and h_y1 > area_min_y
                        and px < area_max_x and py < area_max_y):
                    ctx.draw_filled_rect(px, py, h_x1, h_y1,
                                         *Theme.UI_EDITOR_HANDLE_COLOR, 0.0)
                    ctx.draw_rect(px, py, h_x1, h_y1,
                                  *Theme.UI_EDITOR_ELEMENT_SELECT, 1.0, 0.0)
        else:
            self._selection_geometry = None

        ctx.pop_draw_list_clip_rect()

        self._update_hover_cursor(ctx, area_hovered, inp.mouse_x, inp.mouse_y)

        # ══════════════════════════════════════════════════════════════
        #  Keyboard shortcuts
        # ══════════════════════════════════════════════════════════════
        if inp.wants_deselect():
            self._select_element(None)
        if inp.wants_delete() and self._selected_element_comp is not None:
            self._delete_selected_element()

        if (self._selected_element_comp is not None
                and focused_canvas is not None
                and ctx.is_window_focused(0)
                and not ctx.want_text_input()):
            dx = dy = 0
            if ctx.is_key_pressed(KEY_LEFT_ARROW):
                dx = -1
            elif ctx.is_key_pressed(KEY_RIGHT_ARROW):
                dx = 1
            if ctx.is_key_pressed(KEY_UP_ARROW):
                dy = -1
            elif ctx.is_key_pressed(KEY_DOWN_ARROW):
                dy = 1
            if dx != 0 or dy != 0:
                self._nudge_selected(dx, dy, foc_ref_w, foc_ref_h)

        # ══════════════════════════════════════════════════════════════
        #  Click handling
        # ══════════════════════════════════════════════════════════════
        _cycle_trigger = (
            (inp.lmb_double_clicked or (inp.ctrl_down and inp.lmb_clicked))
            and not inp.space_down and hovered_all and len(hovered_all) > 1
        )
        if _cycle_trigger:
            # Focus the canvas containing the hovered element
            if hovered_canvas_id:
                self._focused_canvas_id = hovered_canvas_id
            _CYCLE_TOL = 3.0 / self._zoom
            cx, cy = self._screen_to_canvas(inp.mouse_x, inp.mouse_y,
                                            foc_origin_x, foc_origin_y)
            lx, ly = self._pick_cycle_last_canvas_pos
            same_spot = (abs(cx - lx) < _CYCLE_TOL and abs(cy - ly) < _CYCLE_TOL)
            if same_spot and self._pick_cycle_candidates == hovered_all:
                self._pick_cycle_index = (self._pick_cycle_index + 1) % len(hovered_all)
            else:
                self._pick_cycle_candidates = hovered_all
                try:
                    cur_idx = hovered_all.index(self._selected_element_comp)
                    self._pick_cycle_index = (cur_idx + 1) % len(hovered_all)
                except ValueError:
                    self._pick_cycle_index = 0
            self._pick_cycle_last_canvas_pos = (cx, cy)
            self._select_element(hovered_all[self._pick_cycle_index])

        elif inp.lmb_clicked and not inp.space_down:
            # Check if clicking on canvas header (or empty-canvas body) → start canvas drag
            clicked_canvas_header = None
            if hovered_canvas_id and not hovered_elem:
                for cgo, cv in all_canvases:
                    if cgo.id != hovered_canvas_id:
                        continue
                    pp = self._canvas_panel_positions.get(cgo.id, [0.0, 0.0])
                    ox = area_min_x + pp[0] * self._zoom
                    oy = area_min_y + pp[1] * self._zoom
                    ct_y = round(oy + self._pan_y)
                    in_header = inp.mouse_y < ct_y
                    canvas_empty = not any(cv.iter_ui_elements())
                    if in_header or canvas_empty:
                        clicked_canvas_header = cgo
                        self._dragging_canvas = True
                        self._drag_canvas_id = cgo.id
                        self._drag_canvas_start_mx = inp.mouse_x
                        self._drag_canvas_start_my = inp.mouse_y
                        self._drag_canvas_start_wx = pp[0]
                        self._drag_canvas_start_wy = pp[1]
                    break

            # Focus the hovered canvas
            if hovered_canvas_id:
                self._focused_canvas_id = hovered_canvas_id

            if clicked_canvas_header is not None:
                self._select_canvas(clicked_canvas_header)
            elif not self._dragging_canvas:
                # Recalculate focused origin after possible focus change
                foc_origin_x, foc_origin_y = self._get_focused_canvas_origin(
                    area_min_x, area_min_y, all_canvases)
                _, focused_canvas = self._get_focused_canvas(all_canvases)
                if focused_canvas is not None:
                    foc_ref_w = float(focused_canvas.reference_width)
                    foc_ref_h = float(focused_canvas.reference_height)

                clicked_kind, clicked_handle = self._hit_test_handle(inp.mouse_x, inp.mouse_y)
                if clicked_kind in ("corner", "edge") and self._selected_element_comp is not None:
                    self._resizing = True
                    self._resize_handle_idx = clicked_handle
                    self._resize_start_mx = inp.mouse_x
                    self._resize_start_my = inp.mouse_y
                    sel = self._selected_element_comp
                    self._prepare_resize_element(sel)
                    self._resize_start_rect = sel.get_rect(foc_ref_w, foc_ref_h)
                    self._resize_start_rotation = float(getattr(sel, 'rotation', 0.0))
                    self._resize_start_corners = sel.get_rotated_corners(foc_ref_w, foc_ref_h)
                    self._undo_pre_resize = (float(sel.x), float(sel.y),
                                             float(sel.width), float(sel.height))
                elif clicked_kind == "rotate" and self._selected_element_comp is not None:
                    self._rotating = True
                    self._dragging = False
                    self._resizing = False
                    self._resize_handle_idx = -1
                    sel = self._selected_element_comp
                    center_x, center_y = self._selection_geometry['center']
                    self._rotate_center_sx = center_x
                    self._rotate_center_sy = center_y
                    self._rotate_start_angle = math.degrees(
                        math.atan2(inp.mouse_y - center_y, inp.mouse_x - center_x))
                    self._rotate_start_rotation = float(getattr(sel, 'rotation', 0.0))
                    self._undo_pre_rotate = float(getattr(sel, 'rotation', 0.0))
                elif clicked_kind == "inside" and self._selected_element_comp is not None:
                    sel = self._selected_element_comp
                    self._dragging = True
                    self._drag_start_x = inp.mouse_x
                    self._drag_start_y = inp.mouse_y
                    drag_x, drag_y, _, _ = sel.get_visual_rect(foc_ref_w, foc_ref_h)
                    self._drag_elem_start_x = drag_x
                    self._drag_elem_start_y = drag_y
                    self._undo_pre_drag = (float(sel.x), float(sel.y))
                elif hovered_elem is not None:
                    self._select_element(hovered_elem)
                    self._dragging = True
                    self._drag_start_x = inp.mouse_x
                    self._drag_start_y = inp.mouse_y
                    drag_x, drag_y, _, _ = hovered_elem.get_visual_rect(foc_ref_w, foc_ref_h)
                    self._drag_elem_start_x = drag_x
                    self._drag_elem_start_y = drag_y
                    self._undo_pre_drag = (float(hovered_elem.x), float(hovered_elem.y))
                elif hovered_canvas_id:
                    # Clicked on canvas body (no element) → select the canvas
                    for cgo, _cv in all_canvases:
                        if cgo.id == hovered_canvas_id:
                            self._select_canvas(cgo)
                            break
                else:
                    self._select_element(None)

    # ------------------------------------------------------------------
    # Snap helpers
    # ------------------------------------------------------------------

    def _drag_snap_step(self) -> float:
        """Return a 'nice' snap step in canvas pixels based on current zoom.

        Uses Theme.UI_EDITOR_SNAP_TABLE for configurable thresholds.
        """
        z = self._zoom
        for threshold, step in Theme.UI_EDITOR_SNAP_TABLE:
            if z >= threshold:
                return step
        return Theme.UI_EDITOR_SNAP_DEFAULT

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


    # ------------------------------------------------------------------
    # Resize handle helpers
    # ------------------------------------------------------------------

    def _prepare_resize_element(self, elem):
        if elem is None:
            return
        if hasattr(elem, "resize_mode"):
            self._undo_pre_resize_mode = getattr(elem, "resize_mode", None)
            if self._undo_pre_resize_mode != TextResizeMode.FixedSize:
                elem.resize_mode = TextResizeMode.FixedSize
        else:
            self._undo_pre_resize_mode = None

    def _hit_test_handle(self, mx: float, my: float):
        """Return (kind, index) for corner, edge, or rotate zones."""
        geom = getattr(self, '_selection_geometry', None)
        if geom is None:
            return None, -1
        hs = Theme.UI_EDITOR_HANDLE_SIZE + 2

        for idx, (cx, cy) in enumerate(geom['corners']):
            if abs(mx - cx) <= hs and abs(my - cy) <= hs:
                return "corner", idx

        rotate_x, rotate_y = geom['rotate_handle']
        if math.hypot(mx - rotate_x, my - rotate_y) <= Theme.UI_EDITOR_ROTATE_HIT_R:
            return "rotate", -1

        inside = self._point_in_quad(mx, my, geom['corners'])
        edge_tol = max(Theme.UI_EDITOR_EDGE_HIT_TOL, hs)
        edges = [(0, 1, 4), (1, 2, 7), (2, 3, 5), (3, 0, 6)]
        for a_idx, b_idx, handle_idx in edges:
            ax, ay = geom['corners'][a_idx]
            bx, by = geom['corners'][b_idx]
            if self._distance_point_to_segment(mx, my, ax, ay, bx, by) <= edge_tol:
                return "edge", handle_idx

        if inside:
            return "inside", -1

        return None, -1

    def _update_hover_cursor(self, ctx: InxGUIContext, area_hovered: bool, mouse_x: float, mouse_y: float):
        if not area_hovered or self._selected_element_comp is None:
            return
        kind, idx = self._hit_test_handle(mouse_x, mouse_y)
        if kind == "corner":
            ctx.set_mouse_cursor(ImGuiMouseCursor.ResizeNWSE if idx in (0, 3) else ImGuiMouseCursor.ResizeNESW)
        elif kind == "edge":
            ctx.set_mouse_cursor(ImGuiMouseCursor.ResizeNS if idx in (4, 5) else ImGuiMouseCursor.ResizeEW)
        elif kind == "rotate":
            ctx.set_mouse_cursor(ImGuiMouseCursor.Hand)

    def _apply_rotation_drag(self, inp):
        elem = self._selected_element_comp
        if elem is None:
            return
        angle = math.degrees(math.atan2(inp.mouse_y - self._rotate_center_sy,
                                        inp.mouse_x - self._rotate_center_sx))
        elem.rotation = float(self._rotate_start_rotation + (angle - self._rotate_start_angle))

    def _apply_resize(self, inp):
        """Update element rect based on current resize handle drag.

        Rotation-aware: mouse deltas are projected onto the element's local
        axes and the opposite rotated corner is preserved.

        Handle index mapping (from hit test):
            Corners: 0=TL, 1=TR, 2=BR, 3=BL
            Edges:   4=top, 5=bottom, 6=left, 7=right
        """
        elem = self._selected_element_comp
        if elem is None:
            return

        _, canvas = self._get_focused_canvas()
        if canvas is None:
            return
        cw = float(canvas.reference_width)
        ch = float(canvas.reference_height)

        # Canvas-space mouse delta
        dx_canvas = (inp.mouse_x - self._resize_start_mx) / self._zoom
        dy_canvas = (inp.mouse_y - self._resize_start_my) / self._zoom

        # Project onto element's local axes using rotation at drag start
        rot = math.radians(self._resize_start_rotation)
        cos_r = math.cos(rot)
        sin_r = math.sin(rot)
        dlx = dx_canvas * cos_r + dy_canvas * sin_r   # delta along local X
        dly = -dx_canvas * sin_r + dy_canvas * cos_r  # delta along local Y

        snap = self._drag_snap_step()
        dlx = round(dlx / snap) * snap
        dly = round(dly / snap) * snap

        _, _, sw, sh = self._resize_start_rect
        idx = self._resize_handle_idx
        MIN_SIZE = Theme.UI_EDITOR_MIN_ELEM_SIZE

        # Handle → (width_delta_sign, height_delta_sign, fixed_corner_index)
        # Corner indices: TL=0, TR=1, BR=2, BL=3
        _HANDLE_INFO = {
            0: (-1, -1, 2),  # TL handle → fix BR
            1: (+1, -1, 3),  # TR handle → fix BL
            2: (+1, +1, 0),  # BR handle → fix TL
            3: (-1, +1, 1),  # BL handle → fix TR
            4: ( 0, -1, 3),  # top edge  → fix BL
            5: ( 0, +1, 0),  # bot edge  → fix TL
            6: (-1,  0, 1),  # left edge → fix TR
            7: (+1,  0, 0),  # right edge→ fix TL
        }
        w_sign, h_sign, fixed_idx = _HANDLE_INFO.get(idx, (0, 0, 0))

        new_w = sw + dlx * w_sign if w_sign != 0 else sw
        new_h = sh + dly * h_sign if h_sign != 0 else sh
        new_w = max(new_w, MIN_SIZE)
        new_h = max(new_h, MIN_SIZE)

        # Aspect ratio lock
        if bool(getattr(elem, 'lock_aspect_ratio', False)) and sw > 0.0 and sh > 0.0:
            aspect = sw / max(sh, 1e-6)
            if w_sign != 0:  # width changed → adjust height
                new_h = max(MIN_SIZE, new_w / max(aspect, 1e-6))
            else:            # height only  → adjust width
                new_w = max(MIN_SIZE, new_h * aspect)

        # Alignment snapping for resize (snap moving edges to guides)
        new_w, new_h = self._apply_resize_alignment_snapping(
            canvas, elem, new_w, new_h, w_sign, h_sign, fixed_idx, cw, ch,
        )

        # Preserve the fixed corner using stored initial positions
        fixed_cx, fixed_cy = self._resize_start_corners[fixed_idx]
        off_x, off_y = elem._rotated_corner_offset(new_w, new_h, fixed_idx)
        new_rx = fixed_cx - off_x
        new_ry = fixed_cy - off_y
        anchor_x, anchor_y = elem._anchor_origin(cw, ch)
        elem.x = new_rx - anchor_x
        elem.y = new_ry - anchor_y
        elem.width = new_w
        elem.height = new_h

    # ------------------------------------------------------------------
    # Suppressed interaction wrappers (undo batching)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_undo_mgr():
        from Infernux.engine.undo import UndoManager
        return UndoManager.instance()

    def _apply_drag_suppressed(self, vis_x, vis_y, ref_w, ref_h):
        """Apply drag with auto-undo suppressed."""
        mgr = self._get_undo_mgr()
        if mgr:
            with mgr.suppress():
                self._selected_element_comp.set_visual_position(vis_x, vis_y, ref_w, ref_h)
        else:
            self._selected_element_comp.set_visual_position(vis_x, vis_y, ref_w, ref_h)

    def _nudge_selected(self, dx: int, dy: int, ref_w: float, ref_h: float):
        """Move selected element by (dx, dy) pixels and record undo."""
        elem = self._selected_element_comp
        if elem is None:
            return
        vx, vy, _, _ = elem.get_visual_rect(ref_w, ref_h)
        old_x, old_y = float(elem.x), float(elem.y)
        elem.set_visual_position(vx + dx, vy + dy, ref_w, ref_h)
        new_x, new_y = float(elem.x), float(elem.y)
        if old_x == new_x and old_y == new_y:
            return
        from Infernux.engine.undo import SetPropertyCommand, CompoundCommand
        cmds = []
        if old_x != new_x:
            cmds.append(SetPropertyCommand(elem, 'x', old_x, new_x, 'Set x'))
        if old_y != new_y:
            cmds.append(SetPropertyCommand(elem, 'y', old_y, new_y, 'Set y'))
        if cmds:
            mgr = self._get_undo_mgr()
            if mgr:
                mgr.record(CompoundCommand(cmds, "Nudge UI Element"))

    def _apply_resize_suppressed(self, inp):
        """Apply resize with auto-undo suppressed."""
        mgr = self._get_undo_mgr()
        if mgr:
            with mgr.suppress():
                self._apply_resize(inp)
        else:
            self._apply_resize(inp)

    def _apply_rotation_drag_suppressed(self, inp):
        """Apply rotation with auto-undo suppressed."""
        mgr = self._get_undo_mgr()
        if mgr:
            with mgr.suppress():
                self._apply_rotation_drag(inp)
        else:
            self._apply_rotation_drag(inp)

    def _record_drag_undo(self):
        """Record a single compound undo command for the completed drag."""
        elem = self._selected_element_comp
        if elem is None:
            return
        old_x, old_y = self._undo_pre_drag
        new_x, new_y = float(elem.x), float(elem.y)
        if old_x == new_x and old_y == new_y:
            return
        from Infernux.engine.undo import SetPropertyCommand, CompoundCommand
        cmds = []
        if old_x != new_x:
            cmds.append(SetPropertyCommand(elem, 'x', old_x, new_x, 'Set x'))
        if old_y != new_y:
            cmds.append(SetPropertyCommand(elem, 'y', old_y, new_y, 'Set y'))
        if cmds:
            mgr = self._get_undo_mgr()
            if mgr:
                mgr.record(CompoundCommand(cmds, "Move UI Element"))

    def _record_resize_undo(self):
        """Record a single compound undo command for the completed resize."""
        elem = self._selected_element_comp
        if elem is None:
            return
        old_x, old_y, old_w, old_h = self._undo_pre_resize
        new_x, new_y = float(elem.x), float(elem.y)
        new_w, new_h = float(elem.width), float(elem.height)
        from Infernux.engine.undo import SetPropertyCommand, CompoundCommand
        cmds = []
        # Include resize_mode change when a TextUI was switched to FixedSize
        old_mode = self._undo_pre_resize_mode
        if old_mode is not None:
            new_mode = getattr(elem, 'resize_mode', None)
            if old_mode != new_mode:
                cmds.append(SetPropertyCommand(elem, 'resize_mode', old_mode, new_mode, 'Set resize_mode'))
        if old_x != new_x:
            cmds.append(SetPropertyCommand(elem, 'x', old_x, new_x, 'Set x'))
        if old_y != new_y:
            cmds.append(SetPropertyCommand(elem, 'y', old_y, new_y, 'Set y'))
        if old_w != new_w:
            cmds.append(SetPropertyCommand(elem, 'width', old_w, new_w, 'Set width'))
        if old_h != new_h:
            cmds.append(SetPropertyCommand(elem, 'height', old_h, new_h, 'Set height'))
        if cmds:
            mgr = self._get_undo_mgr()
            if mgr:
                mgr.record(CompoundCommand(cmds, "Resize UI Element"))

    def _record_rotate_undo(self):
        """Record a single undo command for the completed rotation."""
        elem = self._selected_element_comp
        if elem is None:
            return
        old_rot = self._undo_pre_rotate
        new_rot = float(elem.rotation)
        if old_rot == new_rot:
            return
        from Infernux.engine.undo import SetPropertyCommand
        mgr = self._get_undo_mgr()
        if mgr:
            mgr.record(SetPropertyCommand(elem, 'rotation', old_rot, new_rot,
                                           'Rotate UI Element'))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_element(self, elem_comp):
        """Select a UI element and sync with hierarchy/inspector."""
        self._selected_element_comp = elem_comp
        if self._on_selection_changed:
            if elem_comp is not None:
                go = elem_comp.game_object
                self._focus_canvas_for_object(go)
                self._on_selection_changed(go)
                # Auto-expand hierarchy to reveal this object
                if self._hierarchy_panel and go is not None:
                    self._hierarchy_panel.expand_to_object(go.id)
            else:
                self._on_selection_changed(None)

    def _select_canvas(self, canvas_go):
        """Select a canvas GameObject and sync with hierarchy/inspector."""
        self._clear_interaction_state()
        if canvas_go is None:
            if self._on_selection_changed:
                self._on_selection_changed(None)
            return

        self._focused_canvas_id = canvas_go.id
        if self._on_selection_changed:
            self._on_selection_changed(canvas_go)
        if self._hierarchy_panel is not None:
            self._hierarchy_panel.expand_to_object(canvas_go.id)

    def _delete_selected_element(self):
        """Delete the currently selected UI element's GameObject via undo system."""
        elem = self._selected_element_comp
        if elem is None:
            return
        go = elem.game_object
        self._selected_element_comp = None
        self._dragging = False
        if self._on_selection_changed:
            self._on_selection_changed(None)
        if go is not None:
            from Infernux.engine.undo import UndoManager, DeleteGameObjectCommand
            mgr = UndoManager.instance()
            if mgr:
                mgr.execute(DeleteGameObjectCommand(go.id, "Delete UI Element"))
            else:
                from Infernux.lib import SceneManager
                scene = SceneManager.instance().get_active_scene()
                if scene is not None:
                    scene.destroy_game_object(go)

    # ------------------------------------------------------------------
    # Creation helpers
    # ------------------------------------------------------------------

    def _record_ui_create(self, object_id: int, description: str = "Create UI Element"):
        """Record a UI object creation through the undo system."""
        from Infernux.engine.undo import UndoManager, CreateGameObjectCommand
        mgr = UndoManager.instance()
        if mgr:
            mgr.record(CreateGameObjectCommand(object_id, description))

    def _create_canvas(self):
        """Create a new Canvas GameObject in the scene."""
        from Infernux.lib import SceneManager
        from Infernux.ui import UICanvas as UICanvasCls
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            return
        mgr = self._get_undo_mgr()
        ctx_mgr = mgr.suppress() if mgr else _nullcontext()
        go = None
        with ctx_mgr:
            go = scene.create_game_object("Canvas")
            if go:
                go.add_py_component(UICanvasCls())
                self._focused_canvas_id = go.id
                invalidate_canvas_cache()
                # Select the new canvas in hierarchy
                if self._hierarchy_panel:
                    self._hierarchy_panel.set_selected_object_by_id(go.id)
                elif self._on_selection_changed:
                    self._on_selection_changed(go)
        if go:
            self._record_ui_create(go.id, "Create Canvas")

    def _create_ui_element(self, canvas_go, component_cls, go_name: str,
                           default_size=None, default_pos=None,
                           undo_label: str = "Create UI Element"):
        """Generic helper to create a UI element under a canvas.

        Args:
            canvas_go: Parent canvas game-object.
            component_cls: UI component class to instantiate.
            go_name: Name for the new game-object.
            default_size: Optional (w, h) to set before adding the component.
            default_pos: Optional (x, y) centered-anchor offset.
            undo_label: Description for the undo system.
        """
        from Infernux.lib import SceneManager
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            return
        mgr = self._get_undo_mgr()
        ctx_mgr = mgr.suppress() if mgr else _nullcontext()
        go = None
        with ctx_mgr:
            go = scene.create_game_object(go_name)
            if go:
                go.set_parent(canvas_go)
                comp = component_cls()
                if default_size is not None:
                    comp.width = float(default_size[0])
                    comp.height = float(default_size[1])
                go.add_py_component(comp)
                if default_pos is not None:
                    # Find parent canvas component to set centered alignment
                    canvas_comp = None
                    for c in canvas_go.get_py_components():
                        from Infernux.ui import UICanvas
                        if isinstance(c, UICanvas):
                            canvas_comp = c
                            break
                    if canvas_comp:
                        from Infernux.ui.enums import ScreenAlignH, ScreenAlignV
                        comp.align_h = ScreenAlignH.Center
                        comp.align_v = ScreenAlignV.Center
                        comp.x = float(default_pos[0])
                        comp.y = float(default_pos[1])
                self._select_element(comp)
                invalidate_canvas_cache()
                if self._hierarchy_panel:
                    self._hierarchy_panel.set_selected_object_by_id(go.id)
                    self._hierarchy_panel.set_pending_expand_id(canvas_go.id)
        if go:
            self._record_ui_create(go.id, undo_label)

    def _create_text_element(self, canvas_go):
        """Create a UIText child under the given canvas GameObject."""
        from Infernux.ui import UIText as UITextCls
        self._create_ui_element(
            canvas_go, UITextCls, "Text",
            default_pos=Theme.UI_EDITOR_NEW_TEXT_POS,
            undo_label="Create Text",
        )

    def _create_image_element(self, canvas_go):
        """Create a UIImage child under the given canvas GameObject."""
        from Infernux.ui import UIImage as UIImageCls
        self._create_ui_element(
            canvas_go, UIImageCls, "Image",
            default_size=Theme.UI_EDITOR_NEW_IMAGE_SIZE,
            default_pos=Theme.UI_EDITOR_NEW_IMAGE_POS,
            undo_label="Create Image",
        )

    def _create_button_element(self, canvas_go):
        """Create a UIButton child under the given canvas GameObject."""
        from Infernux.ui import UIButton as UIButtonCls
        self._create_ui_element(
            canvas_go, UIButtonCls, "Button",
            default_size=Theme.UI_EDITOR_NEW_BUTTON_SIZE,
            default_pos=Theme.UI_EDITOR_NEW_BUTTON_POS,
            undo_label="Create Button",
        )

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def _fit_zoom(self, ctx: InxGUIContext, canvas):
        """Fit all canvases into the available area."""
        avail_w = ctx.get_content_region_avail_width()
        avail_h = ctx.get_content_region_avail_height()
        if avail_w < 1 or avail_h < 1:
            return
        # Compute bounding box of all canvases in workspace space
        all_canvases = self._get_all_canvases()
        if not all_canvases:
            return
        self._ensure_canvas_layout(all_canvases)
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        for cgo, cv in all_canvases:
            pos = self._canvas_panel_positions.get(cgo.id, [0.0, 0.0])
            wx, wy = pos[0], pos[1]
            rw = float(cv.reference_width)
            rh = float(cv.reference_height)
            hdr_h = Theme.UI_EDITOR_CANVAS_HEADER_H  # approximate in workspace
            min_x = min(min_x, wx)
            min_y = min(min_y, wy - hdr_h)
            max_x = max(max_x, wx + rw)
            max_y = max(max_y, wy + rh)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        if bbox_w < 1 or bbox_h < 1:
            return
        margin = Theme.UI_EDITOR_FIT_MARGIN
        zoom_w = (avail_w - margin) / bbox_w
        zoom_h = (avail_h - margin) / bbox_h
        self._zoom = max(Theme.UI_EDITOR_MIN_ZOOM,
                         min(Theme.UI_EDITOR_MAX_ZOOM, min(zoom_w, zoom_h)))
        # Center all canvases
        self._pan_x = (avail_w - bbox_w * self._zoom) / 2 - min_x * self._zoom
        self._pan_y = (avail_h - bbox_h * self._zoom) / 2 - min_y * self._zoom
        self._save_view_settings()
