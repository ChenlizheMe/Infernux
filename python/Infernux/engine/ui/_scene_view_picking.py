"""SceneViewPickingMixin — extracted from SceneViewPanel."""
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

# Module constants from scene_view_panel
from .scene_view_panel import _GIZMO_IDS

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


class SceneViewPickingMixin:
    """SceneViewPickingMixin method group for SceneViewPanel."""

    def _handle_picking_and_selection(self, ctx, vp, gizmo_consumed, overlay_hovered,
                                      is_scene_hovered, play_border_clr):
        """Handle object picking, box-select, and play-mode border drawing."""
        if (is_scene_hovered and not gizmo_consumed
                and not overlay_hovered
                and ctx.is_mouse_button_clicked(0)
                and not self._box_select_active):
            picked_id = self._pick_scene_object(ctx, vp)
            if picked_id:
                if self._on_object_picked:
                    self._on_object_picked(picked_id, False)
            else:
                if self._on_object_picked:
                    self._on_object_picked(0, False)

        # Box-select
        if self._box_select_active:
            lx, ly = vp.mouse_local(ctx)
            self._box_select_end = (lx, ly)

            if not ctx.is_mouse_button_down(0):
                self._finalize_box_select(ctx, vp)
                self._box_select_active = False
            else:
                sx, sy = self._box_select_start
                ex, ey = self._box_select_end
                min_x = vp.image_min_x + min(sx, ex)
                min_y = vp.image_min_y + min(sy, ey)
                max_x = vp.image_min_x + max(sx, ex)
                max_y = vp.image_min_y + max(sy, ey)
                ctx.draw_filled_rect(min_x, min_y, max_x, max_y,
                                     0.3, 0.5, 0.9, 0.15)
                ctx.draw_rect(min_x, min_y, max_x, max_y,
                              0.3, 0.5, 0.9, 0.8, thickness=1.0)

        # Play-mode border
        if play_border_clr is not None:
            ctx.draw_rect(
                vp.image_min_x, vp.image_min_y,
                vp.image_max_x, vp.image_max_y,
                *play_border_clr,
                thickness=Theme.BORDER_THICKNESS,
            )

    def _finalize_box_select(self, ctx: InxGUIContext, vp: ViewportInfo):
        """Complete a box-select drag: find objects inside the rectangle."""
        sx, sy = self._box_select_start
        ex, ey = self._box_select_end
        min_x, max_x = min(sx, ex), max(sx, ex)
        min_y, max_y = min(sy, ey), max(sy, ey)

        # Too small? Treat as a deselect click
        if abs(max_x - min_x) < 5 and abs(max_y - min_y) < 5:
            if self._on_object_picked:
                self._on_object_picked(0, False)
            return

        # Gather all scene objects and project them to screen space
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene or not self._engine:
            return

        native = self._engine.get_native_engine()
        if not native:
            return

        all_objects = scene.get_all_objects()
        selected_ids = []
        for obj in all_objects:
            t = obj.get_transform()
            if t is None:
                continue
            # Skip screen-space UI elements (canvas children with _hide_transform_)
            try:
                _skip = False
                for _pc in obj.get_py_components():
                    if getattr(type(_pc), '_hide_transform_', False):
                        _skip = True
                        break
                if _skip:
                    continue
            except RuntimeError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
            pos = t.position
            sp = native.editor_camera.world_to_screen_point(pos.x, pos.y, pos.z)
            if min_x <= sp.x <= max_x and min_y <= sp.y <= max_y:
                selected_ids.append(obj.id)

        from .imgui_keys import KEY_LEFT_CTRL, KEY_RIGHT_CTRL
        ctrl = ctx.is_key_down(KEY_LEFT_CTRL) or ctx.is_key_down(KEY_RIGHT_CTRL)

        from .selection_manager import SelectionManager
        sel = SelectionManager.instance()
        if selected_ids:
            sel.box_select(selected_ids, additive=ctrl)
        elif not ctrl:
            sel.clear()

        # Update outline — combined for multi-select
        all_ids = sel.get_ids()
        if native:
            if len(all_ids) > 1:
                native.set_selection_outlines(all_ids)
            elif all_ids:
                native.set_selection_outline(all_ids[0])
            else:
                native.clear_selection_outline()

        # Resolve primary object for inspector
        primary_id = sel.get_primary()
        primary_obj = scene.find_by_id(primary_id) if primary_id else None
        if self._on_box_select:
            self._on_box_select(primary_obj)

    def _pick_scene_object(self, ctx: InxGUIContext, vp: ViewportInfo) -> int:
        """Pick scene object under mouse cursor with repeated-click cycling."""
        if not self._engine:
            return 0

        local_x, local_y = vp.mouse_local(ctx)

        # Clamp within viewport
        if local_x < 0 or local_y < 0 or local_x > vp.width or local_y > vp.height:
            return 0

        candidates = self._engine.pick_scene_object_ids(local_x, local_y, vp.width, vp.height)

        # Filter invalid IDs and gizmo axis pseudo-IDs.
        ids = []
        for candidate in candidates:
            object_id = int(candidate)
            if object_id > 0 and object_id not in _GIZMO_IDS:
                ids.append(object_id)

        if not ids:
            self._pick_cycle_candidates = []
            self._pick_cycle_index = -1
            return 0

        same_viewport = self._pick_cycle_last_viewport == (int(vp.width), int(vp.height))
        last_x, last_y = self._pick_cycle_last_mouse
        same_spot = abs(local_x - last_x) <= 3.0 and abs(local_y - last_y) <= 3.0
        same_candidates = ids == self._pick_cycle_candidates

        if same_viewport and same_spot and same_candidates and self._pick_cycle_index >= 0:
            index = (self._pick_cycle_index + 1) % len(ids)
        else:
            index = 0

        self._pick_cycle_candidates = ids
        self._pick_cycle_index = index
        self._pick_cycle_last_mouse = (local_x, local_y)
        self._pick_cycle_last_viewport = (int(vp.width), int(vp.height))

        return ids[index]

