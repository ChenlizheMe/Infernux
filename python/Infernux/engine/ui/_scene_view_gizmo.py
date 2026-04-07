"""SceneViewGizmoMixin — extracted from SceneViewPanel."""
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

# Tool mode constants — imported from scene_view_panel
from .scene_view_panel import (
    TOOL_NONE, TOOL_TRANSLATE, TOOL_ROTATE, TOOL_SCALE,
    TRANSLATE_SNAP_STEP, ROTATE_SNAP_DEGREES, SCALE_SNAP_FACTOR,
    _GIZMO_IDS, _AXIS_DIRS, _PLANE_AXIS_PAIRS,
    _euler_deg_to_quat, _quat_to_euler_deg, _quat_mul, _axis_angle_to_quat,
)

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


class SceneViewGizmoMixin:
    """SceneViewGizmoMixin method group for SceneViewPanel."""

    def _process_gizmo_and_camera(self, ctx, vp, delta_time, is_scene_hovered, overlay_hovered):
        """Handle gizmo interaction and camera drag. Returns whether gizmo consumed the input."""
        left_down = ctx.is_mouse_button_down(0)
        left_clicked = left_down and not self._was_left_down
        self._was_left_down = left_down
        gizmo_consumed = False

        if self._engine:
            local_mx, local_my = vp.mouse_local(ctx)
            gizmo_consumed = self._update_gizmo_interaction(
                ctx, local_mx, local_my, vp.width, vp.height,
                left_down, left_clicked, is_scene_hovered)

        # Camera drag
        mgr = InputManager.instance()
        right_down = mgr.get_mouse_button(1)
        middle_down = mgr.get_mouse_button(2)
        if is_scene_hovered and not overlay_hovered and (right_down or middle_down) and not self._is_camera_dragging:
            self._is_camera_dragging = True
            self._fly_to_active = False
            self._begin_camera_capture(ctx)

        if is_scene_hovered or self._is_camera_dragging:
            self._process_camera_input(ctx, delta_time)

        if self._is_camera_dragging and not right_down and not middle_down:
            self._is_camera_dragging = False
            self._end_camera_capture(ctx)

        return gizmo_consumed

    def _update_gizmo_interaction(self, ctx, local_mx, local_my, scene_w, scene_h,
                                   left_down, left_clicked, is_hovered):
        """Python-side hover highlight + axis-constrained drag for all tool modes.

        Returns True if the gizmo consumed the input this frame.
        """
        engine = self._engine
        if not engine:
            return False

        mode = self._gizmo_tool_mode
        if mode == TOOL_NONE:
            return False

        # -----------------------------------------------------------
        # DRAG CONTINUATION (dispatches to mode-specific handler)
        # -----------------------------------------------------------
        if self._is_gizmo_dragging:
            if not left_down:
                # Release drag — record undo command for the completed operation
                self._record_gizmo_undo(mode)
                self._is_gizmo_dragging = False
                self._gizmo_snap_active = False
                engine.set_editor_tool_highlight(0)
                return False

            self._gizmo_snap_active = self._is_ctrl_down(ctx)

            if mode == TOOL_TRANSLATE:
                self._drag_translate(engine, local_mx, local_my, scene_w, scene_h)
            elif mode == TOOL_ROTATE:
                self._drag_rotate(engine, local_mx, local_my, scene_w, scene_h)
            elif mode == TOOL_SCALE:
                self._drag_scale(engine, local_mx, local_my, scene_w, scene_h)

            return True  # consumed

        # -----------------------------------------------------------
        # HOVER DETECTION (using existing picking infrastructure)
        # -----------------------------------------------------------
        if not is_hovered:
            engine.set_editor_tool_highlight(0)
            self._hover_pick_cache_pos = (-1.0, -1.0)
            return False

        # Cache: skip the gizmo axis test when the mouse hasn't moved.
        pos_key = (local_mx, local_my)
        if pos_key == self._hover_pick_cache_pos:
            picked = self._hover_pick_cache_result
        else:
            picked = engine.pick_gizmo_axis(local_mx, local_my, scene_w, scene_h)
            self._hover_pick_cache_pos = pos_key
            self._hover_pick_cache_result = picked

        handle = _GIZMO_IDS.get(picked, 0)
        engine.set_editor_tool_highlight(handle)

        if handle == 0:
            return False  # not hovering any gizmo handle

        # -----------------------------------------------------------
        # DRAG START (common for all modes)
        # Only initiate drag on a fresh press — not when the button
        # was already held and the cursor drifted over the gizmo.
        # -----------------------------------------------------------
        if left_clicked:
            # Block gizmo edits on prefab children (they are locked in Inspector).
            from Infernux.lib._Infernux import SceneManager as _SM
            scene = _SM.instance().get_active_scene()
            sel_id = engine.get_selected_object_id()
            if scene and sel_id:
                _obj = scene.find_by_id(sel_id)
                if _obj is not None:
                    _is_prefab_child = (
                        bool(getattr(_obj, 'prefab_guid', None))
                        and not bool(getattr(_obj, 'prefab_root', False))
                    )
                    if _is_prefab_child:
                        return True  # consume input but refuse to start drag

            self._is_gizmo_dragging = True
            self._gizmo_drag_axis = handle
            self._gizmo_snap_active = self._is_ctrl_down(ctx)
            self._gizmo_drag_start_screen = (local_mx, local_my)
            obj_pos = (0.0, 0.0, 0.0)
            obj_euler = (0.0, 0.0, 0.0)
            obj_scale = (1.0, 1.0, 1.0)
            if scene and sel_id:
                obj = scene.find_by_id(sel_id)
                if obj:
                    p = obj.transform.position
                    obj_pos = (p[0], p[1], p[2])
                    e = obj.transform.euler_angles
                    obj_euler = (e[0], e[1], e[2])
                    s = obj.transform.local_scale
                    obj_scale = (s[0], s[1], s[2])

                    basis_axes = self._gizmo_basis_axes(obj)
                else:
                    basis_axes = self._gizmo_basis_axes(None)
            else:
                basis_axes = self._gizmo_basis_axes(None)

            if handle in _PLANE_AXIS_PAIRS:
                plane_axes = _PLANE_AXIS_PAIRS[handle]
                self._gizmo_drag_plane_axes = plane_axes
                self._gizmo_drag_plane_u = basis_axes[plane_axes[0]]
                self._gizmo_drag_plane_v = basis_axes[plane_axes[1]]
                start_uv = self._plane_hit_coords(
                    engine,
                    local_mx,
                    local_my,
                    scene_w,
                    scene_h,
                    obj_pos,
                    self._gizmo_drag_plane_u,
                    self._gizmo_drag_plane_v,
                )
                self._gizmo_drag_plane_start_uv = start_uv if start_uv is not None else (0.0, 0.0)
                self._gizmo_drag_axis_dir = self._gizmo_drag_plane_u
            else:
                self._gizmo_drag_plane_axes = (0, 0)
                self._gizmo_drag_plane_start_uv = (0.0, 0.0)
                self._gizmo_drag_axis_dir = basis_axes.get(handle, _AXIS_DIRS[1])
            self._gizmo_drag_obj_id = sel_id
            self._gizmo_drag_start_pos = obj_pos
            self._gizmo_drag_start_euler = obj_euler
            self._gizmo_drag_start_scale = obj_scale

            # For translate/scale: record initial axis or plane parameter
            if mode in (TOOL_TRANSLATE, TOOL_SCALE) and handle not in _PLANE_AXIS_PAIRS:
                ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
                self._gizmo_drag_start_t = self._closest_param_on_axis(
                    ray[:3], ray[3:], self._gizmo_drag_start_pos, self._gizmo_drag_axis_dir)

            return True  # consumed

        return True  # hovering a gizmo handle — consume to suppress picking

    def _gizmo_basis_axes(self, obj=None):
        if self._coord_space == 1 and obj is not None:
            r = obj.transform.right
            u = obj.transform.up
            f = obj.transform.forward
            return {
                1: (r[0], r[1], r[2]),
                2: (u[0], u[1], u[2]),
                3: (f[0], f[1], f[2]),
            }
        return dict(_AXIS_DIRS)

    def _plane_hit_coords(self, engine, local_mx, local_my, scene_w, scene_h, plane_origin, axis_u, axis_v):
        ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
        ray_o = ray[:3]
        ray_d = ray[3:]
        normal = self._cross3(axis_u, axis_v)
        denom = self._dot3(ray_d, normal)
        if abs(denom) < 1e-8:
            return None

        t = self._dot3(self._sub3(plane_origin, ray_o), normal) / denom
        if t < 0.0:
            return None

        hit = self._add3(ray_o, self._scale3(ray_d, t))
        rel = self._sub3(hit, plane_origin)
        return (self._dot3(rel, axis_u), self._dot3(rel, axis_v))

    def _is_ctrl_down(self, ctx: InxGUIContext) -> bool:
        return ctx.is_key_down(_keys.KEY_LEFT_CTRL) or ctx.is_key_down(_keys.KEY_RIGHT_CTRL)

    def _record_gizmo_undo(self, mode: int):
        """Record an undo command for the gizmo drag that just finished."""
        from Infernux.lib._Infernux import SceneManager as _SM, Vector3
        from Infernux.engine.undo import UndoManager, SetPropertyCommand

        scene = _SM.instance().get_active_scene()
        if not scene or not self._gizmo_drag_obj_id:
            return
        obj = scene.find_by_id(self._gizmo_drag_obj_id)
        if not obj:
            return

        transform = obj.transform

        if mode == TOOL_TRANSLATE:
            old_val = Vector3(*self._gizmo_drag_start_pos)
            new_val_raw = transform.position
            new_val = Vector3(new_val_raw[0], new_val_raw[1], new_val_raw[2])
            if self._vec3_approx_equal(old_val, new_val):
                return
            cmd = SetPropertyCommand(transform, "position",
                                     old_val, new_val, "Translate")
        elif mode == TOOL_ROTATE:
            old_val = Vector3(*self._gizmo_drag_start_euler)
            new_val_raw = transform.euler_angles
            new_val = Vector3(new_val_raw[0], new_val_raw[1], new_val_raw[2])
            if self._vec3_approx_equal(old_val, new_val):
                return
            cmd = SetPropertyCommand(transform, "euler_angles",
                                     old_val, new_val, "Rotate")
        elif mode == TOOL_SCALE:
            old_val = Vector3(*self._gizmo_drag_start_scale)
            new_val_raw = transform.local_scale
            new_val = Vector3(new_val_raw[0], new_val_raw[1], new_val_raw[2])
            if self._vec3_approx_equal(old_val, new_val):
                return
            cmd = SetPropertyCommand(transform, "local_scale",
                                     old_val, new_val, "Scale")
        else:
            return

        UndoManager.instance().record(cmd)

    def _set_tool_mode(self, mode: int):
        """Switch the active editor tool (syncs to C++ and resets drag)."""
        if mode == self._gizmo_tool_mode:
            return
        self._gizmo_tool_mode = mode
        self._is_gizmo_dragging = False
        if self._engine:
            self._engine.set_editor_tool_mode(mode)
            self._engine.set_editor_tool_highlight(0)

    def _drag_translate(self, engine, local_mx, local_my, scene_w, scene_h):
        """Axis-constrained translation: project mouse ray onto drag axis."""
        if self._gizmo_drag_axis in _PLANE_AXIS_PAIRS:
            uv = self._plane_hit_coords(
                engine,
                local_mx,
                local_my,
                scene_w,
                scene_h,
                self._gizmo_drag_start_pos,
                self._gizmo_drag_plane_u,
                self._gizmo_drag_plane_v,
            )
            if uv is None:
                return

            du = uv[0] - self._gizmo_drag_plane_start_uv[0]
            dv = uv[1] - self._gizmo_drag_plane_start_uv[1]
            if self._gizmo_snap_active:
                du = self._snap_delta(du, TRANSLATE_SNAP_STEP)
                dv = self._snap_delta(dv, TRANSLATE_SNAP_STEP)

            delta_u = self._scale3(self._gizmo_drag_plane_u, du)
            delta_v = self._scale3(self._gizmo_drag_plane_v, dv)
            new_pos = self._add3(self._gizmo_drag_start_pos, self._add3(delta_u, delta_v))

            from Infernux.lib._Infernux import SceneManager as _SM, Vector3
            scene = _SM.instance().get_active_scene()
            if scene:
                obj = scene.find_by_id(self._gizmo_drag_obj_id)
                if obj:
                    obj.transform.position = Vector3(new_pos[0], new_pos[1], new_pos[2])
            return

        ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
        ad = self._gizmo_drag_axis_dir
        sp = self._gizmo_drag_start_pos

        cur_t = self._closest_param_on_axis(ray[:3], ray[3:], sp, ad)
        delta = cur_t - self._gizmo_drag_start_t
        if self._gizmo_snap_active:
            delta = self._snap_delta(delta, TRANSLATE_SNAP_STEP)

        new_pos = (sp[0] + ad[0] * delta,
                   sp[1] + ad[1] * delta,
                   sp[2] + ad[2] * delta)
        from Infernux.lib._Infernux import SceneManager as _SM, Vector3
        scene = _SM.instance().get_active_scene()
        if scene:
            obj = scene.find_by_id(self._gizmo_drag_obj_id)
            if obj:
                obj.transform.position = Vector3(new_pos[0], new_pos[1], new_pos[2])

    def _drag_rotate(self, engine, local_mx, local_my, scene_w, scene_h):
        """Rotation around the drag axis (world or local depending on coord space)."""
        # Screen-space delta from drag start → rotation angle.
        # 200 pixels of horizontal movement ≈ 180°, like Unity.
        dx = local_mx - self._gizmo_drag_start_screen[0]

        ad = self._gizmo_drag_axis_dir  # world-space axis (global or local)

        # Camera-relative sign correction so the visible ring always follows
        # the mouse drag direction.
        #
        # Derivation: the front-most point on the ring (nearest the camera)
        # moves by  δθ · cross(A, P_front).  The horizontal screen component
        # of that movement must have the same sign as the mouse dx.
        # Working through the projection math:
        #   sign = sign( dot(A, camera_up) )
        # where camera_up = cross(camera_right, view_fwd) and
        #       camera_right = normalize(cross(view_fwd, world_up)).
        cam_pos = engine.editor_camera.position
        op = self._gizmo_drag_start_pos
        vf = (op[0] - cam_pos.x, op[1] - cam_pos.y, op[2] - cam_pos.z)
        vf_len = math.sqrt(vf[0]**2 + vf[1]**2 + vf[2]**2)
        if vf_len > 1e-9:
            vf = (vf[0]/vf_len, vf[1]/vf_len, vf[2]/vf_len)
            # camera_right = normalize(cross(view_fwd, world_up=(0,1,0)))
            #              = normalize((-vf_z, 0, vf_x))
            cr_x, cr_z = -vf[2], vf[0]
            cr_len = math.sqrt(cr_x**2 + cr_z**2)
            if cr_len > 1e-9:
                cr = (cr_x/cr_len, 0.0, cr_z/cr_len)
            else:
                cr = (1.0, 0.0, 0.0)  # camera looking straight up/down
            # camera_up = cross(camera_right, view_fwd)
            cu = (cr[1]*vf[2] - cr[2]*vf[1],
                  cr[2]*vf[0] - cr[0]*vf[2],
                  cr[0]*vf[1] - cr[1]*vf[0])
            sign_val = ad[0]*cu[0] + ad[1]*cu[1] + ad[2]*cu[2]
            sign = 1.0 if sign_val >= 0 else -1.0
        else:
            sign = 1.0

        angle_deg = -dx * (180.0 / 200.0) * sign
        if self._gizmo_snap_active:
            angle_deg = self._snap_delta(angle_deg, ROTATE_SNAP_DEGREES)

        se = self._gizmo_drag_start_euler
        q_start = _euler_deg_to_quat(se[0], se[1], se[2])
        q_delta = _axis_angle_to_quat(ad[0], ad[1], ad[2], angle_deg)

        # Always pre-multiply: the axis in q_delta is already expressed in
        # world space for both Global mode (world unit axis) and Local mode
        # (object's local axis mapped to world space).
        q_new = _quat_mul(q_delta, q_start)
        new_euler = _quat_to_euler_deg(q_new)

        from Infernux.lib._Infernux import SceneManager as _SM, Vector3
        scene = _SM.instance().get_active_scene()
        if scene:
            obj = scene.find_by_id(self._gizmo_drag_obj_id)
            if obj:
                obj.transform.euler_angles = Vector3(new_euler[0], new_euler[1], new_euler[2])

    def _drag_scale(self, engine, local_mx, local_my, scene_w, scene_h):
        """Scale along the drag axis. In Local mode, scale applies directly to
        the corresponding local_scale component. In Global mode, the world-axis
        scale factor is decomposed onto local axes."""
        if self._gizmo_drag_axis in _PLANE_AXIS_PAIRS:
            uv = self._plane_hit_coords(
                engine,
                local_mx,
                local_my,
                scene_w,
                scene_h,
                self._gizmo_drag_start_pos,
                self._gizmo_drag_plane_u,
                self._gizmo_drag_plane_v,
            )
            if uv is None:
                return

            factor_u = self._plane_factor(uv[0], self._gizmo_drag_plane_start_uv[0])
            factor_v = self._plane_factor(uv[1], self._gizmo_drag_plane_start_uv[1])
            if self._gizmo_snap_active:
                factor_u = 1.0 + self._snap_delta(factor_u - 1.0, SCALE_SNAP_FACTOR)
                factor_v = 1.0 + self._snap_delta(factor_v - 1.0, SCALE_SNAP_FACTOR)
            factor_u = max(factor_u, 0.01)
            factor_v = max(factor_v, 0.01)

            from Infernux.lib._Infernux import SceneManager as _SM, Vector3
            scene = _SM.instance().get_active_scene()
            if not scene:
                return
            obj = scene.find_by_id(self._gizmo_drag_obj_id)
            if not obj:
                return

            ss = self._gizmo_drag_start_scale
            new_scale = list(ss)
            axis_a, axis_b = self._gizmo_drag_plane_axes
            if self._coord_space == 1:
                new_scale[axis_a - 1] = max(ss[axis_a - 1] * factor_u, 0.001)
                new_scale[axis_b - 1] = max(ss[axis_b - 1] * factor_v, 0.001)
            else:
                r = obj.transform.right
                u = obj.transform.up
                f = obj.transform.forward
                local_axes = [
                    (r[0], r[1], r[2]),
                    (u[0], u[1], u[2]),
                    (f[0], f[1], f[2]),
                ]
                for i in range(3):
                    dot_u = self._dot3(self._gizmo_drag_plane_u, local_axes[i])
                    dot_v = self._dot3(self._gizmo_drag_plane_v, local_axes[i])
                    local_factor_u = 1.0 + (factor_u - 1.0) * dot_u * dot_u
                    local_factor_v = 1.0 + (factor_v - 1.0) * dot_v * dot_v
                    new_scale[i] = max(ss[i] * local_factor_u * local_factor_v, 0.001)

            obj.transform.local_scale = Vector3(new_scale[0], new_scale[1], new_scale[2])
            return

        ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
        ad = self._gizmo_drag_axis_dir
        sp = self._gizmo_drag_start_pos

        cur_t = self._closest_param_on_axis(ray[:3], ray[3:], sp, ad)
        start_t = self._gizmo_drag_start_t

        # Scale factor: ratio of current projection to initial projection
        if abs(start_t) < 1e-6:
            factor = 1.0 + (cur_t - start_t)
        else:
            factor = cur_t / start_t
        if self._gizmo_snap_active:
            factor = 1.0 + self._snap_delta(factor - 1.0, SCALE_SNAP_FACTOR)
        factor = max(factor, 0.01)

        from Infernux.lib._Infernux import SceneManager as _SM, Vector3
        scene = _SM.instance().get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(self._gizmo_drag_obj_id)
        if not obj:
            return

        ss = self._gizmo_drag_start_scale
        new_scale = list(ss)

        if self._coord_space == 1:
            # Local mode: scale directly on the axis component (1=X, 2=Y, 3=Z)
            axis_idx = self._gizmo_drag_axis - 1  # 0, 1, or 2
            new_scale[axis_idx] = max(ss[axis_idx] * factor, 0.001)
        else:
            # Global mode: decompose world-axis scale onto local axes
            r = obj.transform.right
            u = obj.transform.up
            f = obj.transform.forward
            local_axes = [
                (r[0], r[1], r[2]),
                (u[0], u[1], u[2]),
                (f[0], f[1], f[2]),
            ]
            for i in range(3):
                dot_val = (ad[0] * local_axes[i][0] +
                           ad[1] * local_axes[i][1] +
                           ad[2] * local_axes[i][2])
                local_factor = 1.0 + (factor - 1.0) * dot_val * dot_val
                new_scale[i] = max(ss[i] * local_factor, 0.001)

        obj.transform.local_scale = Vector3(new_scale[0], new_scale[1], new_scale[2])

