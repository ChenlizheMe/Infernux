"""
Unity-style Scene View panel with 3D viewport and camera controls.
"""

import math
import os
from InfEngine.lib import InfGUIContext, TextureLoader
from InfEngine.engine.i18n import t
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar
from .viewport_utils import ViewportInfo, capture_viewport_info
from . import imgui_keys as _keys
from InfEngine.resources import file_type_icons_dir

# Gizmo axis IDs — must match C++ EditorTools constants
from InfEngine.lib._InfEngine import GIZMO_X_AXIS_ID, GIZMO_Y_AXIS_ID, GIZMO_Z_AXIS_ID

_GIZMO_IDS = {GIZMO_X_AXIS_ID: 1, GIZMO_Y_AXIS_ID: 2, GIZMO_Z_AXIS_ID: 3}
_AXIS_DIRS = {1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}

# Tool mode constants — must match C++ EditorTools::ToolMode
TOOL_NONE = 0
TOOL_TRANSLATE = 1
TOOL_ROTATE = 2
TOOL_SCALE = 3


# ======================================================================
# Quaternion math helpers  (matches GLM convention: ZYX intrinsic order,
# euler = (pitch/X, yaw/Y, roll/Z) in degrees)
# ======================================================================

def _euler_deg_to_quat(ex, ey, ez):
    """Euler angles (degrees, YXZ intrinsic) → quaternion (w,x,y,z).

    Matches C++ ``EulerYXZToQuat``:  q = qY * qX * qZ.
    """
    rx = math.radians(ex) * 0.5
    ry = math.radians(ey) * 0.5
    rz = math.radians(ez) * 0.5
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        cy * cx * cz + sy * sx * sz,   # w
        cy * sx * cz + sy * cx * sz,   # x
        sy * cx * cz - cy * sx * sz,   # y
        cy * cx * sz - sy * sx * cz,   # z
    )


def _quat_to_euler_deg(q):
    """Quaternion (w,x,y,z) → Euler angles (degrees, YXZ intrinsic).

    Matches C++ ``QuatToEulerYXZ``.
    """
    w, x, y, z = q
    # sinX = 2(w*x - y*z)
    sin_x = 2.0 * (w * x - y * z)
    if abs(sin_x) < 0.9999:
        ex = math.asin(max(-1.0, min(1.0, sin_x)))
        ey = math.atan2(2.0 * (x * z + w * y),
                        1.0 - 2.0 * (x * x + y * y))
        ez = math.atan2(2.0 * (x * y + w * z),
                        1.0 - 2.0 * (x * x + z * z))
    else:
        # Gimbal lock at pitch = ±90°
        ex = math.copysign(math.pi / 2.0, sin_x)
        ey = math.atan2(-(2.0 * (x * z - w * y)),
                        1.0 - 2.0 * (y * y + z * z))
        ez = 0.0
    return (math.degrees(ex), math.degrees(ey), math.degrees(ez))


def _quat_mul(a, b):
    """Hamilton product of two quaternions (w,x,y,z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _axis_angle_to_quat(ax, ay, az, angle_deg):
    """Axis-angle → quaternion (w,x,y,z).  Axis must be unit-length."""
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


@editor_panel("Scene", type_id="scene_view", title_key="panel.scene")
class SceneViewPanel(EditorPanel):
    """
    Unity-style Scene View panel with 3D viewport and camera controls.
    
    Controls (Unity-style):
    - Right-click + drag: Rotate camera (look around)
    - Middle-click + drag: Pan camera
    - Scroll wheel: Zoom in/out (dolly)
    - Right-click + WASD: Fly mode movement
    - Right-click + QE: Up/Down in fly mode
    - Shift: Speed boost in fly mode
    """
    
    WINDOW_TYPE_ID = "scene_view"
    WINDOW_DISPLAY_NAME = "Scene"

    # Key codes imported from shared imgui_keys module
    KEY_W = _keys.KEY_W
    KEY_A = _keys.KEY_A
    KEY_S = _keys.KEY_S
    KEY_D = _keys.KEY_D
    KEY_Q = _keys.KEY_Q
    KEY_E = _keys.KEY_E
    KEY_R = _keys.KEY_R
    KEY_LEFT_SHIFT = _keys.KEY_LEFT_SHIFT
    KEY_RIGHT_SHIFT = _keys.KEY_RIGHT_SHIFT
    
    def __init__(self, title: str = "Scene", engine=None):
        super().__init__(title, window_id="scene_view")
        self._engine = engine
        self._play_mode_manager = None
        self._last_frame_time = 0.0
        self._on_object_picked = None
        self._on_box_select = None  # callback(primary_obj_or_None) after box-select
        
        # Scene render target size tracking
        self._last_scene_width = 0
        self._last_scene_height = 0
        
        # Mouse button state tracking for detecting press/release
        self._was_left_down = False
        self._was_right_down = False
        self._was_middle_down = False
        
        # Mouse position tracking for delta calculation
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Drag-outside-panel state (Unity-style: keep controlling camera
        # even when cursor leaves the panel, wrap at screen edges)
        self._is_camera_dragging = False
        self._last_global_x = 0.0
        self._last_global_y = 0.0
        self._screen_bounds = None  # (x, y, w, h)  cached display bounds

        # Editor gizmo drag state (shared across translate/rotate/scale)
        self._is_gizmo_dragging = False
        self._gizmo_drag_axis = 0          # 1=X, 2=Y, 3=Z
        self._gizmo_drag_axis_dir = (1.0, 0.0, 0.0)
        self._gizmo_drag_start_t = 0.0     # parameter along axis at grab (translate/scale)
        self._gizmo_drag_start_pos = (0.0, 0.0, 0.0)  # object pos at grab
        self._gizmo_drag_start_euler = (0.0, 0.0, 0.0)  # object euler at grab (rotate)
        self._gizmo_drag_start_scale = (1.0, 1.0, 1.0)  # object local_scale at grab (scale)
        self._gizmo_drag_start_screen = (0.0, 0.0) # screen pos at grab (rotate)
        self._gizmo_drag_obj_id = 0        # object being dragged
        self._gizmo_tool_mode = TOOL_TRANSLATE  # current tool mode (Python tracking)
        self._coord_space = 0  # 0=Global, 1=Local

        # Scene picking cycle state (Unity-style repeated click cycling)
        self._pick_cycle_candidates = []
        self._pick_cycle_index = -1
        self._pick_cycle_last_mouse = (-1.0, -1.0)
        self._pick_cycle_last_viewport = (0, 0)

        # Box-select state
        self._box_select_active = False
        self._box_select_start = (0.0, 0.0)  # screen-space start (local to viewport)
        self._box_select_end = (0.0, 0.0)
        self._box_select_vp = None  # ViewportInfo when box-select started

        # Focus tracking for auto-exit UI Mode
        self._was_focused: bool = False
        self._on_focus_gained = None   # callback() when panel gains focus

        # Smooth camera fly-to animation state
        self._fly_to_active: bool = False
        self._fly_to_start_focus = (0.0, 0.0, 0.0)
        self._fly_to_start_dist = 0.0
        self._fly_to_start_yaw = 0.0
        self._fly_to_start_pitch = 0.0
        self._fly_to_target_focus = (0.0, 0.0, 0.0)
        self._fly_to_target_dist = 0.0
        self._fly_to_target_yaw = 0.0
        self._fly_to_target_pitch = 0.0
        self._fly_to_elapsed = 0.0
        self._fly_to_duration = 0.5  # seconds
        # Near/far toggle: alternate between two distances on repeated
        # double-clicks of the same object (like Unity).
        self._fly_to_last_obj_id: int = 0
        self._fly_to_close: bool = False

        # Tool icon texture ids (lazily loaded)
        self._tool_icon_ids: dict[int, int] = {}  # tool_mode -> tex_id
        self._tool_icons_loaded: bool = False

        # Gizmo hover pick cache — avoids expensive per-triangle raycast
        # every frame when the mouse hasn't moved.
        self._hover_pick_cache_pos: tuple[float, float] = (-1.0, -1.0)
        self._hover_pick_cache_result: int = 0
    
    def set_engine(self, engine):
        """Set the engine reference for camera control."""
        self._engine = engine

    def set_play_mode_manager(self, manager):
        """Set the PlayModeManager so the panel can show play-mode border."""
        self._play_mode_manager = manager

    def set_on_object_picked(self, callback):
        """Set callback for scene object picking (receives object ID or 0)."""
        self._on_object_picked = callback

    def set_on_box_select(self, callback):
        """Set callback after box-select completes (receives primary obj or None)."""
        self._on_box_select = callback
    
    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (800, 600)

    def _window_flags(self) -> int:
        return Theme.WINDOW_FLAGS_VIEWPORT | Theme.WINDOW_FLAGS_NO_SCROLL

    def on_disable(self):
        """Panel closed — shrink render target to save GPU memory."""
        if self._engine:
            self._engine.set_scene_view_visible(False)
            if self._last_scene_width != 1 or self._last_scene_height != 1:
                self._engine.resize_scene_render_target(1, 1)
                self._last_scene_width = 1
                self._last_scene_height = 1

    def _on_not_visible(self, ctx):
        """Window collapsed/tabbed out — mark invisible for C++ side."""
        if self._engine:
            self._engine.set_scene_view_visible(False)

    def _pre_render(self, ctx):
        if self._engine:
            self._engine.set_scene_view_visible(True)

        import time
        current_time = time.time()
        self._delta_time = current_time - self._last_frame_time if self._last_frame_time > 0 else 0.016
        self._last_frame_time = current_time
        self._delta_time = min(self._delta_time, 0.1)

        if self._fly_to_active:
            self._tick_fly_to(self._delta_time)

        # Determine play-mode border colour
        self._play_border_clr = None
        pm = self._play_mode_manager
        if pm is None:
            from InfEngine.engine.play_mode import PlayModeManager, PlayModeState
            pm = PlayModeManager.instance()
        if pm and not pm.is_edit_mode:
            from InfEngine.engine.play_mode import PlayModeState
            self._play_border_clr = Theme.BORDER_PAUSE if pm.state == PlayModeState.PAUSED else Theme.BORDER_PLAY

    def _on_visible_pre(self, ctx):
        # Track focus to auto-exit UI Mode
        focused = ctx.is_window_focused(0)
        if focused and not self._was_focused:
            if self._on_focus_gained:
                self._on_focus_gained()
        self._was_focused = focused

    def on_render_content(self, ctx: InfGUIContext):
        delta_time = getattr(self, '_delta_time', 0.016)
        _play_border_clr = getattr(self, '_play_border_clr', None)

        # Get content region for scene viewport
        avail_width = ctx.get_content_region_avail_width()
        avail_height = ctx.get_content_region_avail_height()
        
        scene_width = max(int(avail_width), 64)
        scene_height = max(int(avail_height), 64)
        
        cursor_start_x = ctx.get_cursor_pos_x()
        cursor_start_y = ctx.get_cursor_pos_y()
        
        # Resize scene render target if size changed
        if self._engine and (scene_width != self._last_scene_width or scene_height != self._last_scene_height):
            self._engine.resize_scene_render_target(scene_width, scene_height)
            self._last_scene_width = scene_width
            self._last_scene_height = scene_height
        
        # Get and display scene texture
        scene_texture_id = 0
        if self._engine:
            scene_texture_id = self._engine.get_scene_texture_id()
        
        if scene_texture_id != 0:
            ctx.image(scene_texture_id, float(scene_width), float(scene_height), 0.0, 0.0, 1.0, 1.0)

            vp = capture_viewport_info(ctx)
            is_scene_hovered = vp.is_hovered

            ctx.set_cursor_pos_x(cursor_start_x + 8)
            ctx.set_cursor_pos_y(cursor_start_y + 8)
            overlay_hovered = self._draw_gizmo_overlay(ctx)

            self._draw_pos_overlay(ctx, vp)

            # Unity-style tool switching shortcuts (Q/W/E/R)
            if not ctx.want_text_input() and not ctx.is_mouse_button_down(1):
                if ctx.is_key_pressed(self.KEY_Q):
                    self._set_tool_mode(TOOL_NONE)
                elif ctx.is_key_pressed(self.KEY_W):
                    self._set_tool_mode(TOOL_TRANSLATE)
                elif ctx.is_key_pressed(self.KEY_E):
                    self._set_tool_mode(TOOL_ROTATE)
                elif ctx.is_key_pressed(self.KEY_R):
                    self._set_tool_mode(TOOL_SCALE)

                ctrl = ctx.is_key_down(_keys.KEY_LEFT_CTRL) or ctx.is_key_down(_keys.KEY_RIGHT_CTRL)
                if ctrl and ctx.is_key_pressed(_keys.KEY_F):
                    self._align_object_to_camera()

            # Gizmo interaction
            left_down = ctx.is_mouse_button_down(0)
            left_clicked = left_down and not self._was_left_down
            self._was_left_down = left_down
            gizmo_consumed = False

            if self._engine:
                local_mx, local_my = vp.mouse_local(ctx)
                gizmo_consumed = self._update_gizmo_interaction(
                    local_mx, local_my, vp.width, vp.height,
                    left_down, left_clicked, is_scene_hovered)
            
            # Camera drag
            right_down = ctx.is_mouse_button_down(1)
            middle_down = ctx.is_mouse_button_down(2)
            if is_scene_hovered and (right_down or middle_down) and not self._is_camera_dragging:
                self._is_camera_dragging = True
                self._fly_to_active = False
                self._screen_bounds = ctx.get_display_bounds()
                self._last_global_x = ctx.get_global_mouse_pos_x()
                self._last_global_y = ctx.get_global_mouse_pos_y()
            
            if self._is_camera_dragging and not right_down and not middle_down:
                self._is_camera_dragging = False
            
            if is_scene_hovered or self._is_camera_dragging:
                self._process_camera_input(ctx, delta_time)

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
            if _play_border_clr is not None:
                ctx.draw_rect(
                    vp.image_min_x, vp.image_min_y,
                    vp.image_max_x, vp.image_max_y,
                    *_play_border_clr,
                    thickness=Theme.BORDER_THICKNESS,
                )

        else:
            # Placeholder when texture not ready
            ctx.invisible_button("scene_placeholder", float(scene_width), float(scene_height))
            ctx.set_cursor_pos_x(cursor_start_x + 8)
            ctx.set_cursor_pos_y(cursor_start_y + 8)
            ctx.label(t("scene_view.loading"))
    
    def _draw_gizmo_overlay(self, ctx: InfGUIContext) -> bool:
        """Draw the top-left gizmo controls and return whether they are hovered."""
        hovered = self._draw_coord_space_dropdown(ctx)
        # Measure the combo height so tool buttons match exactly
        combo_h = ctx.get_item_rect_max_y() - ctx.get_item_rect_min_y()
        ctx.same_line(0, Theme.SCENE_GIZMO_TOOL_BTN_GAP)
        hovered = self._draw_tool_mode_buttons(ctx, combo_h) or hovered
        return hovered

    def _draw_coord_space_dropdown(self, ctx: InfGUIContext) -> bool:
        """Draw Global/Local coordinate-space dropdown in the top-left corner."""
        _SPACE_LABELS = [t("scene_view.global"), t("scene_view.local")]
        ctx.push_id_str("coord_space_dropdown")
        # Style the combo to look like a semi-transparent overlay control
        ctx.push_style_color(ImGuiCol.FrameBg, *Theme.SCENE_OVERLAY_COMBO_BG)
        ctx.push_style_color(ImGuiCol.FrameBgHovered, *Theme.SCENE_OVERLAY_COMBO_HOVER)
        ctx.push_style_color(ImGuiCol.FrameBgActive, *Theme.SCENE_OVERLAY_COMBO_ACTIVE)
        ctx.push_style_var_float(ImGuiStyleVar.FrameRounding, Theme.SCENE_OVERLAY_ROUNDING)
        ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, Theme.SCENE_OVERLAY_BORDER_SIZE)
        ctx.set_next_item_width(Theme.SCENE_COORD_DROPDOWN_W)
        new_val = ctx.combo("##coord_space", self._coord_space, _SPACE_LABELS)
        hovered = ctx.is_item_hovered()
        ctx.pop_style_var(2)
        ctx.pop_style_color(3)
        if new_val != self._coord_space:
            self._coord_space = new_val
            # Sync local mode to C++ so gizmo visuals align to object rotation
            if self._engine:
                self._engine.set_editor_tool_local_mode(self._coord_space == 1)
        ctx.pop_id()
        return hovered

    def _ensure_tool_icons(self):
        """Lazily upload tool icon textures to GPU."""
        if self._tool_icons_loaded or not self._engine:
            return
        native = self._engine.get_native_engine() if hasattr(self._engine, 'get_native_engine') else self._engine
        if native is None:
            return
        _ICON_MAP = {
            TOOL_NONE:      "tool_none.png",
            TOOL_TRANSLATE: "tool_move.png",
            TOOL_ROTATE:    "tool_rotate.png",
            TOOL_SCALE:     "tool_scale.png",
        }
        for mode, filename in _ICON_MAP.items():
            tex_name = f"__toolicon__{filename}"
            if native.has_imgui_texture(tex_name):
                self._tool_icon_ids[mode] = native.get_imgui_texture_id(tex_name)
                continue
            icon_path = os.path.join(file_type_icons_dir, filename)
            if not os.path.isfile(icon_path):
                continue
            tex_data = TextureLoader.load_from_file(icon_path)
            if tex_data and tex_data.is_valid():
                pixels = tex_data.get_pixels_list()
                tid = native.upload_texture_for_imgui(
                    tex_name, pixels, tex_data.width, tex_data.height)
                if tid != 0:
                    self._tool_icon_ids[mode] = tid
        self._tool_icons_loaded = True

    def _draw_tool_mode_buttons(self, ctx: InfGUIContext, combo_h: float = 20.0) -> bool:
        """Draw horizontally aligned gizmo-tool icon buttons matching the combo height."""
        self._ensure_tool_icons()
        items = [
            (TOOL_NONE,      t("scene_view.tool_select"), "##tool_none"),
            (TOOL_TRANSLATE, t("scene_view.tool_move"),   "##tool_move"),
            (TOOL_ROTATE,    t("scene_view.tool_rotate"), "##tool_rotate"),
            (TOOL_SCALE,     t("scene_view.tool_scale"),  "##tool_scale"),
        ]
        pad = Theme.SCENE_GIZMO_TOOL_BTN_PAD
        icon_size = max(combo_h - pad[1] * 2, 8.0)
        gap = Theme.SCENE_GIZMO_TOOL_BTN_GAP
        hovered = False
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *pad)
        ctx.push_style_var_float(ImGuiStyleVar.FrameRounding, Theme.SCENE_OVERLAY_ROUNDING)
        for i, (mode, label, btn_id) in enumerate(items):
            if i > 0:
                ctx.same_line(0, gap)
            active = (self._gizmo_tool_mode == mode)
            if active:
                ctx.push_style_color(ImGuiCol.Button, 235.0 / 255.0, 87.0 / 255.0, 87.0 / 255.0, 0.95)
                ctx.push_style_color(ImGuiCol.ButtonHovered, 1.0, 107.0 / 255.0, 107.0 / 255.0, 1.0)
                ctx.push_style_color(ImGuiCol.ButtonActive, 220.0 / 255.0, 67.0 / 255.0, 67.0 / 255.0, 1.0)
            else:
                ctx.push_style_color(ImGuiCol.Button, *Theme.SCENE_OVERLAY_COMBO_BG)
                ctx.push_style_color(ImGuiCol.ButtonHovered, *Theme.SCENE_OVERLAY_COMBO_HOVER)
                ctx.push_style_color(ImGuiCol.ButtonActive, *Theme.SCENE_OVERLAY_COMBO_ACTIVE)
            tex_id = self._tool_icon_ids.get(mode, 0)
            clicked = False
            if tex_id != 0:
                clicked = ctx.image_button(btn_id, tex_id, icon_size, icon_size)
            else:
                clicked = ctx.button(label, lambda m=mode: self._set_tool_mode(m),
                                     width=combo_h, height=combo_h)
            if clicked:
                self._set_tool_mode(mode)
            hovered = ctx.is_item_hovered() or hovered
            ctx.pop_style_color(3)
        ctx.pop_style_var(2)
        return hovered

    def _draw_pos_overlay(self, ctx: InfGUIContext, vp: ViewportInfo):
        """Draw a Unity-style orientation gizmo in the top-right corner."""
        if not self._engine:
            return
        self._draw_orientation_gizmo(ctx, vp)

    # ------------------------------------------------------------------
    # Orientation Gizmo  (Unity-style axis widget)
    # ------------------------------------------------------------------

    # Orientation gizmo constants now come from Theme
    _GIZMO_AXIS_COLORS = {
        'X': (0.9, 0.2, 0.2),      # red
        'Y': (0.2, 0.85, 0.2),     # green
        'Z': (0.3, 0.4, 0.95),     # blue
    }
    _GIZMO_AXIS_VIEWS = {
        # axis_label: (yaw, pitch) — camera looks FROM the positive axis direction
        '+X': (-90.0, 0.0),
        '-X': (90.0, 0.0),
        '+Y': (0.0, -89.9),   # avoid exact 90 to prevent gimbal lock
        '-Y': (0.0, 89.9),
        '+Z': (180.0, 0.0),
        '-Z': (0.0, 0.0),
    }

    def _draw_orientation_gizmo(self, ctx: InfGUIContext, vp: ViewportInfo):
        """Draw orientation gizmo with clickable axis endpoints."""
        cam = self._engine.editor_camera
        if not cam:
            return
        yaw, pitch = cam.rotation
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)

        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

        # Reconstruct the actual camera basis used by the Scene camera.
        forward = (sin_y * cos_p, sin_p, cos_y * cos_p)
        right = (cos_y, 0.0, -sin_y)
        up = (
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        )

        r = Theme.SCENE_ORIENT_RADIUS
        margin = Theme.SCENE_ORIENT_MARGIN
        # Use screen-absolute coordinates from the viewport info
        cx = vp.image_max_x - r - margin
        cy = vp.image_min_y + r + margin

        # Project world axis to 2D screen position
        axis_len = Theme.SCENE_ORIENT_AXIS_LEN
        axes = [
            ('X', (1, 0, 0)),
            ('Y', (0, 1, 0)),
            ('Z', (0, 0, 1)),
        ]

        # Collect all endpoints for depth sorting (draw far ones first)
        endpoints = []
        for label, (ax, ay, az) in axes:
            sx = ax * right[0] + ay * right[1] + az * right[2]
            sy = ax * up[0] + ay * up[1] + az * up[2]
            depth = ax * forward[0] + ay * forward[1] + az * forward[2]
            endpoints.append((label, sx, sy, depth, True))    # positive
            endpoints.append((label, -sx, -sy, -depth, False)) # negative

        # Sort by depth (farthest first → painter's order)
        endpoints.sort(key=lambda e: e[3])

        # Draw axis lines first (below circles)
        for label, sx, sy, depth, positive in endpoints:
            if not positive:
                continue
            clr = self._GIZMO_AXIS_COLORS[label]
            ex = cx + sx * axis_len
            ey = cy - sy * axis_len
            ctx.draw_line(cx, cy, ex, ey, *clr, 0.6, 2.0)

        # Draw endpoints
        mouse_x = ctx.get_mouse_pos_x()
        mouse_y = ctx.get_mouse_pos_y()
        clicked_axis = None

        for label, sx, sy, depth, positive in endpoints:
            clr = self._GIZMO_AXIS_COLORS[label]
            if positive:
                ex = cx + sx * axis_len
                ey = cy - sy * axis_len
                er = Theme.SCENE_ORIENT_END_RADIUS
                a = 1.0
                axis_key = '+' + label
            else:
                ex = cx + sx * axis_len
                ey = cy - sy * axis_len
                er = Theme.SCENE_ORIENT_NEG_RADIUS
                a = 0.5
                axis_key = '-' + label

            # Draw filled circle
            ctx.draw_filled_circle(ex, ey, er, clr[0], clr[1], clr[2], a, 16)

            # Draw label on positive endpoints
            if positive:
                ctx.draw_text(ex - 3, ey - 5, label, 1.0, 1.0, 1.0, 1.0)

            # Hit test
            dx = mouse_x - ex
            dy = mouse_y - ey
            if dx * dx + dy * dy <= er * er * 1.5:
                # Highlight on hover
                ctx.draw_circle(ex, ey, er + 1, 1.0, 1.0, 1.0, 0.7, 1.5, 16)
                if ctx.is_mouse_button_clicked(0):
                    clicked_axis = axis_key

        # Handle click — animate camera to axis view
        if clicked_axis is not None:
            target_yaw, target_pitch = self._GIZMO_AXIS_VIEWS[clicked_axis]
            self._start_fly_to_orientation(target_yaw, target_pitch)

    def _start_fly_to_orientation(self, target_yaw: float, target_pitch: float):
        """Start a smooth camera animation to a specific orientation."""
        cam = self._engine.editor_camera
        if not cam:
            return
        cur_pos = cam.position
        cur_dist = cam.focus_distance
        cur_yaw, cur_pitch = cam.rotation

        # Compute consistent focus from actual camera position to avoid
        # stale m_focusPoint causing an initial teleport/flash.
        yr = math.radians(cur_yaw)
        pr = math.radians(cur_pitch)
        cp = math.cos(pr)
        fwd = (math.sin(yr) * cp, math.sin(pr), math.cos(yr) * cp)
        focus = (cur_pos.x + fwd[0] * cur_dist,
                 cur_pos.y + fwd[1] * cur_dist,
                 cur_pos.z + fwd[2] * cur_dist)

        self._fly_to_start_focus = focus
        self._fly_to_start_dist = cur_dist
        self._fly_to_start_yaw = cur_yaw
        self._fly_to_start_pitch = cur_pitch

        self._fly_to_target_focus = focus  # keep same focus point
        self._fly_to_target_dist = cur_dist    # keep same distance
        self._fly_to_target_yaw = target_yaw
        self._fly_to_target_pitch = target_pitch

        self._fly_to_elapsed = 0.0
        self._fly_to_duration = Theme.SCENE_ORIENT_FLY_DURATION
        self._fly_to_active = True

    def _finalize_box_select(self, ctx: InfGUIContext, vp: ViewportInfo):
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
        from InfEngine.lib import SceneManager
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
            except RuntimeError:
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

    def _pick_scene_object(self, ctx: InfGUIContext, vp: ViewportInfo) -> int:
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

        # Fallback to old single-hit API if list API is unavailable/empty.
        if not ids:
            fallback_id = int(self._engine.pick_scene_object_id(local_x, local_y, vp.width, vp.height))

            if fallback_id > 0 and fallback_id not in _GIZMO_IDS:
                self._pick_cycle_candidates = [fallback_id]
                self._pick_cycle_index = 0
                self._pick_cycle_last_mouse = (local_x, local_y)
                self._pick_cycle_last_viewport = (int(vp.width), int(vp.height))
                return fallback_id

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
    
    def _process_camera_input(self, ctx: InfGUIContext, delta_time: float):
        """Process Unity-style camera input using mouse position delta.
        
        When the mouse is being dragged (right/middle button held), we track
        global screen coordinates so the camera keeps moving even when the
        cursor leaves the panel.  When the cursor reaches a screen edge it
        wraps to the opposite side, just like Unity.
        """
        if not self._engine:
            return
        
        # Mouse button states
        right_down = ctx.is_mouse_button_down(1)
        middle_down = ctx.is_mouse_button_down(2)
        
        # Detect button just pressed
        right_just_pressed = right_down and not self._was_right_down
        middle_just_pressed = middle_down and not self._was_middle_down
        
        # ------------------------------------------------------------------
        # Use GLOBAL mouse position when dragging so that leaving the panel
        # doesn't interrupt camera control.
        # ------------------------------------------------------------------
        if self._is_camera_dragging:
            gx = ctx.get_global_mouse_pos_x()
            gy = ctx.get_global_mouse_pos_y()
        else:
            gx = ctx.get_mouse_pos_x()
            gy = ctx.get_mouse_pos_y()
        
        mouse_delta_x = 0.0
        mouse_delta_y = 0.0
        
        if (right_down or middle_down) and not right_just_pressed and not middle_just_pressed:
            raw_dx = gx - self._last_global_x
            raw_dy = gy - self._last_global_y
            
            # Ignore very large jumps caused by screen-edge warp
            if abs(raw_dx) < 400 and abs(raw_dy) < 400:
                if abs(raw_dx) > 0.1:
                    mouse_delta_x = raw_dx
                if abs(raw_dy) > 0.1:
                    mouse_delta_y = raw_dy
            
            # --- Screen-edge wrapping (Unity-style) ---
            if self._is_camera_dragging and self._screen_bounds:
                sx, sy, sw, sh = self._screen_bounds
                margin = 2.0
                warped = False
                new_gx, new_gy = gx, gy
                
                if gx <= sx + margin:
                    new_gx = sx + sw - margin - 1
                    warped = True
                elif gx >= sx + sw - margin:
                    new_gx = sx + margin + 1
                    warped = True
                
                if gy <= sy + margin:
                    new_gy = sy + sh - margin - 1
                    warped = True
                elif gy >= sy + sh - margin:
                    new_gy = sy + margin + 1
                    warped = True
                
                if warped:
                    ctx.warp_mouse_global(new_gx, new_gy)
                    gx, gy = new_gx, new_gy
        
        # Update tracking state
        self._last_global_x = gx
        self._last_global_y = gy
        # Also keep local tracking in sync for picking etc.
        self._last_mouse_x = ctx.get_mouse_pos_x()
        self._last_mouse_y = ctx.get_mouse_pos_y()
        self._was_right_down = right_down
        self._was_middle_down = middle_down
        
        # Scroll wheel: zoom
        scroll_delta = ctx.get_mouse_wheel_delta()
        
        # Keyboard for fly mode (only when right mouse held)
        key_w = right_down and ctx.is_key_down(self.KEY_W)
        key_s = right_down and ctx.is_key_down(self.KEY_S)
        key_a = right_down and ctx.is_key_down(self.KEY_A)
        key_d = right_down and ctx.is_key_down(self.KEY_D)
        key_q = right_down and ctx.is_key_down(self.KEY_Q)
        key_e = right_down and ctx.is_key_down(self.KEY_E)
        key_shift = ctx.is_key_down(self.KEY_LEFT_SHIFT) or ctx.is_key_down(self.KEY_RIGHT_SHIFT)
        
        # Send to engine
        self._engine.process_scene_view_input(
            delta_time,
            right_down,
            middle_down,
            mouse_delta_x,
            mouse_delta_y,
            scroll_delta,
            key_w, key_a, key_s, key_d,
            key_q, key_e, key_shift
        )
    
    # ------------------------------------------------------------------
    # Tool mode management
    # ------------------------------------------------------------------

    def _set_tool_mode(self, mode: int):
        """Switch the active editor tool (syncs to C++ and resets drag)."""
        if mode == self._gizmo_tool_mode:
            return
        self._gizmo_tool_mode = mode
        self._is_gizmo_dragging = False
        if self._engine:
            self._engine.set_editor_tool_mode(mode)
            self._engine.set_editor_tool_highlight(0)

    # ------------------------------------------------------------------
    # Gizmo interaction helpers (all in Python)
    # ------------------------------------------------------------------

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

    def _update_gizmo_interaction(self, local_mx, local_my, scene_w, scene_h,
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
                engine.set_editor_tool_highlight(0)
                return False

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

        axis = _GIZMO_IDS.get(picked, 0)
        engine.set_editor_tool_highlight(axis)

        if axis == 0:
            return False  # not hovering any gizmo handle

        # -----------------------------------------------------------
        # DRAG START (common for all modes)
        # Only initiate drag on a fresh press — not when the button
        # was already held and the cursor drifted over the gizmo.
        # -----------------------------------------------------------
        if left_clicked:
            self._is_gizmo_dragging = True
            self._gizmo_drag_axis = axis
            self._gizmo_drag_start_screen = (local_mx, local_my)

            from InfEngine.lib._InfEngine import SceneManager as _SM
            scene = _SM.instance().get_active_scene()
            sel_id = engine.get_selected_object_id()
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

                    # Compute axis direction based on coordinate space
                    if self._coord_space == 1 and obj:
                        # Local space: use object's local axes
                        if axis == 1:
                            r = obj.transform.right
                            self._gizmo_drag_axis_dir = (r[0], r[1], r[2])
                        elif axis == 2:
                            u = obj.transform.up
                            self._gizmo_drag_axis_dir = (u[0], u[1], u[2])
                        elif axis == 3:
                            f = obj.transform.forward
                            self._gizmo_drag_axis_dir = (f[0], f[1], f[2])
                    else:
                        self._gizmo_drag_axis_dir = _AXIS_DIRS[axis]
            else:
                self._gizmo_drag_axis_dir = _AXIS_DIRS[axis]
            self._gizmo_drag_obj_id = sel_id
            self._gizmo_drag_start_pos = obj_pos
            self._gizmo_drag_start_euler = obj_euler
            self._gizmo_drag_start_scale = obj_scale

            # For translate/scale: record initial axis parameter
            if mode in (TOOL_TRANSLATE, TOOL_SCALE):
                ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
                self._gizmo_drag_start_t = self._closest_param_on_axis(
                    ray[:3], ray[3:], self._gizmo_drag_start_pos, self._gizmo_drag_axis_dir)

            return True  # consumed

        return True  # hovering a gizmo handle — consume to suppress picking

    # ------------------------------------------------------------------
    # Mode-specific drag handlers
    # ------------------------------------------------------------------

    def _drag_translate(self, engine, local_mx, local_my, scene_w, scene_h):
        """Axis-constrained translation: project mouse ray onto drag axis."""
        ray = engine.screen_to_world_ray(local_mx, local_my, scene_w, scene_h)
        ad = self._gizmo_drag_axis_dir
        sp = self._gizmo_drag_start_pos

        cur_t = self._closest_param_on_axis(ray[:3], ray[3:], sp, ad)
        delta = cur_t - self._gizmo_drag_start_t

        new_pos = (sp[0] + ad[0] * delta,
                   sp[1] + ad[1] * delta,
                   sp[2] + ad[2] * delta)
        from InfEngine.lib._InfEngine import SceneManager as _SM, Vector3
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

        se = self._gizmo_drag_start_euler
        q_start = _euler_deg_to_quat(se[0], se[1], se[2])
        q_delta = _axis_angle_to_quat(ad[0], ad[1], ad[2], angle_deg)

        # Always pre-multiply: the axis in q_delta is already expressed in
        # world space for both Global mode (world unit axis) and Local mode
        # (object's local axis mapped to world space).
        q_new = _quat_mul(q_delta, q_start)
        new_euler = _quat_to_euler_deg(q_new)

        from InfEngine.lib._InfEngine import SceneManager as _SM, Vector3
        scene = _SM.instance().get_active_scene()
        if scene:
            obj = scene.find_by_id(self._gizmo_drag_obj_id)
            if obj:
                obj.transform.euler_angles = Vector3(new_euler[0], new_euler[1], new_euler[2])

    def _drag_scale(self, engine, local_mx, local_my, scene_w, scene_h):
        """Scale along the drag axis. In Local mode, scale applies directly to
        the corresponding local_scale component. In Global mode, the world-axis
        scale factor is decomposed onto local axes."""
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
        factor = max(factor, 0.01)

        from InfEngine.lib._InfEngine import SceneManager as _SM, Vector3
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

    def _record_gizmo_undo(self, mode: int):
        """Record an undo command for the gizmo drag that just finished."""
        from InfEngine.lib._InfEngine import SceneManager as _SM, Vector3
        from InfEngine.engine.undo import UndoManager, SetPropertyCommand

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

    @staticmethod
    def _vec3_approx_equal(a, b, eps=1e-5):
        """Check if two Vector3-like objects are approximately equal."""
        return (abs(a[0] - b[0]) < eps and
                abs(a[1] - b[1]) < eps and
                abs(a[2] - b[2]) < eps)

    def reset_camera(self):
        """Reset camera to default position."""
        cam = self._engine.editor_camera if self._engine else None
        if cam:
            cam.reset()
    
    def focus_on(self, x: float, y: float, z: float, distance: float = 10.0):
        """Focus camera on a point."""
        cam = self._engine.editor_camera if self._engine else None
        if cam:
            cam.focus_on(x, y, z, distance)

    # ------------------------------------------------------------------
    # Smooth camera fly-to (animated frame-selected)
    # ------------------------------------------------------------------

    def fly_to_object(self, game_object):
        """Start a smooth camera animation to focus on *game_object*.

        Computes the bounding sphere of all MeshRenderers on the object
        (and children) and derives the camera distance using Unity's
        formula: distance = radius / sin(fov/2).

        Alternates between a *far* (framing) and *close* (detail) distance
        on repeated double-clicks of the same object, like Unity.
        """
        if not self._engine or game_object is None:
            return

        obj_id = game_object.id

        # Toggle near/far on repeated double-click of the same object
        if obj_id == self._fly_to_last_obj_id:
            self._fly_to_close = not self._fly_to_close
        else:
            self._fly_to_close = False
        self._fly_to_last_obj_id = obj_id

        center, radius = self._compute_object_bounds(game_object)

        # Target distance (Unity formula + small padding)
        cam = self._engine.editor_camera
        fov_deg = cam.fov
        half_fov_rad = math.radians(fov_deg * 0.5)
        sin_half = math.sin(half_fov_rad)
        if sin_half < 1e-6:
            sin_half = 1e-6
        far_dist = max(radius / sin_half * 1.2, 0.5)
        close_dist = far_dist * 0.4
        target_dist = close_dist if self._fly_to_close else far_dist

        # Current camera state — compute consistent focus from actual
        # camera position to avoid stale m_focusPoint causing a flash.
        cur_pos = cam.position
        cur_dist = cam.focus_distance
        cur_yaw, cur_pitch = cam.rotation

        yr = math.radians(cur_yaw)
        pr = math.radians(cur_pitch)
        cp = math.cos(pr)
        fwd = (math.sin(yr) * cp, math.sin(pr), math.cos(yr) * cp)
        actual_focus = (cur_pos.x + fwd[0] * cur_dist,
                        cur_pos.y + fwd[1] * cur_dist,
                        cur_pos.z + fwd[2] * cur_dist)

        # Target yaw/pitch: keep current viewing direction
        target_yaw = cur_yaw
        target_pitch = cur_pitch

        # Store animation state
        self._fly_to_start_focus = actual_focus
        self._fly_to_start_dist = cur_dist
        self._fly_to_start_yaw = cur_yaw
        self._fly_to_start_pitch = cur_pitch

        self._fly_to_target_focus = center
        self._fly_to_target_dist = target_dist
        self._fly_to_target_yaw = target_yaw
        self._fly_to_target_pitch = target_pitch

        self._fly_to_elapsed = 0.0
        self._fly_to_duration = 0.5
        self._fly_to_active = True

    def _tick_fly_to(self, dt: float):
        """Advance the fly-to animation by *dt* seconds."""
        self._fly_to_elapsed += dt
        t = min(self._fly_to_elapsed / self._fly_to_duration, 1.0)

        # Cubic ease-out for smooth deceleration
        t = 1.0 - (1.0 - t) ** 3

        # Interpolate focus point
        fx = self._fly_to_start_focus[0] + (self._fly_to_target_focus[0] - self._fly_to_start_focus[0]) * t
        fy = self._fly_to_start_focus[1] + (self._fly_to_target_focus[1] - self._fly_to_start_focus[1]) * t
        fz = self._fly_to_start_focus[2] + (self._fly_to_target_focus[2] - self._fly_to_start_focus[2]) * t

        # Interpolate distance
        dist = self._fly_to_start_dist + (self._fly_to_target_dist - self._fly_to_start_dist) * t

        # Interpolate yaw/pitch (shortest-path for yaw)
        yaw = self._lerp_angle(self._fly_to_start_yaw, self._fly_to_target_yaw, t)
        pitch = self._fly_to_start_pitch + (self._fly_to_target_pitch - self._fly_to_start_pitch) * t

        # Compute camera position from focus - forward * distance.
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        cos_pitch = math.cos(pitch_rad)
        forward_x = math.sin(yaw_rad) * cos_pitch
        forward_y = math.sin(pitch_rad)
        forward_z = math.cos(yaw_rad) * cos_pitch
        px = fx - forward_x * dist
        py = fy - forward_y * dist
        pz = fz - forward_z * dist

        self._engine.editor_camera.restore_state(
            px, py, pz, fx, fy, fz, dist, yaw, pitch
        )

        if t >= 1.0:
            self._fly_to_active = False

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        """Linearly interpolate between angles (degrees) via shortest path."""
        diff = ((b - a + 180.0) % 360.0) - 180.0
        return a + diff * t

    @staticmethod
    def _compute_object_bounds(game_object):
        """Compute bounding-sphere center and radius for *game_object*.

        Merges world-space AABBs of all MeshRenderers on the object and its
        children.  Falls back to the transform position with a default radius
        if no renderers exist.
        """
        bmin = [float('inf')] * 3
        bmax = [float('-inf')] * 3
        found = False

        def _collect(obj):
            nonlocal found
            mr = obj.get_cpp_component("MeshRenderer")
            if mr is not None:
                try:
                    bounds = mr.get_world_bounds()
                    if bounds and len(bounds) == 6:
                        for i in range(3):
                            if bounds[i] < bmin[i]:
                                bmin[i] = bounds[i]
                            if bounds[i + 3] > bmax[i]:
                                bmax[i] = bounds[i + 3]
                        found = True
                except Exception:
                    pass
            for child in obj.get_children():
                _collect(child)

        _collect(game_object)

        if found:
            cx = (bmin[0] + bmax[0]) * 0.5
            cy = (bmin[1] + bmax[1]) * 0.5
            cz = (bmin[2] + bmax[2]) * 0.5
            dx = bmax[0] - bmin[0]
            dy = bmax[1] - bmin[1]
            dz = bmax[2] - bmin[2]
            radius = math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5
            return (cx, cy, cz), max(radius, 0.1)

        # Fallback: use transform position with a default radius
        pos = game_object.transform.position
        return (pos.x, pos.y, pos.z), 1.0

    def _align_object_to_camera(self):
        """Align the selected object's world transform to the editor camera."""
        if not self._engine:
            return

        obj_id = self._engine.get_selected_object_id()
        if not obj_id:
            return

        from InfEngine.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(obj_id)
        if obj is None:
            return

        cam = self._engine.editor_camera
        cam_pos = cam.position
        cam_yaw, cam_pitch = cam.rotation

        from InfEngine.math import Vector3

        transform = obj.transform
        old_pos = (transform.position.x, transform.position.y, transform.position.z)
        old_euler = (transform.euler_angles.x, transform.euler_angles.y, transform.euler_angles.z)

        new_pos = Vector3(cam_pos.x, cam_pos.y, cam_pos.z)
        new_euler = Vector3(cam_pitch, cam_yaw, 0.0)

        transform.position = new_pos
        transform.euler_angles = new_euler

        # Record undo
        from InfEngine.engine.undo import UndoManager, SetPropertyCommand
        mgr = UndoManager.instance()
        if mgr:
            mgr.record(SetPropertyCommand(
                transform, "position",
                Vector3(*old_pos), new_pos, "Align Position"))
            mgr.record(SetPropertyCommand(
                transform, "euler_angles",
                Vector3(*old_euler), new_euler, "Align Rotation"))


