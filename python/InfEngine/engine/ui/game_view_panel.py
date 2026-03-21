"""
Unity-style Game View panel — renders the scene through the game camera.

The Game View uses a separate render target from the Scene View.
It displays what the player would see through the scene's main Camera component.
"""

import os
import configparser
from operator import attrgetter
from typing import Optional
from InfEngine.lib import InfGUIContext
from InfEngine.engine.i18n import t
from InfEngine.input import Input, KeyCode
from InfEngine.timing import Time
from InfEngine.engine.play_mode import PlayModeManager
from InfEngine.engine.project_context import get_project_root
from InfEngine.ui.ui_texture_cache import get_shared_cache as _get_tex_cache
from InfEngine.ui.ui_render_dispatch import dispatch as _ui_dispatch
from InfEngine.ui.ui_event_system import UIEventProcessor
from InfEngine.ui.ui_canvas_utils import collect_sorted_canvases
from InfEngine.ui.inf_ui_screen_component import clear_rect_cache
from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol
from .viewport_utils import capture_viewport_info

_sort_by_sort_order = attrgetter('sort_order')


@editor_panel("Game", type_id="game_view", title_key="panel.game")
class GameViewPanel(EditorPanel):
    """
    Unity-style Game View panel that renders the scene's main Camera output.
    
    The game camera is automatically bound to the first Camera component found
    in the scene (Scene.main_camera). When no camera is present, a helpful
    message is displayed instead.
    """
    
    WINDOW_TYPE_ID = "game_view"
    WINDOW_DISPLAY_NAME = "Game"

    _RESOLUTION_PRESETS = [
        ("1920\u00d71080", 1920, 1080),
        ("1280\u00d7720", 1280, 720),
        ("2560\u00d71440", 2560, 1440),
        ("3840\u00d72160", 3840, 2160),
        ("1080\u00d71920 Portrait", 1080, 1920),
        ("Custom", 1920, 1080),
    ]
    _PRESET_NAMES = [p[0] for p in _RESOLUTION_PRESETS]
    
    def __init__(self, title: str = "Game", engine=None, play_mode_manager: Optional[PlayModeManager] = None):
        super().__init__(title, window_id="game_view")
        self._engine = engine
        self._play_mode_manager = play_mode_manager
        if self._engine and self._play_mode_manager is None:
            self._play_mode_manager = self._engine.get_play_mode_manager()
        self.__is_playing = False
        
        # Game render target size tracking
        self._last_game_width = 0
        self._last_game_height = 0
        self._game_camera_was_enabled = False

        # Focus tracking for auto-exit UI Mode
        self._was_focused: bool = False
        self._on_focus_gained = None   # callback() when panel gains focus

        # UI event processor — dispatches pointer events to UI elements
        self._ui_event_processor = UIEventProcessor()

        # Game resolution selection (Unity-like)
        self._selected_resolution_idx = 0
        self._custom_width = 1920
        self._custom_height = 1080
        self._display_scale = 0.5
        self._fit_mode = True            # When True, scale auto-adjusts to fill area
        self._settings_loaded = False

        # FPS display uses a 1-second rolling update to avoid noisy per-frame jitter.
        self._fps_accum_time = 0.0
        self._fps_accum_frames = 0
        self._display_fps = 0.0
        self._display_frame_ms = 0.0

    def _set_game_render_active(self, active: bool) -> None:
        """Keep C++ game rendering in lockstep with actual panel visibility.

        Hidden dock tabs do not execute the panel body, so all per-frame Game
        view maintenance stops. If C++ keeps rendering the Game graph anyway,
        it can submit work against stale Game-view state and trigger
        VK_ERROR_DEVICE_LOST when the tab is switched away or closed.
        """
        active = bool(active)
        if not active:
            Input.set_game_focused(False)

        if not self._engine:
            if not active:
                self._game_camera_was_enabled = False
            return

        if active and not self._game_camera_was_enabled:
            self._engine.set_game_camera_enabled(True)
            self._game_camera_was_enabled = True
        elif not active and self._game_camera_was_enabled:
            self._engine.set_game_camera_enabled(False)
            self._game_camera_was_enabled = False
    
    def set_engine(self, engine):
        self._engine = engine
        if self._engine:
            self._play_mode_manager = self._engine.get_play_mode_manager()
    
    def set_play_mode_manager(self, manager: PlayModeManager):
        self._play_mode_manager = manager
    
    def _is_playing(self) -> bool:
        if self._play_mode_manager:
            return self._play_mode_manager.is_playing
        return self.__is_playing
    
    def _is_paused(self) -> bool:
        if self._play_mode_manager:
            return self._play_mode_manager.is_paused
        return False
    
    def _on_play_stop_clicked(self):
        if self._play_mode_manager:
            if self._play_mode_manager.is_playing:
                self._play_mode_manager.exit_play_mode()
            else:
                self._play_mode_manager.enter_play_mode()
        else:
            self.__is_playing = not self.__is_playing
    
    def _on_pause_clicked(self):
        if self._play_mode_manager and self._play_mode_manager.is_playing:
            self._play_mode_manager.toggle_pause()

    def _settings_ini_path(self) -> Optional[str]:
        root = get_project_root()
        if not root:
            return None
        return os.path.join(root, "ProjectSettings", "GameView.ini")

    def _load_resolution_settings(self):
        if self._settings_loaded:
            return
        self._settings_loaded = True

        path = self._settings_ini_path()
        if not path:
            return
        if not os.path.isfile(path):
            self._save_resolution_settings()
            return

        cp = configparser.ConfigParser()
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                cp.read_string(f.read())
        except (OSError, configparser.Error):
            return
        if "GameView" not in cp:
            return

        section = cp["GameView"]
        self._selected_resolution_idx = max(0, min(len(self._RESOLUTION_PRESETS) - 1,
                                                   section.getint("preset_index", fallback=0)))
        self._custom_width = max(64, section.getint("custom_width", fallback=1920))
        self._custom_height = max(64, section.getint("custom_height", fallback=1080))
        self._display_scale = max(0.1, min(2.0, section.getfloat("display_scale", fallback=0.5)))
        self._fit_mode = section.getboolean("fit_mode", fallback=True)

    def _save_resolution_settings(self):
        path = self._settings_ini_path()
        if not path:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cp = configparser.ConfigParser()
        cp["GameView"] = {
            "preset_index": str(self._selected_resolution_idx),
            "custom_width": str(max(64, int(self._custom_width))),
            "custom_height": str(max(64, int(self._custom_height))),
            "display_scale": f"{self._display_scale:.3f}",
            "fit_mode": str(self._fit_mode),
        }
        with open(path, "w", encoding="utf-8") as f:
            cp.write(f)

    def _current_target_resolution(self):
        _, w, h = self._RESOLUTION_PRESETS[self._selected_resolution_idx]
        if self._selected_resolution_idx == len(self._RESOLUTION_PRESETS) - 1:
            return max(64, int(self._custom_width)), max(64, int(self._custom_height))
        return int(w), int(h)

    def _fit_scale(self):
        """Toggle Fit mode on."""
        self._fit_mode = True
        self._save_resolution_settings()

    @staticmethod
    def _fit_into_region(src_w: int, src_h: int, region_w: float, region_h: float):
        if src_w <= 0 or src_h <= 0 or region_w <= 0 or region_h <= 0:
            return 0.0, 0.0
        scale = min(region_w / float(src_w), region_h / float(src_h))
        return float(src_w) * scale, float(src_h) * scale
    
    # ------------------------------------------------------------------
    # EditorPanel hooks
    # ------------------------------------------------------------------

    def _initial_size(self):
        return (Theme.UI_EDITOR_INIT_WINDOW_W, Theme.UI_EDITOR_INIT_WINDOW_H)

    def _window_flags(self) -> int:
        return Theme.WINDOW_FLAGS_VIEWPORT | Theme.WINDOW_FLAGS_NO_SCROLL

    def _pre_render(self, ctx):
        self._load_resolution_settings()

    def on_disable(self):
        self._set_game_render_active(False)

    def _on_not_visible(self, ctx):
        self._set_game_render_active(False)

    def _on_visible_pre(self, ctx):
        focused = ctx.is_window_focused(0)
        if focused and not self._was_focused:
            if self._on_focus_gained:
                self._on_focus_gained()
        self._was_focused = focused

    def on_render_content(self, ctx: InfGUIContext):
        if not self._engine:
            ctx.label(t("game_view.engine_not_init"))
            return

        # Clear per-frame rect cache once at the start of each game_view frame.
        clear_rect_cache(id(ctx))

        # Keep native Game rendering active only while this panel is actually visible.
        self._set_game_render_active(True)

        # ── Resolution toolbar row ──
        old_idx = self._selected_resolution_idx
        ctx.set_next_item_width(140)
        self._selected_resolution_idx = ctx.combo("##Resolution", self._selected_resolution_idx, self._PRESET_NAMES, -1)
        if self._selected_resolution_idx != old_idx:
            self._save_resolution_settings()

        if self._selected_resolution_idx == len(self._RESOLUTION_PRESETS) - 1:
            ctx.same_line(0, 8)
            w_old = self._custom_width
            h_old = self._custom_height
            ctx.set_next_item_width(56)
            self._custom_width = int(ctx.drag_int("##CW", self._custom_width, 1.0, 64, 8192))
            ctx.same_line(0, 2)
            ctx.label(Theme.ICON_REMOVE)
            ctx.same_line(0, 2)
            ctx.set_next_item_width(56)
            self._custom_height = int(ctx.drag_int("##CH", self._custom_height, 1.0, 64, 8192))
            if self._custom_width != w_old or self._custom_height != h_old:
                self._save_resolution_settings()

        # ── Scale slider row ──
        avail_width = ctx.get_content_region_avail_width()
        avail_height = ctx.get_content_region_avail_height()
        target_w, target_h = self._current_target_resolution()

        fit_scale = 1.0
        if target_w > 0 and target_h > 0 and avail_width > 0 and avail_height > 0:
            fit_scale = min(avail_width / float(target_w), avail_height / float(target_h))
            fit_scale = max(0.01, fit_scale)

        if self._fit_mode:
            self._display_scale = fit_scale

        ctx.same_line(0, 12)
        pct = int(round(self._display_scale * 100))
        ctx.label(f"{pct}%")
        ctx.same_line(0, 4)
        ctx.set_next_item_width(100)
        old_scale = self._display_scale
        self._display_scale = ctx.float_slider("##Scale", self._display_scale, 0.10, 2.0)
        self._display_scale = round(self._display_scale, 3)
        if abs(old_scale - self._display_scale) > 0.001:
            self._fit_mode = False
            self._save_resolution_settings()
        ctx.same_line(0, 4)
        pushed_fit_style = self._fit_mode
        if pushed_fit_style:
            ctx.push_style_color(ImGuiCol.Button, *Theme.PLAY_ACTIVE)
        ctx.button(t("game_view.fit"), self._fit_scale, width=32, height=0)
        if pushed_fit_style:
            ctx.pop_style_color(1)

        # ── FPS counter (right-aligned) ──
        dt = Time.unscaled_delta_time
        if dt > 0.0:
            self._fps_accum_time += dt
            self._fps_accum_frames += 1
            if self._fps_accum_time >= 1.0:
                self._display_fps = self._fps_accum_frames / self._fps_accum_time
                self._display_frame_ms = (self._fps_accum_time / self._fps_accum_frames) * 1000.0
                self._fps_accum_time = 0.0
                self._fps_accum_frames = 0

        fps_text = f"FPS: {self._display_fps:.0f} ({self._display_frame_ms:.1f} ms)"
        if fps_text != getattr(self, '_cached_fps_text', None):
            self._cached_fps_text = fps_text
            self._cached_fps_text_w, _ = ctx.calc_text_size(fps_text)
        text_w = self._cached_fps_text_w
        fps_x = max(ctx.get_window_width() - text_w - 24.0, 360.0)
        ctx.same_line(fps_x)
        if ctx.get_content_region_avail_width() >= text_w:
            ctx.label(fps_text)

        draw_w = float(target_w) * self._display_scale
        draw_h = float(target_h) * self._display_scale

        if target_w != self._last_game_width or target_h != self._last_game_height:
            self._engine.resize_game_render_target(target_w, target_h)
            self._last_game_width = target_w
            self._last_game_height = target_h

        game_texture_id = self._engine.get_game_texture_id()

        # Pre-fetch scene + canvases once (used by both render and events)
        from InfEngine.lib import SceneManager as _SM
        _scene = _SM.instance().get_active_scene()
        _canvases = collect_sorted_canvases(_scene) if _scene is not None else []

        cursor_start_x = ctx.get_cursor_pos_x()
        cursor_start_y = ctx.get_cursor_pos_y()

        if game_texture_id != 0:
            pad_x = max(0.0, (avail_width - draw_w) * 0.5)
            pad_y = max(0.0, (avail_height - draw_h) * 0.5)
            ctx.set_cursor_pos_x(cursor_start_x + pad_x)
            ctx.set_cursor_pos_y(cursor_start_y + pad_y)
            ctx.image(game_texture_id, float(draw_w), float(draw_h), 0.0, 0.0, 1.0, 1.0)

            vp = capture_viewport_info(ctx)
            Input.set_game_viewport_origin(vp.image_min_x, vp.image_min_y)

            self._render_screen_ui(ctx, vp.image_min_x, vp.image_min_y,
                                   float(draw_w), float(draw_h),
                                   vp.image_min_x, vp.image_min_y,
                                   vp.image_min_x + float(draw_w),
                                   vp.image_min_y + float(draw_h),
                                   scene=_scene, canvases=_canvases)

        else:
            ctx.label("")
            ctx.label("  " + t("game_view.no_camera"))
            ctx.label("  " + t("game_view.no_camera_detail"))
            ctx.label("")
            ctx.label("  " + t("game_view.create_camera_hint_1"))
            ctx.label("  " + t("game_view.create_camera_hint_2"))

        game_hovered = ctx.is_window_hovered()
        is_playing = self._is_playing()

        # Cursor lock is script-driven (Input.set_cursor_locked).
        # Editor provides ESC as a safety unlock.
        cursor_locked = Input.is_cursor_locked()
        if cursor_locked:
            if Input.get_key_down(KeyCode.ESCAPE):
                Input.set_cursor_locked(False)
                cursor_locked = False

        Input.set_game_focused(
            (game_hovered and is_playing) or cursor_locked
        )

        if not is_playing and cursor_locked:
            Input.set_cursor_locked(False)

        if is_playing and game_hovered:
            self._process_ui_events(target_w, target_h, canvases=_canvases)
        elif not is_playing:
            self._ui_event_processor.reset()

    # ------------------------------------------------------------------
    # Screen-space UI overlay
    # ------------------------------------------------------------------

    def _render_screen_ui(self, ctx: InfGUIContext, vp_x: float, vp_y: float,
                          vp_w: float, vp_h: float,
                          clip_min_x: float = 0.0, clip_min_y: float = 0.0,
                          clip_max_x: float = 1e9, clip_max_y: float = 1e9,
                          scene=None, canvases=None):
        """Push screen-space UI commands to the GPU ScreenUI renderer.

        Commands are accumulated during BuildFrame and rendered inside the
        scene render graph as proper Vulkan passes:
        - CameraOverlay elements go to the Camera list (before post-process)
        - ScreenOverlay elements go to the Overlay list (after post-process)

        When the renderer is disabled (e.g. UI editor using the game texture
        as a clean background), falls back to ImGui overlay drawing so the
        Game panel still shows canvas UI on top of the game image.
        """
        from InfEngine.lib import ScreenUIList
        from InfEngine.ui.enums import RenderMode

        if not self._engine:
            return

        renderer = self._engine.get_screen_ui_renderer()
        if renderer is None:
            return

        if scene is None:
            return

        game_w = self._last_game_width
        game_h = self._last_game_height
        if game_w < 1 or game_h < 1:
            return

        use_overlay = not renderer.is_enabled()

        # Always call begin_frame to clear old GPU commands, even when
        # using the overlay path (prevents stale commands from rendering
        # if the renderer is re-enabled later).
        renderer.begin_frame(game_w, game_h)

        if canvases is None:
            canvases = collect_sorted_canvases(scene)

        if not canvases:
            return

        _get_tid = _get_tex_cache().get_bound(self._engine)

        for canvas in canvases:
            # Map RenderMode to ScreenUIList
            if canvas.render_mode == RenderMode.CameraOverlay:
                ui_list = ScreenUIList.Camera
            elif canvas.render_mode == RenderMode.ScreenOverlay:
                ui_list = ScreenUIList.Overlay
            else:
                continue

            ref_w = float(canvas.reference_width)
            ref_h = float(canvas.reference_height)
            if ref_w < 1 or ref_h < 1:
                continue
            # Scale from design resolution to actual game resolution
            scale_x = float(game_w) / ref_w
            scale_y = float(game_h) / ref_h
            text_scale = min(scale_x, scale_y)

            for elem in canvas._get_elements():
                ex, ey, ew, eh = elem.get_rect(ref_w, ref_h)

                if use_overlay:
                    # ImGui overlay: map design coords → screen coords
                    ovl_scale_x = vp_w / ref_w
                    ovl_scale_y = vp_h / ref_h
                    base_sx = vp_x + ex * ovl_scale_x
                    base_sy = vp_y + ey * ovl_scale_y
                    base_sw = ew * ovl_scale_x
                    base_sh = eh * ovl_scale_y
                    ovl_zoom = min(ovl_scale_x, ovl_scale_y)
                    _ui_dispatch(
                        elem, "editor",
                        ctx=ctx,
                        base_sx=base_sx, base_sy=base_sy,
                        base_sw=base_sw, base_sh=base_sh,
                        zoom=ovl_zoom,
                        get_tex_id=_get_tid,
                    )
                else:
                    sx = ex * scale_x
                    sy = ey * scale_y
                    sw = ew * scale_x
                    sh = eh * scale_y
                    _ui_dispatch(
                        elem, "runtime",
                        renderer=renderer,
                        ui_list=ui_list,
                        sx=sx, sy=sy, sw=sw, sh=sh,
                        ref_w=ref_w, ref_h=ref_h,
                        scale_x=scale_x, scale_y=scale_y,
                        text_scale=text_scale,
                        get_tex_id=_get_tid,
                    )

    # ------------------------------------------------------------------
    # UI event processing
    # ------------------------------------------------------------------

    def _process_ui_events(self, game_w: int, game_h: int, canvases=None):
        """Convert Input mouse state to per-canvas pointer events."""
        if canvases is None:
            from InfEngine.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if scene is None:
                return
            canvases = collect_sorted_canvases(scene)
        if not canvases:
            return

        # Mouse position in viewport pixels (relative to game image top-left)
        vp_x, vp_y, scroll_x, scroll_y, mouse_held, mouse_down, mouse_up = Input.get_game_mouse_frame_state(0)
        display_scale = self._display_scale
        if display_scale < 1e-6:
            return

        # Convert viewport pixels → game-resolution pixels
        game_px = vp_x / display_scale
        game_py = vp_y / display_scale

        # Build per-canvas positions in design (canvas) pixels
        canvas_positions = []
        for canvas in canvases:
            ref_w = float(canvas.reference_width)
            ref_h = float(canvas.reference_height)
            if ref_w < 1 or ref_h < 1:
                canvas_positions.append((0.0, 0.0))
                continue
            cx = game_px * ref_w / float(game_w)
            cy = game_py * ref_h / float(game_h)
            canvas_positions.append((cx, cy))

        scroll = (scroll_x, scroll_y)

        from InfEngine.timing import Time
        dt = Time.unscaled_delta_time

        self._ui_event_processor.process(
            canvases, canvas_positions,
            mouse_down, mouse_up, mouse_held,
            scroll, dt,
        )