"""
PlayerGUI — fullscreen-borderless ImGui GUI for standalone game playback.

Registered as a single InxGUIRenderable that fills the entire window with
the game camera render target.  No editor chrome, no docking, no menus.

Optionally shows a **splash sequence** before revealing the game.  The
scene may finish loading while the window is black or showing splash;
Play starts only after that loading is done and the splash (if any) has
finished.  The game must not already be running when the player first
sees it.
"""

from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, List, Optional

from Infernux.debug import Debug
from Infernux.lib import InxGUIRenderable, InxGUIContext
from Infernux.input import Input, KeyCode, TouchPhase
from Infernux.engine.ui.viewport_utils import capture_viewport_info
from Infernux.ui.ui_event_data import PointerType
from Infernux.ui.ui_event_system import UIEventProcessor, UIPointerFrame
from Infernux.engine.runtime_screen_ui import (
    collect_runtime_ui_input_surfaces,
    map_runtime_ui_pointer,
    map_runtime_ui_pointers,
)
from Infernux.engine.runtime_mouse_events import MouseEventDispatcher


def _player_render_scale() -> float:
    try:
        scale = float(os.environ.get("INFERNUX_PLAYER_RENDER_SCALE", "1"))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(scale):
        return 1.0
    return max(0.25, min(1.0, scale))


class PlayerGUI(InxGUIRenderable):
    """Renders the game camera output fullscreen with screen-space UI overlay."""

    def __init__(self, engine, *,
                 splash_items: Optional[List[Dict]] = None,
                 data_root: str = "",
                 control_channel=None,
                 activate_play: Optional[Callable[[], bool]] = None):
        super().__init__()
        self._engine = engine
        self._last_w = 0
        self._last_h = 0
        self._render_scale = _player_render_scale()
        self._ui_event_processor = UIEventProcessor()
        self._mouse_event_dispatcher = MouseEventDispatcher()
        self._has_shared_scene_query = False
        self._last_frame_time = time.time()
        self._control = control_channel
        self._activate_play = activate_play
        self._play_started = False
        self._play_start_failed = False
        self._profile_frames = os.environ.get(
            "_INFERNUX_PLAYER_PROFILE_FRAMES", ""
        ).strip() == "1"
        self._next_profile_time = time.monotonic() + 2.0

        # Splash
        self._splash = None
        if splash_items:
            from Infernux.engine.splash_player import SplashPlayer
            self._splash = SplashPlayer(splash_items, data_root)

    # ------------------------------------------------------------------
    # InxGUIRenderable interface
    # ------------------------------------------------------------------

    def on_render(self, ctx: InxGUIContext):
        self._tick(ctx)

        # Full main viewport
        x0, y0, vp_w, vp_h = ctx.get_main_viewport_bounds()
        ctx.set_next_window_pos(x0, y0, 0, 0.0, 0.0)
        ctx.set_next_window_size(vp_w, vp_h, 0)

        # ImGui flags: NoTitleBar|NoResize|NoMove|NoScrollbar|NoCollapse
        #              |NoSavedSettings|NoNavInputs|NoNavFocus|NoDocking
        flags = (
            (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5)
            | (1 << 8) | (1 << 16) | (1 << 17) | (1 << 19)
        )
        ctx.push_style_var_vec2(2, 0.0, 0.0)   # WindowPadding = (0,0)
        ctx.push_style_var_float(4, 0.0)        # WindowBorderSize = 0

        # ── Splash mode ───────────────────────────────────────────────
        if self._splash and not self._splash.is_finished:
            # Black background during splash
            ctx.push_style_color(2, 0.0, 0.0, 0.0, 1.0)  # ImGuiCol_WindowBg
            visible = ctx.begin_window("##PlayerFullscreen", True, flags)
            if visible:
                native = self._engine.get_native_engine()
                if native:
                    self._splash.update(ctx, native, x0, y0, vp_w, vp_h)
            ctx.end_window()
            ctx.pop_style_color(1)
            ctx.pop_style_var(2)

            if self._splash.is_finished:
                native = self._engine.get_native_engine()
                if native:
                    self._splash.cleanup(native)
                self._splash = None
            return

        # ── Normal game mode ──────────────────────────────────────────
        # Component start() may create Game-relative RenderTextures. Publish
        # the actual output resolution before activating project lifecycle.
        self._prepare_game_target(vp_w, vp_h)
        self.begin_play_when_ready()
        visible = ctx.begin_window("##PlayerFullscreen", True, flags)
        if visible:
            self._render_game(ctx, vp_w, vp_h)
        ctx.end_window()
        ctx.pop_style_var(2)  # WindowPadding + WindowBorderSize

    # ------------------------------------------------------------------

    def begin_play_when_ready(self) -> bool:
        """Start Play only after loading is done and splash (if any) has finished."""
        if self._play_started:
            return True
        if self._play_start_failed:
            return False
        if self._splash is not None and not self._splash.is_finished:
            return False
        activate = self._activate_play
        if activate is None:
            getter = getattr(self._engine, "get_player_runtime", None)
            session = getter() if callable(getter) else None
            if session is None:
                self._play_start_failed = True
                Debug.log_error(
                    "Player cannot start Play: runtime session is unavailable"
                )
                return False
            if getattr(session, "is_playing", False):
                self._play_started = True
                return True
            activate = getattr(session, "activate", None)
        if not callable(activate) or not activate():
            self._play_start_failed = True
            Debug.log_error("Player cannot start Play: initial scene is not ready")
            return False
        self._play_started = True
        return True

    def _tick(self, ctx):
        """Handle Player window input and debug-control polling.

        Play timing is owned by ``Engine.tick_play_mode``. This must not start
        the session before an optional project splash has finished.
        """
        # Standalone Players have no competing Editor viewport.  Establish the
        # gameplay-input contract before any early return caused by splash,
        # camera startup, a missing GUI texture, or background rendering.
        Input.set_game_focused(True)

        if self._engine:
            if (
                self._control is not None
                and self._control.poll(self._engine) == "shutdown"
            ):
                self._engine.request_exit()
                return

            # In player mode there's no MenuBarPanel, so we must handle
            # close requests (Alt+F4 / window X) directly.
            native = self._engine.get_native_engine()
            if native and native.is_close_requested():
                native.confirm_close()
                return

            if self._profile_frames and time.monotonic() >= self._next_profile_time:
                self._next_profile_time = time.monotonic() + 2.0
                try:
                    from Infernux.engine.player_bootstrap import _plog

                    snapshot = dict(native.renderer_frame_snapshot) if native else {}
                    _plog(f"[FrameProfile] {snapshot}")
                except Exception as exc:
                    Debug.log_suppressed("player_gui.frame_profile", exc)

    def _prepare_game_target(self, vp_w: float, vp_h: float):
        display_w = max(1, int(vp_w))
        display_h = max(1, int(vp_h))
        target_w = max(1, int(display_w * self._render_scale))
        target_h = max(1, int(display_h * self._render_scale))

        if target_w != self._last_w or target_h != self._last_h:
            self._engine.resize_game_render_target(target_w, target_h)
            self._last_w = target_w
            self._last_h = target_h

    def _render_game(self, ctx: InxGUIContext, vp_w: float, vp_h: float):
        display_w = max(1, int(vp_w))
        display_h = max(1, int(vp_h))
        self._prepare_game_target(vp_w, vp_h)
        target_w, target_h = self._last_w, self._last_h

        game_tex = self._engine.get_game_texture_id()
        if game_tex == 0:
            ctx.label("Waiting for camera...")
            return

        ctx.image(game_tex, float(display_w), float(display_h), 0.0, 0.0, 1.0, 1.0)
        vp = capture_viewport_info(ctx)
        Input.set_game_viewport_origin(vp.image_min_x, vp.image_min_y)
        Input.set_game_viewport_size(float(display_w), float(display_h))

        # ESC safety: allow user to unlock cursor even if scripts forgot
        cursor_locked = Input.is_cursor_locked()
        if cursor_locked:
            if Input.get_key_down(KeyCode.ESCAPE):
                Input.set_cursor_locked(False)
                cursor_locked = False

        # The standalone Player owns the entire window. Touchscreen contacts
        # do not define an ImGui mouse-hover state, so UI dispatch must not be
        # gated by the desktop hover bit.
        mouse_frame = Input.get_game_mouse_frame_state(0)
        scene_hit = self._process_ui_events(display_w, display_h, mouse_frame=mouse_frame)
        self._process_mouse_events(display_w, display_h, scene_hit=scene_hit, mouse_frame=mouse_frame)

    def _process_mouse_events(self, game_w: int, game_h: int, *, scene_hit=None, mouse_frame=None) -> None:
        dispatcher = getattr(self, "_mouse_event_dispatcher", None)
        if dispatcher is None:
            return
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        camera = scene.effective_game_camera if scene is not None else None
        if camera is None:
            dispatcher.reset()
            return
        x, y, _sx, _sy, held, down, up = Input.get_game_mouse_frame_state(0) if mouse_frame is None else mouse_frame
        if scene_hit is None and not self._has_shared_scene_query:
            dispatcher.process(camera, (x, y), (float(game_w), float(game_h)), button_state=(held, down, up))
        else:
            dispatcher.process(camera, (x, y), (float(game_w), float(game_h)), hit=scene_hit,
                               button_state=(held, down, up))

    def _process_ui_events(self, game_w: int, game_h: int, *, mouse_frame=None):
        """Convert mouse and every active touch to independent UI pointers."""
        from Infernux.lib import SceneManager

        scene = SceneManager.instance().get_active_scene()
        if scene is None:
            self._has_shared_scene_query = False
            return None

        persistent_scene = SceneManager.instance().get_runtime_persistent_scene()
        surfaces = collect_runtime_ui_input_surfaces(scene, persistent_scene)
        if not surfaces:
            self._ui_event_processor.reset()
            # With no world UI there is no second consumer to share a ray
            # with; the dispatcher keeps its direct closest-hit path.
            self._has_shared_scene_query = False
            return None

        camera = scene.effective_game_camera

        gx, gy, scroll_x, scroll_y, mouse_held, mouse_down, mouse_up = (
            Input.get_game_mouse_frame_state(0) if mouse_frame is None else mouse_frame
        )

        mouse_positions, scene_hit = map_runtime_ui_pointer(
            surfaces, camera, gx, gy, game_w, game_h,
            include_scene_hit=True,
        )
        self._has_shared_scene_query = True

        pointers = [
            UIPointerFrame(
                pointer_id=-1,
                pointer_type=PointerType.Mouse,
                canvas_positions=mouse_positions,
                down=mouse_down,
                up=mouse_up,
                held=mouse_held,
                scroll_delta=(scroll_x, scroll_y),
            )
        ]
        touches = tuple(Input.touches)
        touch_points = tuple(
            (
                float(touch.normalized_position[0]) * float(game_w),
                (1.0 - float(touch.normalized_position[1])) * float(game_h),
            )
            for touch in touches
        )
        touch_positions = map_runtime_ui_pointers(
            surfaces, camera, touch_points, game_w, game_h
        )
        same_frame_terminal = tuple(
            touch.began_this_frame and touch.phase in (TouchPhase.ENDED, TouchPhase.CANCELED)
            for touch in touches
        )
        begin_positions = ()
        if any(same_frame_terminal):
            begin_points = tuple(
                (
                    float(touch.begin_normalized_position[0]) * float(game_w),
                    (1.0 - float(touch.begin_normalized_position[1])) * float(game_h),
                )
                for touch in touches
            )
            begin_positions = map_runtime_ui_pointers(
                surfaces, camera, begin_points, game_w, game_h
            )
        for index, (touch, positions) in enumerate(zip(touches, touch_positions)):
            pointers.append(
                UIPointerFrame(
                    pointer_id=int(touch.finger_id),
                    pointer_type=PointerType.Touch,
                    canvas_positions=positions,
                    down=touch.phase is TouchPhase.BEGAN or same_frame_terminal[index],
                    up=touch.phase in (TouchPhase.ENDED, TouchPhase.CANCELED),
                    held=touch.phase
                    in (
                        TouchPhase.BEGAN,
                        TouchPhase.MOVED,
                        TouchPhase.STATIONARY,
                    ),
                    canceled=touch.phase is TouchPhase.CANCELED,
                    press_canvas_positions=(begin_positions[index] if same_frame_terminal[index] else ()),
                )
            )

        from Infernux.timing import Time
        dt = Time.unscaled_delta_time

        self._ui_event_processor.process_pointers(surfaces, pointers, dt)
        return scene_hit
