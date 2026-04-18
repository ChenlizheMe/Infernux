"""
EditorBootstrap — structured editor initialization.

Breaks the monolithic ``release_engine()`` startup path into explicit
startup steps. Each step is a separate method, closures become instance
methods, and panel/manager references live on the bootstrap instance.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional

from Infernux.debug import Debug
from Infernux.lib import TagLayerManager
import Infernux.resources as _resources
from Infernux.engine.engine import Engine, LogLevel
from Infernux.engine.resources_manager import ResourcesManager
from Infernux.engine.play_mode import PlayModeManager, PlayModeState
from Infernux.engine.scene_manager import SceneFileManager
from Infernux.engine.ui import (
    FrameSchedulerPanel,
    SceneViewPanel,
    GameViewPanel,
    WindowManager,
    TagLayerSettingsPanel,
    BuildSettingsPanel,
    UIEditorPanel,
    EditorPanel,
    EditorServices,
    EditorEventBus,
    EditorEvent,
    PanelRegistry,
    editor_panel,
)
from Infernux.engine.ui import panel_state as _panel_state

_log = logging.getLogger("Infernux.bootstrap")

_LAYOUT_VERSION = 5
_TOTAL_STEPS = 12


def _signal_progress(current_step: int, total: int, message: str) -> None:
    """Write bootstrap progress to the launcher splash via the ready-file."""
    ready_file = os.environ.get("_INFERNUX_READY_FILE", "").strip()
    if not ready_file:
        return
    try:
        with open(ready_file, "w", encoding="utf-8") as f:
            f.write(f"LOADING:{current_step}/{total}:{message}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass

from ._bootstrap_panels import BootstrapPanelsMixin
from ._bootstrap_selection import BootstrapSelectionMixin
from ._bootstrap_wiring import BootstrapWiringMixin


class EditorBootstrap(BootstrapPanelsMixin, BootstrapSelectionMixin, BootstrapWiringMixin):
    """Orchestrates the full editor startup sequence."""

    _instance: Optional["EditorBootstrap"] = None

    def __init__(self, project_path: str, engine_log_level=LogLevel.Info):
        EditorBootstrap._instance = self
        self.project_path = project_path
        self.engine_log_level = engine_log_level

        # Managers
        self.engine: Optional[Engine] = None
        self.undo_manager = None
        self.scene_file_manager: Optional[SceneFileManager] = None
        self.window_manager: Optional[WindowManager] = None
        self.services: Optional[EditorServices] = None
        self.event_bus: Optional[EditorEventBus] = None

        # Panels
        self.frame_scheduler = None
        self.menu_bar = None
        self.toolbar = None
        self.hierarchy = None
        self.inspector_panel = None  # C++ InspectorPanel (native)
        self.project_panel = None
        self.console = None  # C++ ConsolePanel (native)
        self.status_bar = None
        self.scene_view: Optional[SceneViewPanel] = None
        self.game_view: Optional[GameViewPanel] = None
        self.ui_editor: Optional[UIEditorPanel] = None

        # Selection state
        self._prev_selection = [0]  # kept for scene-change cleanup
        self._prev_selection_ids: list = []  # for undo recording
        self._prev_selected_file: str = ""

        # Progress tracking for launcher splash
        self._progress_step = 0

    @classmethod
    def instance(cls) -> Optional["EditorBootstrap"]:
        return cls._instance

    # ── Public entry point ─────────────────────────────────────────────

    def run(self):
        """Execute all bootstrap steps and start the main loop."""
        self._report_progress("Checking project requirements\u2026")
        self._ensure_project_requirements()

        self._report_progress("Initializing renderer\u2026")
        self._init_engine()

        self._report_progress("Loading tag/layer settings\u2026")
        self._load_tag_layer_settings()

        self._report_progress("Creating managers\u2026")
        self._create_managers()

        self._report_progress("Loading layout\u2026")
        self._setup_layout_persistence()

        self._report_progress("Registering window types\u2026")
        self._register_window_types()

        self._report_progress("Creating editor panels\u2026")
        self._create_panels()

        self._report_progress("Wiring selection system\u2026")
        self._wire_selection_system()

        self._report_progress("Setting up UI editor\u2026")
        self._wire_ui_editor()

        self._report_progress("Preparing scene system\u2026")
        self._setup_scene_change_cleanup()

        self._report_progress("Loading scene\u2026")
        self._load_initial_scene()

    def _report_progress(self, message: str):
        """Notify the launcher splash of the current bootstrap step."""
        self._progress_step += 1
        _signal_progress(self._progress_step, _TOTAL_STEPS, message)


    def _ensure_project_requirements(self):
        from Infernux.engine.project_requirements import ensure_project_requirements

        ensure_project_requirements(self.project_path, auto_install=True)

    def _init_engine(self):
        self.engine = Engine(self.engine_log_level)
        self.engine.init_renderer(
            width=1600, height=900, project_path=self.project_path
        )
        self.engine.set_gui_font(_resources.engine_font_path, 15)

    def _load_tag_layer_settings(self):
        path = os.path.join(self.project_path, "ProjectSettings", "TagLayerSettings.json")
        if os.path.isfile(path):
            TagLayerManager.instance().load_from_file(path)

    def _create_managers(self):
        from Infernux.engine.undo import UndoManager

        self.undo_manager = UndoManager()

        self.scene_file_manager = SceneFileManager()
        self.scene_file_manager.set_asset_database(self.engine.get_asset_database())
        self.scene_file_manager.set_engine(self.engine.get_native_engine())

        self.window_manager = WindowManager(self.engine)

        self.services = EditorServices()
        self.services._engine = self.engine
        self.services._undo_manager = self.undo_manager
        self.services._scene_file_manager = self.scene_file_manager
        self.services._play_mode_manager = self.engine._play_mode_manager
        self.services._window_manager = self.window_manager
        self.services._asset_database = self.engine.get_asset_database()
        self.services._project_path = self.project_path

        self.event_bus = EditorEventBus()

        # Inject serialized-field callbacks (breaks circular dep chain)
        self._inject_field_change_hooks()

    def _inject_field_change_hooks(self):
        """Wire serialized-field undo/dirty callbacks without circular imports."""
        from Infernux.engine.undo import UndoManager, SetPropertyCommand
        from Infernux.engine.play_mode import PlayModeManager, PlayModeState

        def will_change(instance, field_name, old_value, new_value):
            mgr = UndoManager.instance()
            if (mgr and not mgr.is_executing and mgr.enabled
                    and hasattr(instance, 'game_object')
                    and instance.game_object is not None):
                pmm = PlayModeManager.instance()
                if pmm is None or pmm.is_edit_mode:
                    mgr.execute(SetPropertyCommand(
                        instance, field_name,
                        old_value, new_value, f"Set {field_name}"))
                    return True
            return False

        def did_change(instance, field_name, old_value, new_value):
            # Scene dirty state is managed exclusively by UndoManager._sync_dirty().
            # No direct mark_dirty() call needed here — every edited property
            # either went through will_change (undo-recorded → _sync_dirty)
            # or is a play-mode / undo-driven write that shouldn't dirty.
            pass

        from Infernux.components.serialized_field import set_field_change_hooks
        set_field_change_hooks(will_change=will_change, did_change=did_change)


    def _wire_toolbar_callbacks_on(self, tb, engine):
        """Shared helper: attach play/camera/grid callbacks to a ToolbarPanel."""
        pmm = engine._play_mode_manager if engine else None
        from Infernux.lib import PlayState
        from Infernux.engine.play_mode import PlayModeState
        from Infernux.engine.ui.closable_panel import ClosablePanel

        def _on_play():
            if not pmm:
                return
            if pmm.is_playing:
                pmm.exit_play_mode()
            else:
                if pmm.enter_play_mode():
                    ClosablePanel.focus_panel_by_id("game_view")
                    if engine:
                        engine.select_docked_window("game_view")
        def _on_pause():
            if pmm:
                pmm.toggle_pause()
        def _on_step():
            if pmm:
                pmm.step_frame()
        def _get_play_state():
            if not pmm:
                return PlayState.Edit
            state = pmm.state
            if state == PlayModeState.PLAYING:
                return PlayState.Playing
            elif state == PlayModeState.PAUSED:
                return PlayState.Paused
            return PlayState.Edit
        def _get_play_time_str():
            if not pmm:
                return "00:00.000"
            t = pmm.total_play_time
            return f"{int(t//60):02d}:{t%60:06.3f}"

        tb.on_play = _on_play
        tb.on_pause = _on_pause
        tb.on_step = _on_step
        tb.get_play_state = _get_play_state
        tb.get_play_time_str = _get_play_time_str

        native = engine.get_native_engine() if engine else None
        if native:
            tb.is_show_grid = lambda: native.is_show_grid()
            tb.set_show_grid = lambda v: native.set_show_grid(v)

        def _sync_camera():
            cam = engine.editor_camera if engine else None
            if not cam:
                return tb.get_camera_settings()
            return {
                "fov": float(cam.fov),
                "rotation_speed": float(cam.rotation_speed),
                "pan_speed": float(cam.pan_speed),
                "zoom_speed": float(cam.zoom_speed),
                "move_speed": float(cam.move_speed),
                "move_speed_boost": float(cam.move_speed_boost),
            }
        def _apply_camera(settings):
            cam = engine.editor_camera if engine else None
            if not cam:
                return
            cam.fov = settings["fov"]
            cam.rotation_speed = settings["rotation_speed"]
            cam.pan_speed = settings["pan_speed"]
            cam.zoom_speed = settings["zoom_speed"]
            cam.move_speed = settings["move_speed"]
            cam.move_speed_boost = settings["move_speed_boost"]

        tb.sync_camera_from_engine = _sync_camera
        tb.apply_camera_to_engine = _apply_camera


    # ── Native panel callback wiring ───────────────────────────────────

    def _inspector_set_selected_file(self, path):
        """Compute asset category and call C++ SetSelectedFile."""
        if path:
            import os
            from Infernux.core.asset_types import asset_category_from_extension
            ext = os.path.splitext(path)[1].lower()
            cat = asset_category_from_extension(ext) or ""
            self.inspector_panel.set_selected_file(path, cat)
        else:
            self.inspector_panel.clear_selected_file()

    # ── Selection undo helpers ─────────────────────────────────────────

    # Structural command types whose associated selection changes should
    # not be recorded as separate undo entries.
    _STRUCTURAL_CMD_TYPES = None  # lazily populated

    def _setup_scene_change_cleanup(self):
        def on_scene_changed():
            self._prev_selection[0] = 0
            from Infernux.engine.ui.selection_manager import SelectionManager
            SelectionManager.instance().clear()
            self.hierarchy.clear_selection_and_notify()
            self.inspector_panel.set_selected_object_id(0)
            self._set_outline(0)
            self.scene_view._fly_to_active = False
            self.scene_view._fly_to_last_obj_id = 0
            self.scene_view._fly_to_close = False

        self.scene_file_manager.set_on_scene_changed(on_scene_changed)

    def _setup_layout_persistence(self):
        project_name = os.path.basename(self.project_path)

        docs_dir = None
        if os.name == "nt":
            try:
                import ctypes
                import ctypes.wintypes
                buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
                ctypes.windll.shell32.SHGetFolderPathW(None, 0x0005, None, 0, buf)
                if buf.value:
                    docs_dir = pathlib.Path(buf.value)
            except (OSError, ValueError) as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
        if docs_dir is None:
            docs_dir = pathlib.Path.home() / "Documents"

        layout_dir = docs_dir / "Infernux" / project_name
        os.makedirs(layout_dir, exist_ok=True)
        _panel_state.init(str(layout_dir))

        layout_ver_path = str(layout_dir / ".layout_version")
        imgui_ini_path = str(layout_dir / "imgui.ini")
        self.window_manager.set_imgui_ini_path(imgui_ini_path)

        # Clean up old project-local imgui.ini
        old_ini = os.path.join(self.project_path, "imgui.ini")
        if os.path.isfile(old_ini):
            try:
                os.remove(old_ini)
            except OSError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

        need_reset = True
        if os.path.isfile(layout_ver_path):
            try:
                with open(layout_ver_path, "r") as f:
                    if f.read().strip() == str(_LAYOUT_VERSION):
                        need_reset = False
            except OSError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
        if need_reset:
            if os.path.isfile(imgui_ini_path):
                os.remove(imgui_ini_path)
            os.makedirs(os.path.dirname(layout_ver_path), exist_ok=True)
            with open(layout_ver_path, "w") as f:
                f.write(str(_LAYOUT_VERSION))

    def _persist_editor_state(self):
        if self.console is None or self.project_panel is None or self.window_manager is None:
            return
        if self.toolbar is not None:
            _panel_state.put("toolbar", {
                "camera_settings": self.toolbar.get_camera_settings(),
            })
        if self.console is not None:
            _panel_state.put("console", {
                "show_info": self.console.show_info,
                "show_warnings": self.console.show_warnings,
                "show_errors": self.console.show_errors,
                "collapse": self.console.collapse,
                "clear_on_play": self.console.clear_on_play,
                "error_pause": self.console.error_pause,
                "auto_scroll": self.console.auto_scroll,
            })
        _panel_state.put("project", {"current_path": self.project_panel.get_current_path()})
        _panel_state.put("window_manager", self.window_manager.save_state())

        # Persist individual panel states for every window id we still track
        # (singletons live in _default_instances; dynamically opened ids may only
        # appear in _window_instances until closed — those must still save).
        wm = self.window_manager
        seen_ids: set[str] = set()
        for wid in set(wm._default_instances.keys()) | set(wm._window_instances.keys()):
            if wid in seen_ids:
                continue
            seen_ids.add(wid)
            inst = wm._window_instances.get(wid) or wm._default_instances.get(wid)
            if inst is None:
                continue
            if hasattr(inst, "save_state") and callable(inst.save_state):
                try:
                    data = inst.save_state()
                    if data:
                        _panel_state.put(f"panel:{wid}", data)
                except Exception:
                    pass

        _panel_state.save()

    def _load_initial_scene(self):
        import Infernux.renderstack  # noqa: F401 — ensure RenderStack is discoverable
        self.scene_file_manager.load_last_scene_or_default()