"""
EditorBootstrap — structured editor initialization.

Replaces the monolithic ``release_engine()`` god-function with organized
lifecycle phases.  Each phase is a separate method, closures become
instance methods, and all panel/manager references are instance attributes.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional

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
_TOTAL_PHASES = 13


def _signal_progress(phase: int, total: int, message: str) -> None:
    """Write bootstrap progress to the launcher splash via the ready-file."""
    ready_file = os.environ.get("_INFERNUX_READY_FILE", "").strip()
    if not ready_file:
        return
    try:
        with open(ready_file, "w", encoding="utf-8") as f:
            f.write(f"LOADING:{phase}/{total}:{message}\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


class EditorBootstrap:
    """Orchestrates the full editor startup sequence."""

    def __init__(self, project_path: str, engine_log_level=LogLevel.Info):
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
        self._phase = 0

    # ── Public entry point ─────────────────────────────────────────────

    def run(self):
        """Execute all bootstrap phases and start the main loop."""
        self._report_progress("Checking project requirements\u2026")
        self._ensure_project_requirements()

        self._report_progress("Compiling JIT kernels\u2026")
        self._precompile_jit()

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
        """Notify the launcher splash of the current bootstrap phase."""
        self._phase += 1
        _signal_progress(self._phase, _TOTAL_PHASES, message)


    def _ensure_project_requirements(self):
        from Infernux.engine.project_requirements import ensure_project_requirements

        ensure_project_requirements(self.project_path, auto_install=True)

    @staticmethod
    def _precompile_jit():
        from Infernux.jit import precompile_jit

        precompile_jit()


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


    def _register_window_types(self):
        """Register all @editor_panel-decorated panels with WindowManager.

        Panels that require constructor arguments have their factory
        overridden here before apply_all flushes them into the
        WindowManager.
        """
        wm = self.window_manager
        engine = self.engine
        project_path = self.project_path

        from Infernux.lib import InspectorPanel as NativeInspectorPanel

        wm.register_window_type(
            type_id="inspector",
            window_class=NativeInspectorPanel,
            display_name="Inspector",
            factory=self._create_native_inspector,
            singleton=True,
            title_key="panel.inspector",
        )

        # Override factories for panels that need runtime dependencies.
        # Panels with no-arg constructors use the default factory (cls()).
        _factories = {
            "scene_view":         lambda: SceneViewPanel(engine=engine),
            "game_view":          lambda: GameViewPanel(engine=engine),
            "project":            lambda: self._create_native_project_panel(),
            "toolbar":            lambda: self._create_native_toolbar(engine),
            "console":            lambda: self._create_native_console(),
            "hierarchy":          lambda: self._create_native_hierarchy(),
            "tag_layer_settings": lambda: self._create_tag_layer_panel(),
        }
        for reg in PanelRegistry.get_registrations():
            if reg.type_id in _factories:
                reg.factory = _factories[reg.type_id]

        PanelRegistry.apply_all(wm)

    def _create_tag_layer_panel(self):
        panel = TagLayerSettingsPanel()
        panel.set_project_path(self.project_path)
        return panel

    def _create_native_inspector(self):
        """Create a fresh C++ InspectorPanel with all callbacks wired."""
        from Infernux.lib import InspectorPanel as NativeInspectorPanel
        ip = NativeInspectorPanel()
        old = self.inspector_panel
        self.inspector_panel = ip
        self._wire_inspector_callbacks()
        self.inspector_panel = old
        return ip

    def _create_native_project_panel(self):
        """Create a fresh C++ ProjectPanel with all callbacks wired."""
        from Infernux.lib import ProjectPanel as NativeProjectPanel
        pp = NativeProjectPanel()
        # Re-use the same wiring logic; temporarily swap so the method
        # wires the new panel, then restore.
        old = self.project_panel
        self.project_panel = pp
        self._wire_project_callbacks()
        self.project_panel = old
        return pp

    def _create_native_console(self):
        """Create a fresh C++ ConsolePanel for WindowManager re-open."""
        from Infernux.lib import ConsolePanel as NativeConsolePanel
        panel = NativeConsolePanel()
        _project_path = self.project_path
        def _on_dbl(source_file, source_line):
            if not source_file:
                return
            from Infernux.engine.ui import project_utils
            project_utils.open_file_with_system(
                source_file, project_root=_project_path)
        panel.on_double_click_entry = _on_dbl
        return panel

    def _create_native_hierarchy(self):
        """Create a fresh C++ HierarchyPanel with all callbacks wired."""
        from Infernux.lib import HierarchyPanel as NativeHierarchyPanel
        hp = NativeHierarchyPanel()
        old = self.hierarchy
        self.hierarchy = hp
        self._wire_hierarchy_callbacks()
        self.hierarchy = old
        return hp

    def _create_native_toolbar(self, engine):
        """Create a fresh C++ ToolbarPanel with all callbacks wired."""
        from Infernux.lib import ToolbarPanel as NativeToolbarPanel
        from Infernux.engine.i18n import t as _t
        tb = NativeToolbarPanel()
        tb.translate = _t
        self._wire_toolbar_callbacks_on(tb, engine)
        return tb

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


    def _create_panels(self):
        engine = self.engine
        wm = self.window_manager

        # Per-frame scheduler
        self.frame_scheduler = FrameSchedulerPanel(engine=engine)
        engine.register_gui("frame_scheduler", self.frame_scheduler)

        # Menu bar (C++ native panel — replaces Python MenuBarPanel)
        from Infernux.lib import MenuBarPanel as NativeMenuBarPanel
        from Infernux.engine.i18n import t as _t
        self.menu_bar = NativeMenuBarPanel()
        self.menu_bar.translate = _t
        self._wire_menu_bar_callbacks(wm)
        engine.register_gui("menu_bar", self.menu_bar)

        # Toolbar (C++ native panel — replaces Python ToolbarPanel)
        from Infernux.lib import ToolbarPanel as NativeToolbarPanel, PlayState
        self.toolbar = NativeToolbarPanel()
        self.toolbar.translate = _t
        self._wire_toolbar_callbacks(engine)
        engine.register_gui("toolbar", self.toolbar)
        wm.register_existing_window("toolbar", self.toolbar, "toolbar")

        ts = _panel_state.get("toolbar")
        if ts:
            cam_settings = ts.get("camera_settings")
            if cam_settings:
                self.toolbar.set_camera_settings(cam_settings)

        # Hierarchy (C++ native panel — replaces Python HierarchyPanel)
        from Infernux.lib import HierarchyPanel as NativeHierarchyPanel
        self.hierarchy = NativeHierarchyPanel()
        self._wire_hierarchy_callbacks()
        engine.register_gui("hierarchy", self.hierarchy)
        wm.register_existing_window("hierarchy", self.hierarchy, "hierarchy")

        # Inspector (C++ native panel — replaces Python InspectorPanel)
        from Infernux.lib import InspectorPanel as NativeInspectorPanel
        self.inspector_panel = NativeInspectorPanel()
        self._wire_inspector_callbacks()
        engine.register_gui("inspector", self.inspector_panel)
        wm.register_existing_window("inspector", self.inspector_panel, "inspector")

        # Project (C++ native panel — replaces Python ProjectPanel)
        from Infernux.lib import ProjectPanel as NativeProjectPanel
        self.project_panel = NativeProjectPanel()
        self._wire_project_callbacks()
        engine.register_gui("project", self.project_panel)
        wm.register_existing_window("project", self.project_panel, "project")

        ps = _panel_state.get("project")
        if ps:
            path = ps.get("current_path", "")
            if path:
                self.project_panel.set_current_path(path)

        # Console (C++ native panel — replaces Python ConsolePanel)
        from Infernux.lib import ConsolePanel as NativeConsolePanel
        from Infernux.debug import DebugConsole
        self.console = NativeConsolePanel()
        # Bridge Python Debug.log() → C++ ConsolePanel
        DebugConsole.instance().set_native_console(self.console)
        engine.register_gui("console", self.console)
        wm.register_existing_window("console", self.console, "console")

        cs = _panel_state.get("console")
        if cs:
            self.console.show_info = cs.get("show_info", True)
            self.console.show_warnings = cs.get("show_warnings", True)
            self.console.show_errors = cs.get("show_errors", True)
            self.console.collapse = cs.get("collapse", False)
            self.console.clear_on_play = cs.get("clear_on_play", True)
            self.console.error_pause = cs.get("error_pause", False)
            self.console.auto_scroll = cs.get("auto_scroll", True)

        # Wire play-mode clear-on-play
        if engine._play_mode_manager is not None:
            _native_console = self.console
            def _on_play_clear(event):
                from Infernux.engine.play_mode import PlayModeState
                if event.new_state == PlayModeState.PLAYING and _native_console.clear_on_play:
                    _native_console.clear()
            engine._play_mode_manager.add_state_change_listener(_on_play_clear)

        # Wire console double-click → open source file
        _console_project_path = self.project_path
        def _on_console_double_click(source_file, source_line):
            if not source_file:
                return
            from Infernux.engine.ui import project_utils
            project_utils.open_file_with_system(
                source_file, project_root=_console_project_path)
        self.console.on_double_click_entry = _on_console_double_click

        # Status bar (C++ native panel — replaces Python StatusBarPanel)
        from Infernux.lib import StatusBarPanel as NativeStatusBarPanel
        self.status_bar = NativeStatusBarPanel()
        self.status_bar.set_console_panel(self.console)
        self._wire_status_bar_listener()
        engine.register_gui("status_bar", self.status_bar)

        # Scene view
        self.scene_view = SceneViewPanel(engine=engine)
        self.scene_view.set_window_manager(wm)
        if engine._play_mode_manager is not None:
            self.scene_view.set_play_mode_manager(engine._play_mode_manager)
        engine.register_gui("scene_view", self.scene_view)
        wm.register_existing_window("scene_view", self.scene_view, "scene_view")

        # Game view
        self.game_view = GameViewPanel(engine=engine)
        self.game_view.set_window_manager(wm)
        engine.register_gui("game_view", self.game_view)
        wm.register_existing_window("game_view", self.game_view, "game_view")

        # UI Editor
        self.ui_editor = UIEditorPanel()
        self.ui_editor.set_window_manager(wm)
        self.ui_editor.set_hierarchy_panel(self.hierarchy)
        self.ui_editor.set_engine(engine)
        engine.register_gui("ui_editor", self.ui_editor)
        wm.register_existing_window("ui_editor", self.ui_editor, "ui_editor")

        self.project_panel.on_state_changed = self._persist_editor_state
        wm.set_on_state_changed(self._persist_editor_state)

        ws = _panel_state.get("window_manager")
        if ws:
            wm.load_state(ws)

        self._persist_editor_state()

    # ── Native panel callback wiring ───────────────────────────────────

    def _wire_menu_bar_callbacks(self, wm):
        """Wire C++ MenuBarPanel callbacks to Python managers."""
        mb = self.menu_bar
        sfm = self.scene_file_manager
        engine = self.engine

        # Scene file operations
        if sfm:
            mb.on_save = lambda: sfm.save_current_scene()
            mb.on_new_scene = lambda: sfm.new_scene()
            mb.on_request_close = lambda: sfm.request_close()

        # Undo
        def _undo():
            from Infernux.engine.undo import UndoManager
            mgr = UndoManager.instance()
            if mgr and mgr.can_undo:
                mgr.undo()
        def _redo():
            from Infernux.engine.undo import UndoManager
            mgr = UndoManager.instance()
            if mgr and mgr.can_redo:
                mgr.redo()
        def _can_undo():
            from Infernux.engine.undo import UndoManager
            mgr = UndoManager.instance()
            return bool(mgr and mgr.can_undo)
        def _can_redo():
            from Infernux.engine.undo import UndoManager
            mgr = UndoManager.instance()
            return bool(mgr and mgr.can_redo)

        mb.on_undo = _undo
        mb.on_redo = _redo
        mb.can_undo = _can_undo
        mb.can_redo = _can_redo

        # Window management
        from Infernux.lib import WindowTypeInfo
        def _get_registered_types():
            types = wm.get_registered_types()
            result = []
            for type_id, info in types.items():
                wti = WindowTypeInfo()
                wti.type_id = type_id
                wti.display_name = info.display_name
                wti.singleton = info.singleton
                result.append(wti)
            return result
        def _get_open_windows():
            return wm.get_open_windows()

        mb.get_registered_types = _get_registered_types
        mb.get_open_windows = _get_open_windows
        mb.open_window = lambda tid: wm.open_window(tid)
        mb.close_window = lambda tid: wm.close_window(tid)
        mb.reset_layout = lambda: wm.reset_layout()

        # Close request from C++ engine
        native = engine.get_native_engine() if engine else None
        if native:
            mb.is_close_requested = lambda: native.is_close_requested()

        # Floating sub-panels (still rendered from Python)
        from Infernux.engine.ui.build_settings_panel import BuildSettingsPanel
        from Infernux.engine.ui.preferences_panel import PreferencesPanel
        from Infernux.engine.ui.tag_layer_settings import PhysicsLayerMatrixPanel
        from Infernux.engine.project_context import get_project_root
        self._build_settings = BuildSettingsPanel()
        self._preferences = PreferencesPanel()
        self._physics_layer_matrix = PhysicsLayerMatrixPanel()
        self._physics_layer_matrix.set_project_path(get_project_root() or "")

        mb.toggle_build_settings = lambda: (
            self._build_settings.close() if self._build_settings.is_open
            else self._build_settings.open()
        )
        mb.toggle_preferences = lambda: (
            self._preferences.close() if self._preferences.is_open
            else self._preferences.open()
        )
        mb.toggle_physics_layer_matrix = lambda: (
            self._physics_layer_matrix.close() if self._physics_layer_matrix.is_open
            else self._physics_layer_matrix.open()
        )
        mb.is_build_settings_open = lambda: self._build_settings.is_open
        mb.is_preferences_open = lambda: self._preferences.is_open
        mb.is_physics_layer_matrix_open = lambda: self._physics_layer_matrix.is_open

        # Register a secondary renderable that draws the floating sub-panels
        # and save-confirmation popup after the menu bar.
        from Infernux.lib import InxGUIRenderable, InxGUIContext
        _bs = self._build_settings
        _pref = self._preferences
        _plm = self._physics_layer_matrix
        _sfm = sfm

        class _MenuBarFloatingPanels(InxGUIRenderable):
            def on_render(self, ctx: InxGUIContext):
                _bs.render(ctx)
                _pref.render(ctx)
                _plm.render(ctx)
                if _sfm:
                    _sfm.render_confirmation_popup(ctx)

        self._menu_bar_floats = _MenuBarFloatingPanels()
        engine.register_gui("menu_bar_floats", self._menu_bar_floats)

    def _wire_toolbar_callbacks(self, engine):
        """Wire C++ ToolbarPanel callbacks to Python PlayModeManager."""
        self._wire_toolbar_callbacks_on(self.toolbar, engine)

    def _wire_status_bar_listener(self):
        """Wire C++ StatusBarPanel to DebugConsole listener + EngineStatus."""
        sb = self.status_bar

        # Subscribe to DebugConsole for latest message + count updates
        from Infernux.debug import DebugConsole, LogType
        from Infernux.engine.ui.console_utils import is_internal, sanitize_text

        def _on_log_entry(entry):
            if is_internal(entry):
                return
            msg = sanitize_text(getattr(entry, 'message', ''))
            level_map = {
                LogType.LOG: "info",
                LogType.WARNING: "warning",
                LogType.ERROR: "error",
                LogType.ASSERT: "error",
                LogType.EXCEPTION: "error",
            }
            level = level_map.get(entry.log_type, "info")
            sb.set_latest_message(msg, level)
            if level == "warning":
                sb.increment_warn_count()
            elif level == "error":
                sb.increment_error_count()

        console = DebugConsole.instance()
        for entry in console.get_entries():
            _on_log_entry(entry)
        console.add_listener(_on_log_entry)

        # Register a lightweight renderable that syncs EngineStatus each frame
        from Infernux.lib import InxGUIRenderable, InxGUIContext
        from Infernux.engine.ui.engine_status import EngineStatus

        class _EngineStatusSync(InxGUIRenderable):
            def on_render(self, ctx: InxGUIContext):
                text, progress = EngineStatus.get()
                sb.set_engine_status(text, progress)

        self._engine_status_sync = _EngineStatusSync()
        self.engine.register_gui("engine_status_sync", self._engine_status_sync)

    def _wire_hierarchy_callbacks(self):
        """Wire C++ HierarchyPanel callbacks to Python managers."""
        hp = self.hierarchy
        from Infernux.engine.ui.selection_manager import SelectionManager
        from Infernux.engine.i18n import t as _t
        from Infernux.debug import Debug

        sel = SelectionManager.instance()

        # -- Selection integration --
        hp.is_selected = lambda oid: sel.is_selected(oid)
        hp.select_id = lambda oid: sel.select(oid)
        hp.toggle_id = lambda oid: sel.toggle(oid)
        hp.range_select_id = lambda oid: sel.range_select(oid)
        hp.clear_selection = lambda: sel.clear()
        hp.get_primary = lambda: sel.get_primary()
        hp.get_selected_ids = lambda: sel.get_ids()
        hp.selection_count = lambda: sel.count()
        hp.is_selection_empty = lambda: sel.is_empty()
        hp.set_ordered_ids = lambda ids: sel.set_ordered_ids(ids)

        # -- Translation --
        hp.translate = _t

        # -- Warning --
        hp.show_warning = lambda msg: Debug.log_warning(msg)

        # -- Undo --
        from Infernux.engine.undo import HierarchyUndoTracker
        undo = HierarchyUndoTracker()
        hp.undo_record_create = lambda oid, desc: undo.record_create(oid, desc)
        hp.undo_record_delete = lambda oid, desc: undo.record_delete(oid, desc)
        hp.undo_record_move = lambda oid, opid, npid, oidx, nidx: undo.record_move(oid, opid, npid, oidx, nidx)

        # -- Scene info --
        def _get_scene_display_name():
            sfm = self.scene_file_manager
            return sfm.get_display_name() if sfm else ""

        def _is_prefab_mode():
            sfm = self.scene_file_manager
            return bool(sfm and sfm.is_prefab_mode)

        def _get_prefab_display_name():
            sfm = self.scene_file_manager
            if sfm:
                name = sfm.get_display_name()
                return _t("hierarchy.prefab_mode_header").format(name=name)
            return "Prefab"

        hp.get_scene_display_name = _get_scene_display_name
        hp.is_prefab_mode = _is_prefab_mode
        hp.get_prefab_display_name = _get_prefab_display_name

        # -- Runtime hidden IDs --
        def _get_runtime_hidden_ids():
            try:
                mgr = PlayModeManager.instance()
                if mgr is not None:
                    return mgr.get_runtime_hidden_object_ids()
            except Exception:
                pass
            return set()

        hp.get_runtime_hidden_ids = _get_runtime_hidden_ids

        # -- Canvas / UI-mode queries (need Python py_components) --
        def _go_has_canvas(oid):
            from Infernux.lib import SceneManager as _SM
            from Infernux.ui import UICanvas
            scene = _SM.instance().get_active_scene()
            if not scene:
                return False
            go = scene.find_by_id(oid)
            if not go:
                return False
            for comp in go.get_py_components():
                if isinstance(comp, UICanvas):
                    return True
            return False

        def _go_has_ui_screen_component(oid):
            from Infernux.lib import SceneManager as _SM
            from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
            scene = _SM.instance().get_active_scene()
            if not scene:
                return False
            go = scene.find_by_id(oid)
            if not go:
                return False
            for comp in go.get_py_components():
                if isinstance(comp, InxUIScreenComponent):
                    return True
            return False

        def _parent_has_canvas_ancestor(oid):
            from Infernux.lib import SceneManager as _SM
            from Infernux.ui import UICanvas
            scene = _SM.instance().get_active_scene()
            if not scene:
                return False
            go = scene.find_by_id(oid)
            if not go:
                return False
            cur = go
            while cur is not None:
                for comp in cur.get_py_components():
                    if isinstance(comp, UICanvas):
                        return True
                cur = cur.get_parent()
            return False

        def _has_canvas_descendant(oid):
            from Infernux.lib import SceneManager as _SM
            from Infernux.ui import UICanvas
            scene = _SM.instance().get_active_scene()
            if not scene:
                return False
            go = scene.find_by_id(oid)
            if not go:
                return False
            stack = [go]
            while stack:
                cur = stack.pop()
                for comp in cur.get_py_components():
                    if isinstance(comp, UICanvas):
                        return True
                stack.extend(cur.get_children())
            return False

        hp.go_has_canvas = _go_has_canvas
        hp.go_has_ui_screen_component = _go_has_ui_screen_component
        hp.parent_has_canvas_ancestor = _parent_has_canvas_ancestor
        hp.has_canvas_descendant = _has_canvas_descendant

        # -- Context-menu creation callbacks --
        def _create_primitive(type_idx, parent_id):
            from Infernux.lib import SceneManager, PrimitiveType
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            types = [PrimitiveType.Cube, PrimitiveType.Sphere, PrimitiveType.Capsule,
                     PrimitiveType.Cylinder, PrimitiveType.Plane]
            if type_idx < 0 or type_idx >= len(types):
                return
            new_obj = scene.create_primitive(types[type_idx])
            if new_obj:
                _finalize(new_obj, parent_id, "Create Primitive")

        def _create_light(type_idx, parent_id):
            from Infernux.lib import SceneManager, LightType, LightShadows, Vector3
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            names = ["Directional Light", "Point Light", "Spot Light"]
            light_types = [LightType.Directional, LightType.Point, LightType.Spot]
            if type_idx < 0 or type_idx >= len(light_types):
                return
            new_obj = scene.create_game_object(names[type_idx])
            if not new_obj:
                return
            light_comp = new_obj.add_component("Light")
            if light_comp:
                light_comp.light_type = light_types[type_idx]
                light_comp.shadows = LightShadows.Hard
                light_comp.shadow_bias = 0.0
                if light_types[type_idx] == LightType.Directional:
                    trans = new_obj.transform
                    if trans:
                        trans.euler_angles = Vector3(50.0, -30.0, 0.0)
                elif light_types[type_idx] == LightType.Point:
                    light_comp.range = 10.0
                elif light_types[type_idx] == LightType.Spot:
                    light_comp.range = 10.0
                    light_comp.outer_spot_angle = 45.0
                    light_comp.spot_angle = 30.0
            _finalize(new_obj, parent_id, "Create Light")

        def _create_camera(parent_id):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            new_obj = scene.create_game_object("Camera")
            if new_obj:
                cam = new_obj.add_component("Camera")
                _finalize(new_obj, parent_id, "Create Camera")

        def _create_render_stack(parent_id):
            from Infernux.lib import SceneManager
            from Infernux.renderstack import RenderStack as RenderStackCls
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            new_obj = scene.create_game_object("RenderStack")
            if not new_obj:
                return
            stack = new_obj.add_py_component(RenderStackCls())
            if stack is None:
                scene.destroy_game_object(new_obj)
                return
            _finalize(new_obj, parent_id, "Create RenderStack")

        def _create_empty(parent_id):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if scene:
                new_obj = scene.create_game_object("GameObject")
                if new_obj:
                    _finalize(new_obj, parent_id, "Create Empty")

        def _create_ui_canvas(parent_id):
            from Infernux.lib import SceneManager
            from Infernux.ui import UICanvas as UICanvasCls
            from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            go = scene.create_game_object("Canvas")
            if go:
                go.add_py_component(UICanvasCls())
                invalidate_canvas_cache()
                _finalize(go, parent_id, "Create Canvas")

        def _create_ui_text(parent_id):
            from Infernux.lib import SceneManager
            from Infernux.ui import UIText as UITextCls
            from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            canvas_pid = _find_canvas_parent_id(scene, parent_id)
            go = scene.create_game_object("Text")
            if go:
                go.add_py_component(UITextCls())
                _finalize(go, canvas_pid, "Create Text")
                invalidate_canvas_cache()

        def _create_ui_button(parent_id):
            from Infernux.lib import SceneManager
            from Infernux.ui import UIButton as UIButtonCls
            from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            canvas_pid = _find_canvas_parent_id(scene, parent_id)
            go = scene.create_game_object("Button")
            if go:
                btn = UIButtonCls()
                btn.width = 160.0
                btn.height = 40.0
                go.add_py_component(btn)
                _finalize(go, canvas_pid, "Create Button")
                invalidate_canvas_cache()

        def _find_canvas_parent_id(scene, parent_id):
            if parent_id == 0:
                return 0
            from Infernux.ui import UICanvas
            obj = scene.find_by_id(parent_id)
            if not obj:
                return parent_id
            cur = obj
            while cur is not None:
                for c in cur.get_py_components():
                    if isinstance(c, UICanvas):
                        return cur.id
                cur = cur.get_parent()
            return parent_id

        def _finalize(new_obj, parent_id, description):
            """Parent, select, record undo, notify."""
            if parent_id and parent_id != 0:
                from Infernux.lib import SceneManager
                scene = SceneManager.instance().get_active_scene()
                if scene:
                    parent = scene.find_by_id(parent_id)
                    if parent:
                        new_obj.set_parent(parent)
            sel.select(new_obj.id)
            undo.record_create(new_obj.id, description)
            if hp.on_selection_changed:
                hp.on_selection_changed(new_obj.id)

        hp.create_primitive = _create_primitive
        hp.create_light = _create_light
        hp.create_camera = _create_camera
        hp.create_render_stack = _create_render_stack
        hp.create_empty = _create_empty
        hp.create_ui_canvas = _create_ui_canvas
        hp.create_ui_text = _create_ui_text
        hp.create_ui_button = _create_ui_button

        # -- Prefab actions --
        def _save_as_prefab(oid):
            from Infernux.lib import SceneManager, AssetRegistry
            from Infernux.engine.project_context import get_project_root
            from Infernux.engine.prefab_manager import save_prefab, PREFAB_EXTENSION
            import os
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            go = scene.find_by_id(oid)
            if not go:
                return
            root = get_project_root()
            if not root:
                return
            assets_dir = os.path.join(root, "Assets")
            os.makedirs(assets_dir, exist_ok=True)
            adb = None
            registry = AssetRegistry.instance()
            if registry:
                adb = registry.get_asset_database()
            from Infernux.engine.ui.project_file_ops import get_unique_name
            prefab_name = get_unique_name(assets_dir, go.name, PREFAB_EXTENSION)
            file_path = os.path.join(assets_dir, prefab_name + PREFAB_EXTENSION)
            if save_prefab(go, file_path, asset_database=adb):
                Debug.log_internal(f"Prefab saved: {file_path}")

        def _prefab_select_asset(oid):
            from Infernux.lib import SceneManager, AssetRegistry
            scene = SceneManager.instance().get_active_scene()
            go = scene.find_by_id(oid) if scene else None
            if not go:
                return
            guid = getattr(go, 'prefab_guid', '')
            path = _resolve_prefab(guid)
            if path:
                EditorEventBus.instance().emit("select_asset", path)

        def _prefab_open_asset(oid):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            go = scene.find_by_id(oid) if scene else None
            if not go:
                return
            guid = getattr(go, 'prefab_guid', '')
            path = _resolve_prefab(guid)
            if path:
                EditorEventBus.instance().emit("open_asset", path)

        def _prefab_apply_overrides(oid):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            go = scene.find_by_id(oid) if scene else None
            if not go:
                return
            guid = getattr(go, 'prefab_guid', '')
            path = _resolve_prefab(guid)
            if path:
                from Infernux.engine.prefab_overrides import apply_overrides_to_prefab
                apply_overrides_to_prefab(go, path)

        def _prefab_revert_overrides(oid):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            go = scene.find_by_id(oid) if scene else None
            if not go:
                return
            guid = getattr(go, 'prefab_guid', '')
            path = _resolve_prefab(guid)
            if path:
                from Infernux.engine.prefab_overrides import revert_overrides
                revert_overrides(go, path)

        def _prefab_unpack(oid):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            go = scene.find_by_id(oid) if scene else None
            if not go:
                return
            _unpack_recursive(go)
            Debug.log_internal(f"Unpacked prefab instance: {go.name}")

        def _unpack_recursive(obj):
            try:
                obj.prefab_guid = ""
                obj.prefab_root = False
            except Exception:
                pass
            try:
                for child in obj.get_children():
                    _unpack_recursive(child)
            except Exception:
                pass

        def _resolve_prefab(guid):
            if not guid:
                return None
            try:
                from Infernux.lib import AssetRegistry
                registry = AssetRegistry.instance()
                if registry:
                    adb = registry.get_asset_database()
                    if adb:
                        return adb.get_path_from_guid(guid)
            except Exception:
                pass
            return None

        hp.save_as_prefab = _save_as_prefab
        hp.prefab_select_asset = _prefab_select_asset
        hp.prefab_open_asset = _prefab_open_asset
        hp.prefab_apply_overrides = _prefab_apply_overrides
        hp.prefab_revert_overrides = _prefab_revert_overrides
        hp.prefab_unpack = _prefab_unpack

        # -- Clipboard --
        _clipboard = {"entries": [], "cut": False}

        def _copy_selected(cut):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return False
            ids = sel.get_ids()
            if not ids:
                return False
            selected_set = set(ids)
            roots = []
            for oid in ids:
                obj = scene.find_by_id(oid)
                if obj is None:
                    continue
                parent = obj.get_parent()
                skip = False
                while parent is not None:
                    if parent.id in selected_set:
                        skip = True
                        break
                    parent = parent.get_parent()
                if not skip:
                    roots.append(obj)
            if not roots:
                return False
            entries = []
            for obj in roots:
                parent = obj.get_parent()
                transform = getattr(obj, "transform", None)
                entries.append({
                    "json": obj.serialize(),
                    "source_parent_id": parent.id if parent else None,
                    "source_sibling_index": transform.get_sibling_index() if transform else 0,
                })
            _clipboard["entries"] = entries
            _clipboard["cut"] = bool(cut)
            if cut:
                from Infernux.engine.undo import CompoundCommand, DeleteGameObjectCommand, UndoManager
                commands = [DeleteGameObjectCommand(obj.id, "Cut GameObject") for obj in roots]
                mgr = UndoManager.instance()
                if mgr:
                    cmd = commands[0] if len(commands) == 1 else CompoundCommand(commands, "Cut GameObjects")
                    mgr.execute(cmd)
                else:
                    sfm2 = self.scene_file_manager
                    for obj in roots:
                        live = scene.find_by_id(obj.id)
                        if live:
                            scene.destroy_game_object(live)
                    if sfm2:
                        sfm2.mark_dirty()
                sel.clear()
                if hp.on_selection_changed:
                    hp.on_selection_changed(0)
            return True

        def _paste_clipboard():
            if not _clipboard["entries"]:
                return False
            from Infernux.lib import SceneManager
            from Infernux.engine.undo import CompoundCommand, CreateGameObjectCommand, UndoManager
            from Infernux.engine.prefab_manager import _restore_pending_py_components, _strip_prefab_runtime_fields
            import json
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return False
            explicit_parent = None
            if sel.count() == 1:
                explicit_parent = scene.find_by_id(sel.get_primary())
            created = []
            for entry in _clipboard["entries"]:
                parent = explicit_parent
                if parent is None:
                    src_pid = entry.get("source_parent_id")
                    if src_pid is not None:
                        parent = scene.find_by_id(src_pid)
                try:
                    obj_data = json.loads(entry["json"])
                except Exception:
                    continue
                _strip_prefab_runtime_fields(obj_data)
                new_obj = scene.instantiate_from_json(json.dumps(obj_data), parent)
                if new_obj:
                    created.append(new_obj)
            if created and scene.has_pending_py_components():
                sfm2 = self.scene_file_manager
                adb = getattr(sfm2, "_asset_database", None) if sfm2 else None
                _restore_pending_py_components(scene, asset_database=adb)
            if not created:
                return False
            cids = [o.id for o in created]
            cmds = [CreateGameObjectCommand(cid, "Paste GameObject") for cid in cids]
            mgr = UndoManager.instance()
            if mgr:
                cmd = cmds[0] if len(cmds) == 1 else CompoundCommand(cmds, "Paste GameObjects")
                mgr.record(cmd)
            else:
                sfm2 = self.scene_file_manager
                if sfm2:
                    sfm2.mark_dirty()
            sel.set_ids(cids)
            if hp.on_selection_changed:
                hp.on_selection_changed(cids[-1] if cids else 0)
            if _clipboard["cut"]:
                _clipboard["entries"] = []
                _clipboard["cut"] = False
            return True

        def _has_clipboard_data():
            return bool(_clipboard["entries"])

        hp.copy_selected = _copy_selected
        hp.paste_clipboard = _paste_clipboard
        hp.has_clipboard_data = _has_clipboard_data

        # -- External drop (from Project panel) --
        def _instantiate_prefab(ref, parent_id, is_guid):
            from Infernux.lib import SceneManager, AssetRegistry
            from Infernux.engine.prefab_manager import instantiate_prefab
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            adb = None
            registry = AssetRegistry.instance()
            if registry:
                adb = registry.get_asset_database()
            parent = scene.find_by_id(parent_id) if parent_id else None
            try:
                if is_guid:
                    new_obj = instantiate_prefab(guid=ref, scene=scene, parent=parent, asset_database=adb)
                else:
                    new_obj = instantiate_prefab(file_path=ref, scene=scene, parent=parent, asset_database=adb)
            except Exception as exc:
                Debug.log_error(f"Prefab instantiation failed: {exc}")
                return
            if new_obj:
                sel.select(new_obj.id)
                undo.record_create(new_obj.id, "Instantiate Prefab")
                if hp.on_selection_changed:
                    hp.on_selection_changed(new_obj.id)

        def _create_model_object(ref, parent_id, is_guid):
            from Infernux.lib import SceneManager, AssetRegistry
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            guid = ref if is_guid else ""
            if not guid:
                registry = AssetRegistry.instance()
                adb = registry.get_asset_database() if registry else None
                if not adb:
                    return
                guid = adb.get_guid_from_path(ref)
            if not guid:
                return
            new_obj = scene.create_from_model(guid)
            if new_obj:
                _finalize(new_obj, parent_id, "Create Model")

        hp.instantiate_prefab = _instantiate_prefab
        hp.create_model_object = _create_model_object

        # -- Delete selected --
        def _delete_selected_objects():
            from Infernux.lib import SceneManager
            from Infernux.engine.undo import CompoundCommand, DeleteGameObjectCommand, UndoManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            ids = list(sel.get_ids())
            if not ids:
                return
            commands = [DeleteGameObjectCommand(oid, "Delete GameObject") for oid in ids]
            mgr = UndoManager.instance()
            if mgr:
                cmd = commands[0] if len(commands) == 1 else CompoundCommand(commands, "Delete GameObjects")
                mgr.execute(cmd)
            else:
                for oid in ids:
                    obj = scene.find_by_id(oid)
                    if obj:
                        scene.destroy_game_object(obj)
                sfm2 = self.scene_file_manager
                if sfm2:
                    sfm2.mark_dirty()
            sel.clear()
            if hp.on_selection_changed:
                hp.on_selection_changed(0)

        hp.delete_selected_objects = _delete_selected_objects

    def _wire_project_callbacks(self):
        """Wire C++ ProjectPanel callbacks to Python managers."""
        pp = self.project_panel
        from Infernux.engine.i18n import t as _t
        from Infernux.engine.ui import project_file_ops as file_ops
        from Infernux.engine.ui import project_utils
        from Infernux.engine.scene_manager import SceneFileManager

        # -- Engine subsystems --
        native_engine = self.engine.get_native_engine()
        if native_engine:
            pp.setup_from_engine(native_engine)

        pp.set_root_path(self.project_path)

        import Infernux.resources as _resources
        pp.set_icons_directory(_resources.file_type_icons_dir)

        # -- Translation --
        pp.translate = _t

        # -- Asset database access (via engine) --
        adb = self.engine.get_asset_database()

        pp.get_guid_from_path = lambda path: (
            adb.get_guid_from_path(path) if adb else ""
        )
        pp.get_path_from_guid = lambda guid: (
            adb.get_path_from_guid(guid) if adb else ""
        )

        # -- File operation callbacks --
        from Infernux.debug import Debug

        def _safe_project_create(cb, *args):
            try:
                return cb(*args)
            except Exception as exc:
                Debug.log_error(f"ProjectPanel create failed: {exc}")
                return False, str(exc)

        def _safe_project_path(cb, *args):
            try:
                return cb(*args) or ""
            except Exception as exc:
                Debug.log_error(f"ProjectPanel path operation failed: {exc}")
                return ""

        pp.create_folder = lambda cur, name: _safe_project_create(
            file_ops.create_folder, cur, name)
        pp.create_script = lambda cur, name: _safe_project_create(
            file_ops.create_script, cur, name, adb)
        pp.create_shader = lambda cur, name, typ: _safe_project_create(
            file_ops.create_shader, cur, name, typ, adb)
        pp.create_material = lambda cur, name: _safe_project_create(
            file_ops.create_material, cur, name, adb)
        pp.create_scene = lambda cur, name: _safe_project_create(
            file_ops.create_scene, cur, name, adb)
        pp.do_rename = lambda old, new_name: _safe_project_path(
            file_ops.do_rename, old, new_name, adb)
        pp.get_unique_name = lambda cur, base, ext: (
            file_ops.get_unique_name(cur, base, ext)
        )
        pp.move_item_to_directory = lambda item, dest: _safe_project_path(
            file_ops.move_item_to_directory, item, dest, adb)

        # -- Delete (with Win32 confirmation dialog) --
        def _delete_items(paths):
            import ctypes, os
            valid = []
            seen = set()
            for p in paths or []:
                if not p or not os.path.exists(p) or p in seen:
                    continue
                seen.add(p)
                valid.append(p)
            if not valid:
                return

            title = _t("project.delete_confirm_title")
            if len(valid) == 1:
                msg = _t("project.delete_confirm_msg").replace(
                    "{name}", os.path.basename(valid[0]))
            else:
                msg = _t("project.delete_confirm_multi_msg").replace(
                    "{count}", str(len(valid)))
            # MB_OKCANCEL | MB_ICONWARNING | MB_DEFBUTTON2
            result = ctypes.windll.user32.MessageBoxW(
                0, msg, title, 0x1 | 0x30 | 0x100)
            if result != 1:  # IDOK
                return

            for item_path in sorted(
                valid, key=lambda p: (p.count(os.sep), len(p)), reverse=True
            ):
                if os.path.exists(item_path):
                    file_ops.delete_item(item_path, adb)

        pp.delete_items = _delete_items

        # -- Create prefab from hierarchy drag --
        def _create_prefab_from_hierarchy(obj_id, current_path):
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            game_object = scene.find_by_id(obj_id)
            if not game_object:
                return
            ok, result = file_ops.create_prefab_from_gameobject(
                game_object, current_path, adb)
            if not ok:
                from Infernux.debug import Debug
                Debug.log_warning(f"Failed to create prefab: {result}")

        pp.create_prefab_from_hierarchy = _create_prefab_from_hierarchy

        # -- Open callbacks --
        pp.open_file = lambda path: project_utils.open_file_with_system(
            path, project_root=self.project_path)

        def _open_scene(file_path):
            from Infernux.debug import Debug
            from Infernux.engine.deferred_task import DeferredTaskRunner
            from Infernux.engine.play_mode import PlayModeManager

            def _open_after_stop():
                sfm = SceneFileManager.instance()
                if sfm:
                    return bool(sfm.open_scene(file_path))
                Debug.log_warning("SceneFileManager not initialized")
                return False

            play_mode = PlayModeManager.instance()
            if play_mode and play_mode.is_playing:
                runner = DeferredTaskRunner.instance()
                if runner.is_busy:
                    Debug.log_warning(
                        "Cannot open scene while another deferred task is running")
                    return

                def _on_stop(ok):
                    if not ok:
                        Debug.log_warning(
                            "Play Mode stop did not complete; scene open cancelled")
                        return
                    try:
                        from Infernux.lib import SceneManager as NativeSM
                        nsm = NativeSM.instance()
                    except Exception:
                        nsm = None
                    if play_mode.is_playing:
                        Debug.log_warning(
                            "Scene open cancelled — Play Mode still active")
                        return
                    if nsm and nsm.is_playing():
                        Debug.log_warning(
                            "Scene open cancelled — native Play Mode still active")
                        return
                    _open_after_stop()

                if not play_mode.exit_play_mode(on_complete=_on_stop):
                    Debug.log_warning(
                        "Failed to stop Play Mode before opening scene")
                return

            sfm = SceneFileManager.instance()
            if sfm:
                sfm.open_scene(file_path)
            else:
                Debug.log_warning("SceneFileManager not initialized")

        pp.open_scene = _open_scene

        pp.open_prefab_mode = lambda path: (
            SceneFileManager.instance().open_prefab_mode_with_undo(path)
            if SceneFileManager.instance() else None
        )

        pp.reveal_in_explorer = lambda path: (
            project_utils.reveal_in_file_explorer(path)
        )

        # -- Script validation for drag-drop --
        def _validate_script_component(file_path):
            try:
                from Infernux.components.script_loader import (
                    load_component_from_file, ScriptLoadError)
                load_component_from_file(file_path)
                return True
            except Exception:
                return False

        pp.validate_script_component = _validate_script_component

        # -- Inspector invalidation --
        def _invalidate_asset_inspector(path):
            try:
                from Infernux.engine.ui.asset_inspector import (
                    invalidate_asset)
                invalidate_asset(path)
            except Exception:
                pass

        pp.invalidate_asset_inspector = _invalidate_asset_inspector

    def _wire_inspector_callbacks(self):
        """Wire C++ InspectorPanel callbacks to Python managers."""
        ip = self.inspector_panel
        engine = self.engine
        import time as _time
        from Infernux.engine.i18n import t as _t
        from Infernux.engine.ui import inspector_support as _inspector_support
        _bump_inspector_values = _inspector_support.bump_inspector_value_generation
        _record_profile_count = _inspector_support.record_inspector_profile_count
        _record_profile_timing = _inspector_support.record_inspector_profile_timing

        # ── Translation ────────────────────────────────────────────────
        ip.translate = _t

        # ── Selection ──────────────────────────────────────────────────
        from Infernux.engine.ui.selection_manager import SelectionManager

        ip.is_multi_selection = lambda: SelectionManager.instance().is_multi()
        ip.get_selected_ids = lambda: SelectionManager.instance().get_ids()
        ip.get_value_generation = _inspector_support.get_inspector_value_generation
        ip.consume_component_body_profile = _inspector_support.consume_inspector_profile_metrics

        # ── Object info ────────────────────────────────────────────────
        from Infernux.lib import SceneManager, InspectorObjectInfo

        _component_cache = {
            "object_id": 0,
            "scene_version": -1,
            "structure_version": -1,
            "items": [],
            "native_map": {},
            "py_map": {},
        }
        _material_section_cache = {
            "object_id": 0,
            "scene_version": -1,
            "structure_version": -1,
            "signature": (),
            "entries": [],
        }

        def _invalidate_material_section_cache():
            _material_section_cache["object_id"] = 0
            _material_section_cache["scene_version"] = -1
            _material_section_cache["structure_version"] = -1
            _material_section_cache["signature"] = ()
            _material_section_cache["entries"] = []

        def _invalidate_component_cache():
            _component_cache["object_id"] = 0
            _component_cache["scene_version"] = -1
            _component_cache["structure_version"] = -1
            _component_cache["items"] = []
            _component_cache["native_map"] = {}
            _component_cache["py_map"] = {}
            _invalidate_material_section_cache()

        def _current_scene_and_versions():
            scene = SceneManager.instance().get_active_scene()
            scene_version = getattr(scene, 'structure_version', -1) if scene else -1
            structure_version = _inspector_support.get_component_structure_version()
            return scene, scene_version, structure_version

        def _get_component_payload(obj_id):
            scene, scene_version, structure_version = _current_scene_and_versions()
            if (
                _component_cache["object_id"] == obj_id
                and _component_cache["scene_version"] == scene_version
                and _component_cache["structure_version"] == structure_version
            ):
                items = _component_cache["items"]
                native_map = _component_cache["native_map"]
                py_map = _component_cache["py_map"]
                stale = False
                for item in items:
                    comp = native_map.get(item.component_id) if item.is_native else py_map.get(item.component_id)
                    if comp is None:
                        stale = True
                        break
                    item.enabled = bool(getattr(comp, 'enabled', True))
                    if not item.is_native:
                        item.is_broken = bool(getattr(comp, '_is_broken', False))
                        item.broken_error = (
                            getattr(comp, '_broken_error', '') or ''
                            if item.is_broken else ''
                        )
                if not stale:
                    return scene, _component_cache["items"], native_map, py_map

            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                _invalidate_component_cache()
                return scene, [], {}, {}

            items = []
            native_map = {}
            py_map = {}

            for comp in (obj.get_components() or []):
                if _is_python_component_entry(comp):
                    continue
                comp_type_name = getattr(comp, 'type_name', type(comp).__name__)
                if comp_type_name == "Transform":
                    continue
                comp_id = getattr(comp, 'component_id', id(comp))
                ci = InspectorComponentInfo()
                ci.type_name = comp_type_name
                ci.component_id = comp_id
                ci.enabled = bool(getattr(comp, 'enabled', True))
                ci.is_native = True
                ci.is_script = False
                ci.is_broken = False
                ci.icon_id = _get_component_icon_id(comp_type_name, False)
                items.append(ci)
                native_map[comp_id] = comp

            for py_comp in (obj.get_py_components() or []):
                comp_id = getattr(py_comp, 'component_id', id(py_comp))
                ci = InspectorComponentInfo()
                ci.type_name = getattr(py_comp, 'type_name', type(py_comp).__name__)
                ci.component_id = comp_id
                ci.enabled = bool(getattr(py_comp, 'enabled', True))
                ci.is_native = False
                ci.is_script = True
                ci.is_broken = bool(getattr(py_comp, '_is_broken', False))
                ci.broken_error = (
                    getattr(py_comp, '_broken_error', '') or ''
                    if ci.is_broken else ''
                )
                ci.icon_id = _get_component_icon_id(ci.type_name, True)
                items.append(ci)
                py_map[comp_id] = py_comp

            _component_cache["object_id"] = obj_id
            _component_cache["scene_version"] = scene_version
            _component_cache["structure_version"] = structure_version
            _component_cache["items"] = items
            _component_cache["native_map"] = native_map
            _component_cache["py_map"] = py_map
            return scene, items, native_map, py_map

        def _get_cached_component_maps(obj_id):
            scene, scene_version, structure_version = _current_scene_and_versions()
            if (
                _component_cache["object_id"] == obj_id
                and _component_cache["scene_version"] == scene_version
                and _component_cache["structure_version"] == structure_version
            ):
                return scene, _component_cache["items"], _component_cache["native_map"], _component_cache["py_map"]
            return _get_component_payload(obj_id)

        def _resolve_component(obj_id, comp_id, is_native):
            _scene, _items, native_map, py_map = _get_cached_component_maps(obj_id)
            if is_native:
                return native_map.get(comp_id)
            return py_map.get(comp_id)

        def _get_object_info(obj_id):
            info = InspectorObjectInfo()
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return info
            info.name = obj.name
            info.active = obj.active
            info.tag = getattr(obj, 'tag', 'Untagged')
            info.layer = getattr(obj, 'layer', 0)
            info.prefab_guid = getattr(obj, 'prefab_guid', '') or ''
            info.hide_transform = getattr(obj, 'hide_transform', False)
            return info

        ip.get_object_info = _get_object_info

        def _set_object_property(obj_id, prop_name, value_str):
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return
            from Infernux.engine.undo import UndoManager, SetPropertyCommand
            mgr = UndoManager.instance()
            old_val = getattr(obj, prop_name, None)
            if prop_name == "active":
                new_val = value_str.lower() in ("true", "1")
            elif prop_name == "name":
                new_val = value_str
            elif prop_name == "tag":
                new_val = value_str
            elif prop_name == "layer":
                new_val = int(value_str)
            else:
                new_val = value_str
            if mgr:
                mgr.execute(SetPropertyCommand(obj, prop_name, old_val, new_val,
                                            f"Set {prop_name}"))
            else:
                setattr(obj, prop_name, new_val)
                _bump_inspector_values()
            # DEBUG: verify active property change took effect
            if prop_name == "active":
                actual = getattr(obj, prop_name, None)
                if actual != new_val:
                    from Infernux.debug import Debug
                    Debug.log_warning(
                        f"[Inspector] SetActive failed: old={old_val}, "
                        f"requested={new_val}, actual={actual}, obj={obj_id}")

        ip.set_object_property = _set_object_property

        # ── Transform ──────────────────────────────────────────────────
        from Infernux.lib import InspectorTransformData

        def _get_transform_data(obj_id):
            td = InspectorTransformData()
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return td
            trans = obj.get_transform()
            if trans is None:
                return td
            lp = trans.local_position
            le = trans.local_euler_angles
            ls = trans.local_scale
            td.px, td.py_, td.pz = lp.x, lp.y, lp.z
            td.rx, td.ry, td.rz = le.x, le.y, le.z
            td.sx, td.sy, td.sz = ls.x, ls.y, ls.z
            return td

        ip.get_transform_data = _get_transform_data

        def _set_transform_data(obj_id, td):
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return
            trans = obj.get_transform()
            if trans is None:
                return
            from Infernux.lib import Vector3
            trans.local_position = Vector3(td.px, td.py_, td.pz)
            trans.local_euler_angles = Vector3(td.rx, td.ry, td.rz)
            trans.local_scale = Vector3(td.sx, td.sy, td.sz)
            _bump_inspector_values()

        ip.set_transform_data = _set_transform_data

        # ── Component enumeration ──────────────────────────────────────
        from Infernux.lib import InspectorComponentInfo
        from Infernux.components.component import InxComponent

        def _is_python_component_entry(component) -> bool:
            return isinstance(component, InxComponent) or hasattr(component, 'get_py_component')

        def _get_component_list(obj_id):
            _scene, items, _native_map, _py_map = _get_component_payload(obj_id)
            return items

        ip.get_component_list = _get_component_list

        # ── Component icons ────────────────────────────────────────────
        # Lazy-init icon cache from InspectorPanel Python class
        _icon_cache = {}
        _icons_loaded = [False]

        def _ensure_icons():
            if _icons_loaded[0]:
                return
            _icons_loaded[0] = True
            native_engine = engine.get_native_engine()
            if not native_engine:
                return
            import os
            import Infernux.resources as _resources
            from Infernux.lib import TextureLoader
            icons_dir = _resources.component_icons_dir
            if not os.path.isdir(icons_dir):
                return
            for fname in os.listdir(icons_dir):
                if not fname.startswith("component_") or not fname.endswith(".png"):
                    continue
                key = fname[len("component_"):-len(".png")]
                tex_name = f"__compicon__{key}"
                if native_engine.has_imgui_texture(tex_name):
                    _icon_cache[key] = native_engine.get_imgui_texture_id(tex_name)
                    continue
                icon_path = os.path.join(icons_dir, fname)
                tex_data = TextureLoader.load_from_file(icon_path)
                if tex_data and tex_data.is_valid():
                    pixels, w, h = _inspector_support.prepare_component_icon_pixels(tex_data)
                    if w > 0 and h > 0:
                        tid = native_engine.upload_texture_for_imgui(tex_name, pixels, w, h)
                        if tid != 0:
                            _icon_cache[key] = tid

        def _get_component_icon_id(type_name, is_script):
            _ensure_icons()
            tid = _icon_cache.get(type_name.lower(), 0)
            if tid == 0 and is_script:
                tid = _icon_cache.get("script", 0)
            return tid

        ip.get_component_icon_id = _get_component_icon_id

        # ── Component body rendering ───────────────────────────────────
        from Infernux.engine.ui import inspector_components as comp_ui

        def _render_component_body(ctx, obj_id, type_name, comp_id, is_native):
            _record_profile_count("bodyResolve_count")
            _resolve_t0 = _time.perf_counter()
            comp = _resolve_component(obj_id, comp_id, is_native)
            _record_profile_timing("bodyResolve", (_time.perf_counter() - _resolve_t0) * 1000.0)
            if comp is None:
                return
            if is_native:
                _record_profile_count("bodyNativeDispatch_count")
                _native_t0 = _time.perf_counter()
                comp_ui.render_component(ctx, comp)
                _record_profile_timing("bodyNativeDispatch", (_time.perf_counter() - _native_t0) * 1000.0)
                return
            else:
                _record_profile_count("bodyPyCheck_count")
                _py_check_t0 = _time.perf_counter()
                _script_err = None
                if getattr(comp, '_is_broken', False):
                    _script_err = getattr(comp, '_broken_error', '') or 'Script failed to load'
                else:
                    _py_guid = getattr(comp, '_script_guid', None)
                    adb = engine.get_asset_database()
                    if _py_guid and adb:
                        from Infernux.components.script_loader import get_script_error_by_path
                        _py_path = adb.get_path_from_guid(_py_guid)
                        if _py_path:
                            _script_err = get_script_error_by_path(_py_path)
                _record_profile_timing("bodyPyCheck", (_time.perf_counter() - _py_check_t0) * 1000.0)
                if _script_err:
                    from Infernux.engine.ui.theme import Theme, ImGuiCol
                    ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
                    ctx.text_wrapped(_script_err)
                    ctx.pop_style_color(1)
                else:
                    from Infernux.engine.ui.inspector_components import render_py_component
                    _record_profile_count("bodyPyDispatch_count")
                    _py_render_t0 = _time.perf_counter()
                    render_py_component(ctx, comp)
                    _record_profile_timing("bodyPyDispatch", (_time.perf_counter() - _py_render_t0) * 1000.0)
                return

        ip.render_component_body = _render_component_body

        # ── Component clipboard & context menu ─────────────────────────
        # Shared clipboard for Copy / Paste as New / Paste as Values.
        _comp_clipboard = {
            "type_name": "",
            "is_native": True,
            "script_guid": "",
            "json": "",           # serialize() for native, _serialize_fields() for py
        }

        def _copy_component_to_clipboard(comp, type_name, is_native):
            _comp_clipboard["type_name"] = type_name
            _comp_clipboard["is_native"] = is_native
            _comp_clipboard["script_guid"] = getattr(comp, '_script_guid', '') or ''
            try:
                if is_native and hasattr(comp, "serialize"):
                    _comp_clipboard["json"] = comp.serialize()
                elif hasattr(comp, "_serialize_fields"):
                    _comp_clipboard["json"] = comp._serialize_fields()
                else:
                    _comp_clipboard["json"] = ""
            except Exception:
                _comp_clipboard["json"] = ""

        def _has_comp_clipboard():
            return bool(_comp_clipboard["type_name"] and _comp_clipboard["json"])

        def _can_paste_values(comp, type_name, is_native):
            """True when clipboard data can be applied to *comp*."""
            if not _has_comp_clipboard():
                return False
            return (_comp_clipboard["type_name"] == type_name and
                    _comp_clipboard["is_native"] == is_native)

        def _paste_as_new_component(obj):
            """Add a new component from clipboard data."""
            from Infernux.engine.undo import UndoManager
            tn = _comp_clipboard["type_name"]
            native = _comp_clipboard["is_native"]
            json_data = _comp_clipboard["json"]
            guid = _comp_clipboard["script_guid"]
            mgr = UndoManager.instance()
            if native:
                result = obj.add_component(tn)
                if result and json_data and hasattr(result, "deserialize"):
                    try:
                        result.deserialize(json_data)
                    except Exception:
                        pass
            else:
                from Infernux.engine.component_restore import create_component_instance
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                asset_db = sfm._asset_database if sfm else None
                instance, _sp = create_component_instance(
                    guid, tn, asset_database=asset_db)
                if instance is None:
                    from Infernux.debug import Debug
                    Debug.log_warning(f"Cannot paste: failed to create '{tn}'")
                    return
                if json_data:
                    try:
                        instance._deserialize_fields(json_data, _skip_on_after_deserialize=True)
                    except TypeError:
                        instance._deserialize_fields(json_data)
                    except Exception:
                        pass
                if guid:
                    try:
                        instance._script_guid = guid
                    except Exception:
                        pass
                obj.add_py_component(instance)
            _invalidate_component_cache()

        def _paste_values_to_component(comp, is_native):
            """Apply clipboard data to an existing component."""
            json_data = _comp_clipboard["json"]
            if not json_data:
                return
            from Infernux.engine.undo import UndoManager
            if is_native and hasattr(comp, "deserialize"):
                try:
                    comp.deserialize(json_data)
                except Exception:
                    pass
            elif hasattr(comp, "_deserialize_fields"):
                try:
                    comp._deserialize_fields(json_data, _skip_on_after_deserialize=True)
                except TypeError:
                    comp._deserialize_fields(json_data)
                except Exception:
                    pass
            _bump_inspector_values()

        def _get_script_path_for_component(comp):
            """Return the file path for a Python script component, or ''."""
            guid = getattr(comp, '_script_guid', None)
            if not guid:
                return ''
            adb = engine.get_asset_database()
            if adb:
                path = adb.get_path_from_guid(guid)
                if path:
                    return path
            return ''

        def _can_remove_component(obj, comp, type_name, is_native):
            if is_native:
                blockers = []
                if hasattr(obj, 'get_remove_component_blockers'):
                    try:
                        blockers = list(obj.get_remove_component_blockers(comp) or [])
                    except RuntimeError:
                        blockers = []
                can_remove = not blockers
                if can_remove and hasattr(obj, 'can_remove_component'):
                    can_remove = bool(obj.can_remove_component(comp))
                if not can_remove:
                    from Infernux.debug import Debug
                    suffix = (
                        f" required by: {', '.join(blockers)}"
                        if blockers else
                        "another component depends on it"
                    )
                    Debug.log_warning(f"Cannot remove '{type_name}' — {suffix}")
                    return False
            return True

        _project_path = self.project_path

        def _render_component_context_menu(ctx, obj_id, type_name, comp_id, is_native):
            # C++ handles BeginPopupContextItem/EndPopup — render menu items only.
            # Return True if component was removed (C++ skips EndPopup in that case).
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return False
            comp = _resolve_component(obj_id, comp_id, is_native)
            if comp is None:
                return False

            # ── View Script (only for Python script components) ────────
            if not is_native:
                script_path = _get_script_path_for_component(comp)
                if script_path:
                    if ctx.selectable(_t("inspector.show_script")):
                        from Infernux.engine.ui import project_utils
                        project_utils.open_file_with_system(
                            script_path, project_root=_project_path)
                        ctx.close_current_popup()
                        return False
                    ctx.separator()

            # ── Copy Properties ────────────────────────────────────────
            if ctx.selectable(_t("inspector.copy_properties")):
                _copy_component_to_clipboard(comp, type_name, is_native)
                ctx.close_current_popup()
                return False

            # ── Paste as New Component ─────────────────────────────────
            has_clip = _has_comp_clipboard()
            if not has_clip:
                ctx.begin_disabled()
            if ctx.selectable(_t("inspector.paste_as_new")):
                _paste_as_new_component(obj)
                ctx.close_current_popup()
                if not has_clip:
                    ctx.end_disabled()
                return False
            if not has_clip:
                ctx.end_disabled()

            # ── Paste as Values ────────────────────────────────────────
            can_paste_vals = _can_paste_values(comp, type_name, is_native)
            if not can_paste_vals:
                ctx.begin_disabled()
            if ctx.selectable(_t("inspector.paste_properties")):
                _paste_values_to_component(comp, is_native)
                ctx.close_current_popup()
                if not can_paste_vals:
                    ctx.end_disabled()
                return False
            if not can_paste_vals:
                ctx.end_disabled()

            ctx.separator()

            # ── Remove ─────────────────────────────────────────────────
            if ctx.selectable(_t("inspector.remove")):
                if not _can_remove_component(obj, comp, type_name, is_native):
                    return False
                if is_native:
                    from Infernux.engine.undo import UndoManager, RemoveNativeComponentCommand
                    mgr = UndoManager.instance()
                    if mgr:
                        mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
                        _invalidate_component_cache()
                    elif obj.remove_component(comp) is False:
                        return False
                    else:
                        _invalidate_component_cache()
                else:
                    from Infernux.engine.undo import UndoManager, RemovePyComponentCommand
                    mgr = UndoManager.instance()
                    if mgr:
                        mgr.execute(RemovePyComponentCommand(obj.id, comp))
                        _invalidate_component_cache()
                    elif obj.remove_py_component(comp) is False:
                        return False
                    else:
                        _invalidate_component_cache()
                ctx.close_current_popup()
                return True
            return False

        ip.render_component_context_menu = _render_component_context_menu

        # ── Component enabled toggle ───────────────────────────────────
        def _set_component_enabled(obj_id, comp_id, new_enabled, is_native):
            comp = _resolve_component(obj_id, comp_id, is_native)
            if comp is None:
                return
            from Infernux.engine.undo import UndoManager, SetPropertyCommand
            mgr = UndoManager.instance()
            old_val = comp.enabled
            if mgr:
                mgr.execute(SetPropertyCommand(
                    comp, "enabled", old_val, new_enabled,
                    f"Toggle {getattr(comp, 'type_name', '?')}"))
            else:
                comp.enabled = new_enabled
                _bump_inspector_values()
            for item in _component_cache["items"]:
                if item.component_id == comp_id:
                    item.enabled = bool(new_enabled)
                    break

        ip.set_component_enabled = _set_component_enabled

        # ── Add Component ──────────────────────────────────────────────
        from Infernux.lib import InspectorAddComponentEntry

        def _get_add_component_entries():
            entries = []
            # Native component types (from C++ registry)
            from Infernux.lib import get_registered_component_types
            for type_name in sorted(get_registered_component_types()):
                if type_name == "Transform":
                    continue
                e = InspectorAddComponentEntry()
                e.display_name = type_name
                e.category = "Built-in"
                e.is_native = True
                entries.append(e)
            # Engine-level Python components (e.g. RenderStack)
            from Infernux.renderstack.render_stack import RenderStack
            for display_name, comp_cls in [("RenderStack", RenderStack)]:
                e = InspectorAddComponentEntry()
                e.display_name = display_name
                e.category = "Engine"
                e.is_native = False
                e.script_path = ""
                entries.append(e)
            # User script components (scan project folder)
            import os
            from Infernux.engine.project_context import get_project_root
            from Infernux.components.script_loader import load_component_from_file, ScriptLoadError
            project_root = get_project_root()
            if project_root and os.path.isdir(project_root):
                for dirpath, _dirnames, filenames in os.walk(project_root):
                    rel = os.path.relpath(dirpath, project_root)
                    if any(part.startswith('.') or part in (
                            '__pycache__', 'build', 'Library',
                            'ProjectSettings', 'Logs', 'Temp')
                           for part in rel.split(os.sep)):
                        continue
                    for fn in filenames:
                        if not fn.endswith('.py') or fn.startswith('_'):
                            continue
                        full = os.path.join(dirpath, fn)
                        with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(4096)
                        if 'InxComponent' not in content:
                            continue
                        try:
                            comp_class = load_component_from_file(full)
                        except (ScriptLoadError, Exception):
                            continue
                        e = InspectorAddComponentEntry()
                        e.display_name = comp_class.__name__
                        e.category = "Scripts"
                        e.is_native = False
                        e.script_path = full
                        entries.append(e)
            return entries

        ip.get_add_component_entries = _get_add_component_entries

        def _add_component(type_name_or_path, is_native, script_path):
            sel = SelectionManager.instance()
            primary = sel.get_primary()
            if not primary:
                return
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            if obj is None:
                return
            from Infernux.engine.ui.inspector_components import (
                _record_add_component_compound, _get_component_ids
            )
            from Infernux.debug import Debug
            if is_native:
                before_ids = _get_component_ids(obj)
                result = obj.add_component(type_name_or_path)
                if result is not None:
                    Debug.log_internal(f"Added component: {type_name_or_path}")
                    _record_add_component_compound(
                        obj, type_name_or_path, result, before_ids, is_py=False)
                    _invalidate_component_cache()
                    _bump_inspector_values()
                else:
                    Debug.log_error(f"Failed to add component: {type_name_or_path}")
            else:
                # Engine-level Python component (no script_path) vs user script
                if not script_path:
                    # Engine Python component — look up by display_name
                    _engine_py_map = {"RenderStack": None}
                    try:
                        from Infernux.renderstack.render_stack import RenderStack as _RS
                        _engine_py_map["RenderStack"] = _RS
                    except ImportError:
                        pass
                    comp_cls = _engine_py_map.get(type_name_or_path)
                    if comp_cls is None:
                        Debug.log_error(f"Unknown engine component: {type_name_or_path}")
                        return
                    # Enforce @disallow_multiple
                    if getattr(comp_cls, '_disallow_multiple_', False):
                        for pc in (obj.get_py_components() or []):
                            if isinstance(pc, comp_cls):
                                Debug.log_warning(
                                    f"Cannot add another '{comp_cls.__name__}' — "
                                    f"only one per scene is allowed")
                                return
                    instance = comp_cls()
                    before_ids = _get_component_ids(obj)
                    obj.add_py_component(instance)
                    _record_add_component_compound(
                        obj, comp_cls.__name__, instance, before_ids, is_py=True)
                    _invalidate_component_cache()
                    _bump_inspector_values()
                    Debug.log_internal(f"Added component {comp_cls.__name__}")
                else:
                    from Infernux.components import load_and_create_component
                    adb = engine.get_asset_database()
                    try:
                        component_instance = load_and_create_component(
                            script_path, asset_database=adb)
                    except Exception as exc:
                        Debug.log_error(f"Failed to load script '{script_path}': {exc}")
                        return
                    if component_instance is None:
                        Debug.log_error(f"No InxComponent found in '{script_path}'")
                        return
                    before_ids = _get_component_ids(obj)
                    obj.add_py_component(component_instance)
                    _record_add_component_compound(
                        obj, component_instance.type_name,
                        component_instance, before_ids, is_py=True)
                    _invalidate_component_cache()
                    _bump_inspector_values()
                    Debug.log_internal(f"Added component {component_instance.type_name}")

        ip.add_component = _add_component

        # ── Remove Component ───────────────────────────────────────────
        def _remove_component(obj_id, type_name, comp_id, is_native):
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return False
            comp = _resolve_component(obj_id, comp_id, is_native)
            if comp is not None:
                if not _can_remove_component(obj, comp, type_name, is_native):
                    return False
                from Infernux.engine.undo import UndoManager
                mgr = UndoManager.instance()
                if is_native:
                    from Infernux.engine.undo import RemoveNativeComponentCommand
                    if mgr:
                        mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
                        _invalidate_component_cache()
                        _bump_inspector_values()
                        return True
                    ok = obj.remove_component(comp) is not False
                    if ok:
                        _invalidate_component_cache()
                        _bump_inspector_values()
                    return ok
                else:
                    from Infernux.engine.undo import RemovePyComponentCommand
                    if mgr:
                        mgr.execute(RemovePyComponentCommand(obj.id, comp))
                        _invalidate_component_cache()
                        _bump_inspector_values()
                        return True
                    ok = obj.remove_py_component(comp) is not False
                    if ok:
                        _invalidate_component_cache()
                        _bump_inspector_values()
                    return ok
            return False

        ip.remove_component = _remove_component

        # ── Asset / File preview ───────────────────────────────────────
        def _render_asset_inspector(ctx, file_path, category):
            from Infernux.engine.ui.asset_inspector import render_asset_inspector
            try:
                render_asset_inspector(ctx, ip, file_path, category)
            except Exception as exc:
                from Infernux.debug import Debug
                Debug.log_error(f"Asset inspector render failed for '{file_path}': {exc}")

        ip.render_asset_inspector = _render_asset_inspector

        def _render_file_preview(ctx, file_path):
            import os
            if os.path.isdir(file_path):
                ctx.label(_t("inspector.folder_label").format(name=os.path.basename(file_path)))
                ctx.separator()
                ctx.label(_t("inspector.path_label").format(path=file_path))
            else:
                ctx.label(_t("inspector.file_label").format(name=os.path.basename(file_path)))
                ctx.separator()
                ctx.label(_t("inspector.no_previewer"))

        ip.render_file_preview = _render_file_preview

        # ── Material sections ──────────────────────────────────────────
        _inline_material_state = {
            "cache": {},
            "exec_layer": None,
        }

        def _make_inline_material_panel_adapter():
            class _Adapter:
                def __init__(self):
                    self._inline_material_cache = _inline_material_state["cache"]
                    self._inline_material_exec_layer = _inline_material_state["exec_layer"]

                def _get_native_engine(self):
                    return engine.get_native_engine()

                def _ensure_material_file_path(self, material):
                    return _inspector_support.ensure_material_file_path(material)

                def _sync_back(self):
                    _inline_material_state["cache"] = self._inline_material_cache
                    _inline_material_state["exec_layer"] = self._inline_material_exec_layer

            return _Adapter()

        def _render_material_sections(ctx, obj_id):
            from Infernux.components.builtin_component import BuiltinComponent
            from Infernux.engine.ui import inspector_material as mat_ui
            from Infernux.engine.ui.inspector_utils import render_compact_section_header, render_info_text
            from Infernux.engine.ui.theme import Theme, ImGuiCol, ImGuiStyleVar

            scene, items, native_map, _py_map = _get_cached_component_maps(obj_id)
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return

            wrapper_cls = BuiltinComponent._builtin_registry.get("MeshRenderer")
            renderers = []
            signature_parts = []
            for item in items:
                if not item.is_native or item.type_name != "MeshRenderer":
                    continue
                renderer = native_map.get(item.component_id)
                if renderer is None:
                    continue
                if wrapper_cls is not None and not isinstance(renderer, BuiltinComponent):
                    try:
                        renderer = wrapper_cls._get_or_create_wrapper(renderer, obj)
                    except Exception:
                        pass
                mat_count = getattr(renderer, 'material_count', 0) or 1
                try:
                    material_guids = tuple(renderer.get_material_guids() or [])
                except Exception:
                    material_guids = ()
                try:
                    slot_names = tuple(renderer.get_material_slot_names() or [])
                except Exception:
                    slot_names = ()
                renderers.append((renderer, mat_count, material_guids, slot_names))
                signature_parts.append((
                    getattr(renderer, 'component_id', id(renderer)),
                    mat_count,
                    material_guids,
                    slot_names,
                ))

            if not renderers:
                return

            ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP * 1.5)
            ctx.separator()
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT)
            ctx.label(_t("inspector.material_overrides"))
            ctx.pop_style_color(1)
            ctx.separator()
            if not render_compact_section_header(
                ctx, "Materials##obj_mat_sections", level="primary", default_open=True
            ):
                return

            _scene, scene_version, structure_version = _current_scene_and_versions()
            signature = tuple(signature_parts)
            if (
                _material_section_cache["object_id"] == obj_id
                and _material_section_cache["scene_version"] == scene_version
                and _material_section_cache["structure_version"] == structure_version
                and _material_section_cache["signature"] == signature
            ):
                valid_entries = _material_section_cache["entries"]
            else:
                valid_entries = []
                for renderer, mat_count, material_guids, slot_names in renderers:
                    for slot_idx in range(mat_count):
                        try:
                            mat = renderer.get_effective_material(slot_idx)
                        except Exception:
                            mat = None
                        if mat is None:
                            continue
                        if slot_idx < len(slot_names) and slot_names[slot_idx]:
                            label = f"{slot_names[slot_idx]} (Slot {slot_idx})"
                        else:
                            label = f"Element {slot_idx}"
                        is_default = slot_idx >= len(material_guids) or not material_guids[slot_idx]
                        valid_entries.append({
                            "label": label,
                            "material": mat,
                            "is_default": is_default,
                        })
                _material_section_cache["object_id"] = obj_id
                _material_section_cache["scene_version"] = scene_version
                _material_section_cache["structure_version"] = structure_version
                _material_section_cache["signature"] = signature
                _material_section_cache["entries"] = valid_entries

            owner_name = getattr(obj, 'name', '') or ''
            multiple_renderers = len(renderers) > 1

            if not valid_entries:
                return

            ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
            ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)
            for index, entry in enumerate(valid_entries):
                title = entry["label"]
                if multiple_renderers and owner_name:
                    title = f"{owner_name} / {title}"
                if not render_compact_section_header(
                    ctx, f"{title}##mat_entry_{index}", level="secondary", default_open=True
                ):
                    continue

                if entry["is_default"]:
                    render_info_text(ctx, "Using the renderer's effective default material")

                adapter = _make_inline_material_panel_adapter()
                ctx.push_id(index)
                try:
                    mat_ui.render_inline_material_body(ctx, adapter, entry["material"], cache_key=f"obj_mat_{obj_id}_{index}")
                finally:
                    ctx.pop_id()
                    adapter._sync_back()

                if index != len(valid_entries) - 1:
                    ctx.separator()
            ctx.pop_style_var(2)

        ip.render_material_sections = _render_material_sections

        # ── Prefab ─────────────────────────────────────────────────────
        from Infernux.lib import InspectorPrefabInfo

        def _get_prefab_info(obj_id):
            pinfo = InspectorPrefabInfo()
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return pinfo
            guid = getattr(obj, 'prefab_guid', '') or ''
            if not guid:
                return pinfo
            from Infernux.engine.scene_manager import SceneFileManager
            sfm = SceneFileManager.instance()
            if sfm and not sfm.is_prefab_mode:
                pinfo.is_readonly = True
                pinfo.is_transform_readonly = True
            return pinfo

        ip.get_prefab_info = _get_prefab_info

        def _prefab_action(obj_id, action):
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(obj_id) if scene else None
            if obj is None:
                return
            guid = getattr(obj, 'prefab_guid', '') or ''
            if not guid:
                return
            adb = engine.get_asset_database()
            if action == "select":
                if adb:
                    path = adb.get_path_from_guid(guid)
                    if path:
                        self.project_panel.set_current_path(
                            __import__('os').path.dirname(path))
            elif action == "open":
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm and adb:
                    path = adb.get_path_from_guid(guid)
                    if path:
                        sfm.open_prefab_mode_with_undo(path)
            elif action == "apply":
                from Infernux.engine.prefab_overrides import apply_prefab_overrides
                apply_prefab_overrides(obj)
            elif action == "revert":
                from Infernux.engine.prefab_overrides import revert_prefab_overrides
                revert_prefab_overrides(obj)

        ip.prefab_action = _prefab_action

        # ── Tags & Layers ──────────────────────────────────────────────
        from Infernux.lib import TagLayerManager

        ip.get_all_tags = lambda: TagLayerManager.instance().get_all_tags()
        ip.get_all_layers = lambda: TagLayerManager.instance().get_all_layers()

        # ── Script drop ────────────────────────────────────────────────
        def _handle_script_drop(script_path):
            sel = SelectionManager.instance()
            primary = sel.get_primary()
            if not primary:
                return
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            if obj is None:
                return
            from Infernux.components import load_and_create_component
            from Infernux.debug import Debug
            adb = engine.get_asset_database()
            try:
                instance = load_and_create_component(
                    script_path, asset_database=adb)
            except Exception as exc:
                Debug.log_error(f"Failed to load script '{script_path}': {exc}")
                return
            if instance is None:
                Debug.log_error(f"No InxComponent found in '{script_path}'")
                return
            from Infernux.engine.ui.inspector_components import (
                _record_add_component_compound, _get_component_ids
            )
            before_ids = _get_component_ids(obj)
            obj.add_py_component(instance)
            _record_add_component_compound(
                obj, instance.type_name, instance, before_ids, is_py=True)
            _invalidate_component_cache()
            _bump_inspector_values()

        ip.handle_script_drop = _handle_script_drop

        # ── Window manager ─────────────────────────────────────────────
        wm = self.window_manager

        def _open_window(win_id):
            if wm:
                wm.open_window(win_id)

        ip.open_window = _open_window

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

    def _wire_selection_system(self):
        hierarchy = self.hierarchy
        inspector = self.inspector_panel
        project = self.project_panel
        scene_view = self.scene_view
        event_bus = self.event_bus

        hierarchy.on_selection_changed = self._on_hierarchy_selected
        project.on_file_selected = self._on_project_selected
        project.on_empty_area_clicked = self._on_project_panel_empty_clicked
        scene_view.set_on_object_picked(self._on_scene_view_picked)
        scene_view.set_on_box_select(self._on_box_select_done)
        hierarchy.on_double_click_focus = (
            lambda oid: self._fly_to_object_by_id(oid)
        )

        # Let structural undo commands restore selection via the same
        # pipeline as SelectionCommand (updates inspector, outline, etc.).
        from Infernux.engine.undo import (
            CreateGameObjectCommand, DeleteGameObjectCommand)
        CreateGameObjectCommand._selection_restore_fn = self._apply_selection_undo
        DeleteGameObjectCommand._selection_restore_fn = self._apply_selection_undo

    def _set_outline(self, object_id: int):
        native = self.engine.get_native_engine()
        if not native:
            return
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        ids = sel.get_ids()
        if len(ids) > 1:
            native.set_selection_outlines(ids)
        elif object_id:
            native.set_selection_outline(object_id)
        else:
            native.clear_selection_outline()

    def _fly_to_object_by_id(self, object_id: int):
        """Resolve object ID and fly scene view to it."""
        if not object_id:
            return
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(object_id) if scene else None
        if obj:
            self.scene_view.fly_to_object(obj)

    def _on_hierarchy_selected(self, object_id: int):
        """C++ HierarchyPanel calls this with uint64_t primary ID (0 = none)."""
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        new_ids = sel.get_ids()
        primary_id = sel.get_primary()

        # Record selection change for undo (skip if caused by undo/redo itself)
        self._record_editor_selection_change(new_ids, "")

        # Resolve ID → game object for inspector & event bus
        obj = None
        if primary_id:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary_id) if scene else None

        self.inspector_panel.set_selected_object_id(primary_id or 0)
        if primary_id:
            self.project_panel.clear_selection()
        self._set_outline(primary_id)
        self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)

    def _on_project_selected(self, path):
        self._record_editor_selection_change([], path or "")
        self._inspector_set_selected_file(path)
        if path:
            self.hierarchy.clear_selection_and_notify()
        self.event_bus.emit(EditorEvent.FILE_SELECTED, path)

    def _on_project_panel_empty_clicked(self):
        self._record_editor_selection_change([], "")
        self.project_panel.clear_selection()
        self.hierarchy.clear_selection_and_notify()

    def _on_scene_view_picked(self, object_id: int, ctrl: bool = False):
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()

        if ctrl and object_id:
            sel.toggle(object_id)
        elif object_id:
            sel.select(object_id)
        else:
            sel.clear()

        new_ids = sel.get_ids()
        primary = sel.get_primary()

        # Record selection change for undo
        self._record_editor_selection_change(new_ids, "")

        self._set_outline(primary)

        if primary:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            self.inspector_panel.set_selected_object_id(primary)
            self.project_panel.clear_selection()
            # Expand hierarchy to reveal the picked object
            if obj:
                self.hierarchy.expand_to_object(obj.id)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)
        else:
            self.project_panel.clear_selection()
            self.inspector_panel.set_selected_object_id(0)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, None)

    def _on_box_select_done(self, primary_obj):
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        new_ids = sel.get_ids()
        self._record_editor_selection_change(new_ids, "")

        self.inspector_panel.set_selected_object_id(primary_obj.id if primary_obj else 0)
        if primary_obj:
            self.project_panel.clear_selection()
            self.hierarchy.expand_to_object(primary_obj.id)
        else:
            self.project_panel.clear_selection()
        self._set_outline(primary_obj.id if primary_obj else 0)
        self.event_bus.emit(EditorEvent.SELECTION_CHANGED, primary_obj)

    def _navigate_console_entry_to_object(self, object_id: int) -> bool:
        """Reveal a console-targeted scene object in Hierarchy and Inspector."""
        if not object_id:
            return False

        from Infernux.lib import SceneManager

        scene = SceneManager.instance().get_active_scene()
        obj = scene.find_by_id(object_id) if scene else None
        if obj is None:
            return False

        if self.window_manager is not None:
            if not self.window_manager.is_window_open("hierarchy"):
                self.window_manager.open_window("hierarchy")
            if not self.window_manager.is_window_open("inspector"):
                self.window_manager.open_window("inspector")

        self.hierarchy.set_selected_object_by_id(object_id, clear_search=True)

        if not self.hierarchy.get_ui_mode():
            self.inspector_panel.set_selected_object_id(object_id)
            self.project_panel.clear_selection()
            self._set_outline(object_id)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)

        return True

    # ── Selection undo helpers ─────────────────────────────────────────

    # Structural command types whose associated selection changes should
    # not be recorded as separate undo entries.
    _STRUCTURAL_CMD_TYPES = None  # lazily populated

    @classmethod
    def _get_structural_types(cls):
        if cls._STRUCTURAL_CMD_TYPES is None:
            from Infernux.engine.undo import (
                CompoundCommand,
                CreateGameObjectCommand, DeleteGameObjectCommand,
                ReparentCommand, MoveGameObjectCommand,
                AddNativeComponentCommand, RemoveNativeComponentCommand,
                AddPyComponentCommand, RemovePyComponentCommand,
            )
            cls._STRUCTURAL_CMD_TYPES = (
                CompoundCommand,
                CreateGameObjectCommand, DeleteGameObjectCommand,
                ReparentCommand, MoveGameObjectCommand,
                AddNativeComponentCommand, RemoveNativeComponentCommand,
                AddPyComponentCommand, RemovePyComponentCommand,
            )
        return cls._STRUCTURAL_CMD_TYPES

    def _record_editor_selection_change(self, new_ids: list, file_path: str):
        """Push an EditorSelectionCommand when hierarchy/project selection changes.

        Skipped when:
        - The change is triggered by undo/redo (``is_executing``).
        - A structural command (create/delete/…) was just pushed in the
          same synchronous call chain, i.e. the stack top is a structural
          command with a timestamp < 50 ms ago.  This avoids recording a
          spurious SelectionCommand that is really a side-effect of the
          structural operation.
        """
        import time
        from Infernux.engine.undo import UndoManager, EditorSelectionCommand
        mgr = UndoManager.instance()
        next_file = file_path or ""
        if not mgr or mgr.is_executing:
            self._prev_selection_ids = list(new_ids)
            self._prev_selected_file = next_file
            return
        if new_ids == self._prev_selection_ids and next_file == self._prev_selected_file:
            return

        # Skip if the stack top is a structural command from this frame.
        if mgr._undo_stack:
            top = mgr._undo_stack[-1]
            if (isinstance(top, self._get_structural_types())
                    and (time.time() - top.timestamp) < 0.05):
                self._prev_selection_ids = list(new_ids)
                self._prev_selected_file = next_file
                return

        mgr.record(EditorSelectionCommand(
            self._prev_selection_ids, self._prev_selected_file,
            new_ids, next_file,
            self._apply_editor_selection_undo))
        self._prev_selection_ids = list(new_ids)
        self._prev_selected_file = next_file

    def _record_selection_change(self, new_ids: list):
        self._record_editor_selection_change(new_ids, "")

    def _apply_editor_selection_undo(self, ids: list, file_path: str):
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        file_path = file_path or ""

        if file_path:
            sel.clear()
            self._prev_selection_ids = []
            self._prev_selected_file = file_path
            self._set_outline(0)
            self.hierarchy.clear_selection_and_notify()
            self.project_panel.set_selected_file(file_path)
            self._inspector_set_selected_file(file_path)
            self.event_bus.emit(EditorEvent.FILE_SELECTED, file_path)
            return

        sel.set_ids(ids)
        self._prev_selection_ids = list(ids)
        self._prev_selected_file = ""

        primary = sel.get_primary()
        self._set_outline(primary)
        self.project_panel.clear_selection()
        self._inspector_set_selected_file("")

        if primary:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            self.inspector_panel.set_selected_object_id(primary)
            if obj:
                self.hierarchy.expand_to_object(obj.id)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)
        else:
            self.inspector_panel.set_selected_object_id(0)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, None)

    def _apply_selection_undo(self, ids: list):
        """Restore a selection state during undo/redo."""
        self._apply_editor_selection_undo(ids, "")


    def _wire_ui_editor(self):
        ui_editor = self.ui_editor
        hierarchy = self.hierarchy
        scene_view = self.scene_view
        game_view = self.game_view
        from Infernux.engine.ui.closable_panel import ClosablePanel

        def on_ui_mode_request(enter: bool):
            hierarchy.set_ui_mode(enter)

        ui_editor.set_on_request_ui_mode(on_ui_mode_request)

        def on_ui_editor_selected(go):
            if go is not None:
                hierarchy.set_selected_object_by_id(go.id)
            else:
                hierarchy.clear_selection_and_notify()

        ui_editor.set_on_selection_changed(on_ui_editor_selected)

        def on_hierarchy_ui_sync(oid):
            """C++ sends uint64_t; resolve to object for UIEditorPanel."""
            obj = None
            if oid:
                from Infernux.lib import SceneManager
                scene = SceneManager.instance().get_active_scene()
                obj = scene.find_by_id(oid) if scene else None
            ui_editor.notify_hierarchy_selection(obj)

        hierarchy.on_selection_changed_ui_editor = on_hierarchy_ui_sync

        def on_panel_focus_changed(_old_panel_id: str, new_panel_id: str):
            if self.window_manager is not None:
                self.window_manager.note_panel_focus(new_panel_id)

        ClosablePanel.set_on_panel_focus_changed(on_panel_focus_changed)


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
            except (OSError, ValueError):
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
            except OSError:
                pass

        need_reset = True
        if os.path.isfile(layout_ver_path):
            try:
                with open(layout_ver_path, "r") as f:
                    if f.read().strip() == str(_LAYOUT_VERSION):
                        need_reset = False
            except OSError:
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
        _panel_state.save()

    def _load_initial_scene(self):
        import Infernux.renderstack  # noqa: F401 — ensure RenderStack is discoverable
        self.scene_file_manager.load_last_scene_or_default()
