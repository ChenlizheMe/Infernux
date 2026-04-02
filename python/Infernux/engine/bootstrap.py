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
    MenuBarPanel,
    FrameSchedulerPanel,
    ToolbarPanel,
    HierarchyPanel,
    InspectorPanel,
    SceneViewPanel,
    GameViewPanel,
    ProjectPanel,
    WindowManager,
    TagLayerSettingsPanel,
    StatusBarPanel,
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
_TOTAL_PHASES = 12


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
        self.hierarchy: Optional[HierarchyPanel] = None
        self.inspector_panel: Optional[InspectorPanel] = None
        self.project_panel: Optional[ProjectPanel] = None
        self.console = None  # C++ ConsolePanel (native)
        self.status_bar = None
        self.scene_view: Optional[SceneViewPanel] = None
        self.game_view: Optional[GameViewPanel] = None
        self.ui_editor: Optional[UIEditorPanel] = None

        # Selection state
        self._prev_selection = [0]  # kept for scene-change cleanup
        self._prev_selection_ids: list = []  # for undo recording

        # Progress tracking for launcher splash
        self._phase = 0

    # ── Public entry point ─────────────────────────────────────────────

    def run(self):
        """Execute all bootstrap phases and start the main loop."""
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

    # ── Phase 1: JIT pre-compilation ───────────────────────────────────

    @staticmethod
    def _precompile_jit():
        from Infernux._jit_kernels import precompile as _jit_precompile
        _jit_precompile()

    # ── Phase 2: Engine initialization ─────────────────────────────────

    def _init_engine(self):
        self.engine = Engine(self.engine_log_level)
        self.engine.init_renderer(
            width=1600, height=900, project_path=self.project_path
        )
        self.engine.set_gui_font(_resources.engine_font_path, 15)

    # ── Phase 3: Tag/layer settings ────────────────────────────────────

    def _load_tag_layer_settings(self):
        path = os.path.join(self.project_path, "ProjectSettings", "TagLayerSettings.json")
        if os.path.isfile(path):
            TagLayerManager.instance().load_from_file(path)

    # ── Phase 4: Create managers ───────────────────────────────────────

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

    # ── Phase 5: Register window types ─────────────────────────────────

    def _register_window_types(self):
        """Register all @editor_panel-decorated panels with WindowManager.

        Panels that require constructor arguments have their factory
        overridden here before apply_all flushes them into the
        WindowManager.
        """
        wm = self.window_manager
        engine = self.engine
        project_path = self.project_path

        # Override factories for panels that need runtime dependencies.
        # Panels with no-arg constructors use the default factory (cls()).
        _factories = {
            "inspector":          lambda: InspectorPanel(engine=engine),
            "scene_view":         lambda: SceneViewPanel(engine=engine),
            "game_view":          lambda: GameViewPanel(engine=engine),
            "project":            lambda: ProjectPanel(root_path=project_path, engine=engine),
            "toolbar":            lambda: self._create_native_toolbar(engine),
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

    # ── Phase 6: Create and register panels ────────────────────────────

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

        # Inspector
        self.inspector_panel = InspectorPanel(engine=engine)
        self.inspector_panel.set_window_manager(wm)
        engine.register_gui("inspector", self.inspector_panel)
        wm.register_existing_window("inspector", self.inspector_panel, "inspector")

        # Project
        self.project_panel = ProjectPanel(root_path=self.project_path, engine=engine)
        self.project_panel.set_window_manager(wm)
        engine.register_gui("project", self.project_panel)
        wm.register_existing_window("project", self.project_panel, "project")

        ps = _panel_state.get("project")
        if ps:
            self.project_panel.load_state(ps)

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

        self.project_panel.set_on_state_changed(self._persist_editor_state)
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
        from Infernux.engine.ui.console_panel import ConsolePanel as PyConsolePanel

        def _on_log_entry(entry):
            if PyConsolePanel._is_internal(entry):
                return
            msg = PyConsolePanel._sanitize_text(getattr(entry, 'message', ''))
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

    # ── Phase 7: Wire selection system ─────────────────────────────────

    def _wire_selection_system(self):
        hierarchy = self.hierarchy
        inspector = self.inspector_panel
        project = self.project_panel
        scene_view = self.scene_view
        event_bus = self.event_bus

        hierarchy.on_selection_changed = self._on_hierarchy_selected
        project.set_on_file_selected(self._on_project_selected)
        project.set_on_empty_area_clicked(self._on_project_panel_empty_clicked)
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
        self._record_selection_change(new_ids)

        # Resolve ID → game object for inspector & event bus
        obj = None
        if primary_id:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary_id) if scene else None

        self.inspector_panel.set_selected_object(obj)
        if obj is not None:
            self.project_panel.clear_selection()
        self._set_outline(primary_id)
        self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)

    def _on_project_selected(self, path):
        self.inspector_panel.set_selected_file(path)
        if path:
            self.hierarchy.clear_selection_and_notify()
        self.event_bus.emit(EditorEvent.FILE_SELECTED, path)

    def _on_project_panel_empty_clicked(self):
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
        self._record_selection_change(new_ids)

        self._set_outline(primary)

        if primary:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            self.inspector_panel.set_selected_object(obj)
            self.project_panel.clear_selection()
            # Expand hierarchy to reveal the picked object
            if obj:
                self.hierarchy.expand_to_object(obj.id)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)
        else:
            self.project_panel.clear_selection()
            self.inspector_panel.set_selected_object(None)
            self.event_bus.emit(EditorEvent.SELECTION_CHANGED, None)

    def _on_box_select_done(self, primary_obj):
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        new_ids = sel.get_ids()
        self._record_selection_change(new_ids)

        self.inspector_panel.set_selected_object(primary_obj)
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
            self.inspector_panel.set_selected_object(obj)
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
                AddPyComponentCommand, RemovePyComponentCommand,
            )
            cls._STRUCTURAL_CMD_TYPES = (
                CompoundCommand,
                CreateGameObjectCommand, DeleteGameObjectCommand,
                ReparentCommand, MoveGameObjectCommand,
                AddPyComponentCommand, RemovePyComponentCommand,
            )
        return cls._STRUCTURAL_CMD_TYPES

    def _record_selection_change(self, new_ids: list):
        """Push a SelectionCommand if the selection actually changed.

        Skipped when:
        - The change is triggered by undo/redo (``is_executing``).
        - A structural command (create/delete/…) was just pushed in the
          same synchronous call chain, i.e. the stack top is a structural
          command with a timestamp < 50 ms ago.  This avoids recording a
          spurious SelectionCommand that is really a side-effect of the
          structural operation.
        """
        import time
        from Infernux.engine.undo import UndoManager, SelectionCommand
        mgr = UndoManager.instance()
        if not mgr or mgr.is_executing:
            self._prev_selection_ids = list(new_ids)
            return
        if new_ids == self._prev_selection_ids:
            return

        # Skip if the stack top is a structural command from this frame.
        if mgr._undo_stack:
            top = mgr._undo_stack[-1]
            if (isinstance(top, self._get_structural_types())
                    and (time.time() - top.timestamp) < 0.05):
                self._prev_selection_ids = list(new_ids)
                return

        mgr.record(SelectionCommand(
            self._prev_selection_ids, new_ids,
            self._apply_selection_undo))
        self._prev_selection_ids = list(new_ids)

    def _apply_selection_undo(self, ids: list):
        """Restore a selection state during undo/redo."""
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        sel.set_ids(ids)
        self._prev_selection_ids = list(ids)

        primary = sel.get_primary()
        self._set_outline(primary)

        if primary:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            obj = scene.find_by_id(primary) if scene else None
            self.inspector_panel.set_selected_object(obj)
            # Expand hierarchy to reveal without re-triggering selection callback
            if obj:
                self.hierarchy.expand_to_object(obj.id)
        else:
            self.inspector_panel.set_selected_object(None)

    # ── Phase 8: Wire UI Editor ────────────────────────────────────────

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

    # ── Phase 9: Scene-change cleanup ──────────────────────────────────

    def _setup_scene_change_cleanup(self):
        def on_scene_changed():
            self._prev_selection[0] = 0
            from Infernux.engine.ui.selection_manager import SelectionManager
            SelectionManager.instance().clear()
            self.hierarchy.clear_selection_and_notify()
            self.inspector_panel.set_selected_object(None)
            self._set_outline(0)
            self.scene_view._fly_to_active = False
            self.scene_view._fly_to_last_obj_id = 0
            self.scene_view._fly_to_close = False

        self.scene_file_manager.set_on_scene_changed(on_scene_changed)

    # ── Phase 11: Layout persistence ───────────────────────────────────

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
        _panel_state.put("project", self.project_panel.save_state())
        _panel_state.put("window_manager", self.window_manager.save_state())
        _panel_state.save()

    # ── Phase 12: Load initial scene ───────────────────────────────────

    def _load_initial_scene(self):
        import Infernux.renderstack  # noqa: F401 — ensure RenderStack is discoverable
        self.scene_file_manager.load_last_scene_or_default()
