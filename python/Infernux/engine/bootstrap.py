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
            "toolbar":            lambda: ToolbarPanel(engine=engine),
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

    # ── Phase 6: Create and register panels ────────────────────────────

    def _create_panels(self):
        engine = self.engine
        wm = self.window_manager

        # Per-frame scheduler
        self.frame_scheduler = FrameSchedulerPanel(engine=engine)
        engine.register_gui("frame_scheduler", self.frame_scheduler)

        # Menu bar
        self.menu_bar = MenuBarPanel(engine)
        self.menu_bar.set_window_manager(wm)
        self.menu_bar.set_scene_file_manager(self.scene_file_manager)
        engine.register_gui("menu_bar", self.menu_bar)

        # Toolbar
        self.toolbar = ToolbarPanel(engine=engine)
        self.toolbar.set_window_manager(wm)
        engine.register_gui("toolbar", self.toolbar)
        wm.register_existing_window("toolbar", self.toolbar, "toolbar")

        ts = _panel_state.get("toolbar")
        if ts:
            self.toolbar.load_state(ts)

        # Hierarchy
        self.hierarchy = HierarchyPanel()
        self.hierarchy.set_window_manager(wm)
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

        # Status bar
        self.status_bar = StatusBarPanel()
        self.status_bar.set_console_panel(self.console)
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

    # ── Phase 7: Wire selection system ─────────────────────────────────

    def _wire_selection_system(self):
        hierarchy = self.hierarchy
        inspector = self.inspector_panel
        project = self.project_panel
        scene_view = self.scene_view
        event_bus = self.event_bus

        hierarchy.set_on_selection_changed(self._on_hierarchy_selected)
        project.set_on_file_selected(self._on_project_selected)
        project.set_on_empty_area_clicked(self._on_project_panel_empty_clicked)
        scene_view.set_on_object_picked(self._on_scene_view_picked)
        scene_view.set_on_box_select(self._on_box_select_done)
        hierarchy.set_on_double_click_focus(
            lambda obj: self.scene_view.fly_to_object(obj)
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

    def _on_hierarchy_selected(self, obj):
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        new_ids = sel.get_ids()
        primary_id = sel.get_primary()

        # Record selection change for undo (skip if caused by undo/redo itself)
        self._record_selection_change(new_ids)

        self.inspector_panel.set_selected_object(obj)
        if obj is not None:
            self.project_panel.clear_selection()
        self._set_outline(primary_id)
        self.event_bus.emit(EditorEvent.SELECTION_CHANGED, obj)

    def _on_project_selected(self, path):
        self.inspector_panel.set_selected_file(path)
        if path:
            self.hierarchy.clear_selection()
        self.event_bus.emit(EditorEvent.FILE_SELECTED, path)

    def _on_project_panel_empty_clicked(self):
        self.project_panel.clear_selection()
        self.hierarchy.clear_selection()

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
                self.hierarchy.expand_to_object(obj)
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
            self.hierarchy.expand_to_object(primary_obj)
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

        if self.hierarchy.ui_mode and not self.hierarchy._is_in_canvas_tree(obj):
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
                self.hierarchy.expand_to_object(obj)
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
                hierarchy.clear_selection()

        ui_editor.set_on_selection_changed(on_ui_editor_selected)

        def on_hierarchy_ui_sync(obj):
            ui_editor.notify_hierarchy_selection(obj)

        hierarchy.set_on_selection_changed_ui_editor(on_hierarchy_ui_sync)

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
            self.hierarchy.clear_selection()
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
            _panel_state.put("toolbar", self.toolbar.save_state())
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
