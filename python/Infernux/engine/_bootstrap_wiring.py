"""BootstrapWiringMixin — extracted from EditorBootstrap."""
from __future__ import annotations

"""
EditorBootstrap — structured editor initialization.

Breaks the monolithic ``release_engine()`` startup path into explicit
startup steps. Each step is a separate method, closures become instance
methods, and panel/manager references live on the bootstrap instance.
"""


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


class BootstrapWiringMixin:
    """BootstrapWiringMixin method group for EditorBootstrap."""

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
        from Infernux.engine.bootstrap_hierarchy import wire_hierarchy_callbacks
        wire_hierarchy_callbacks(self)

    def _wire_project_callbacks(self):
        """Wire C++ ProjectPanel callbacks to Python managers."""
        from Infernux.engine.bootstrap_project import wire_project_callbacks
        wire_project_callbacks(self)

    def _wire_inspector_callbacks(self):
        """Wire C++ InspectorPanel callbacks to Python managers."""
        from Infernux.engine.bootstrap_inspector import wire_inspector_callbacks
        wire_inspector_callbacks(self)

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

