"""BootstrapSelectionMixin — extracted from EditorBootstrap."""
from __future__ import annotations

"""
EditorBootstrap — structured editor initialization.

Replaces the monolithic ``release_engine()`` god-function with organized
lifecycle phases.  Each phase is a separate method, closures become
instance methods, and all panel/manager references are instance attributes.
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


class BootstrapSelectionMixin:
    """BootstrapSelectionMixin method group for EditorBootstrap."""

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

