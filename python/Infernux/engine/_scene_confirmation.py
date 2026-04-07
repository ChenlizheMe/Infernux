"""SceneConfirmationMixin — extracted from SceneFileManager."""
from __future__ import annotations

"""
Scene file management for Infernux.

Handles:
- Tracking the current scene file path (.scene)
- Saving / loading scene files (delegates to C++ Scene::SaveToFile / LoadFromFile)
- Python component serialization during save, recreation during load
- Remembering last opened scene per project (EditorSettings.json)
- Default scene fallback when a scene file is missing
- File-dialog for "Save As" when the scene has no file yet
- Enforcing that scenes must be saved under Assets/

The C++ layer already provides ``Scene.serialize / deserialize / save_to_file /
load_from_file`` and ``PendingPyComponent`` for Python component recreation.
This module orchestrates those primitives into a complete workflow.
"""

import os
import json
import threading
from typing import Optional, Callable

from Infernux.debug import Debug
from Infernux.engine.project_context import get_project_root
from Infernux.engine.path_utils import safe_path as _safe_path


class SceneConfirmationMixin:
    """SceneConfirmationMixin method group for SceneFileManager."""

    def _request_save_confirmation(self, action: str, open_path: Optional[str] = None):
        """Set up the confirmation popup state."""
        self._pending_action = action
        self._pending_open_path = open_path
        self._show_confirm = True

    def _execute_pending_action(self) -> bool:
        """Run the action that was deferred by the confirmation dialog."""
        action = self._pending_action
        path = self._pending_open_path
        self._pending_action = None
        self._pending_open_path = None

        if action == 'new':
            self._begin_deferred_new()
            return True
        elif action == 'open' and path:
            self._begin_deferred_open(path)
            return True
        elif action == 'close' and self._engine:
            self._engine.confirm_close()
            return True
        elif action == 'close':
            native = self._native_engine_for_close()
            if native:
                native.confirm_close()
                return True
        return False

    def _clear_pending_action(self):
        self._pending_action = None
        self._pending_open_path = None

    def render_confirmation_popup(self, ctx):
        """Must be called every frame (by menu_bar).

        Draws the modal "Save before …?" dialog when ``_show_confirm`` is set.
        """
        POPUP_ID = "Save Scene?##save_confirm"

        if not self._show_confirm and self._pending_action is None:
            return

        if self._show_confirm:
            ctx.open_popup(POPUP_ID)
            self._show_confirm = False

        # ImGuiWindowFlags_AlwaysAutoResize = 1 << 6 = 64
        if ctx.begin_popup_modal(POPUP_ID, 64):
            ctx.label("当前场景有未保存的修改。")
            ctx.label("The current scene has unsaved changes.")
            ctx.label("")
            ctx.separator()
            ctx.label("")

            def _on_save():
                if self._current_scene_path:
                    action = self._pending_action
                    if self._do_save(self._current_scene_path):
                        if not self._execute_pending_action():
                            native = self._native_engine_for_close()
                            if native and action == 'close':
                                native.confirm_close()
                    else:
                        native = self._native_engine_for_close()
                        if self._pending_action == 'close' and native:
                            native.cancel_close()
                        self._close_in_progress = False
                        self._clear_pending_action()
                else:
                    # On close with an untitled scene, auto-save into Assets/
                    # to avoid Save-As dialog platform differences.
                    if self._pending_action == 'close':
                        default_path = self._default_scene_save_path()
                        if default_path and self._do_save(default_path):
                            if not self._execute_pending_action():
                                native = self._native_engine_for_close()
                                if native:
                                    native.confirm_close()
                        else:
                            native = self._native_engine_for_close()
                            if native:
                                native.cancel_close()
                            self._close_in_progress = False
                            self._clear_pending_action()
                    else:
                        self._post_save_callback = self._execute_pending_action
                        self._show_save_as_dialog()
                ctx.close_current_popup()

            def _on_dont_save():
                action = self._pending_action
                if action == 'close':
                    self._dirty = False
                    self._execute_pending_action()
                else:
                    self._execute_pending_action()
                ctx.close_current_popup()

            def _on_cancel():
                native = self._native_engine_for_close()
                if self._pending_action == 'close' and native:
                    native.cancel_close()
                self._close_in_progress = False
                self._clear_pending_action()
                ctx.close_current_popup()

            ctx.button("  保存  Save  ", _on_save)
            ctx.same_line()
            ctx.button("  不保存  Don't Save  ", _on_dont_save)
            ctx.same_line()
            ctx.button("  取消  Cancel  ", _on_cancel)

            ctx.end_popup()

    def poll_pending_save(self):
        """Check if the file dialog has produced a result and perform the save."""
        if self._pending_save_path is not None:
            path = self._pending_save_path
            self._pending_save_path = None  # consume
            if path:
                success = self._do_save(path)
                if success and self._post_save_callback:
                    cb = self._post_save_callback
                    self._post_save_callback = None
                    cb()
                elif not success:
                    # Save failed — cancel pending close/open/new chain so user can retry.
                    if self._post_save_callback is not None:
                        if self._pending_action == 'close' and self._engine:
                            self._engine.cancel_close()
                        self._close_in_progress = False
                        self._clear_pending_action()
                    self._post_save_callback = None
            else:
                # User cancelled the Save As dialog — cancel pending close/open/new chain.
                if self._post_save_callback is not None:
                    if self._pending_action == 'close' and self._engine:
                        self._engine.cancel_close()
                    self._close_in_progress = False
                    self._clear_pending_action()
                self._post_save_callback = None

