"""
SceneSaveMixin for strict, durable scene persistence.

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
from __future__ import annotations

import json
import os
import copy
from typing import Optional

from infernux.debug import Debug
from infernux.engine.path_utils import (
    is_path_within,
    path_key,
    resolved_path,
)
from .scene_manager import (
    SCENE_EXTENSION,
    DEFAULT_SCENE_FILE_BASE,
    LAST_OPENED_SCENE_GUID_KEY,
    _effective_project_root,
    _load_editor_settings,
    _save_editor_settings,
    _get_scene_root_objects,
)


def is_synthetic_input_frame() -> bool:
    """Resolve the editor input probe lazily so headless imports stay UI-free."""
    from infernux.engine.ui._dialogs import is_synthetic_input_frame as _probe

    return _probe()


def save_file_dialog(**kwargs):
    """Resolve the native editor dialog lazily so headless imports stay UI-free."""
    from infernux.engine.ui._dialogs import save_file_dialog as _save_file_dialog

    return _save_file_dialog(**kwargs)


class SceneSaveMixin:
    """SceneSaveMixin method group for SceneFileManager."""

    _SAVE_AS_MODAL_ID = "scene.save_as"
    _SAVE_AS_ALLOWED_PARENT_IDS = ("editor.unsaved_changes",)

    def _ensure_save_as_modal_presenter(self):
        """Lazily bind Scene Save As to the current Editor session."""
        from infernux.engine.interaction import EditorInteractionCore

        core = EditorInteractionCore.instance()
        if core is None:
            return None
        service = core.modals
        if self._save_as_modal_service is service:
            return service
        if self._save_as_modal_service is not None:
            self._save_as_modal_service.unregister(
                self._SAVE_AS_MODAL_ID,
                cancel=True,
            )
        self._save_as_modal_service = service
        service.register(
            self._SAVE_AS_MODAL_ID,
            is_active=lambda: bool(
                self._save_as_popup_open
                or self._save_as_native_dialog_pending
            ),
            render=self.render_save_as_popup,
            cancel=self._cancel_save_as,
            allowed_parent_ids=self._SAVE_AS_ALLOWED_PARENT_IDS,
        )
        return service

    def save_current_scene(self) -> bool:
        """Save the current scene.  If no file is associated, show a Save-As dialog.

        Returns True if the save happened synchronously, False if a dialog was
        opened (the actual save happens asynchronously via the dialog callback).
        """
        if self._is_play_mode():
            Debug.log_warning("Cannot save scene while in Play mode.")
            return False

        from infernux.engine.interaction import (
            DocumentActionStatus,
            DocumentRegistry,
        )

        result = DocumentRegistry.instance().request_save(self.document_id)
        return result.status in {
            DocumentActionStatus.APPLIED,
            DocumentActionStatus.NO_OP,
        }

    def save(self, *, ticket, save_as: bool = False):
        from infernux.engine.interaction import (
            DocumentActionResult,
            DocumentActionStatus,
            DocumentKind,
            DocumentRegistry,
        )

        if self._is_play_mode():
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "cannot save a scene while in Play mode",
            )
        registry = DocumentRegistry.instance()
        document = registry.get(ticket.document_id)
        if document is None or document.controller is not self or not ticket.is_pending:
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "the document is no longer open",
            )
        if document.kind is DocumentKind.PREFAB:
            if save_as:
                return DocumentActionResult(
                    DocumentActionStatus.REJECTED,
                    "prefab Save As is not supported in Prefab Mode",
                )
            return self._save_prefab(ticket_id=ticket.ticket_id)
        binding = self._binding_for_document(document.document_id)
        suspended = (document.document_id == self._previous_scene_document_id
                     and isinstance(self._previous_scene_document, dict))
        if document.kind is not DocumentKind.SCENE or (binding is None and not suspended):
            return DocumentActionResult(DocumentActionStatus.REJECTED, "the scene document is not resident or suspended")
        path = binding.resource_path if binding is not None else document.resource_path
        if save_as or not path:
            if self._save_as_ticket_id:
                return DocumentActionResult(DocumentActionStatus.REJECTED, "another Scene Save As is already open")
            self._save_as_ticket_id = ticket.ticket_id
            try:
                shown = self._show_save_as_dialog(document_id=document.document_id)
            except Exception:
                self._save_as_ticket_id = ""
                raise
            if shown:
                return DocumentActionResult(DocumentActionStatus.PENDING)
            self._save_as_ticket_id = ""
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "no project root is available",
            )
        return self.save_to_resource(ticket=ticket, resource_path=path)

    def save_to_resource(self, *, ticket, resource_path: str):
        """Persist an automation-selected path under the normal save contract."""
        from infernux.engine.interaction import (
            DocumentActionResult,
            DocumentActionStatus,
            DocumentKind,
            DocumentRegistry,
        )

        if self._is_play_mode():
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "cannot save a scene while in Play mode",
            )
        document = DocumentRegistry.instance().get(ticket.document_id)
        binding = self._binding_for_document(ticket.document_id)
        suspended = (ticket.document_id == self._previous_scene_document_id
                     and isinstance(self._previous_scene_document, dict))
        if (document is None or document.controller is not self or not ticket.is_pending
                or (binding is None and not suspended)):
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "the scene document is not resident or suspended",
            )
        if document.kind is not DocumentKind.SCENE:
            return DocumentActionResult(
                DocumentActionStatus.REJECTED,
                "explicit resource save only supports Scene documents",
            )
        return self._do_save(
            resource_path,
            ticket_id=ticket.ticket_id,
            document_id=document.document_id,
        )

    def discard(self, *, document_id: str) -> bool:
        from infernux.engine.interaction import DocumentKind, DocumentRegistry

        registry = DocumentRegistry.instance()
        document = registry.get(document_id)
        if document is None:
            return False
        if document.document_id != self.document_id:
            # Suspended scenes are resolved by replace/exit transactions; a
            # view-level discard must never mutate whichever scene is active.
            return False
        if document.kind is DocumentKind.PREFAB:
            # Prefab replacement uses CloseCoordinator's explicit discard
            # authorization and then exits to the suspended Scene. Reloading
            # a live Prefab in-place requires its own command transaction.
            return False
        if self._current_scene_path:
            return bool(
                self._do_open_scene(
                    self._current_scene_path,
                    record_navigation=False,
                )
            )
        return True

    def _save_prefab(self, *, ticket_id: str = "") -> bool:
        """Save the currently-edited prefab in Prefab Mode."""
        prefab_guid = str(self.prefab_mode_guid or "").strip().casefold()
        prefab_path = (
            str(self._asset_database.get_path_from_guid(prefab_guid) or "").strip()
            if self._asset_database is not None and prefab_guid
            else ""
        )
        if not prefab_path:
            Debug.log_warning("Prefab Mode asset GUID has no current file projection.")
            return False
        self.prefab_mode_path = resolved_path(prefab_path)
        prefab_path = self.prefab_mode_path

        from infernux.lib import SceneManager
        from infernux.engine.prefab_manager import (
            save_prefab_document,
        )
        from infernux.engine.interaction import (
            DocumentRegistry,
            document_content_token,
        )

        scene = SceneManager.instance().get_active_scene()
        roots = _get_scene_root_objects(scene)
        if not roots:
            Debug.log_warning("No root objects in Prefab Mode scene.")
            return False

        registry = DocumentRegistry.instance()
        document = registry.get(self.document_id)
        active_ticket_id = str(ticket_id or "")
        if document is None:
            Debug.log_error("Prefab save requires a bound editor document.")
            return False
        if not active_ticket_id:
            active_ticket_id = registry.begin_save(document.document_id).ticket_id
        try:
            from infernux.engine.prefab_manager import _read_resolved_prefab_document
            if _read_resolved_prefab_document(prefab_path, self._asset_database) != self.prefab_envelope:
                raise RuntimeError("Prefab source or base changed since opening; reopen before saving")
            prefab_document, _, _, _ = self.capture_prefab_mode_document()
            serialized_token = document_content_token(prefab_document)
            registry.capture_save_revision(
                active_ticket_id,
                content_token=serialized_token,
            )
        except Exception as exc:
            registry.complete_save(
                active_ticket_id,
                success=False,
                message=f"failed to serialize prefab: {exc}",
            )
            return False
        try:
            if self._asset_database is not None:
                from infernux.engine.prefab_overrides import build_prefab_asset_edit_command
                from infernux.engine.prefab_manager import _read_prefab_document

                if _read_prefab_document(prefab_path) != prefab_document:
                    command = build_prefab_asset_edit_command(
                        prefab_path, prefab_document, self._asset_database,
                    )
                    # Save persists the existing authoring history, rather than
                    # inserting another edit. The command still provides the
                    # same all-source/instance publication and compensation.
                    command.execute()
                    for world_id in command.scene_world_ids():
                        owner = self.document_id_for_scene(world_id)
                        if owner and owner != document.document_id:
                            registry.mark_changed(owner)
            elif not save_prefab_document(prefab_document, prefab_path):
                raise RuntimeError("Prefab document write failed")
        except Exception as exc:
            registry.complete_save(
                active_ticket_id,
                success=False,
                message=f"failed to save prefab: {exc}",
            )
            return False

        self.prefab_envelope = prefab_document
        self._prefab_variant_overrides = copy.deepcopy(prefab_document.get("variant", {}).get("property_overrides", []))
        from infernux.engine.prefab_manager import _link_prefab_hierarchy
        _link_prefab_hierarchy(roots[0], prefab_document["root_object"], roots[0].prefab_guid)
        current_token = None
        try:
            current_document, _, _, _ = self.capture_prefab_mode_document()
            current_token = document_content_token(current_document)
        except Exception as exc:
            Debug.log_suppressed("prefab_save.current_content_token", exc)
        from infernux.engine.interaction import DocumentKey, DocumentKind
        key = DocumentKey.asset(DocumentKind.PREFAB, prefab_guid)
        registry.complete_save(
            active_ticket_id,
            success=True,
            key=key,
            resource_path=prefab_path,
            title=os.path.splitext(os.path.basename(prefab_path))[0],
            content_token=current_token,
        )
        return True

    def save_scene_as(self):
        """Force a Save-As dialog regardless of whether a path exists."""
        if self._is_play_mode():
            Debug.log_warning("Cannot save scene while in Play mode.")
            return False
        if self.is_prefab_mode:
            return False
        from infernux.engine.interaction import DocumentRegistry

        return DocumentRegistry.instance().request_save(
            self.document_id,
            save_as=True,
        ).accepted

    def _do_save(self, path: str, *, ticket_id: str, document_id: str) -> bool:
        """Persist exactly the Scene and attempt selected by the Registry."""
        from infernux.engine.interaction import (
            DocumentIdentityKind, DocumentKind, DocumentRegistry, SaveTicketStatus,
            document_content_token,
        )
        from infernux.engine.ui.engine_status import EngineStatus

        registry = DocumentRegistry.instance()
        document = registry.get(document_id)
        ticket = registry.get_save_ticket(ticket_id)
        if (document is None or document.controller is not self or document.kind is not DocumentKind.SCENE
                or ticket is None or not ticket.is_pending or ticket.document_id != document_id):
            return False
        binding = self._binding_for_document(document_id)
        suspended = (document_id == self._previous_scene_document_id
                     and isinstance(self._previous_scene_document, dict))
        if binding is None and not suspended:
            registry.complete_save(ticket_id, success=False, cancelled=True,
                                   message="the scene document is no longer resident or suspended")
            return False
        # Prefab Mode retains the binding, but its native world is empty until
        # exit restores this snapshot. Identity does not imply live contents.
        scene = binding.scene if binding is not None and not suspended else None
        previous_name = scene.name if scene is not None else ""
        target = resolved_path(path)
        if not target.lower().endswith(SCENE_EXTENSION):
            target += SCENE_EXTENSION
        try:
            if not self._is_under_assets(target):
                raise ValueError("Scenes must be saved under the project's Assets directory")
            title = os.path.splitext(os.path.basename(target))[0]
            if scene is not None:
                scene.name = title
                payload = json.loads(scene.serialize_asset())
            else:
                from infernux.engine.scene_authoring import encode_scene_document

                payload = encode_scene_document(dict(self._previous_scene_document, name=title))
            serialized_token, committed_state = self._persist_scene_document(target, payload, ticket)
            if not ticket.is_pending:
                return False
            if registry.get(document_id) is not document or document.controller is not self:
                raise RuntimeError("the Scene document changed during asset publication")
            key = self._document_key("scene", target)
            if key.identity_kind is not DocumentIdentityKind.ASSET_GUID:
                raise RuntimeError("saved scene was not registered")
            current_token = serialized_token
            if scene is not None:
                current_token = document_content_token(scene.serialize_asset_document())
            completed = registry.complete_save(
                ticket_id, success=True, key=key, resource_path=target, title=title,
                content_token=current_token, committed_file_state=committed_state,
            )
            if completed.status is not SaveTicketStatus.SUCCEEDED:
                return False
        except Exception as exc:
            if scene is not None:
                scene.name = previous_name
            registry.complete_save(
                ticket_id, success=False,
                conflict="changed outside the editor" in str(exc).casefold(),
                message=str(exc),
            )
            Debug.log_error(f"Failed to save Scene '{target}': {exc}")
            EngineStatus.flash("保存失败 Save Failed", 0.0, duration=2.0)
            return False

        if binding is not None:
            binding.resource_path = target
            binding.asset_guid = document.key.identity
        if suspended:
            self._previous_scene_path = target
            self._previous_scene_document = dict(self._previous_scene_document, name=title)
        if document_id == self.document_id:
            self._current_scene_path = target
            for label, persist in (("camera state", self._save_camera_state),
                                   ("recent Scene", self._remember_last_scene)):
                try:
                    persist(target)
                except Exception as exc:
                    Debug.log_warning(f"Scene saved, but Editor {label} was not recorded: {exc}")
        EngineStatus.flash("保存完成 Saved", 1.0, duration=1.5)
        return True

    def _persist_scene_document(self, path: str, payload: dict, ticket):
        """Use one conditional write/commit path for resident and suspended Scenes."""
        from infernux.core.assets import AssetManager
        from infernux.core.document_store import capture_document_file_state, write_document_text
        from infernux.engine.interaction import DocumentRegistry, document_content_token

        registry = DocumentRegistry.instance()
        content_token = document_content_token(payload)
        registry.capture_save_revision(ticket.ticket_id, content_token=content_token)
        expected_state = registry.capture_save_target(ticket.ticket_id, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_document_text(
            path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            expected_file_state=expected_state, commit_chain_token=ticket.commit_token,
        )
        committed_state = AssetManager.register_local_commit(
            path, commit_token=ticket.commit_token, content_token=content_token,
            file_state=capture_document_file_state(path),
            edit_revision=ticket.captured_revision, document_id=ticket.document_id,
        )
        if self._asset_database is not None:
            publish = (AssetManager.reimport_asset if self._asset_database.contains_path(path)
                       else AssetManager.import_asset)
            result = publish(path, database=self._asset_database)
            if not result:
                raise RuntimeError(getattr(result, "error", "") or "Scene asset publication was rejected")
        return content_token, committed_state

    def _default_scene_save_path(self) -> Optional[str]:
        """Return a unique default scene path under Assets/ for untitled saves."""
        root = _effective_project_root()
        if not root:
            return None

        assets_dir = os.path.join(root, "Assets")
        os.makedirs(assets_dir, exist_ok=True)

        base_name = DEFAULT_SCENE_FILE_BASE
        candidate = os.path.join(assets_dir, f"{base_name}{SCENE_EXTENSION}")
        if not os.path.exists(candidate):
            return candidate

        index = 1
        while True:
            candidate = os.path.join(assets_dir, f"{base_name} {index}{SCENE_EXTENSION}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _show_save_as_dialog(self, *, document_id: str) -> bool:
        """Open the appropriate Save As workflow for a user or automation agent."""
        if self._save_as_popup_open or self._save_as_native_dialog_pending:
            return False
        root = _effective_project_root()
        if not root:
            Debug.log_warning("No project root set — cannot save scene.")
            return False

        modal_service = self._ensure_save_as_modal_presenter()
        if modal_service is None:
            Debug.log_warning("Scene Save As requires an active Editor interaction session.")
            return False
        if not modal_service.activate(
            self._SAVE_AS_MODAL_ID,
            owner_id="scene",
        ):
            return False

        from infernux.engine.interaction import DocumentRegistry

        document = DocumentRegistry.instance().get(document_id)
        document_path = document.resource_path if document is not None else ""
        if document_path:
            default_name = os.path.splitext(os.path.basename(document_path))[0]
        else:
            default_name = DEFAULT_SCENE_FILE_BASE

        self._save_as_folder = "Assets"
        self._save_as_name = default_name
        self._save_as_error = ""
        self._save_as_agent_modal = is_synthetic_input_frame()
        self._save_as_focus_name = self._save_as_agent_modal
        self._save_as_popup_requested = self._save_as_agent_modal
        self._save_as_popup_open = self._save_as_agent_modal
        self._save_as_native_dialog_pending = not self._save_as_agent_modal
        return True

    def render_save_as_popup(self, ctx) -> None:
        """Render Scene Save As through the global ModalPortal."""
        if self._save_as_native_dialog_pending:
            self._save_as_native_dialog_pending = False
            self._save_with_native_dialog()
            return

        if not self._save_as_popup_open:
            return

        from infernux.engine.ui.editor_modal import (
            EditorModalAction,
            begin_editor_modal,
            end_editor_modal,
            render_editor_modal_actions,
        )

        request_open = self._save_as_popup_requested
        self._save_as_popup_requested = False
        if not begin_editor_modal(
            ctx,
            popup_id="Save Scene As###scene_save_as",
            title="Save Scene As",
            semantic_id="scene.save_as",
            request_open=request_open,
            height=280.0,
        ):
            return
        ctx.label("保存场景到项目 Assets 目录")
        ctx.label("Save the scene under this project's Assets directory.")
        ctx.spacing()

        self._save_as_folder = ctx.text_input(
            "Folder##scene_save_as_folder", self._save_as_folder, 512
        )
        ctx.record_semantic_item("text_input", "Folder", True, "scene.save_as.folder")
        if self._save_as_focus_name:
            ctx.set_keyboard_focus_here()
            self._save_as_focus_name = False
        self._save_as_name = ctx.text_input(
            "Name##scene_save_as_name", self._save_as_name, 256
        )
        ctx.record_semantic_item("text_input", "Name", True, "scene.save_as.name")

        if self._save_as_error:
            ctx.spacing()
            ctx.text_wrapped(self._save_as_error)

        ctx.spacing()
        ctx.separator()
        ctx.spacing()

        def _save() -> None:
            path, error = self._resolve_save_as_path()
            if error:
                self._save_as_error = error
                return
            if not self._save_as_path(path):
                return
            self._close_save_as_popup(ctx)

        def _cancel() -> None:
            ctx.close_current_popup()
            self._cancel_save_as()

        render_editor_modal_actions(
            ctx,
            [
                EditorModalAction("Save", "confirm", _save),
                EditorModalAction("Cancel", "cancel", _cancel),
            ],
            semantic_prefix="scene.save_as",
        )
        end_editor_modal(ctx)

    def _resolve_save_as_path(self) -> tuple[str, str]:
        root = _effective_project_root()
        if not root:
            return "", "No project root is available."

        folder = str(self._save_as_folder or "").strip().replace("\\", "/")
        if not folder:
            folder = "Assets"
        if os.path.isabs(folder):
            return "", "Folder must be a project-relative path under Assets."

        target_folder = resolved_path(os.path.join(root, folder))
        if not self._is_under_assets(target_folder):
            return "", "Scenes must be saved under the project's Assets directory."

        name = str(self._save_as_name or "").strip()
        if name.lower().endswith(SCENE_EXTENSION):
            name = name[: -len(SCENE_EXTENSION)]
        if not name:
            return "", "Enter a scene name."
        if name != os.path.basename(name) or any(ch in name for ch in '<>:"/\\|?*'):
            return "", "Scene name contains an invalid path or filename character."

        return os.path.join(target_folder, name + SCENE_EXTENSION), ""

    def _resolve_native_save_as_path(self, path: str) -> tuple[str, str]:
        """Validate a platform dialog result using the same Assets boundary."""
        target = resolved_path(str(path or ""))
        if not target.lower().endswith(SCENE_EXTENSION):
            target += SCENE_EXTENSION
        if not self._is_under_assets(target):
            return "", "Scenes must be saved under the project's Assets directory."
        return target, ""

    def _save_with_native_dialog(self) -> None:
        root = _effective_project_root()
        if not root:
            return

        path = save_file_dialog(
            title="Save Scene As",
            win32_filter="Scene (*.scene)\0*.scene\0\0",
            initial_dir=os.path.join(root, "Assets"),
            default_filename=f"{self._save_as_name}{SCENE_EXTENSION}",
            default_ext=SCENE_EXTENSION.lstrip("."),
            tk_filetypes=[("Scene (*.scene)", "*.scene")],
        )
        if not path:
            self._cancel_save_as()
            return

        path, error = self._resolve_native_save_as_path(path)
        if error:
            Debug.log_warning(error)
            self._cancel_save_as(message=error)
            return
        if not self._save_as_path(path):
            message = self._save_as_error or "The scene could not be saved. Check the Console for details."
            Debug.log_warning(message)
            self._cancel_save_as(message=message)
            return
        self._finish_save_as()

    def _save_as_path(self, path: str) -> bool:
        from infernux.engine.interaction import DocumentRegistry, SaveTicketStatus

        registry = DocumentRegistry.instance()
        anchor = registry.get_save_ticket(self._save_as_ticket_id)
        document = registry.get(anchor.document_id) if anchor is not None else None
        if document is None or document.controller is not self:
            self._cancel_save_as(message="the Save As Scene is no longer open")
            return False
        if os.path.exists(path) and path_key(path) != path_key(document.resource_path):
            self._save_as_error = "A scene already exists at this location. Choose another name to avoid overwriting it."
            return False
        if anchor.is_pending:
            result = registry._invoke_save_controller(
                anchor, lambda: self.save_to_resource(ticket=anchor, resource_path=path),
            )
        elif anchor.status is SaveTicketStatus.FAILED:
            if registry.active_save_ticket(document.document_id) is not None:
                self._save_as_error = "A Scene save is already pending."
                return False
            result = registry.request_save_to_resource(document.document_id, path)
        else:
            self._save_as_error = "The Scene Save As session has ended."
            return False
        if not result.accepted:
            self._save_as_error = result.message or "The scene could not be saved. Check the Console for details."
            return False
        self._save_as_ticket_id = ""
        return True

    def _cancel_save_as(self, *, message: str = "save was cancelled") -> None:
        ticket_id = self._save_as_ticket_id
        self._reset_save_as_state()
        if ticket_id:
            from infernux.engine.interaction import DocumentRegistry

            DocumentRegistry.instance().complete_save(
                ticket_id, success=False, cancelled=True, message=message,
            )

    def _close_save_as_popup(self, ctx) -> None:
        ctx.close_current_popup()
        self._finish_save_as()

    def _finish_save_as(self) -> None:
        self._reset_save_as_state()

    def _reset_save_as_state(self) -> None:
        self._save_as_ticket_id = ""
        self._save_as_popup_open = False
        self._save_as_popup_requested = False
        self._save_as_focus_name = False
        self._save_as_agent_modal = False
        self._save_as_native_dialog_pending = False
        self._save_as_error = ""
        if self._save_as_modal_service is not None:
            self._save_as_modal_service.deactivate(self._SAVE_AS_MODAL_ID)

    def _is_under_assets(self, path: str) -> bool:
        """Check if *path* is within the project's Assets/ directory."""
        root = _effective_project_root()
        if not root:
            return False
        return is_path_within(path, os.path.join(root, "Assets"))

    def _remember_last_scene(self, path: str):
        guid = ""
        get_guid = getattr(self._asset_database, "get_guid_from_path", None)
        if callable(get_guid):
            guid = str(get_guid(path) or "").strip()
        if not guid:
            Debug.log_warning(f"Cannot remember scene without an AssetDatabase GUID: {path}")
            return
        settings = _load_editor_settings()
        settings[LAST_OPENED_SCENE_GUID_KEY] = guid
        _save_editor_settings(settings)

