"""One project-wide authoring document for Editor project settings."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Optional
from uuid import uuid4

from infernux.core.document_store import read_document_text_snapshot, submit_document_text
from infernux.engine.build_settings import (
    BUILD_SETTINGS_DEFAULTS,
    _json_copy,
    normalize_build_settings,
)
from infernux.engine.path_utils import resolved_path

from .documents import (
    DocumentActionResult,
    DocumentActionStatus,
    DocumentCapability,
    DocumentKey,
    DocumentKind,
    DocumentRegistry,
    DocumentState,
    document_content_token,
)


_SECTION_FILENAMES = {
    "build": "BuildSettings.json",
    "tag_layers": "TagLayerSettings.json",
    "physics": "PhysicsSettings.json",
}


@dataclass(frozen=True)
class _PendingSettingsWrite:
    tickets: dict[str, Any]
    document: dict[str, Any]
    save_ticket_id: str = ""
    submission_errors: dict[str, str] = field(default_factory=dict)


class ProjectSettingsDocumentController:
    """Own all project setting sections and their live runtime projection."""

    def __init__(
        self,
        project_path: str,
        *,
        tag_layer_manager: Any = None,
        physics_module: Any = None,
        submitter: Callable[..., Any] = submit_document_text,
    ) -> None:
        root = resolved_path(project_path)
        if not root:
            raise ValueError("project settings require a project path")
        self.project_path = root
        self.settings_path = os.path.join(root, "ProjectSettings")
        self.document_id = ""
        self._tag_layer_manager = tag_layer_manager
        self._physics_module = physics_module
        self._submitter = submitter
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._pending_writes: dict[int, _PendingSettingsWrite] = {}
        self._failed_sections: set[str] = set()
        self._commit_chain_token = uuid4().hex
        self._minimum_revision = 0
        self._document, self._file_states = self._load_document()
        self._saved_document = copy.deepcopy(self._document)
        self._apply_runtime(self._document)

    def _manager(self):
        if self._tag_layer_manager is None:
            from infernux.lib import TagLayerManager

            self._tag_layer_manager = TagLayerManager.instance()
        return self._tag_layer_manager

    def _physics(self):
        if self._physics_module is None:
            from infernux.physics import settings

            self._physics_module = settings
        return self._physics_module

    def _section_path(self, section: str) -> str:
        return os.path.join(self.settings_path, _SECTION_FILENAMES[section])

    def _load_document(self) -> tuple[dict[str, Any], dict[str, Any]]:
        from infernux.physics.settings import DEFAULT_PHYSICS_SETTINGS

        defaults = {
            "build": BUILD_SETTINGS_DEFAULTS,
            "tag_layers": json.loads(self._manager().serialize_defaults()),
            "physics": DEFAULT_PHYSICS_SETTINGS,
        }
        sections, states = {}, {}
        for section in _SECTION_FILENAMES:
            text, states[section] = read_document_text_snapshot(self._section_path(section))
            sections[section] = json.loads(text) if text is not None else copy.deepcopy(defaults[section])
        return self._normalize_document(sections), states

    def _normalize_document(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != set(_SECTION_FILENAMES):
            raise ValueError(
                "project settings must contain build, tag_layers, and physics sections"
            )
        build = normalize_build_settings(value["build"], project_path=self.project_path)
        tag_layers = _json_copy(value["tag_layers"])
        if not isinstance(tag_layers, dict):
            raise TypeError("tag/layer settings must be a JSON object")
        physics = self._physics().normalize(value["physics"])
        return {
            "build": build,
            "tag_layers": tag_layers,
            "physics": _json_copy(physics),
        }

    def _apply_runtime(
        self,
        document: dict[str, Any],
        previous: Optional[dict[str, Any]] = None,
    ) -> None:
        if previous is None or previous["tag_layers"] != document["tag_layers"]:
            tag_payload = json.dumps(
                document["tag_layers"], ensure_ascii=False, allow_nan=False
            )
            if self._manager().deserialize(tag_payload) is False:
                raise ValueError("tag/layer settings were rejected by the runtime")
        if previous is None or previous["physics"] != document["physics"]:
            self._physics().apply(document["physics"])

    def add_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass

    def _notify(self) -> None:
        snapshot = self.capture_document()
        for callback in tuple(self._listeners):
            callback(copy.deepcopy(snapshot))

    def capture_document(self) -> dict[str, Any]:
        return copy.deepcopy(self._document)

    def section(self, name: str) -> dict[str, Any]:
        if name not in _SECTION_FILENAMES:
            raise KeyError(f"unknown project settings section: {name}")
        return copy.deepcopy(self._document[name])

    def restore_document(
        self,
        document: dict[str, Any],
        revision: Optional[int],
        *,
        persist: bool = True,
    ) -> None:
        if revision is not None and revision < self._minimum_revision:
            raise RuntimeError("Project settings history predates the last durable reload")
        normalized = self._normalize_document(document)
        changed = tuple(
            section for section in _SECTION_FILENAMES
            if normalized[section] != self._document[section]
        )
        self._apply_runtime(normalized, self._document)
        self._document = normalized
        if revision is not None:
            DocumentRegistry.instance().restore_content_revision(
                self.document_id, int(revision)
            )
        self._notify()
        if persist:
            self.schedule_autosave(sections=changed)

    def apply_document(
        self,
        document: dict[str, Any],
        *,
        edit_key: str,
        description: str,
        view_id: str = "",
    ) -> bool:
        from infernux.engine.undo import EditableDocumentDraftCommand, UndoManager

        previous = self.capture_document()
        following = self._normalize_document(document)
        if previous == following:
            return False
        registry = DocumentRegistry.instance()
        editor_document = registry.require(self.document_id)
        manager = UndoManager.instance()
        if manager is None or not manager.enabled or manager.is_executing:
            return False
        next_revision = registry.reserve_changed_revision(
            self.document_id,
            view_id=view_id,
        )
        return bool(
            manager.execute(
                EditableDocumentDraftCommand(
                    self,
                    previous,
                    following,
                    editor_document.revision,
                    next_revision,
                    edit_key=f"{self._minimum_revision}:{edit_key or 'project_settings'}",
                    description=str(description or "Edit Project Settings"),
                )
            )
        )

    def apply_section(
        self,
        name: str,
        value: dict[str, Any],
        *,
        edit_key: str,
        description: str,
        view_id: str = "",
    ) -> bool:
        document = self.capture_document()
        if name not in document:
            raise KeyError(f"unknown project settings section: {name}")
        document[name] = copy.deepcopy(value)
        return self.apply_document(
            document,
            edit_key=edit_key,
            description=description,
            view_id=view_id,
        )

    def apply_derived_section(self, name: str, value: dict[str, Any]) -> bool:
        """Apply a consequence already owned by another journal command.

        Asset rename/move commands, for example, also update Build Settings
        scene references. The asset command remains the one user action; this
        method keeps the shared settings document and asynchronous persistence
        current without publishing a duplicate action.
        """
        document = self.capture_document()
        if name not in document:
            raise KeyError(f"unknown project settings section: {name}")
        document[name] = copy.deepcopy(value)
        following = self._normalize_document(document)
        if following == self._document:
            return False
        registry = DocumentRegistry.instance()
        revision = registry.reserve_changed_revision(
            self.document_id,
            view_id="build_settings" if name == "build" else "",
        )
        self.restore_document(following, revision, persist=True)
        return True

    @staticmethod
    def _ticket_status(ticket: Any) -> str:
        return str(getattr(ticket, "status", "") or "").rsplit(".", 1)[-1].lower()

    def _submit_snapshot(
        self,
        document: dict[str, Any],
        *,
        sections: tuple[str, ...],
        save_ticket_id: str = "",
    ) -> _PendingSettingsWrite:
        payloads = {
            section: json.dumps(document[section], indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            for section in sections
        }
        tickets, errors = {}, {}
        if sections:
            try:
                os.makedirs(self.settings_path, exist_ok=True)
            except OSError as exc:
                errors.update({section: str(exc) for section in sections})
                self._failed_sections.update(sections)
                payloads.clear()
        for section, payload in payloads.items():
            try:
                tickets[section] = self._submitter(
                    self._section_path(section), payload,
                    expected_file_state=self._file_states[section],
                    commit_chain_token=self._commit_chain_token,
                )
            except Exception as exc:
                # Retain already submitted tickets: a partial submission must
                # never lose the actual durable state of another section.
                errors[section] = str(exc)
                self._failed_sections.add(section)
        pending = _PendingSettingsWrite(
            tickets=tickets,
            document={section: copy.deepcopy(document[section]) for section in sections},
            save_ticket_id=str(save_ticket_id or ""),
            submission_errors=errors,
        )
        self._pending_writes[id(pending)] = pending
        return pending

    def _unsaved_sections(self) -> tuple[str, ...]:
        in_flight = {
            section for pending in self._pending_writes.values()
            for section in (pending.tickets.keys() | pending.submission_errors.keys())
        }
        return tuple(
            section for section in _SECTION_FILENAMES
            if self._document[section] != self._saved_document[section]
            or section in in_flight or section in self._failed_sections
        )

    def schedule_autosave(self, *, sections: Optional[tuple[str, ...]] = None) -> bool:
        DocumentRegistry.instance().require(self.document_id)
        pending = self._submit_snapshot(
            self.capture_document(),
            sections=self._unsaved_sections() if sections is None else sections,
        )
        return not pending.submission_errors

    def poll_pending_writes(self) -> int:
        registry = DocumentRegistry.instance()
        completed = 0
        for key, pending in tuple(self._pending_writes.items()):
            if not all(ticket.is_complete for ticket in pending.tickets.values()):
                # Consume snapshots in submission order, including batches
                # involving different files. Never regress a saved baseline.
                break
            statuses = tuple(self._ticket_status(ticket) for ticket in pending.tickets.values())
            succeeded = not pending.submission_errors and all(status == "succeeded" for status in statuses)
            errors = [f"{_SECTION_FILENAMES[section]}: {error}"
                      for section, error in pending.submission_errors.items()]
            self._failed_sections.update(pending.submission_errors)
            for section, ticket in pending.tickets.items():
                status = self._ticket_status(ticket)
                if status == "succeeded":
                    self._saved_document[section] = copy.deepcopy(pending.document[section])
                    self._file_states[section] = ticket.committed_file_state
                    self._failed_sections.discard(section)
                elif status != "superseded":
                    self._failed_sections.add(section)
                    errors.append(f"{_SECTION_FILENAMES[section]}: {status}: {ticket.error}")
            message = "project settings persistence failed: " + "; ".join(errors or statuses)
            if pending.save_ticket_id:
                save_ticket = registry.get_save_ticket(pending.save_ticket_id)
                if save_ticket is not None and save_ticket.is_pending:
                    registry.complete_save(
                        pending.save_ticket_id,
                        success=succeeded,
                        content_token=(
                            document_content_token(self.capture_document())
                            if succeeded
                            else None
                        ),
                        message="" if succeeded else message,
                    )
            if errors:
                from infernux.debug import Debug

                Debug.log_error(message)
                if any("target changed" in error for error in errors):
                    registry.mark_conflict(self.document_id)
            self._pending_writes.pop(key, None)
            completed += 1
        editor_document = registry.get(self.document_id)
        if (
            completed and editor_document is not None and not self._pending_writes
            and registry.active_save_ticket(self.document_id) is None
        ):
            if self._document == self._saved_document and not self._failed_sections:
                if editor_document.state is not DocumentState.CONFLICT:
                    registry.mark_saved(self.document_id)
            elif not editor_document.is_dirty:
                # Undo can return to an old revision while a newer autosave
                # has already reached disk. A failed undo write is still dirty.
                registry.mark_changed(self.document_id)
        return completed

    def save(self, *, ticket, save_as: bool = False):
        if save_as:
            return False
        registry = DocumentRegistry.instance()
        document = self.capture_document()
        registry.capture_save_revision(
            ticket.ticket_id,
            content_token=document_content_token(document),
        )
        self._submit_snapshot(
            document,
            sections=self._unsaved_sections(),
            save_ticket_id=ticket.ticket_id,
        )
        self.poll_pending_writes()
        return (
            ticket.status.value == "succeeded"
            if not ticket.is_pending
            else DocumentActionResult(DocumentActionStatus.PENDING)
        )

    def poll_save(self, ticket):
        self.poll_pending_writes()
        current = DocumentRegistry.instance().get_save_ticket(ticket.ticket_id)
        if current is None or current.is_pending:
            return None
        return current.status.value == "succeeded"

    def reload_from_resource(self, *, document_id: str, resource_path: str = ""):
        if str(document_id or "") != self.document_id:
            return False
        self.poll_pending_writes()
        if self._pending_writes:
            return DocumentActionResult(DocumentActionStatus.REJECTED, "Wait for pending settings writes before reloading")
        registry = DocumentRegistry.instance()
        if registry.active_save_ticket(self.document_id) is not None:
            return False
        document, states = self._load_document()
        self._apply_runtime(document, self._document)
        self._document = document
        self._saved_document = copy.deepcopy(document)
        self._file_states = states
        self._failed_sections.clear()
        self._commit_chain_token = uuid4().hex
        self._minimum_revision = registry.establish_loaded_baseline(self.document_id)
        self._notify()
        return True

    def discard(self, *, document_id: str):
        # Discard is a read. It must not restore cached bytes over another
        # author's version, even if only one of the settings files changed.
        return self.reload_from_resource(document_id=document_id)

    def resource_moved(self, **_kwargs) -> None:
        raise RuntimeError("project settings cannot be moved as an asset")


def ensure_project_settings_document(
    project_path: str,
    *,
    view_id: str = "",
    tag_layer_manager: Any = None,
    physics_module: Any = None,
) -> ProjectSettingsDocumentController:
    root = resolved_path(project_path)
    settings_path = os.path.join(root, "ProjectSettings")
    key = DocumentKey.resource(DocumentKind.PROJECT_SETTINGS, settings_path)
    registry = DocumentRegistry.instance()
    document = registry.get_by_key(key)
    controller = document.controller if document is not None else None
    if controller is not None and not isinstance(
        controller, ProjectSettingsDocumentController
    ):
        # Older workspace snapshots could claim this shared document through a
        # generic Panel before that Panel had a chance to bind the real
        # ProjectSettings controller.  The document identity is authoritative;
        # the generic controller owns no durable Project Settings state, so
        # replace it instead of making editor startup depend on stale UI state.
        controller = ProjectSettingsDocumentController(
            root,
            tag_layer_manager=tag_layer_manager,
            physics_module=physics_module,
        )
        registry.update_metadata(
            document.document_id,
            title="Project Settings",
            resource_path=settings_path,
            capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
            controller=controller,
        )
        controller.document_id = document.document_id
    if controller is None:
        controller = ProjectSettingsDocumentController(
            root,
            tag_layer_manager=tag_layer_manager,
            physics_module=physics_module,
        )
        document = registry.create(
            DocumentKind.PROJECT_SETTINGS,
            "Project Settings",
            key=key,
            resource_path=settings_path,
            capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
            controller=controller,
        )
        controller.document_id = document.document_id
    if view_id:
        registry.attach_view(document.document_id, view_id)
    return controller


__all__ = [
    "BUILD_SETTINGS_DEFAULTS",
    "ProjectSettingsDocumentController",
    "ensure_project_settings_document",
    "normalize_build_settings",
]
