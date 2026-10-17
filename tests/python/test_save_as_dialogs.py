from __future__ import annotations
from types import SimpleNamespace
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.interaction import (
    DocumentCapability,
    DocumentKind,
    DocumentRegistry,
    EditorInteractionCore,
)
from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator
import infernux.engine._scene_save as scene_save


class _TestResourceDocumentController:
    def __init__(self, discard=None) -> None:
        self._discard = discard

    def discard(self, *, document_id):
        del document_id
        if not callable(self._discard):
            return False
        return self._discard()


def _open_test_dirty_document(panel_id: str, title: str, *, discard=None):
    registry = DocumentRegistry.instance()
    registry.close_view(panel_id)
    document = registry.create(
        DocumentKind.GENERIC,
        title,
        revision=1,
        saved_revision=0,
        capabilities=(DocumentCapability.DISCARD if callable(discard) else DocumentCapability.NONE),
        controller=_TestResourceDocumentController(discard),
    )
    registry.attach_view(document.document_id, panel_id)
    return document


def _mark_test_document_saved(panel_id: str) -> None:
    registry = DocumentRegistry.instance()
    document = registry.document_for_view(panel_id)
    assert document is not None
    registry.mark_saved(document.document_id)


def _scene_manager() -> SceneFileManager:
    previous = SceneFileManager._instance
    previous_core = EditorInteractionCore._instance
    previous_confirmation = DirtyPanelConfirmationCoordinator._instance
    core = EditorInteractionCore()
    DirtyPanelConfirmationCoordinator._instance = None
    manager = SceneFileManager()
    manager._test_previous_instance = previous
    manager._test_interaction_core = core
    manager._test_previous_interaction_core = previous_core
    manager._test_previous_confirmation = previous_confirmation
    return manager


def _restore_scene_manager(manager: SceneFileManager) -> None:
    DirtyPanelConfirmationCoordinator._instance = manager._test_previous_confirmation
    manager._test_interaction_core.shutdown()
    EditorInteractionCore._instance = manager._test_previous_interaction_core
    SceneFileManager._instance = manager._test_previous_instance


def _set_scene_dirty(manager: SceneFileManager, dirty: bool) -> None:
    registry = DocumentRegistry.instance()
    document = registry.require(manager.document_id)
    if dirty and not document.is_dirty:
        registry.mark_changed(document.document_id)
    elif not dirty and document.is_dirty:
        registry.restore_saved_revision(document.document_id)


def test_unsaved_scene_agent_save_uses_editor_owned_save_as_state(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_save, "_effective_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(scene_save, "is_synthetic_input_frame", lambda: True)
    manager = _scene_manager()
    try:
        manager._current_scene_path = None
        manager._show_save_as_dialog()

        assert manager._save_as_popup_open is True
        assert manager._save_as_popup_requested is True
        assert manager._save_as_focus_name is True
        assert manager._save_as_folder == "Assets"
        assert manager._save_as_name == "UntitledScene"
        assert manager._test_interaction_core.modals.active_modal_id == "scene.save_as"
    finally:
        _restore_scene_manager(manager)


def test_unsaved_scene_user_save_uses_native_dialog(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_save, "_effective_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(scene_save, "is_synthetic_input_frame", lambda: False)
    target = tmp_path / "Assets" / "Scenes" / "RacingEntry.scene"
    saved: list[str] = []
    monkeypatch.setattr(scene_save, "save_file_dialog", lambda **_kwargs: str(target))
    manager = _scene_manager()
    try:
        manager._current_scene_path = None
        manager._do_save = lambda path, **_kwargs: saved.append(path) or True
        manager._show_save_as_dialog()

        assert manager._save_as_popup_open is False
        assert manager._save_as_native_dialog_pending is True

        manager._test_interaction_core.modals.render(None)

        assert saved == [str(target)]
        assert manager._test_interaction_core.modals.active_modal_id == ""
    finally:
        _restore_scene_manager(manager)


def test_scene_save_as_nests_under_unsaved_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_save, "_effective_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(scene_save, "is_synthetic_input_frame", lambda: True)
    manager = _scene_manager()
    parent_active = {"value": True}
    modals = manager._test_interaction_core.modals
    modals.register(
        "editor.unsaved_changes",
        is_active=lambda: parent_active["value"],
        render=lambda _ctx: None,
        cancel=lambda: parent_active.update(value=False),
    )
    try:
        assert modals.activate("editor.unsaved_changes", owner_id="scene")
        assert manager._show_save_as_dialog()
        assert [entry.modal_id for entry in modals.active_stack] == [
            "editor.unsaved_changes",
            "scene.save_as",
        ]
    finally:
        _restore_scene_manager(manager)


def _enter_fake_prefab_mode(manager: SceneFileManager, prefab_path: str) -> tuple[str, str]:
    from infernux.engine.interaction import DocumentKey, DocumentKind

    previous_document_id = manager.document_id
    manager._previous_scene_document_id = previous_document_id
    manager._previous_scene_path = manager._current_scene_path
    manager.is_prefab_mode = True
    manager.prefab_mode_guid = "interaction-prefab-guid"
    manager.prefab_mode_path = prefab_path
    manager._current_scene_path = prefab_path
    manager._replace_scene_document(
        kind="prefab",
        resource_path=prefab_path,
        title="InteractionPrefab",
        dirty=True,
        preserve_previous=True,
        key_override=DocumentKey.asset(DocumentKind.PREFAB, manager.prefab_mode_guid),
    )
    return previous_document_id, manager.document_id


def test_dirty_prefab_exit_waits_for_explicit_close_decision(tmp_path, monkeypatch):
    manager = _scene_manager()
    coordinator = DirtyPanelConfirmationCoordinator.instance()
    scheduled: list[object] = []
    saved: list[str] = []
    try:
        if coordinator.is_active:
            coordinator.choose_cancel()
        prefab_path = str(tmp_path / "Assets" / "Interaction.prefab")
        _previous_id, prefab_id = _enter_fake_prefab_mode(manager, prefab_path)
        monkeypatch.setattr(
            manager,
            "_schedule_prefab_exit",
            lambda callback=None, **_kwargs: scheduled.append(callback) or True,
        )

        def save_prefab(*, ticket_id=""):
            saved.append(ticket_id)
            registry = DocumentRegistry.instance()
            registry.capture_save_revision(ticket_id)
            registry.complete_save(ticket_id, success=True)
            return True

        monkeypatch.setattr(manager, "_save_prefab", save_prefab)

        assert manager.exit_prefab_mode() is True
        assert coordinator.active_document_id == prefab_id
        assert scheduled == []
        assert saved == []

        coordinator.choose_cancel()
        assert coordinator.is_active is False
        assert scheduled == []
        assert saved == []

        assert manager.exit_prefab_mode() is True
        coordinator.choose_discard()
        assert coordinator.is_active is False
        assert len(scheduled) == 1
        assert saved == []

        scheduled.clear()
        assert manager.exit_prefab_mode() is True
        coordinator.choose_save()
        assert coordinator.is_active is False
        assert len(saved) == 1
        assert len(scheduled) == 1
    finally:
        if coordinator.is_active:
            coordinator.choose_cancel()
        _restore_scene_manager(manager)


def test_prefab_exit_records_history_only_after_deferred_transition_completes(
    tmp_path,
    monkeypatch,
):
    from infernux.engine.interaction import EditorContextSnapshot, PrefabCommandService
    from infernux.engine.undo import PrefabModeCommand, UndoManager

    manager = _scene_manager()
    previous_undo = UndoManager._instance
    undo = UndoManager()
    previous_service = PrefabCommandService._instance
    service = PrefabCommandService(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        context_provider=lambda: EditorContextSnapshot(),
    )
    requested: list[tuple[object, bool]] = []
    try:
        prefab_path = str(tmp_path / "Assets" / "Interaction.prefab")
        _enter_fake_prefab_mode(manager, prefab_path)
        monkeypatch.setattr(
            manager,
            "_request_prefab_exit",
            lambda on_complete=None, *, preserve_undo_history=False: (
                requested.append((on_complete, preserve_undo_history)) or True
            ),
        )

        assert service.exit() is True
        assert undo.action_journal.applied_entries() == ()
        assert len(requested) == 1
        callback, preserve_history = requested[0]
        assert preserve_history is True

        manager.is_prefab_mode = False
        callback()

        entries = undo.action_journal.applied_entries()
        assert len(entries) == 1
        assert isinstance(entries[0].action, PrefabModeCommand)
        assert entries[0].action.description == "Exit Prefab Mode"
    finally:
        service.shutdown()
        PrefabCommandService._instance = previous_service
        UndoManager._instance = previous_undo
        _restore_scene_manager(manager)


def test_dirty_panel_cancel_releases_native_close_request(monkeypatch):
    manager = _scene_manager()
    panel_id = "scene_close_dirty_panel_cancel"

    class _Native:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def confirm_close(self) -> None:
            self.calls.append("confirm")

        def cancel_close(self) -> None:
            self.calls.append("cancel")

    native = _Native()
    coordinator = DirtyPanelConfirmationCoordinator.instance()
    try:
        assert coordinator.is_active is False
        manager._engine = native
        _set_scene_dirty(manager, False)
        monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
        _open_test_dirty_document(panel_id, "Animation Editor")
        manager.request_close()
        coordinator.choose_cancel()

        assert native.calls == ["cancel"]
        assert manager._close_in_progress is False
        assert coordinator.is_active is False
    finally:
        if coordinator.is_active:
            coordinator.choose_cancel()
        DocumentRegistry.instance().close_view(panel_id)
        _restore_scene_manager(manager)
