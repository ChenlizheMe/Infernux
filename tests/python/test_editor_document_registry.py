from __future__ import annotations

from pathlib import Path

import pytest

from infernux.engine.interaction import (
    DocumentActionResult,
    DocumentActionStatus,
    DocumentCapability,
    DocumentKey,
    DocumentKind,
    DocumentRegistry,
    DocumentState,
    SaveTicketStatus,
)


class _AssetDatabase:
    def __init__(self) -> None:
        self._paths: dict[str, str] = {}

    def register(self, guid: str, path) -> None:
        self._paths[str(guid)] = str(path)

    def get_path_from_guid(self, guid: str) -> str:
        return self._paths.get(str(guid), "")

    def get_guid_from_path(self, path: str) -> str:
        target = str(path)
        return next(
            (guid for guid, candidate in self._paths.items() if candidate == target),
            "",
        )


@pytest.fixture
def asset_database(monkeypatch):
    from infernux.core.assets import AssetManager

    database = _AssetDatabase()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    return database


class _Controller:
    def __init__(self, registry: DocumentRegistry, document_id: str) -> None:
        self.registry = registry
        self.document_id = document_id
        self.pending = False
        self.saved_revision = None
        self.discarded = False
        self.reloaded = False
        self.save_calls = 0
        self.ticket = None
        self.resource_path = ""

    def save(self, *, ticket, save_as: bool = False):
        del save_as
        self.save_calls += 1
        self.ticket = ticket
        if self.pending:
            return DocumentActionResult(DocumentActionStatus.PENDING)
        self.registry.capture_save_revision(ticket.ticket_id)
        if self.saved_revision is not None:
            ticket.captured_revision = self.saved_revision
        self.registry.complete_save(ticket.ticket_id, success=True)
        return True

    def discard(self, *, document_id: str):
        assert document_id == self.document_id
        self.discarded = True
        return True

    def reload_from_resource(self, *, document_id: str, resource_path: str):
        assert document_id == self.document_id
        self.resource_path = resource_path
        self.reloaded = True
        return True

    def save_to_resource(self, *, ticket, resource_path: str):
        self.resource_path = resource_path
        return self.save(ticket=ticket, save_as=True)


def _document(
    registry: DocumentRegistry,
    *,
    dirty: bool = False,
    kind: DocumentKind = DocumentKind.PARTICLE_GRAPH,
    document_id: str = "particle:smoke",
):
    document = registry.create(
        kind,
        "Smoke",
        document_id=document_id,
        revision=1 if dirty else 0,
        saved_revision=0,
        capabilities=(DocumentCapability.SAVE | DocumentCapability.SAVE_AS | DocumentCapability.DISCARD),
    )
    controller = _Controller(registry, document.document_id)
    registry.update_metadata(document.document_id, controller=controller)
    return document, controller


def test_external_change_replaces_dirty_asset_document_without_prompt(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Smoke.particlegraph"
    path.write_text("baseline", encoding="utf-8")
    document, controller = _document(registry, dirty=True)
    registry.rekey(
        document.document_id,
        DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid"),
        resource_path=str(path),
    )
    path.write_text("external-change", encoding="utf-8")

    assert registry.publish_external_resource_change(str(path), guid="particle-guid") == (document.document_id,)
    assert document.external_revision == 1
    assert document.state is DocumentState.READY
    assert controller.reloaded
    assert not document.is_dirty
    assert not controller.discarded

    result = registry.request_save(document.document_id)
    assert result.status is DocumentActionStatus.NO_OP
    assert controller.save_calls == 0


def test_sidecar_change_reloads_only_import_document_and_cancels_its_save(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Source.scene"
    path.write_text("unchanged source", encoding="utf-8")
    Path(str(path) + ".meta").write_text("changed sidecar", encoding="utf-8")
    source = registry.create(
        DocumentKind.SCENE,
        "Source",
        key=DocumentKey.asset(DocumentKind.SCENE, "shared-guid"),
        resource_path=str(path),
        revision=1,
        saved_revision=0,
    )
    settings = registry.create(
        DocumentKind.IMPORT_SETTINGS,
        "Import",
        key=DocumentKey.asset(DocumentKind.IMPORT_SETTINGS, "shared-guid"),
        resource_path=str(path),
        revision=1,
        saved_revision=0,
    )
    source_controller = _Controller(registry, source.document_id)
    settings_controller = _Controller(registry, settings.document_id)
    registry.update_metadata(source.document_id, controller=source_controller)
    registry.update_metadata(settings.document_id, controller=settings_controller)
    ticket = registry.begin_save(settings.document_id)
    assert registry.durable_resource_content_changed(str(path), guid="shared-guid") is False
    assert registry.preflight_external_resource_change(str(path), guid="shared-guid", metadata_only=True)
    assert ticket.status is SaveTicketStatus.CANCELLED
    registry.publish_external_resource_change(str(path), guid="shared-guid")
    assert settings_controller.reloaded and not settings.is_dirty
    assert source.is_dirty and source.state is DocumentState.READY
    assert not source_controller.reloaded


def test_external_asset_deletion_is_terminal_without_a_reload_conflict(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Smoke.particlegraph"
    path.write_text("baseline", encoding="utf-8")
    document, controller = _document(registry, dirty=True)
    registry.rekey(
        document.document_id,
        DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid"),
        resource_path=str(path),
    )
    path.unlink()

    assert registry.preflight_external_resource_change(str(path), guid="particle-guid", deleted=True)
    assert registry.publish_external_resource_change(
        str(path),
        guid="particle-guid",
        deleted=True,
    ) == (document.document_id,)
    assert document.state is DocumentState.READY
    assert not controller.reloaded


def test_external_scene_deletion_keeps_explicit_conflict_arbitration(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Level.scene"
    path.write_text("baseline", encoding="utf-8")
    document, _controller = _document(
        registry,
        kind=DocumentKind.SCENE,
    )
    registry.rekey(
        document.document_id,
        DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=str(path),
    )
    path.unlink()

    assert not registry.preflight_external_resource_change(
        str(path),
        guid="scene-guid",
        deleted=True,
    )
    assert document.state is DocumentState.CONFLICT


def test_external_change_during_asset_save_cancels_stale_write_and_reloads(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Smoke.particlegraph"
    path.write_text("baseline", encoding="utf-8")
    document, controller = _document(registry, dirty=True)
    registry.rekey(
        document.document_id,
        DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid"),
        resource_path=str(path),
    )
    controller.pending = True

    result = registry.request_save(document.document_id)
    assert result.status is DocumentActionStatus.PENDING
    ticket = registry.active_save_ticket(document.document_id)
    assert ticket is not None

    path.write_text("external-change", encoding="utf-8")
    registry.publish_external_resource_change(str(path), guid="particle-guid")
    completed = registry.complete_save(ticket.ticket_id, success=True)

    assert completed.status is SaveTicketStatus.CANCELLED
    assert controller.reloaded
    assert document.state is DocumentState.READY
    assert document.saved_revision == document.revision
    assert not document.is_dirty


def test_keep_local_acknowledges_conflict_before_intentional_overwrite(tmp_path):
    registry = DocumentRegistry()
    path = tmp_path / "Smoke.scene"
    path.write_text("baseline", encoding="utf-8")
    document, controller = _document(
        registry,
        dirty=True,
        kind=DocumentKind.SCENE,
    )
    registry.rekey(
        document.document_id,
        DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=str(path),
    )
    path.write_text("external-change", encoding="utf-8")
    registry.publish_external_resource_change(str(path), guid="scene-guid")

    resolved = registry.resolve_conflict_keep_local(document.document_id)
    saved = registry.request_save(document.document_id)

    assert resolved.status is DocumentActionStatus.APPLIED
    assert saved.status is DocumentActionStatus.APPLIED
    assert controller.save_calls == 1
    assert document.state is DocumentState.READY
    assert not document.is_dirty


def test_document_edit_save_undo_and_branch_preserve_save_authority():
    registry = DocumentRegistry()
    document, controller = _document(registry)
    original_revision = document.revision
    assert registry.request_save(document.document_id).status is DocumentActionStatus.NO_OP
    assert controller.save_calls == 0

    edited_revision = registry.mark_changed(document.document_id)
    assert edited_revision == 1 and document.saved_revision == 0
    assert document.is_dirty
    assert registry.request_save(document.document_id).status is DocumentActionStatus.APPLIED
    assert controller.save_calls == 1
    assert document.saved_revision == edited_revision
    assert not document.is_dirty

    registry.restore_content_revision(document.document_id, original_revision)
    assert document.revision == original_revision
    assert document.saved_revision == edited_revision and document.is_dirty
    registry.restore_content_revision(document.document_id, edited_revision)
    assert not document.is_dirty

    registry.restore_content_revision(document.document_id, original_revision)
    replacement_revision = registry.mark_changed(document.document_id)
    assert replacement_revision > edited_revision
    assert replacement_revision != document.saved_revision
    assert document.is_dirty


def test_deferred_save_preserves_edits_made_after_an_async_save_starts():
    registry = DocumentRegistry()
    document, controller = _document(registry, dirty=True)
    controller.pending = True

    registry.defer_save(document.document_id)
    captured_revision = registry.mark_changed(document.document_id)
    result = registry.process_deferred_saves()[0]
    newer_revision = registry.mark_changed(document.document_id)
    registry.complete_save(controller.ticket.ticket_id, success=True)

    assert result.status is DocumentActionStatus.PENDING
    assert document.saved_revision == captured_revision
    assert document.revision == newer_revision
    assert document.is_dirty is True


def test_async_save_dialog_commits_the_revision_actually_serialized():
    registry = DocumentRegistry()
    document, _controller = _document(registry, dirty=True)
    ticket = registry.begin_save(document.document_id, save_as=True)
    serialized_revision = registry.mark_changed(document.document_id)

    assert ticket.captured_revision != serialized_revision
    assert registry.capture_save_revision(ticket.ticket_id) == serialized_revision
    registry.complete_save(ticket.ticket_id, success=True)

    assert document.saved_revision == serialized_revision
    assert document.revision == serialized_revision
    assert document.is_dirty is False


def test_async_save_records_the_exact_captured_revision_after_undo():
    registry = DocumentRegistry()
    document, _controller = _document(registry)
    saved_revision = registry.mark_changed(document.document_id)
    ticket = registry.begin_save(document.document_id)
    newer_revision = registry.mark_changed(document.document_id)
    registry.restore_content_revision(document.document_id, 0)

    registry.complete_save(ticket.ticket_id, success=True)

    assert document.saved_revision == saved_revision
    assert document.revision == 0
    assert newer_revision > saved_revision
    assert document.is_dirty


def test_shared_asset_document_deduplicates_views_and_retires_after_last_close(tmp_path):
    registry = DocumentRegistry()
    key = DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid")
    resource_path = str(tmp_path / "Smoke.particlegraph")
    document, created = registry.open_or_create(key, "Smoke", resource_path=resource_path)
    reopened, created_again = registry.open_or_create(
        DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid"),
        "Smoke",
        resource_path=resource_path,
    )
    assert created and not created_again
    assert reopened is document and registry.get_by_key(key) is document
    registry.mark_changed(document.document_id)
    registry.attach_view(document.document_id, "particle-left")
    registry.attach_view(document.document_id, "particle-right")
    assert registry.document_for_view("particle-left") is document
    assert registry.document_for_view("particle-right") is document
    assert registry.dirty_documents() == (document,)

    registry.detach_view("particle-left")
    assert document.view_ids == {"particle-right"}
    registry.attach_view(document.document_id, "particle-left")
    assert registry.close_view("particle-left") == document.document_id
    assert registry.get(document.document_id) is document
    locator = registry.locate(document.document_id)
    assert registry.close_view("particle-right") == document.document_id
    assert registry.document_for_view("particle-right") is None
    assert registry.get(document.document_id) is None
    assert registry.resolve_locator(locator) is None


def test_save_as_rekeys_document_atomically_without_changing_document_id(tmp_path):
    registry = DocumentRegistry()
    document, _ = _document(registry, dirty=True)
    old_key = document.key
    new_path = tmp_path / "Saved.particlegraph"
    new_key = DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "saved-particle-guid")
    ticket = registry.begin_save(document.document_id, save_as=True)

    registry.complete_save(
        ticket.ticket_id,
        success=True,
        key=new_key,
        resource_path=str(new_path),
        title="Saved",
    )

    assert document.document_id == "particle:smoke"
    assert registry.get_by_key(old_key) is None
    assert registry.get_by_key(new_key) is document
    assert document.title == "Saved"
    assert not document.is_dirty


def test_save_as_key_collision_fails_without_rekeying_or_clearing_dirty(tmp_path):
    registry = DocumentRegistry()
    document, _ = _document(registry, dirty=True)
    original_key = document.key
    occupied_key = DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "occupied-particle-guid")
    registry.create(DocumentKind.PARTICLE_GRAPH, "Occupied", key=occupied_key)
    ticket = registry.begin_save(document.document_id, save_as=True)

    registry.complete_save(ticket.ticket_id, success=True, key=occupied_key)

    assert ticket.status is SaveTicketStatus.FAILED
    assert registry.get_by_key(original_key) is document
    assert document.is_dirty


def test_discard_is_rejected_while_the_document_has_an_active_save_ticket():
    registry = DocumentRegistry()
    document, controller = _document(registry, dirty=True)
    controller.pending = True

    assert registry.request_save(document.document_id).status is DocumentActionStatus.PENDING
    result = registry.request_discard(document.document_id)

    assert result.status is DocumentActionStatus.REJECTED
    assert "saving" in result.message
    assert controller.discarded is False
    assert document.is_dirty is True


def test_scene_undo_crosses_a_registry_save_point_without_moving_it():
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.engine.undo import SetPropertyCommand, UndoManager

    class Target:
        value = 0

    registry = DocumentRegistry.instance()
    previous_scene = SceneFileManager._instance
    previous_undo = UndoManager.instance()
    try:
        scene_files = SceneFileManager()
        manager = UndoManager()
        document = registry.require(scene_files.document_id)
        target = Target()

        first = SetPropertyCommand(target, "value", 0, 1)
        first.timestamp = 0.0
        assert manager.execute(first)
        saved_revision = document.revision
        registry.mark_saved(document.document_id, saved_revision)

        second = SetPropertyCommand(target, "value", 1, 2)
        second.timestamp = 10.0
        assert manager.execute(second)
        second_revision = document.revision
        assert second_revision > saved_revision

        manager.undo()
        assert target.value == 1
        assert document.revision == saved_revision
        assert document.saved_revision == saved_revision
        assert not document.is_dirty

        manager.undo()
        assert target.value == 0
        assert document.revision != saved_revision
        assert document.saved_revision == saved_revision
        assert document.is_dirty

        manager.redo()
        assert target.value == 1
        assert document.revision == saved_revision
        assert document.saved_revision == saved_revision
        assert not document.is_dirty
    finally:
        UndoManager._instance = previous_undo
        SceneFileManager._instance = previous_scene


def test_scene_command_merge_cannot_cross_completed_or_pending_save_points():
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.engine.undo import SetPropertyCommand, UndoManager

    class Target:
        value = 0

    registry = DocumentRegistry.instance()
    previous_scene = SceneFileManager._instance
    previous_undo = UndoManager.instance()
    try:
        scene_files = SceneFileManager()
        manager = UndoManager()
        document = registry.require(scene_files.document_id)
        target = Target()

        first = SetPropertyCommand(target, "value", 0, 1)
        first.timestamp = 1.0
        assert manager.execute(first)
        first_revision = document.revision

        pending = registry.begin_save(document.document_id)
        second = SetPropertyCommand(target, "value", 1, 2)
        second.timestamp = 1.1
        assert manager.execute(second)
        assert len(manager.action_journal.applied_entries()) == 2
        registry.complete_save(pending.ticket_id, success=True)
        assert document.saved_revision == first_revision
        assert document.is_dirty

        third = SetPropertyCommand(target, "value", 2, 3)
        third.timestamp = 2.0
        assert manager.execute(third)
        saved = registry.begin_save(document.document_id)
        registry.complete_save(saved.ticket_id, success=True)
        saved_revision = document.saved_revision

        fourth = SetPropertyCommand(target, "value", 3, 4)
        fourth.timestamp = 2.1
        assert manager.execute(fourth)
        assert len(manager.action_journal.applied_entries()) == 4
        assert document.saved_revision == saved_revision
        assert document.is_dirty
    finally:
        UndoManager._instance = previous_undo
        SceneFileManager._instance = previous_scene


def test_failed_scene_save_never_moves_the_registry_save_point():
    from infernux.engine.scene_manager import SceneFileManager
    from infernux.engine.undo import SetPropertyCommand, UndoManager

    class Target:
        value = 0

    registry = DocumentRegistry.instance()
    previous_scene = SceneFileManager._instance
    previous_undo = UndoManager.instance()
    try:
        scene_files = SceneFileManager()
        manager = UndoManager()
        document = registry.require(scene_files.document_id)
        target = Target()
        saved_revision = document.saved_revision

        assert manager.execute(SetPropertyCommand(target, "value", 0, 1))
        ticket = registry.begin_save(document.document_id)
        registry.complete_save(ticket.ticket_id, success=False, message="disk full")

        assert document.saved_revision == saved_revision
        assert document.is_dirty
        manager.undo()
        assert document.revision == saved_revision
        assert document.saved_revision == saved_revision
        assert not document.is_dirty
    finally:
        UndoManager._instance = previous_undo
        SceneFileManager._instance = previous_scene


def test_focused_scene_save_queues_every_dirty_resident_scene():
    from infernux.engine.interaction import EditorSaveService, FocusService, PanelInteractionRegistry
    from infernux.engine.scene_manager import SceneFileManager

    registry = DocumentRegistry.instance()
    documents = []
    controllers = []
    for index in range(2):
        document = registry.create(
            DocumentKind.SCENE,
            f"Scene {index}",
            document_id=f"scene:{index}",
            revision=1,
            saved_revision=0,
            capabilities=DocumentCapability.SAVE | DocumentCapability.SAVE_AS,
        )
        controller = _Controller(registry, document.document_id)
        registry.update_metadata(document.document_id, controller=controller)
        documents.append(document)
        controllers.append(controller)

    previous_scene_files = SceneFileManager._instance
    previous_focus = FocusService._instance
    previous_saving = EditorSaveService._instance
    SceneFileManager._instance = type(
        "SceneFiles",
        (),
        {
            "document_id": documents[0].document_id,
            "loaded_scene_document_ids": lambda self: tuple(document.document_id for document in documents),
        },
    )()
    focus = FocusService()
    registry.attach_view(documents[0].document_id, "scene_view")
    focus.activate_panel(
        "scene_view",
        view_id="scene_view",
        document_id=documents[0].document_id,
        record_history=False,
    )
    saving = EditorSaveService(registry, PanelInteractionRegistry())
    try:
        result = saving.save_focused()
        assert result.accepted
        assert result.target == "scenes"
        assert registry.process_deferred_saves() == (
            DocumentActionResult(DocumentActionStatus.APPLIED),
            DocumentActionResult(DocumentActionStatus.APPLIED),
        )
        assert [controller.save_calls for controller in controllers] == [1, 1]
        assert all(not document.is_dirty for document in documents)
    finally:
        saving.shutdown()
        EditorSaveService._instance = previous_saving
        FocusService._instance = previous_focus
        SceneFileManager._instance = previous_scene_files


@pytest.fixture
def primary_saving(monkeypatch):
    from infernux.engine.interaction import EditorSaveService, FocusService, PanelInteractionRegistry

    monkeypatch.setattr(FocusService, "_instance", None)
    monkeypatch.setattr(EditorSaveService, "_instance", None)
    registry = DocumentRegistry.instance()
    focus = FocusService()
    panels = PanelInteractionRegistry()
    saving = EditorSaveService(registry, panels)
    try:
        yield registry, focus, panels, saving
    finally:
        saving.shutdown()


def test_save_and_save_as_follow_primary_editor_through_auxiliary_focus(primary_saving):
    from infernux.engine.interaction import PanelInteractionDescriptor

    registry, focus, panels, saving = primary_saving
    inspector_doc, inspector_controller = _document(
        registry, dirty=True, kind=DocumentKind.MATERIAL, document_id="inspector-material",
    )
    registry.attach_view(inspector_doc.document_id, "inspector")
    for view_id, kind in (
        ("animclip2d_editor", DocumentKind.ANIMATION_CLIP),
        ("animfsm_editor", DocumentKind.ANIMATION_FSM),
        ("animtimeline_editor", DocumentKind.TIMELINE),
        ("particle_graph_editor", DocumentKind.PARTICLE_GRAPH),
        ("scene_view", DocumentKind.PREFAB),
    ):
        document, controller = _document(registry, dirty=True, kind=kind, document_id=view_id)
        panels.register_type(view_id, PanelInteractionDescriptor(document_backed=kind is not DocumentKind.PREFAB))
        registry.attach_view(document.document_id, view_id)
        focus.activate_panel(view_id, view_id=view_id, document_id=document.document_id)
        for auxiliary_view, save_as in (("project", False), ("inspector", True)):
            focus.activate_panel(
                auxiliary_view, view_id=auxiliary_view,
                document_id=inspector_doc.document_id if auxiliary_view == "inspector" else "",
            )
            result = saving.save_focused(save_as=save_as)
            assert result.accepted
            assert result.document_id == document.document_id
            assert result.panel_id == view_id
            assert focus.snapshot.active_view_id == auxiliary_view
            assert registry.process_deferred_saves() == (DocumentActionResult(DocumentActionStatus.APPLIED),)
            assert controller.ticket.save_as is save_as
        assert controller.save_calls == 2
    assert inspector_controller.save_calls == 0


def test_primary_save_resolves_rebound_view_and_rejects_detached_view(primary_saving):
    from infernux.engine.interaction import PanelInteractionDescriptor

    registry, focus, panels, saving = primary_saving
    view_id = "animclip2d_editor"
    panels.register_type(view_id, PanelInteractionDescriptor(document_backed=True))
    original, original_controller = _document(registry, dirty=True, kind=DocumentKind.ANIMATION_CLIP)
    registry.attach_view(original.document_id, view_id)
    focus.activate_panel(view_id, view_id=view_id, document_id=original.document_id)
    focus.activate_panel("project", view_id="project")
    replacement, replacement_controller = _document(
        registry, dirty=True, kind=DocumentKind.ANIMATION_CLIP, document_id="new-clip",
    )
    registry.attach_view(replacement.document_id, view_id)

    assert saving.save_focused().document_id == replacement.document_id
    registry.process_deferred_saves()
    assert replacement_controller.save_calls == 1
    assert original_controller.save_calls == 0
    registry.detach_view(view_id)
    result = saving.save_focused()
    assert result.result.status is DocumentActionStatus.REJECTED
    assert result.panel_id == view_id
    assert registry.process_deferred_saves() == ()


def test_guid_locator_follows_resource_rekey_and_a_replacement_live_entry(tmp_path):
    registry = DocumentRegistry()
    key = DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, "particle-guid")
    document = registry.create(
        DocumentKind.PARTICLE_GRAPH,
        "Draft",
        document_id="first-live-entry",
        key=key,
        resource_path=str(tmp_path / "Draft.particlegraph"),
    )
    locator = registry.locate(document.document_id)
    assert locator is not None and locator.resource_path == ""
    new_path = str(tmp_path / "Smoke.particlegraph")
    registry.rekey(document.document_id, key, resource_path=new_path)
    registry.unregister(document.document_id)
    reopened = registry.create(
        DocumentKind.PARTICLE_GRAPH,
        "Smoke",
        document_id="replacement-live-entry",
        key=key,
        resource_path=new_path,
    )
    assert reopened.document_id != document.document_id
    assert reopened.stable_id == locator.stable_id
    assert registry.resolve_locator(locator) is reopened


def test_reused_asset_path_with_new_guid_does_not_inherit_stable_identity(tmp_path):
    registry = DocumentRegistry()
    asset_path = tmp_path / "Walk.animclip2d"
    first = registry.create(
        DocumentKind.ANIMATION_CLIP,
        "Walk",
        key=DocumentKey.asset(DocumentKind.ANIMATION_CLIP, "old-asset-guid"),
        resource_path=str(asset_path),
    )
    locator = registry.locate(first.document_id)
    registry.unregister(first.document_id)

    reopened = registry.create(
        DocumentKind.ANIMATION_CLIP,
        "Walk",
        key=DocumentKey.asset(DocumentKind.ANIMATION_CLIP, "new-asset-guid"),
        resource_path=str(asset_path),
    )

    assert locator is not None
    assert reopened.stable_id != locator.stable_id
    assert registry.resolve_locator(locator) is None


def test_history_locator_uses_post_save_as_identity_after_scene_navigation(tmp_path):
    registry = DocumentRegistry()
    draft_path = tmp_path / "Draft.scene"
    saved_path = tmp_path / "Saved.scene"

    draft, _ = registry.open_or_create(
        DocumentKey.session(DocumentKind.SCENE, "unsaved-scene"),
        "Draft",
        resource_path="",
    )
    history_locator = registry.locate(draft.document_id)
    assert history_locator is not None

    save_ticket = registry.begin_save(draft.document_id, save_as=True)
    saved_key = DocumentKey.asset(DocumentKind.SCENE, "saved-scene-guid")
    registry.complete_save(
        save_ticket.ticket_id,
        success=True,
        key=saved_key,
        resource_path=str(saved_path),
        title="Saved",
    )
    assert draft.stable_id == history_locator.stable_id

    registry.unregister(draft.document_id)
    other, _ = registry.open_or_create(
        DocumentKey.asset(DocumentKind.SCENE, "replacement-scene-guid"),
        "Other",
        resource_path=str(draft_path),
    )
    assert other.stable_id != history_locator.stable_id

    restored_locator = registry.canonical_locator(history_locator)

    assert restored_locator.stable_id == history_locator.stable_id
    assert restored_locator.key_hint == saved_key
    reopened, created = registry.open_or_create(
        restored_locator.key_hint,
        "Saved",
        resource_path=str(saved_path),
    )
    assert not created
    assert reopened.stable_id == history_locator.stable_id
    assert reopened is draft


def test_dormant_document_reopens_by_guid_without_losing_command_identity(tmp_path):
    registry = DocumentRegistry()
    asset_path = tmp_path / "Walk.animclip2d"
    first, _ = registry.open_or_create(
        DocumentKey.asset(DocumentKind.ANIMATION_CLIP, "asset-guid"),
        "Walk",
        resource_path=str(asset_path),
    )
    original_id = first.document_id
    registry.unregister(original_id)

    guid_key = DocumentKey.asset(DocumentKind.ANIMATION_CLIP, "asset-guid")
    reopened, created = registry.open_or_create(
        guid_key,
        "Walk",
        resource_path=str(asset_path),
    )

    assert created is False
    assert reopened is first
    assert reopened.document_id == original_id
    assert reopened.key == guid_key


def test_dormant_document_restores_original_identity_revision_and_draft():
    class _Controller:
        def __init__(self, value):
            self.value = value

        def capture_document_restore_state(self, _document_id):
            return {"value": self.value}

    registry = DocumentRegistry()
    controller = _Controller(17)
    document = registry.create(
        DocumentKind.TIMELINE,
        "Timeline",
        document_id="timeline-session",
        key=DocumentKey.session(DocumentKind.TIMELINE, "timeline-session-key"),
        revision=4,
        saved_revision=1,
        controller=controller,
    )
    locator = registry.locate(document.document_id)
    assert locator is not None

    registry.unregister(document.document_id)
    controller.value = 99

    assert registry.dormant_restore_state(locator) == {"value": 17}
    restored = registry.restore_dormant(locator, controller=controller)
    assert restored is document
    assert restored.document_id == "timeline-session"
    assert restored.revision == 4
    assert restored.saved_revision == 1
    assert restored.controller is controller


def test_restore_capture_failure_does_not_leave_registry_half_closed():
    class _Controller:
        @staticmethod
        def capture_document_restore_state(_document_id):
            raise RuntimeError("capture failed")

    registry = DocumentRegistry()
    document = registry.create(
        DocumentKind.TIMELINE,
        "Timeline",
        controller=_Controller(),
    )
    locator = registry.locate(document.document_id)
    assert locator is not None

    assert registry.unregister(document.document_id)
    assert registry.get(document.document_id) is None
    assert registry.dormant_restore_state(locator) is None
    assert registry.restore_dormant(locator) is document


def test_scene_document_dirty_ownership_is_scoped_per_authoring_view():
    registry = DocumentRegistry()
    document = registry.create(DocumentKind.SCENE, "Main")
    for view_id in ("scene_view", "game_view", "ui_editor"):
        registry.attach_view(document.document_id, view_id)

    ui_revision = registry.mark_changed(
        document.document_id,
        view_id="ui_editor",
    )

    assert document.is_dirty_for_view("ui_editor")
    assert not document.is_dirty_for_view("scene_view")
    assert not document.is_dirty_for_view("game_view")

    scene_revision = registry.mark_changed(
        document.document_id,
        view_id="scene_view",
    )
    assert document.is_dirty_for_view("ui_editor")
    assert document.is_dirty_for_view("scene_view")
    assert not document.is_dirty_for_view("game_view")

    registry.restore_content_revision(document.document_id, ui_revision)
    assert document.is_dirty_for_view("ui_editor")
    assert not document.is_dirty_for_view("scene_view")

    registry.restore_content_revision(document.document_id, scene_revision)
    registry.mark_saved(document.document_id)
    assert not document.is_dirty
    assert not document.is_dirty_for_view("ui_editor")
    assert not document.is_dirty_for_view("scene_view")


def test_scene_dirty_view_ownership_survives_session_restore():
    class SessionController:
        @staticmethod
        def capture_document_restore_state(_document_id):
            return {"scene": "snapshot"}

    controller = SessionController()
    source = DocumentRegistry()
    document = source.create(
        DocumentKind.SCENE,
        "Main",
        controller=controller,
    )
    for view_id in ("scene_view", "game_view", "ui_editor"):
        source.attach_view(document.document_id, view_id)
    source.mark_changed(document.document_id, view_id="ui_editor")
    state = source.capture_session_state()

    restored_registry = DocumentRegistry()
    assert restored_registry.queue_session_restore(state) == 1
    restored, _ = restored_registry.claim_session_document(
        "scene_view",
        controller=controller,
    )
    restored_registry.claim_session_document("game_view", controller=controller)
    restored_registry.claim_session_document("ui_editor", controller=controller)

    assert restored.is_dirty_for_view("ui_editor")
    assert not restored.is_dirty_for_view("scene_view")
    assert not restored.is_dirty_for_view("game_view")


def test_pending_session_prunes_closed_views_and_drops_orphan_drafts():
    class SessionController:
        @staticmethod
        def capture_document_restore_state(_document_id):
            return {"draft": True}

    source = DocumentRegistry()
    shared = source.create(
        DocumentKind.SCENE,
        "Shared",
        controller=SessionController(),
    )
    source.attach_view(shared.document_id, "scene_view")
    source.attach_view(shared.document_id, "ui_editor")
    source.mark_changed(shared.document_id, view_id="ui_editor")
    orphan = source.create(
        DocumentKind.PARTICLE_GRAPH,
        "Orphan",
        controller=SessionController(),
    )
    source.attach_view(orphan.document_id, "particle_graph_editor")
    source.mark_changed(orphan.document_id, view_id="particle_graph_editor")

    restored = DocumentRegistry()
    assert restored.queue_session_restore(source.capture_session_state()) == 2
    removed = restored.prune_pending_session_views({"scene_view"})

    assert removed == ("particle_graph_editor", "ui_editor")
    assert restored.pending_session_view_ids() == ("scene_view",)
    assert restored.is_session_restore_suppressed("ui_editor")
    assert restored.is_session_restore_suppressed("particle_graph_editor")
    session = restored.capture_session_state()
    assert len(session["documents"]) == 1
    record = session["documents"][0]
    assert record["view_ids"] == ["scene_view"]
    assert record["dirty_view_ids"] == ["scene_view"]
