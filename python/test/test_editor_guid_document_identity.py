from __future__ import annotations

from types import SimpleNamespace


class _Database:
    def __init__(self):
        self.paths_by_guid: dict[str, str] = {}
        self.guids_by_path: dict[str, str] = {}

    def register(self, path: str, guid: str) -> None:
        self.paths_by_guid[guid] = path
        self.guids_by_path[path] = guid

    def get_guid_from_path(self, path: str) -> str:
        return self.guids_by_path.get(str(path), "")

    def get_path_from_guid(self, guid: str) -> str:
        return self.paths_by_guid.get(str(guid), "")


class _RestoreController:
    def capture_document_restore_state(self, _document_id: str):
        return {"model": "current"}


def test_asset_move_preserves_guid_selection_and_dormant_document_identity(tmp_path):
    from Infernux.engine.interaction import (
        AssetMutationService,
        DocumentKey,
        DocumentKind,
        DocumentRegistry,
        SelectionService,
        SelectionTarget,
    )

    old_path = str(tmp_path / "Old.particlegraph")
    new_path = str(tmp_path / "Moved.particlegraph")
    guid = "particle-guid"
    registry = DocumentRegistry()
    selection = SelectionService()
    mutations = AssetMutationService(registry, selection)
    document = registry.create(
        DocumentKind.PARTICLE_GRAPH,
        "Old",
        key=DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, guid),
        resource_path=old_path,
        revision=3,
        saved_revision=2,
        controller=_RestoreController(),
    )
    registry.attach_view(document.document_id, "particle_graph_editor")
    selection.select(SelectionTarget.asset(guid), owner_id="project")
    locator = registry.locate(document.document_id)
    assert locator is not None and locator.resource_path == ""
    registry.unregister(document.document_id, preserve_dormant=True)

    plan = mutations.prepare_relocation(((old_path, new_path, guid),))
    change = mutations.commit_relocation(plan)
    restored = registry.restore_dormant(locator, controller=_RestoreController())

    assert change.changes[0].selection_changed is False
    assert selection.snapshot.primary == SelectionTarget.asset(guid)
    assert restored is not None
    assert restored.key == DocumentKey.asset(DocumentKind.PARTICLE_GRAPH, guid)
    assert restored.stable_id == locator.stable_id
    assert restored.resource_path == new_path
    assert (restored.revision, restored.saved_revision) == (3, 2)


def test_new_guid_at_deleted_path_does_not_inherit_dormant_state(tmp_path, monkeypatch):
    from Infernux.engine.interaction import DocumentKey, DocumentKind, DocumentRegistry
    from Infernux.engine.interaction import documents as document_module

    path = str(tmp_path / "Reused.scene")
    database = _Database()
    database.register(path, "new-guid")
    monkeypatch.setattr(document_module, "_asset_database", lambda: database)
    registry = DocumentRegistry()
    old = registry.create(
        DocumentKind.SCENE,
        "Old",
        key=DocumentKey.asset(DocumentKind.SCENE, "old-guid"),
        resource_path=path,
        controller=_RestoreController(),
    )
    old_stable_id = old.stable_id
    registry.unregister(old.document_id, preserve_dormant=True)

    locator = registry.locate_resource(DocumentKind.SCENE, path)
    current, created = registry.open_or_create(
        locator.key_hint,
        "Current",
        stable_id=locator.stable_id,
        resource_path=path,
    )

    assert created is True
    assert current.key == DocumentKey.asset(DocumentKind.SCENE, "new-guid")
    assert current.stable_id != old_stable_id


def test_delete_preflight_matches_registered_documents_by_guid_not_path(tmp_path):
    from Infernux.engine.interaction import DocumentKey, DocumentKind, DocumentRegistry

    old_path = str(tmp_path / "Reused.scene")
    moved_path = str(tmp_path / "Moved.scene")
    registry = DocumentRegistry()
    document = registry.create(
        DocumentKind.SCENE,
        "Moved",
        key=DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=moved_path,
    )

    assert registry.documents_under_resource(old_path, guids={"new-guid"}) == ()
    assert registry.documents_under_resource(old_path, guids={"scene-guid"}) == (
        document,
    )


def test_scene_session_persists_guid_and_ignores_legacy_path_record(
    tmp_path, monkeypatch
):
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.engine.scene_manager import SceneFileManager

    path = str(tmp_path / "Main.scene")
    database = _Database()
    database.register(path, "main-scene-guid")
    manager = SceneFileManager()
    manager.set_asset_database(database)
    manager._current_scene_path = path
    document = manager._replace_scene_document(
        kind="scene",
        resource_path=path,
        title="Main",
        dirty=True,
    )
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)

    state = manager.save_session_state()

    assert state["asset_guid"] == "main-scene-guid"
    assert "current_scene_path" not in state
    assert DocumentRegistry.instance().require(document.document_id).is_dirty
    assert not manager.restore_session_state(
        {
            "dirty": True,
            "current_scene_path": path,
            "document": {"name": "Legacy"},
        }
    )


def test_path_keyed_registered_asset_session_record_is_silently_skipped():
    from Infernux.engine.interaction import DocumentKind, DocumentRegistry

    source = DocumentRegistry()
    document = source.create(
        DocumentKind.PARTICLE_GRAPH,
        "Draft",
        controller=_RestoreController(),
    )
    source.attach_view(document.document_id, "particle_graph_editor")
    state = source.capture_session_state()
    state["documents"][0]["key"] = {
        "identity_kind": "resource_path",
        "identity": "Assets/Legacy.particlegraph",
    }
    state["documents"][0]["resource_path"] = "Assets/Legacy.particlegraph"

    restored = DocumentRegistry()
    assert restored.queue_session_restore(state) == 0
    assert restored.capture_session_state() == {"documents": []}


def test_authoring_save_as_rekeys_to_published_guid(tmp_path, monkeypatch):
    from Infernux.core.assets import AssetManager
    from Infernux.engine.interaction import (
        AuthoringAssetSnapshot,
        AuthoringDocumentController,
        DocumentActionStatus,
        DocumentIdentityKind,
        DocumentKind,
        DocumentRegistry,
    )
    from Infernux.engine.interaction import authoring_documents as authoring_module

    target = str(tmp_path / "Saved.particlegraph")
    database = _Database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(
        authoring_module,
        "submit_document_text",
        lambda *_args, **_kwargs: SimpleNamespace(
            is_complete=True,
            status="succeeded",
            error="",
            committed_file_state=None,
        ),
    )

    class _View:
        def capture_authoring_save_snapshot(self, path):
            return AuthoringAssetSnapshot(path, "{}", "token", "Saved")

        def publish_authoring_save_snapshot(self, snapshot):
            database.register(snapshot.target_path, "saved-guid")
            return ""

        def current_authoring_content_token(self):
            return "token"

    view = _View()
    controller = AuthoringDocumentController(view)
    registry = DocumentRegistry()
    document = registry.create(
        DocumentKind.PARTICLE_GRAPH,
        "Draft",
        revision=1,
        saved_revision=0,
        controller=controller,
    )
    ticket = registry.begin_save(document.document_id, save_as=True)

    result = controller.continue_save_to_resource(ticket.ticket_id, target)

    assert result.status is DocumentActionStatus.APPLIED
    assert document.key.identity_kind is DocumentIdentityKind.ASSET_GUID
    assert document.key.identity == "saved-guid"
    assert document.resource_path == target
