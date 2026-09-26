from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from Infernux.engine.interaction import (
    DocumentKey,
    DocumentKind,
    DocumentRegistry,
    DocumentState,
    ExternalDocumentConflictService,
)
from Infernux.engine.scene_manager import SceneFileManager
from Infernux.host import EditorAutomationHost, OperationError


@pytest.mark.parametrize("field", ["__type_name__", "__component_id__"])
@pytest.mark.parametrize("entry", ["startup", "deferred", "conflict"])
def test_scene_entry_points_reject_invalid_disk_fields_and_publish_reason(
    scene, tmp_path, monkeypatch, field, entry,
):
    import Infernux.engine.scene_manager as scene_files

    owner = scene.create_game_object("KeepLocalTank")
    original = scene.serialize_document()
    invalid = json.loads(json.dumps(original))
    record = {
        "component_id": 981725,
        "type_id": "python:script-guid:type-guid:TankBattle:TankBattle",
        "enabled": True,
        "execution_order": 0,
        "data": {field: "invalid-runtime-metadata"},
    }
    invalid["objects"][0]["components"].append(record)
    path = tmp_path / "TankBattle.scene"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    monkeypatch.setattr(SceneFileManager, "_instance", None)
    manager = SceneFileManager()
    monkeypatch.setattr(manager, "_is_under_assets", lambda _path: True)
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    monkeypatch.setattr(manager, "_remember_last_scene", lambda _path: None)
    monkeypatch.setattr(manager, "_restore_camera_state", lambda _path: None)
    monkeypatch.setattr(manager, "_prepare_native_scene_swap", lambda: None)
    registry = DocumentRegistry.instance()
    document = registry.require(manager.document_id)

    if entry == "startup":
        monkeypatch.setattr(scene_files, "_load_editor_settings", lambda: {
            scene_files.LAST_OPENED_SCENE_GUID_KEY: "scene-guid",
        })
        manager._asset_database = SimpleNamespace(get_path_from_guid=lambda _guid: str(path))
        defaults = []
        monkeypatch.setattr(manager, "_do_new_scene", lambda: defaults.append(True))
        manager.load_last_scene_or_default()
        assert defaults == [True]
    elif entry == "deferred":
        assert manager.open_scene(str(path))
        assert manager.last_scene_load["status"] == "pending"
        deadline = time.monotonic() + 5.0
        while manager.is_loading and time.monotonic() < deadline:
            manager.poll_deferred_load()
            time.sleep(0.001)
        assert not manager.is_loading
    else:
        manager.set_asset_database(SimpleNamespace(
            get_guid_from_path=lambda candidate: (
                "scene-guid" if str(candidate) == str(path.resolve()) else ""
            ),
            get_path_from_guid=lambda guid: (
                str(path.resolve()) if guid == "scene-guid" else ""
            ),
        ))
        manager._current_scene_path = str(path.resolve())
        registry.rekey(
            document.document_id,
            DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
            resource_path=str(path.resolve()),
        )
        registry.update_metadata(document.document_id, resource_path=str(path.resolve()))
        registry.mark_changed(document.document_id)
        registry.mark_conflict(document.document_id)
        conflicts = ExternalDocumentConflictService(registry)
        conflicts.poll()
        result = conflicts.reload(conflicts.active.conflict_id)
        assert not result.accepted
        assert field in result.message
        assert field in conflicts.error
        assert document.state is DocumentState.CONFLICT
        assert conflicts.keep_local(conflicts.active.conflict_id).accepted
        assert document.is_dirty

    assert manager.last_scene_load["status"] == "failed"
    assert manager.last_scene_load["path"] == str(path.resolve())
    assert field in manager.last_scene_load["error"]
    assert scene.serialize_document() == original
    assert scene.find("KeepLocalTank") is owner
    info = EditorAutomationHost().project_info(str(tmp_path))
    assert info["active_scene"]["last_load"] == manager.last_scene_load
    assert info["active_scene"]["loading"] is False


def test_save_conflict_reports_document_state_to_automation(monkeypatch):
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    manager = SceneFileManager()
    registry = DocumentRegistry.instance()
    registry.mark_conflict(manager.document_id)

    with pytest.raises(OperationError) as caught:
        EditorAutomationHost().save_scene()

    assert caught.value.code == "scene.save_rejected"
    assert "outside the Editor" in str(caught.value)
    assert caught.value.details["document_state"] == "conflict"
    assert registry.active_save_ticket(manager.document_id) is None


def test_pending_conflict_reload_preserves_specific_failure(monkeypatch, tmp_path):
    import Infernux.engine.deferred_task as deferred

    monkeypatch.setattr(SceneFileManager, "_instance", None)
    manager = SceneFileManager()
    path = tmp_path / "Pending.scene"
    path.write_text("invalid", encoding="utf-8")
    registry = DocumentRegistry.instance()
    document = registry.require(manager.document_id)
    registry.mark_conflict(document.document_id)
    manager._pending_external_reload = document.document_id
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    monkeypatch.setattr(deferred.DeferredTaskRunner, "instance", lambda: SimpleNamespace(is_busy=False))

    def fail_reload(**_kwargs):
        manager._scene_load_failed(str(path), "Scene.objects[0]: Light.shadowBias is not part of the current format")
        return False

    monkeypatch.setattr(manager, "reload_from_resource", fail_reload)
    assert manager.poll_pending_writes() == 1
    result = registry.take_external_reload_result(document.document_id)
    assert not result.accepted
    assert "Light.shadowBias" in result.message
    assert document.state is DocumentState.CONFLICT


@pytest.fixture
def persisted_scene(scene, engine, tmp_path, monkeypatch):
    from Infernux.core.assets import AssetManager

    owner = scene.create_game_object("LocalTank")
    path = tmp_path / "TankBattle.scene"
    path.write_text(json.dumps(scene.serialize_document()), encoding="utf-8")
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    manager = SceneFileManager()
    manager._current_scene_path = str(path)
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    assert database.import_asset(str(path)).succeeded
    manager.set_asset_database(database)
    monkeypatch.setattr(manager, "_is_under_assets", lambda _path: True)
    monkeypatch.setattr(manager, "_is_play_mode", lambda: False)
    monkeypatch.setattr(manager, "_save_camera_state", lambda _path: None)
    monkeypatch.setattr(manager, "_restore_camera_state", lambda _path: None)
    monkeypatch.setattr(manager, "_remember_last_scene", lambda _path: None)
    monkeypatch.setattr(manager, "_prepare_native_scene_swap", lambda: None)
    monkeypatch.setattr(manager, "sync_all_prefab_instances", lambda _scene: None)
    registry = DocumentRegistry.instance()
    return manager, registry, path, owner


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("newer_write", ["none", "before_choice", "after_choice"])
def test_keep_local_then_scene_save_uses_only_observed_conflict_version(
    persisted_scene, dirty, newer_write,
):
    manager, registry, path, owner = persisted_scene
    document = registry.require(manager.document_id)
    if dirty:
        owner.name = "EditedLocalTank"
        registry.mark_changed(document.document_id)
    path.write_text('{"broken": "external-version"}', encoding="utf-8")
    registry.publish_external_resource_change(str(path))
    assert document.state is DocumentState.CONFLICT
    conflicts = ExternalDocumentConflictService(registry)
    conflicts.poll()
    observed = document.external_file_state
    external_revision = document.external_revision
    # Identical watcher echoes must not invalidate the presented choice.
    registry.publish_external_resource_change(str(path))
    assert document.external_revision == external_revision
    newer_bytes = '{"broken": "newer-unobserved-version"}'
    if newer_write == "before_choice":
        path.write_text(newer_bytes, encoding="utf-8")
    assert conflicts.keep_local(conflicts.active.conflict_id).accepted
    assert document.is_dirty
    assert document.durable_file_state is observed
    if newer_write == "after_choice":
        path.write_text(newer_bytes, encoding="utf-8")

    saved = manager.save_current_scene()

    assert saved is (newer_write == "none")
    assert registry.active_save_ticket(document.document_id) is None
    if newer_write == "none":
        assert document.state is DocumentState.READY
        assert not document.is_dirty
        assert json.loads(path.read_text(encoding="utf-8"))["objects"][0]["name"] == owner.name
    else:
        assert path.read_text(encoding="utf-8") == newer_bytes
        assert document.state is DocumentState.CONFLICT
        assert document.is_dirty


def _drain_scene_load(manager):
    deadline = time.monotonic() + 5.0
    while manager.is_loading and time.monotonic() < deadline:
        manager.poll_deferred_load()
        time.sleep(0.001)
    assert not manager.is_loading


def test_queued_external_reload_reports_newer_disk_revision_as_conflict(
    persisted_scene, scene, monkeypatch,
):
    import Infernux.engine.deferred_task as deferred
    from Infernux.engine.interaction import DocumentActionStatus
    from Infernux.core.document_store import capture_document_file_state

    manager, registry, path, _owner = persisted_scene
    document = registry.require(manager.document_id)
    read_version = scene.serialize_document()
    read_version["objects"][0]["name"] = "QueuedReadVersion"
    path.write_text(json.dumps(read_version), encoding="utf-8")
    registry.mark_conflict(document.document_id)
    monkeypatch.setattr(deferred.DeferredTaskRunner, "instance", lambda: SimpleNamespace(is_busy=False))
    manager._deferred_load_path = str(path)
    result = registry.request_reload_external(document.document_id)
    assert result.status is DocumentActionStatus.PENDING
    assert manager._pending_external_reload is not None
    manager._deferred_load_path = None
    newer = json.loads(json.dumps(read_version))
    newer["objects"][0]["name"] = "QueuedLaterDiskVersion"
    newer_bytes = json.dumps(newer)
    finish = manager._finish_open_scene

    def publish_after_external_write(*args, **kwargs):
        path.write_text(newer_bytes, encoding="utf-8")
        return finish(*args, **kwargs)

    monkeypatch.setattr(manager, "_finish_open_scene", publish_after_external_write)

    assert manager.poll_pending_writes() == 1

    result = registry.take_external_reload_result(document.document_id)
    assert result.status is DocumentActionStatus.FAILED
    assert "changed again while it was loading" in result.message
    assert manager._pending_external_reload is None
    assert document.state is DocumentState.CONFLICT
    assert scene.find("QueuedReadVersion") is not None
    assert scene.find("QueuedLaterDiskVersion") is None
    current = capture_document_file_state(str(path))
    assert document.external_file_state.content_hash == current.content_hash
    assert document.durable_file_state.content_hash != current.content_hash
    assert path.read_text(encoding="utf-8") == newer_bytes


@pytest.mark.parametrize("dirty", [False, True])
def test_successful_same_path_mcp_reload_establishes_savable_disk_baseline(
    persisted_scene, scene, dirty,
):
    manager, registry, path, _owner = persisted_scene
    document = registry.require(manager.document_id)
    if dirty:
        registry.mark_changed(document.document_id)
    changed = scene.serialize_document()
    changed["objects"][0]["name"] = "DiskTank"
    path.write_text(json.dumps(changed), encoding="utf-8")
    registry.mark_conflict(document.document_id)
    result = EditorAutomationHost().reload_scene(discard_changes=True)
    assert result["scheduled"]
    _drain_scene_load(manager)
    assert manager.last_scene_load["status"] == "loaded"
    assert document.state is DocumentState.READY
    assert not document.is_dirty
    restored = scene.find("DiskTank")
    assert restored is not None
    restored.name = "EditedReloadedTank"
    registry.mark_changed(document.document_id)

    assert manager.save_current_scene()

    assert json.loads(path.read_text(encoding="utf-8"))["objects"][0]["name"] == "EditedReloadedTank"
    assert not document.is_dirty


@pytest.mark.parametrize("entry", ["deferred", "conflict"])
def test_scene_baseline_keeps_read_snapshot_when_disk_changes_before_publish(
    persisted_scene, scene, monkeypatch, entry,
):
    manager, registry, path, _owner = persisted_scene
    document = registry.require(manager.document_id)
    initial = scene.serialize_document()
    initial["objects"][0]["name"] = "ReadVersion"
    path.write_text(json.dumps(initial), encoding="utf-8")
    newer = json.loads(json.dumps(initial))
    newer["objects"][0]["name"] = "LaterDiskVersion"
    newer_bytes = json.dumps(newer)
    finish = manager._finish_open_scene

    def publish_after_external_write(*args, **kwargs):
        path.write_text(newer_bytes, encoding="utf-8")
        return finish(*args, **kwargs)

    monkeypatch.setattr(manager, "_finish_open_scene", publish_after_external_write)
    if entry == "deferred":
        assert manager.reload_current_scene(discard_changes=True)
        _drain_scene_load(manager)
    else:
        registry.mark_conflict(document.document_id)
        result = registry.request_reload_external(document.document_id)
        assert not result.accepted
        assert "changed again while it was loading" in result.message
    assert scene.find("ReadVersion") is not None
    assert scene.find("LaterDiskVersion") is None
    assert registry.durable_resource_content_changed(str(path)) is True
    scene.find("ReadVersion").name = "LocalAfterLoad"
    registry.mark_changed(document.document_id)

    assert not manager.save_current_scene()

    assert path.read_text(encoding="utf-8") == newer_bytes
    assert document.state is DocumentState.CONFLICT


@pytest.mark.parametrize("entry", ["new_path", "same_path"])
@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("watcher_delivery", ["before_publish", "after_publish"])
def test_watcher_publishes_disk_change_after_scene_read_before_open_completes(
    persisted_scene, scene, engine, monkeypatch, entry, dirty, watcher_delivery,
):
    from Infernux.engine.resources_manager import ResourceChangeHandler
    from Infernux.engine.scene_document_transaction import SceneDocumentTransactionState

    manager, registry, original_path, _owner = persisted_scene
    previous_document_id = manager.document_id
    path = (original_path.with_name("FirstOpen.scene")
            if entry == "new_path" else original_path)
    read_version = scene.serialize_document()
    read_version["objects"][0]["name"] = "ReadVersion"
    path.write_text(json.dumps(read_version), encoding="utf-8")
    newer = json.loads(json.dumps(read_version))
    newer["objects"][0]["name"] = "LaterDiskVersion"
    newer_bytes = json.dumps(newer)
    database = engine.get_asset_database()
    handler = ResourceChangeHandler(engine, project_path=str(path.parent))
    finish = manager._finish_open_scene
    publication_count = 0

    def publish_after_external_write(*args, **kwargs):
        nonlocal publication_count
        publication_count += 1
        if publication_count == 1 and watcher_delivery == "after_publish":
            # The native ticket already contains A. Publish B on disk before
            # the editor registers either the existing or newly opened path.
            path.write_text(newer_bytes, encoding="utf-8")
        return finish(*args, **kwargs)

    try:
        assert database.import_asset(str(path)).succeeded
        manager.set_asset_database(database)
        monkeypatch.setattr(manager, "_finish_open_scene", publish_after_external_write)
        if entry == "new_path":
            assert manager.open_scene(str(path))
        else:
            assert manager.reload_current_scene(discard_changes=True)
        if watcher_delivery == "before_publish":
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                manager.poll_deferred_load()
                transaction = manager._scene_transaction
                if transaction is not None and transaction.state is SceneDocumentTransactionState.DOCUMENT_READY:
                    break
                time.sleep(0.001)
            assert manager._scene_transaction.state is SceneDocumentTransactionState.DOCUMENT_READY
            path.write_text(newer_bytes, encoding="utf-8")
            handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(path)))
            assert handler.process_pending_reloads(force=True) == 1
            assert handler.pending_count == 0
        _drain_scene_load(manager)
        document = registry.require(manager.document_id)
        assert (document.document_id == previous_document_id) is (entry == "same_path")
        assert scene.find("ReadVersion") is not None
        assert scene.find("LaterDiskVersion") is None
        # Even an already-consumed watcher event cannot make stale A appear
        # current: the publication boundary presents the newer B conflict.
        assert document.state is DocumentState.CONFLICT
        if dirty:
            scene.find("ReadVersion").name = "LocalAfterLoad"
            registry.mark_changed(document.document_id)

        # Exercise the real watcher queue, native AssetDatabase reimport, and
        # document publication, not merely a fingerprint comparison or CAS.
        if watcher_delivery == "after_publish":
            handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(path)))
            assert handler.process_pending_reloads(force=True) == 1
        assert handler.pending_count == 0
        assert path.read_text(encoding="utf-8") == newer_bytes
        if dirty or watcher_delivery == "before_publish":
            assert document.state is DocumentState.CONFLICT
            assert scene.find("LocalAfterLoad" if dirty else "ReadVersion") is not None
            assert scene.find("LaterDiskVersion") is None
            assert not manager.save_current_scene()
            assert path.read_text(encoding="utf-8") == newer_bytes
            info = EditorAutomationHost().project_info(str(path.parent))
            assert info["active_scene"]["document_state"] == "conflict"
        else:
            assert document.state is DocumentState.READY
            assert not document.is_dirty
            assert scene.find("ReadVersion") is None
            loaded = scene.find("LaterDiskVersion")
            assert loaded is not None
            assert publication_count == 2
            loaded.name = "EditedAfterWatcherReload"
            registry.mark_changed(document.document_id)
            assert manager.save_current_scene()
            assert not document.is_dirty
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["objects"][0]["name"] == "EditedAfterWatcherReload"
        assert registry.active_save_ticket(document.document_id) is None
    finally:
        if database.get_guid_from_path(str(path)):
            database.delete_asset(str(path))
