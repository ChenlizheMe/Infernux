from __future__ import annotations

import json
import os
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.engine import Engine
from infernux.engine.import_coordinator import AssetFsEventKind, ImportCoordinator
from infernux.engine.interaction import ActionOrigin, action_origin_scope
from infernux.engine.resources_manager import (
    ResourceChangeHandler,
    ResourcesManager,
    _AssetImportNotReady,
    _is_plugin_editor_translation_catalog,
)
from infernux.engine.path_utils import path_key
from infernux.debug import Debug
from infernux.lib import AssetMutationResult, RuntimeMode


def _mutation(operation, path, guid=""):
    result = AssetMutationResult()
    result.succeeded = True
    result.database_committed = True
    result.changed = True
    result.operation = operation
    result.path = path
    result.guid = guid
    return result


class _AssetDatabaseProbe:
    refresh_pending = False

    def __init__(self):
        self.guid_by_path = {}
        self.queries = []
        self.mutations = []

    def complete_pending_refresh(self):
        if not self.refresh_pending:
            raise AssertionError("completion requires pending refresh work")
        self.refresh_pending = False

    def get_guid_from_path(self, path):
        self.queries.append((path, threading.get_ident()))
        return self.guid_by_path.get(path, "")

    def import_asset(self, path):
        self.mutations.append(("import", path, threading.get_ident()))
        self.guid_by_path[path] = "created-guid"
        return _mutation("import", path, "created-guid")

    def contains_path(self, path):
        return path in self.guid_by_path

    def get_meta_by_path(self, path):
        return None

    def reimport_asset(self, path):
        self.mutations.append(("modified", path, threading.get_ident()))
        return _mutation("reimport", path, self.guid_by_path.get(path, ""))

    def delete_asset(self, path):
        self.mutations.append(("deleted", path, threading.get_ident()))
        self.guid_by_path.pop(path, None)
        return _mutation("delete", path)

    def move_asset(self, old_path, new_path):
        self.mutations.append(("moved", old_path, new_path, threading.get_ident()))
        guid = self.guid_by_path.pop(old_path, "")
        if guid:
            self.guid_by_path[new_path] = guid
        result = _mutation("move", new_path, guid)
        result.previous_path = old_path
        return result


class _EngineProbe:
    def __init__(self, asset_database):
        self.asset_database = asset_database
        self.editor_wakes = 0
        self.full_speed_requests = 0

    def get_asset_database(self):
        return self.asset_database

    def request_editor_wake(self):
        self.editor_wakes += 1

    def request_full_speed_frame(self):
        self.full_speed_requests += 1


def _event(path, *, destination=""):
    values = {"is_directory": False, "src_path": str(path)}
    if destination:
        values["dest_path"] = str(destination)
    return SimpleNamespace(**values)


def _run_script_publication_owner(handler: ResourceChangeHandler) -> int:
    if not handler._frontend_worker_running:
        handler.process_script_worker()
    return handler._drain_script_results()


def _patch_asset_manager(monkeypatch, calls):
    AssetManager._watcher_echo_suppression.clear()
    AssetManager._meta_write_suppression.clear()
    monkeypatch.setattr(
        AssetManager,
        "_get_registry",
        classmethod(lambda _cls: None),
    )
    monkeypatch.setattr(
        AssetManager,
        "invalidate",
        classmethod(lambda _cls, guid: calls.append(("invalidate", guid, threading.get_ident()))),
    )
    monkeypatch.setattr(
        AssetManager,
        "_publish_asset_content_change",
        classmethod(
            lambda _cls, path, event_type="modified", **_kwargs: calls.append(
                (f"asset-{event_type}", path, threading.get_ident())
            )
        ),
    )
    monkeypatch.setattr(AssetManager, "_invalidate_project_panel_cache", classmethod(lambda _cls: None))


def test_due_scripts_do_not_wait_behind_a_bulk_model_import_queue(tmp_path):
    coordinator = ImportCoordinator(debounce_seconds=0.0, clock=lambda: 1.0)
    models = [str(tmp_path / f"model_{index}.glb") for index in range(128)]
    for index, path in enumerate(models):
        coordinator.submit(AssetFsEventKind.CREATED, path, observed_at=index / 1000)
    scripts = [str(tmp_path / name) for name in ("Movement.py", "Player.py")]
    for index, path in enumerate(scripts):
        coordinator.submit(AssetFsEventKind.CREATED, path, observed_at=0.2 + index / 1000)

    for path in (*scripts, models[0]):
        event, = coordinator.drain(max_events=1)
        assert event.path == path
        coordinator.complete(event)
    assert coordinator.pending_count == len(models) - 1


def test_script_priority_preserves_move_metadata_deadlines_and_inflight_order(tmp_path):
    coordinator = ImportCoordinator(debounce_seconds=0.0)
    source = str(tmp_path / "Before.asset")
    destination = str(tmp_path / "After.py")
    dependent = str(tmp_path / "SameIdentity.py")
    coordinator.submit(AssetFsEventKind.META_MODIFIED, source, guid_hint="stable", observed_at=0.1, debounce_seconds=0.5)
    coordinator.submit(AssetFsEventKind.MOVED, source, destination=destination, guid_hint="stable", observed_at=0.2)
    coordinator.submit(AssetFsEventKind.META_MODIFIED, destination, guid_hint="stable", observed_at=0.3)
    coordinator.submit(AssetFsEventKind.MODIFIED, dependent, guid_hint="stable", observed_at=0.4)
    assert coordinator.drain(now=0.5, max_events=1) == []

    metadata, = coordinator.drain(now=0.6, max_events=1)
    assert metadata.kind is AssetFsEventKind.META_MODIFIED and metadata.path == source
    assert coordinator.drain(now=0.6, max_events=1) == []
    coordinator.complete(metadata)
    move, = coordinator.drain(now=0.6, max_events=1)
    assert move.kind is AssetFsEventKind.MOVED and move.destination == destination
    coordinator.defer(move, now=0.6)
    assert coordinator.drain(now=0.7, max_events=1) == []
    move, = coordinator.drain(now=0.8, max_events=1)
    coordinator.complete(move)
    for path in (destination, dependent):
        event, = coordinator.drain(now=0.8, max_events=1)
        assert event.path == path
        coordinator.complete(event)

    coordinator.submit(AssetFsEventKind.META_DELETED, destination, observed_at=1.0)
    coordinator.submit(AssetFsEventKind.DELETED, destination, observed_at=1.1)
    assert coordinator.drain(now=1.5, max_events=1) == []
    metadata, = coordinator.drain(now=1.7, max_events=1)
    assert metadata.kind is AssetFsEventKind.META_DELETED
    coordinator.complete(metadata)
    assert coordinator.drain(now=1.8, max_events=1) == []
    deleted, = coordinator.drain(now=1.9, max_events=1)
    assert deleted.kind is AssetFsEventKind.DELETED
    coordinator.complete(deleted)


def test_watcher_thread_only_submits_and_main_thread_commits(monkeypatch, tmp_path):
    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)

    path = tmp_path / "asset.txt"
    path.write_text("content", encoding="utf-8")
    database.guid_by_path[str(path)] = "stable-guid"
    callback_thread_ids = []

    def submit_many():
        callback_thread_ids.append(threading.get_ident())
        for _ in range(50):
            handler.on_modified(_event(path))

    worker = threading.Thread(target=submit_many)
    worker.start()
    worker.join()

    assert database.mutations == []
    assert asset_calls == []
    assert callback_thread_ids[0] != threading.get_ident()

    assert handler.process_pending_reloads(force=True) == 1
    assert [entry[0] for entry in asset_calls] == ["invalidate", "asset-modified"]
    assert [entry[0] for entry in database.mutations] == ["modified"]
    assert asset_calls[0][-1] == threading.get_ident()
    assert database.mutations[0][-1] == threading.get_ident()


def test_move_query_may_run_on_watcher_but_mutation_waits_for_owner(monkeypatch, tmp_path):
    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)

    old_path = tmp_path / "old.txt"
    new_path = tmp_path / "new.txt"
    new_path.write_text("moved", encoding="utf-8")
    database.guid_by_path[str(old_path)] = "stable-guid"

    worker = threading.Thread(target=lambda: handler.on_moved(_event(old_path, destination=new_path)))
    worker.start()
    worker.join()

    assert database.queries[0][-1] == worker.ident
    assert database.mutations == []
    assert handler.process_pending_reloads(force=True) == 1
    assert database.mutations[0][0] == "moved"
    assert database.mutations[0][-1] == threading.get_ident()
    # A move is published exclusively by AssetMutationService as one typed
    # relocation; it must not also masquerade as a content-change event.
    assert [entry[0] for entry in asset_calls] == ["invalidate"]
    assert all(entry[-1] == threading.get_ident() for entry in asset_calls)


@pytest.mark.parametrize(
    "extension",
    [
        ".glsl",
    ],
)
def test_rejected_dependency_move_is_terminal_and_exact_revert_can_recover(monkeypatch, tmp_path, extension):
    from infernux.engine.interaction import DocumentRegistry

    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    _patch_asset_manager(monkeypatch, [])
    old = tmp_path / ("Before" + extension)
    new = tmp_path / ("After" + extension)
    new.write_text("invalid source", encoding="utf-8")
    database.guid_by_path[str(old)] = "stable-guid"
    publications, notifications, errors = [], [], []
    monkeypatch.setattr(
        AssetManager,
        "_compile_shader_dependency_runtime",
        classmethod(
            lambda _cls, path, previous_path="": publications.append((path, previous_path)) or "dependency rejected"
        ),
    )
    monkeypatch.setattr(AssetManager, "_publish_compile_diagnostic", staticmethod(lambda *_args: None))
    monkeypatch.setattr(handler, "_notify_shader_reloaded", notifications.append)
    monkeypatch.setattr(Debug, "log_error", errors.append)
    handler._rejected_compiled_assets.add(path_key(str(old)))
    handler.on_moved(_event(old, destination=new))
    assert handler.process_pending_reloads(force=True) == 1
    assert handler.pending_count == 0
    assert handler.process_pending_reloads(force=True) == 0
    assert publications == [(str(new), str(old))]
    assert database.guid_by_path[str(new)] == "stable-guid"
    assert handler._rejected_compiled_assets == {path_key(str(new))}
    assert not notifications and not errors

    # A restored durable document may have identical bytes; the outstanding
    # rejected candidate still requires one successful runtime publication.
    monkeypatch.setattr(DocumentRegistry.instance(), "durable_resource_content_changed", lambda *_a, **_kw: False)
    reimports = []
    monkeypatch.setattr(
        AssetManager,
        "reimport_asset",
        classmethod(lambda _cls, path, **_kw: reimports.append(path) or _mutation("reimport", path, "stable-guid")),
    )
    new.write_text("valid original source", encoding="utf-8")
    handler.on_modified(_event(new))
    assert handler.process_pending_reloads(force=True) == 1
    assert reimports == [str(new)]
    assert notifications == [str(new)]
    assert not handler._rejected_compiled_assets


def test_document_store_atomic_replace_ignores_temp_events_and_reimports_target(monkeypatch, tmp_path):
    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)

    target = tmp_path / "atomic.mat"
    temporary = tmp_path / "atomic.mat.tmp.123456.7"
    temporary.write_text("staged", encoding="utf-8")
    database.guid_by_path[str(target)] = "stable-guid"

    handler.on_created(_event(temporary))
    handler.on_modified(_event(temporary))
    target.write_text("published", encoding="utf-8")
    temporary.unlink()
    handler.on_moved(_event(temporary, destination=target))
    handler.on_deleted(_event(temporary))

    assert handler.pending_count == 1
    assert handler.process_pending_reloads(force=True) == 1
    assert [entry[0] for entry in database.mutations] == ["modified"]
    assert database.mutations[0][1] == str(target.resolve())
    assert [entry[0] for entry in asset_calls] == ["invalidate", "asset-modified"]
    assert all(str(temporary) not in str(entry) for entry in database.mutations)


@pytest.mark.parametrize(
    "staging_name,destination_exists",
    [
        ("TankBattle.py.tmp.47892.966d5eeabe93", False),
        (".editor-save", True),
    ],
)
def test_already_imported_staging_file_cannot_replace_target_script_identity(
    engine,
    tmp_path,
    staging_name,
    destination_exists,
):
    from infernux.components.registry import get_component_registrations, unregister_component_script
    from infernux.components.script_loader import retire_script_module
    from infernux.lib import ResourceType

    scripts = tmp_path / "Assets" / "Scripts"
    scripts.mkdir(parents=True)
    target = scripts / "TankBattle.py"
    staging = scripts / staging_name
    source = "from infernux import InxComponent\nclass AtomicPublicationTank(InxComponent):\n    marker = 1\n"
    database = engine.get_asset_database()
    handler = ResourceChangeHandler(_EngineProbe(database), project_path=str(tmp_path))
    try:
        target_guid = ""
        if destination_exists:
            target.write_text(source, encoding="utf-8")
            handler.on_created(_event(target))
            handler.process_pending_reloads(force=True)
            _run_script_publication_owner(handler)
            target_guid = database.get_guid_from_path(str(target))
        before = [entry for entry in get_component_registrations(project_root=str(tmp_path)) if entry.project_script]
        assert len(before) == int(destination_exists)
        staging.write_text(source.replace("marker = 1", "marker = 2"), encoding="utf-8")
        handler.on_created(_event(staging))
        handler.process_pending_reloads(force=True)
        staging_guid = database.get_guid_from_path(str(staging))
        assert staging_guid and staging_guid != target_guid

        os.replace(staging, target)
        handler.on_moved(_event(staging, destination=target))
        handler.process_pending_reloads(force=True)
        _run_script_publication_owner(handler)

        assert database.get_guid_from_path(str(target)) == (target_guid or staging_guid)
        assert database.get_meta_by_path(str(target)).get_resource_type() == ResourceType.Script
        assert not database.get_guid_from_path(str(staging))
        assert not Path(str(staging) + ".meta").exists()
        after = [entry for entry in get_component_registrations(project_root=str(tmp_path)) if entry.project_script]
        assert len(after) == 1
        if destination_exists:
            assert after[0].script_path == before[0].script_path
            assert after[0].type_name == before[0].type_name
        else:
            assert Path(after[0].script_path) == target
            assert after[0].type_name == "AtomicPublicationTank"
        published = handler._script_change_collector.journal.last_known_good(str(target))
        assert published is not None
        assert b"marker = 2" in published.source
    finally:
        unregister_component_script(str(target))
        retire_script_module(str(target))
        database.delete_asset(str(target))
        database.delete_asset(str(staging))


def test_dirty_scene_still_blocks_external_reimport_for_user_arbitration(
    monkeypatch,
    tmp_path,
):
    from infernux.engine.interaction import (
        DocumentCapability,
        DocumentKey,
        DocumentKind,
        DocumentRegistry,
        DocumentState,
    )

    class _SceneController:
        @staticmethod
        def reload_from_resource(*, document_id: str, resource_path: str):
            del document_id, resource_path
            return True

    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)
    target = tmp_path / "Dirty.scene"
    target.write_text("{}", encoding="utf-8")
    target_path = str(target.resolve())
    database.guid_by_path[target_path] = "scene-guid"
    registry = DocumentRegistry()
    document = registry.create(
        DocumentKind.SCENE,
        "Dirty",
        key=DocumentKey.asset(DocumentKind.SCENE, "scene-guid"),
        resource_path=target_path,
        revision=1,
        saved_revision=0,
        capabilities=DocumentCapability.SAVE | DocumentCapability.DISCARD,
        controller=_SceneController(),
    )

    target.write_text('{"changed": true}', encoding="utf-8")
    handler.on_modified(_event(target))

    assert handler.process_pending_reloads(force=True) == 1
    assert database.mutations == []
    assert asset_calls == []
    assert document.state is DocumentState.CONFLICT
    assert document.is_dirty


@pytest.mark.parametrize("kind", ["scene", "render_effect"])
def test_failed_external_import_retries_without_acknowledging_unpublished_bytes(
    monkeypatch,
    tmp_path,
    kind,
):
    from infernux.engine.interaction import (
        DocumentKey,
        DocumentKind,
        DocumentRegistry,
        DocumentState,
    )

    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    _patch_asset_manager(monkeypatch, [])
    path = tmp_path / ("Retry.scene" if kind == "scene" else "Retry.effect")
    content = {
        "$schema": "infernux.render_effect",
        "feature_type": "infernux.post.bloom",
        "parameters": {},
        "dependencies": [],
    }
    path.write_text(json.dumps(content), encoding="utf-8")
    database.guid_by_path[str(path)] = "retry-guid"
    reloads = []
    registry = DocumentRegistry.instance()
    document = registry.create(
        DocumentKind(kind),
        "Retry",
        key=DocumentKey.asset(DocumentKind(kind), "retry-guid"),
        resource_path=str(path),
        controller=SimpleNamespace(reload_from_resource=lambda **_kwargs: reloads.append(True)),
    )
    loaded_baseline = document.durable_file_state
    attempts = []
    reimport = AssetManager.reimport_asset

    def fail_first_import(asset_path, **kwargs):
        attempts.append(asset_path)
        if len(attempts) == 1:
            raise _AssetImportNotReady("source is temporarily unavailable")
        return reimport(asset_path, **kwargs)

    monkeypatch.setattr(AssetManager, "reimport_asset", fail_first_import)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    handler.on_modified(_event(path))
    assert handler.process_pending_reloads(force=True) == 1
    assert handler.pending_count == 1
    assert document.durable_file_state is loaded_baseline
    assert document.external_file_state is not None
    assert document.state is DocumentState.CONFLICT
    assert not reloads
    external_revision = document.external_revision

    assert handler.process_pending_reloads(force=True) == 1

    assert len(attempts) == 2
    assert reloads == [True]
    assert handler.pending_count == 0
    assert document.state is DocumentState.READY
    assert document.external_file_state is None
    assert document.external_revision == external_revision
    assert (
        registry.durable_resource_content_changed(
            str(path),
            guid="retry-guid",
        )
        is False
    )


def test_initial_script_scan_publishes_artifact_for_main_thread(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    valid = assets / "valid.py"
    invalid = assets / "invalid.py"
    particle = assets / "smoke.particle.py"
    valid.write_text("value = 1\n", encoding="utf-8")
    invalid.write_text("def broken(:\n", encoding="utf-8")
    particle.write_text("this is checked by the particle compiler\n", encoding="utf-8")

    manager = ResourcesManager(str(tmp_path), _EngineProbe(_AssetDatabaseProbe()))
    commits = []
    monkeypatch.setattr(
        "infernux.components.script_loader.set_script_error",
        lambda path, message: commits.append(("set", path, message, threading.get_ident())),
    )
    monkeypatch.setattr(
        "infernux.components.script_loader._clear_script_error",
        lambda path: commits.append(("clear", path, threading.get_ident())),
    )

    worker = threading.Thread(target=manager._initial_script_scan)
    worker.start()
    worker.join()
    assert commits == []

    assert manager.process_pending_reloads() == 2
    # Initial scan is one ready-barrier transaction: an invalid member keeps
    # every valid member out of the live registry until the whole scan passes.
    assert {entry[0] for entry in commits} == {"set"}
    assert all(entry[-1] == threading.get_ident() for entry in commits)
    manager.cleanup()


def test_prepare_startup_finishes_refresh_before_the_watcher_loop(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "valid.py").write_text("value = 1\n", encoding="utf-8")
    (assets / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    manager = ResourcesManager(str(tmp_path), _EngineProbe(_AssetDatabaseProbe()))
    published = []
    monkeypatch.setattr(
        "infernux.components.script_loader.set_script_error",
        lambda path, message: published.append(("set", path, message)),
    )
    monkeypatch.setattr(
        "infernux.components.script_loader._clear_script_error",
        lambda path: published.append(("clear", path)),
    )
    started = []
    monkeypatch.setattr(
        manager,
        "start",
        lambda **kwargs: started.append(kwargs.get("skip_initial_scan")),
    )

    manager.prepare_startup()

    assert started == [True]
    assert manager._startup_prepared is True
    assert manager._event_handler is not None
    assert manager._event_handler._initial_scan_transaction_id is None
    assert manager.process_pending_reloads(force=True) == 0
    assert {entry[0] for entry in published} == {"set"}
    manager.cleanup()


def test_background_asset_refresh_uses_native_transaction_result(tmp_path):
    class Database(_AssetDatabaseProbe):
        refresh_pending = True

        def __init__(self):
            super().__init__()
            self.commit_attempts = 0
            self.refresh_restarts = 0

        def try_commit_refresh(self):
            self.commit_attempts += 1
            if self.commit_attempts == 1:
                self.refresh_restarts += 1
                return False
            self.refresh_pending = False
            return True

    database = Database()
    manager = ResourcesManager(str(tmp_path), _EngineProbe(database))
    manager._startup_prepared = True

    assert manager.process_pending_reloads() == 0
    assert database.refresh_restarts == 1
    assert database.refresh_pending is True

    assert manager.process_pending_reloads() == 0
    assert database.refresh_pending is False
    manager.cleanup()


def test_initial_scan_supersede_restarts_barrier_with_current_sources(monkeypatch, tmp_path):
    first = tmp_path / "Assets" / "first.py"
    second = tmp_path / "Assets" / "second.py"
    first.parent.mkdir()
    first.write_text("value = 'old-first'\n", encoding="utf-8")
    second.write_text("value = 'old-second'\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()), project_path=str(tmp_path))
    published = []
    monkeypatch.setattr(
        handler,
        "_publish_valid_script",
        lambda path, **kwargs: published.append((path, kwargs["source"])) or True,
    )

    initial = handler.begin_script_transaction((str(first), str(second)), initial_scan=True)
    handler._check_script(
        str(first),
        origin="initial_scan",
        change_kind="initial_scan",
        transaction_id=initial,
    )
    handler._check_script(
        str(second),
        origin="initial_scan",
        change_kind="initial_scan",
        transaction_id=initial,
    )
    assert handler.process_script_worker(max_items=1) == 1
    handler._drain_script_results()
    assert published == []

    first.write_text("value = 'new-first'\n", encoding="utf-8")
    handler._check_script(str(first), origin="editor", force=True)
    assert handler.process_script_worker(max_items=1) == 1
    handler._drain_script_results()

    # The old barrier is gone and has been replaced by one transaction.  The
    # replacement reads both files, including the newer first source.
    assert handler._initial_scan_transaction_id is not None
    assert handler._initial_scan_transaction_id != initial
    assert published == []
    assert handler.process_script_worker() == 2
    handler._drain_script_results()

    assert {path_key(path) for path, _source in published} == {
        path_key(str(first)),
        path_key(str(second)),
    }
    assert dict((path_key(path), source) for path, source in published)[path_key(str(first))] == first.read_bytes()
    assert handler._initial_scan_transaction_id is None


def test_restarted_initial_scan_removes_deleted_members_and_ignores_late_old_result(monkeypatch, tmp_path):
    first = tmp_path / "Assets" / "first.py"
    deleted = tmp_path / "Assets" / "deleted.py"
    first.parent.mkdir()
    first.write_text("value = 'first'\n", encoding="utf-8")
    deleted.write_text("value = 'deleted'\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()))
    published = []
    handler._publish_valid_script = lambda path, **_kwargs: published.append(path) or True

    initial = handler.begin_script_transaction((str(first), str(deleted)), initial_scan=True)
    handler._check_script(
        str(first),
        origin="initial_scan",
        change_kind="initial_scan",
        transaction_id=initial,
    )
    handler._check_script(
        str(deleted),
        origin="initial_scan",
        change_kind="initial_scan",
        transaction_id=initial,
    )
    assert handler.process_script_worker(max_items=1) == 1
    handler._drain_script_results()

    deleted.unlink()
    first.write_text("value = 'new-first'\n", encoding="utf-8")
    handler._check_script(str(first), origin="editor", force=True)
    handler._restart_initial_scan_transaction(handler._script_transactions[initial])
    replacement_id = handler._initial_scan_transaction_id
    replacement = handler._script_transactions[replacement_id]
    assert replacement.expected_paths == [str(first)]

    # The old deleted member was still queued.  It is consumed as retired
    # work and cannot recreate the old initial barrier.
    handler.process_script_worker()
    handler._drain_script_results()
    assert handler._initial_scan_transaction_id is None
    assert [path_key(path) for path in published] == [path_key(str(first))]


def test_frontend_worker_does_not_publish_or_update_graph(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    database = _AssetDatabaseProbe()
    engine = _EngineProbe(database)
    manager = ResourcesManager(str(tmp_path), engine)
    handler = ResourceChangeHandler(engine, project_path=str(tmp_path))
    manager._event_handler = handler

    path = assets / "worker_probe.py"
    path.write_text("value = 1\n", encoding="utf-8")
    registered = []
    monkeypatch.setattr(
        "infernux.components.registry.register_component_script",
        lambda value, **_kwargs: registered.append(value),
    )
    monkeypatch.setattr(
        "infernux.components.script_loader._clear_script_error",
        lambda _path: None,
    )

    published = []
    publish = handler._publish_valid_script

    def observe_publication(path, **kwargs):
        published.append(kwargs)
        return publish(path, **kwargs)

    monkeypatch.setattr(handler, "_publish_valid_script", observe_publication)
    worker_id = []
    handler._check_script(str(path), origin="watchdog")

    def pump_frontend():
        worker_id.append(threading.get_ident())
        assert handler.process_script_worker() == 1

    worker = threading.Thread(target=pump_frontend)
    worker.start()
    worker.join()

    assert worker_id[0] != threading.get_ident()
    assert handler.dependency_graph_snapshot().modules == ()
    assert registered == []
    assert published == []
    assert engine.editor_wakes == 1
    assert handler._script_change_collector.last_known_good(str(path)) is None

    assert manager.process_pending_reloads() == 0
    assert handler.dependency_graph.module_for_path(str(path)) is not None
    assert registered == [str(path)]

    assert len(published) == 1
    assert published[0]["source"] == path.read_bytes()
    assert isinstance(published[0]["code"], types.CodeType)
    assert handler._script_change_collector.last_known_good(str(path)) is not None


def test_reload_rejection_keeps_lkg_and_live_body_at_resources_safe_point(monkeypatch, tmp_path):
    from infernux.engine.play_mode import PlayModeManager, ScriptReloadOutcome

    database = _AssetDatabaseProbe()
    engine = _EngineProbe(database)
    manager = ResourcesManager(str(tmp_path), engine)
    handler = ResourceChangeHandler(engine)
    manager._event_handler = handler
    path = tmp_path / "controller.py"
    path.write_text("value = 'A'\n", encoding="utf-8")
    monkeypatch.setattr(
        "infernux.components.registry.register_component_script",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(
        "infernux.components.script_loader._clear_script_error",
        lambda _path: None,
    )

    live_body = {"value": "old"}
    publish_results = iter(
        (
            ScriptReloadOutcome(True, True, 1),
            ScriptReloadOutcome(False, True, 0, "schema rejected"),
        )
    )

    class _PlayModeProbe:
        def prepare_script_reload_batch(self, revisions):
            return SimpleNamespace(revisions=tuple(revisions))

        def commit_script_reload_batch(self, batch):
            outcome = next(publish_results)
            if outcome.success:
                live_body["value"] = batch.revisions[0].source.decode("utf-8").split("'")[1]
            return outcome

        def rollback_script_reload_batch(self, _batch):
            return None

        def finalize_script_reload_batch(self, _batch):
            return None

    monkeypatch.setattr(
        PlayModeManager,
        "instance",
        classmethod(lambda _cls: _PlayModeProbe()),
    )

    handler._check_script(str(path))
    _run_script_publication_owner(handler)
    first_lkg = handler._script_change_collector.journal.last_known_good(str(path))
    assert first_lkg is not None
    assert first_lkg.generation == 1
    assert live_body["value"] == "A"

    path.write_text("value = 'B'\n", encoding="utf-8")
    handler._check_script(str(path))
    _run_script_publication_owner(handler)

    current_lkg = handler._script_change_collector.journal.last_known_good(str(path))
    assert current_lkg == first_lkg
    assert live_body["value"] == "A"


def test_real_observer_is_owned_joinable_and_commits_on_main(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    database = _AssetDatabaseProbe()
    manager = ResourcesManager(str(tmp_path), _EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)

    manager.start()
    deadline = time.monotonic() + 3.0
    while manager._observer is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._observer is not None
    assert manager._observer.daemon is False
    assert manager._thread.daemon is False

    path = assets / "watched.txt"
    path.write_text("content", encoding="utf-8")
    while (manager._event_handler is None or manager._event_handler.pending_count == 0) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager._event_handler.pending_count > 0
    assert database.mutations == []
    manager.process_pending_reloads(force=True)
    assert database.mutations
    assert all(entry[-1] == threading.get_ident() for entry in database.mutations)

    manager.cleanup()
    assert manager._thread is None
    assert manager._observer is None


def test_engine_exit_drains_resources_before_native_cleanup():
    order = []
    engine = Engine.__new__(Engine)
    engine._mode = RuntimeMode.Headless
    engine._process_owned_exit = False
    engine._before_exit_callback = None
    engine._play_mode_manager = None
    engine._runtime_scene_manager = None
    engine._runtime_scheduler = SimpleNamespace(clear=lambda: None)
    engine._resources_manager = SimpleNamespace(cleanup=lambda: order.append("resources"))
    engine._engine = SimpleNamespace(cleanup=lambda: order.append("native"))
    engine._gui_objects = {}

    engine.exit()

    assert order == ["resources", "native"]
    assert engine._resources_manager is None
    assert engine._engine is None


def test_dependency_transaction_publishes_helper_and_dependent_only_after_both_ready(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    helper = assets / "helper.py"
    dependent = assets / "dependent.py"
    helper.write_text("value = 1\n", encoding="utf-8")
    dependent.write_text("from helper import value\n", encoding="utf-8")

    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()), project_path=str(tmp_path))
    handler.dependency_graph.index_assets()
    published = []
    monkeypatch.setattr(
        handler,
        "_publish_valid_script",
        lambda path, **_kwargs: published.append(path) or True,
    )

    helper.write_text("value = 2\n", encoding="utf-8")
    handler._check_script(str(helper), origin="editor")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()
    assert published == []

    # The root has staged the closure and queued the dependent, but publication
    # is still blocked until the dependent frontend result is ready.
    assert handler.process_script_worker() == 1
    handler._drain_script_results()
    assert published == [str(helper.resolve()), str(dependent.resolve())]
    assert handler._script_change_collector.last_known_good(str(helper)) is not None
    assert handler._script_change_collector.last_known_good(str(dependent)) is not None


def test_dependency_failure_aborts_root_without_live_graph_or_lkg_change(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    helper = assets / "helper.py"
    dependent = assets / "dependent.py"
    helper.write_text("value = 1\n", encoding="utf-8")
    dependent.write_text("from helper import value\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()), project_path=str(tmp_path))
    handler.dependency_graph.index_assets()
    before = handler.dependency_graph.module_for_path(str(helper)).source_hash
    published = []
    monkeypatch.setattr(
        handler,
        "_publish_valid_script",
        lambda path, **_kwargs: published.append(path) or True,
    )

    def compile_with_dependent_failure(source):
        if b"broken" in source:
            raise SyntaxError("dependent syntax failure")
        return compile(source, "<transaction-test>", "exec")

    handler._script_change_collector._compile_source = compile_with_dependent_failure
    helper.write_text("value = 2\n", encoding="utf-8")
    dependent.write_text("def broken(:\n", encoding="utf-8")
    handler._check_script(str(helper), origin="editor")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()
    assert handler.process_script_worker() == 1
    handler._drain_script_results()

    assert published == []
    assert handler.dependency_graph.module_for_path(str(helper)).source_hash == before
    assert handler._script_change_collector.last_known_good(str(helper)) is None
    assert handler._script_change_collector.last_known_good(str(dependent)) is None


def test_disk_supersede_aborts_transaction_without_publishing(tmp_path):
    path = tmp_path / "controller.py"
    path.write_text("value = 'A'\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()))
    published = []
    handler._publish_valid_script = lambda path, **_kwargs: published.append(path) or True

    handler._check_script(str(path), origin="editor")
    path.write_text("value = 'B'\n", encoding="utf-8")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()

    assert published == []
    assert handler._script_change_collector.last_known_good(str(path)) is None


def test_second_member_publish_failure_rolls_back_batch_and_lkg(monkeypatch, tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()))
    tx = handler.begin_script_transaction((str(first), str(second)))
    attempted = []

    def publish(path, **_kwargs):
        attempted.append(path)
        if path_key(path) == path_key(str(second)):
            raise RuntimeError("second member rejected")
        return True

    handler._publish_valid_script = publish
    handler._check_script(str(first), origin="editor", transaction_id=tx)
    handler._check_script(str(second), origin="editor", transaction_id=tx)
    assert handler.process_script_worker() == 2
    handler._drain_script_results()

    assert attempted == [str(first.resolve()), str(second.resolve())]
    assert handler._script_change_collector.last_known_good(str(first)) is None
    assert handler._script_change_collector.last_known_good(str(second)) is None
    assert handler._script_change_collector.pending_count == 0


@pytest.mark.parametrize("raises", [False, True])
def test_edit_dependency_batch_failure_rolls_back_once_and_keeps_lkg_empty(
    monkeypatch,
    tmp_path,
    raises,
):
    from infernux.engine.play_mode import PlayModeManager

    errors = []
    monkeypatch.setattr(Debug, "log_error", lambda message, *_args, **_kw: errors.append(message))
    first = tmp_path / "first_edit_failure.py"
    second = tmp_path / "second_edit_failure.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    database = _AssetDatabaseProbe()
    database.guid_by_path[str(first.resolve())] = "first-edit-failure-guid"
    database.guid_by_path[str(second.resolve())] = "second-edit-failure-guid"
    handler = ResourceChangeHandler(_EngineProbe(database))
    transaction_id = handler.begin_script_transaction((str(first), str(second)))

    class _Batch:
        committed = False

    class _FailingEditOwner:
        is_playing = False
        is_paused = False

        def __init__(self):
            self.batch = _Batch()
            self.rollback_count = 0

        def prepare_script_reload_batch(self, _revisions):
            return self.batch

        def commit_script_reload_batch(self, _batch):
            if raises:
                raise RuntimeError("second Edit member rejected")
            return SimpleNamespace(success=False, error="second Edit member rejected")

        def rollback_script_reload_batch(self, batch):
            assert batch is self.batch
            self.rollback_count += 1

    owner = _FailingEditOwner()
    monkeypatch.setattr(
        PlayModeManager,
        "instance",
        classmethod(lambda _cls: owner),
    )

    handler._check_script(str(first), origin="editor", transaction_id=transaction_id)
    handler._check_script(str(second), origin="editor", transaction_id=transaction_id)
    assert handler.process_script_worker() == 2
    handler._drain_script_results()

    assert owner.rollback_count == 1
    assert handler._script_change_collector.last_known_good(str(first)) is None
    assert handler._script_change_collector.last_known_good(str(second)) is None

    assert len(errors) == int(raises), errors
    if raises:
        assert "second Edit member rejected" in errors[0]
    assert handler._script_transactions == {}
    assert handler._script_change_collector.current_result(str(first)) is None
    assert handler._script_change_collector.current_result(str(second)) is None


def test_graph_commit_failure_does_not_rollback_uncommitted_graph_stage(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    script = assets / "controller.py"
    script.write_text("value = 1\n", encoding="utf-8")
    handler = ResourceChangeHandler(_EngineProbe(_AssetDatabaseProbe()), project_path=str(tmp_path))
    handler.dependency_graph.index_assets()
    monkeypatch.setattr(
        handler,
        "_publish_valid_script",
        lambda _path, **_kwargs: True,
    )

    rollback_calls = []
    graph = handler.dependency_graph
    monkeypatch.setattr(
        graph,
        "commit_transaction",
        lambda _stage: (_ for _ in ()).throw(RuntimeError("graph commit failed")),
    )
    monkeypatch.setattr(
        graph,
        "rollback_transaction",
        lambda stage: rollback_calls.append(stage),
    )
    errors = []
    monkeypatch.setattr(
        Debug,
        "log_error",
        lambda message, *args, **kwargs: errors.append(str(message)),
    )

    handler._check_script(str(script), origin="editor")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()

    assert rollback_calls == []
    assert not any("Script dependency graph rollback failed" in message for message in errors)


def test_lkg_commit_failure_rolls_back_live_and_committed_dependency_graph(
    monkeypatch,
    tmp_path,
):
    from infernux.engine.play_mode import PlayModeManager

    assets = tmp_path / "Assets"
    assets.mkdir()
    script = assets / "controller.py"
    script.write_text("value = 1\n", encoding="utf-8")
    database = _AssetDatabaseProbe()
    database.guid_by_path[str(script.resolve())] = "lkg-rollback-guid"
    handler = ResourceChangeHandler(_EngineProbe(database), project_path=str(tmp_path))
    handler.dependency_graph.index_assets()
    before_record = handler.dependency_graph.module_for_path(str(script))
    assert before_record is not None
    before_hash = before_record.source_hash
    script.write_text("value = 2\n", encoding="utf-8")

    class _Batch:
        committed = False

    class _Owner:
        is_playing = True
        is_paused = False

        def __init__(self):
            self.batch = _Batch()
            self.commit_count = 0
            self.rollback_count = 0
            self.finalize_count = 0

        def prepare_script_reload_batch(self, _revisions):
            return self.batch

        def commit_script_reload_batch(self, batch):
            assert batch is self.batch
            self.commit_count += 1
            batch.committed = True
            return SimpleNamespace(success=True)

        def rollback_script_reload_batch(self, batch):
            assert batch is self.batch
            self.rollback_count += 1
            batch.committed = False

        def finalize_script_reload_batch(self, _batch):
            self.finalize_count += 1

    owner = _Owner()
    monkeypatch.setattr(
        PlayModeManager,
        "instance",
        classmethod(lambda _cls: owner),
    )
    monkeypatch.setattr(
        handler._script_change_collector,
        "commit_published_batch",
        lambda *_args, **_kwargs: False,
    )

    handler._check_script(str(script), origin="editor")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()

    assert owner.commit_count == 1
    assert owner.rollback_count == 1
    assert owner.finalize_count == 0
    assert handler._script_change_collector.last_known_good(str(script)) is None
    after_record = handler.dependency_graph.module_for_path(str(script))
    assert after_record is not None
    assert after_record.source_hash == before_hash


def test_durable_lkg_retries_finalize_without_republishing_or_rollback(
    monkeypatch,
    tmp_path,
):
    from infernux.engine.play_mode import PlayModeManager

    script = tmp_path / "durable_finalize_retry.py"
    script.write_text("value = 1\n", encoding="utf-8")
    database = _AssetDatabaseProbe()
    database.guid_by_path[str(script.resolve())] = "durable-finalize-guid"
    handler = ResourceChangeHandler(_EngineProbe(database))

    class _Batch:
        committed = False

    class _Owner:
        is_playing = True
        is_paused = False

        def __init__(self):
            self.batch = _Batch()
            self.commit_count = 0
            self.rollback_count = 0
            self.finalize_count = 0

        def prepare_script_reload_batch(self, _revisions):
            return self.batch

        def commit_script_reload_batch(self, batch):
            assert batch is self.batch
            self.commit_count += 1
            batch.committed = True
            return SimpleNamespace(success=True)

        def rollback_script_reload_batch(self, _batch):
            self.rollback_count += 1

        def finalize_script_reload_batch(self, batch):
            assert batch is self.batch
            self.finalize_count += 1
            if self.finalize_count == 1:
                raise RuntimeError("simulated native finalize interruption")

    owner = _Owner()
    monkeypatch.setattr(
        PlayModeManager,
        "instance",
        classmethod(lambda _cls: owner),
    )

    handler._check_script(str(script), origin="editor")
    assert handler.process_script_worker() == 1
    handler._drain_script_results()

    assert owner.commit_count == 1
    assert owner.rollback_count == 0
    assert owner.finalize_count == 1
    assert handler._script_change_collector.last_known_good(str(script)) is not None
    assert len(handler._script_transactions) == 1

    handler._drain_script_results()

    assert owner.commit_count == 1
    assert owner.rollback_count == 0
    assert owner.finalize_count == 2
    assert handler._script_transactions == {}
