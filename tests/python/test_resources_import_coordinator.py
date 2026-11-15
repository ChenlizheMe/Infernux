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
        self._watcher_import = None

    def complete_pending_refresh(self):
        if not self.refresh_pending:
            raise AssertionError("completion requires pending refresh work")
        self.refresh_pending = False

    def get_guid_from_path(self, path):
        self.queries.append((path, threading.get_ident()))
        return self.guid_by_path.get(path, "")

    def import_asset(self, path, *, write_scope=None):
        self.mutations.append(("import", path, threading.get_ident()))
        self.guid_by_path[path] = "created-guid"
        return _mutation("import", path, "created-guid")

    def contains_path(self, path):
        return path in self.guid_by_path

    def get_meta_by_path(self, path):
        return None

    def get_meta_by_guid(self, guid):
        return None

    def _begin_watcher_import(self, path, on_ready, on_finished, *, metadata_only=False):
        assert self._watcher_import is None
        self._watcher_import = path, on_ready, on_finished, metadata_only
        return True

    def _complete_pending_watcher_import(self):
        from infernux.core.document_store import capture_document_file_state

        path, on_ready, on_finished, metadata_only = self._watcher_import
        self._watcher_import = None
        guid = self.guid_by_path.get(path, "")
        proposed = SimpleNamespace(
            operation="reimport" if guid else "import",
            guid=guid,
            _authored_metadata_guid="",
            _observed_metadata=capture_document_file_state(path + ".meta") if metadata_only else None,
        )
        if not on_ready(proposed, True):
            on_finished("skipped", proposed, 0, None)
            return
        result = self.reimport_asset(path) if guid else self.import_asset(path)
        if result:
            outcome = "published" if on_ready(result, False) else "superseded"
        else:
            outcome = "failed"
        on_finished(outcome, result, 0, None)

    def reimport_asset(self, path):
        self.mutations.append(("modified", path, threading.get_ident()))
        return _mutation("reimport", path, self.guid_by_path.get(path, ""))

    def delete_asset(self, path, *, write_scope=None):
        self.mutations.append(("deleted", path, threading.get_ident()))
        self.guid_by_path.pop(path, None)
        return _mutation("delete", path)

    def move_asset(self, old_path, new_path, *, write_scope=None):
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


def test_invalid_native_asset_waits_for_source_change_instead_of_retrying(monkeypatch, tmp_path):
    from infernux.lib import AssetMutationErrorCode

    database = _AssetDatabaseProbe()
    handler = ResourceChangeHandler(_EngineProbe(database))
    _patch_asset_manager(monkeypatch, [])
    path = tmp_path / "Damaged.mat"
    path.write_text("invalid material", encoding="utf-8")
    imports = []

    def reject(source):
        imports.append(source)
        database.guid_by_path[source] = "failed-guid"
        result = _mutation("import", source, "failed-guid")
        result.succeeded = False
        result.error_code = AssetMutationErrorCode.IMPORT_FAILED
        result.error = "Invalid material"
        return result

    database.import_asset = reject
    handler.on_created(_event(path))
    assert handler.process_pending_reloads(force=True) == 1
    assert handler.process_pending_reloads(force=True) == 0
    assert handler.pending_count == 0 and imports == [str(path)]
    path.write_text("corrected material", encoding="utf-8")
    handler.on_modified(_event(path))
    assert handler.process_pending_reloads(force=True) == 1
    assert database.mutations[-1][0] == "modified"


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
    assert not notifications
    assert len(errors) == 1

    # A restored durable document may have identical bytes; the outstanding
    # rejected candidate still requires one successful runtime publication.
    monkeypatch.setattr(DocumentRegistry.instance(), "durable_resource_content_changed", lambda *_a, **_kw: False)
    reimports = []
    def reimport(path):
        reimports.append(path)
        return _mutation("reimport", path, "stable-guid")

    monkeypatch.setattr(database, "reimport_asset", reimport)
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
    temporary = tmp_path / "atomic.mat.inx-tmp.123.123456.7"
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
def test_failed_external_import_waits_for_a_new_event_without_acknowledging_bytes(
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
    reimport = database.reimport_asset

    def fail_first_import(asset_path):
        attempts.append(asset_path)
        if len(attempts) == 1:
            from infernux.lib import AssetMutationErrorCode

            result = _mutation("reimport", asset_path, "retry-guid")
            result.succeeded = False
            result.database_committed = False
            result.error_code = AssetMutationErrorCode.IMPORT_FAILED
            result.error = f"failed to open source file: {asset_path}"
            return result
        return reimport(asset_path)

    monkeypatch.setattr(database, "reimport_asset", fail_first_import)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    handler.on_modified(_event(path))
    assert handler.process_pending_reloads(force=True) == 1
    assert handler.pending_count == 0
    assert document.durable_file_state is loaded_baseline
    assert document.external_file_state is not None
    assert document.state is DocumentState.CONFLICT
    assert not reloads
    external_revision = document.external_revision
    handler.on_modified(_event(path))
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


def test_initial_script_scan_compiles_on_worker_and_publishes_on_owner(tmp_path):
    import sys
    assets = tmp_path / "Assets"
    assets.mkdir()
    valid = assets / "ResourceScanValid.py"
    invalid = assets / "ResourceScanBroken.py"
    valid.write_text("value = 1\n", encoding="utf-8")
    invalid.write_text("def broken(:\n", encoding="utf-8")
    manager = ResourcesManager(str(tmp_path), _EngineProbe(_AssetDatabaseProbe()))
    manager._initial_script_scan()
    worker = threading.Thread(target=manager._event_handler.process_script_worker)
    worker.start()
    worker.join()
    assert "ResourceScanValid" not in sys.modules
    assert manager.process_pending_reloads(force=True) == 2
    states = manager._event_handler.reload.states
    assert states[str(valid.resolve())].status == "active"
    assert states[str(invalid.resolve())].status == "failed"
    assert sys.modules["ResourceScanValid"].value == 1
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










def test_real_observer_is_owned_joinable_and_commits_on_main(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    database = _AssetDatabaseProbe()
    manager = ResourcesManager(str(tmp_path), _EngineProbe(database))
    asset_calls = []
    _patch_asset_manager(monkeypatch, asset_calls)

    try:
        manager.start()
        deadline = time.monotonic() + 3.0
        observer = manager._observer
        while (observer is None or not observer.is_alive()) and time.monotonic() < deadline:
            time.sleep(0.01)
            observer = manager._observer
        assert observer is not None and observer.is_alive()
        assert observer.daemon is False
        assert manager._thread.daemon is False

        path = assets / "watched.txt"
        path.write_text("content", encoding="utf-8")
        deadline = time.monotonic() + 3.0
        while manager._event_handler.pending_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert manager._event_handler.pending_count > 0
        assert database.mutations == []
        # A later filesystem event can supersede the first claimed import.
        deadline = time.monotonic() + 3.0
        while not database.mutations and time.monotonic() < deadline:
            manager.process_pending_reloads(force=True)
            if not database.mutations:
                time.sleep(0.01)
        assert any(entry[0] == "import" and path_key(entry[1]) == path_key(path) for entry in database.mutations)
        assert all(entry[-1] == threading.get_ident() for entry in database.mutations)
    finally:
        manager.cleanup()

    assert manager._thread is None
    assert manager._observer is None


def test_engine_exit_drains_resources_before_plugin_and_native_cleanup(monkeypatch):
    from infernux.application import Application

    order = []
    engine = Engine.__new__(Engine)
    engine._mode = RuntimeMode.Headless
    engine._process_owned_exit = False
    engine._cpu_work_owner = None
    engine._cpu_range_tasks = []
    engine._result_consumers = []
    engine._before_exit_callback = None
    engine._play_mode_manager = None
    engine._player_runtime = None
    engine._runtime_scene_manager = None
    engine._runtime_scheduler = SimpleNamespace(clear=lambda: None)
    engine._resources_manager = SimpleNamespace(cleanup=lambda: order.append("resources"))
    engine._engine = SimpleNamespace(_require_cleanup_boundary=lambda: None,
                                     cleanup=lambda: order.append("native"))
    engine._gui_objects = {}
    monkeypatch.setattr(engine, "_shutdown_plugins_for_exit", lambda: order.append("plugins"))
    monkeypatch.setattr(Application, "_current_engine", lambda: engine)

    engine.exit()

    assert order == ["resources", "plugins", "native"]
    assert engine._resources_manager is None
    assert engine._engine is None


















@pytest.fixture(autouse=True)
def _resource_test_project(tmp_path, monkeypatch):
    monkeypatch.setattr(_AssetDatabaseProbe, "project_root", str(tmp_path), raising=False)
    handlers = []
    original = ResourceChangeHandler.__init__
    def create(self, *args, **kwargs):
        original(self, *args, **kwargs)
        handlers.append(self)
    monkeypatch.setattr(ResourceChangeHandler, "__init__", create)
    yield
    for handler in handlers:
        handler.reload.close()
