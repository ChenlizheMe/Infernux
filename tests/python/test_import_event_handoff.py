"""A watcher successor owns ordering even while an older import is deferred."""
from types import SimpleNamespace
import threading

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.import_coordinator import AssetFsEventKind as K, ImportCoordinator
from infernux.engine.resources_manager import (
    ResourceChangeHandler, _AssetImportNotReady, _AssetLocalWritePending,
)
from infernux.lib import AssetRegistry


CASES = {
    'modify_delete': ([(K.MODIFIED, 'a', '')], [(K.DELETED, 'a', '')], [(K.DELETED, 'a', '')]),
    'modify_modify': ([(K.MODIFIED, 'a', '')], [(K.MODIFIED, 'a', '')], [(K.MODIFIED, 'a', '')]),
    'modify_move': ([(K.MODIFIED, 'a', '')], [(K.MOVED, 'a', 'b')], [(K.MOVED, 'a', 'b')]),
    'move_delete': ([(K.MOVED, 'a', 'b')], [(K.DELETED, 'b', '')], [(K.DELETED, 'a', '')]),
    'move_move': ([(K.MOVED, 'a', 'b')], [(K.MOVED, 'b', 'c')], [(K.MOVED, 'a', 'c')]),
    'move_modify': ([(K.MOVED, 'a', 'b')], [(K.MODIFIED, 'b', '')], [(K.MOVED, 'a', 'b')]),
    'meta_delete': ([(K.META_MODIFIED, 'a', '')], [(K.META_DELETED, 'a', '')], [(K.META_DELETED, 'a', '')]),
    'create_modify': ([(K.CREATED, 'a', '')], [(K.MODIFIED, 'a', '')], [(K.CREATED, 'a', '')]),
    'create_delete': ([(K.CREATED, 'a', '')], [(K.DELETED, 'a', '')], []),
    'modify_replace_delete': ([(K.MODIFIED, 'a', '')], [(K.CREATED, 'a', ''), (K.DELETED, 'a', '')], [(K.DELETED, 'a', '')]),
    'move_replace_delete': ([(K.MOVED, 'a', 'b')], [(K.CREATED, 'b', ''), (K.DELETED, 'b', '')], [(K.DELETED, 'a', '')]),
    'delete_create_guid': ([(K.DELETED, 'a', '')], [(K.CREATED, 'b', '')], [(K.MOVED, 'a', 'b')]),
}


@pytest.mark.parametrize('operation', ['retry', 'defer'])
@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('watcher_thread', [False, True])
def test_handoff_composes_original_identity_with_successor(tmp_path, operation, case, watcher_thread):
    coordinator = ImportCoordinator(clock=lambda: 100.)
    old, following, expected = CASES[case]
    def submit(events, observed):
        for kind, source, target in events:
            coordinator.submit(kind, str(tmp_path / source),
                               destination=str(tmp_path / target) if target else '',
                               guid_hint='identity', observed_at=observed)
    submit(old, 10.)
    claimed, = coordinator.drain(force=True)
    # Arrival order is authoritative even if the watcher supplies an older timestamp.
    if watcher_thread:
        worker = threading.Thread(target=submit, args=(following, 9.))
        worker.start()
        worker.join(timeout=5.)
        assert not worker.is_alive()
    else:
        submit(following, 9.)
    getattr(coordinator, operation)(claimed, now=100.)
    events = coordinator.drain(force=True)
    assert [(e.kind, e.path, e.destination) for e in events] == [
        (kind, str(tmp_path / source), str(tmp_path / target) if target else '')
        for kind, source, target in expected]
    for event in events:
        assert event.guid_hint == 'identity'
        assert event.attempt == 0, 'a new observation has its own import budget'
        assert event.observed_at == 9., 'rescheduling must not fabricate a watcher observation'


@pytest.mark.parametrize('operation', ['retry', 'defer'])
def test_handoff_preserves_delete_grace_and_independent_metadata(tmp_path, operation):
    coordinator = ImportCoordinator(clock=lambda: 100., delete_grace_seconds=2.)
    path = str(tmp_path / 'a')
    coordinator.submit(K.MODIFIED, path)
    claimed, = coordinator.drain(force=True)
    coordinator.submit(K.DELETED, path, guid_hint='deleted', observed_at=100.1)
    coordinator.submit(K.META_MODIFIED, path, observed_at=100.2)
    getattr(coordinator, operation)(claimed, now=100.3)
    metadata, = coordinator.drain(now=100.4)
    assert metadata.kind is K.META_MODIFIED
    assert coordinator.drain(now=102.) == []
    deleted, = coordinator.drain(now=102.11)
    assert deleted.kind is K.DELETED and deleted.guid_hint == 'deleted'
    assert deleted.observed_at == 100.1


@pytest.mark.parametrize('operation', ['retry', 'defer'])
def test_plain_wait_changes_deadline_without_rewriting_observation(tmp_path, operation):
    coordinator = ImportCoordinator(clock=lambda: 100., retry_delay_seconds=.5)
    coordinator.submit(K.MODIFIED, str(tmp_path / 'a'), observed_at=10.)
    claimed, = coordinator.drain(force=True)
    getattr(coordinator, operation)(claimed, now=100.)
    assert coordinator.drain(now=100.4) == []
    event, = coordinator.drain(now=100.6)
    assert event.observed_at == 10.
    assert event.attempt == (1 if operation == 'retry' else 0)


def test_successor_is_not_reported_as_an_exhausted_old_import(tmp_path):
    coordinator = ImportCoordinator(max_attempts=1)
    path = str(tmp_path / 'a')
    coordinator.submit(K.MOVED, path, destination=str(tmp_path / 'b'), guid_hint='identity')
    claimed, = coordinator.drain(force=True)
    coordinator.submit(K.DELETED, str(tmp_path / 'b'))
    assert coordinator.retry(claimed)
    deleted, = coordinator.drain(force=True)
    assert deleted.kind is K.DELETED and deleted.path == path
    assert deleted.guid_hint == 'identity'


@pytest.mark.parametrize('timing', ['ordinary', 'retry', 'defer'])
def test_real_asset_delete_retires_guid_during_inflight_import(engine, tmp_path, monkeypatch, timing):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    source = tmp_path / 'External.txt'
    source.write_text('before', encoding='utf-8')
    imported = database.import_asset(str(source))
    assert imported, imported.error
    handler = ResourceChangeHandler(engine, project_path=str(tmp_path))
    event = SimpleNamespace(src_path=str(source), is_directory=False)
    try:
        if timing == 'ordinary':
            source.unlink()
            handler.on_deleted(event)
        else:
            dispatch = handler._dispatch_event
            first = True
            def interleave(current):
                nonlocal first
                if first:
                    first = False
                    source.unlink()
                    handler.on_deleted(event)
                    error = _AssetImportNotReady if timing == 'retry' else _AssetLocalWritePending
                    raise error('controlled watcher deletion during import')
                return dispatch(current)
            monkeypatch.setattr(handler, '_dispatch_event', interleave)
            handler.on_modified(event)
        for _ in range(5):
            handler.process_pending_reloads(force=True)
        assert not source.exists()
        assert not database.contains_path(str(source))
        assert not database.get_path_from_guid(imported.guid)
        assert handler.pending_count == 0
    finally:
        handler.cleanup()
        if database.contains_path(str(source)):
            database.delete_asset(str(source))


@pytest.mark.parametrize('operation', ['retry', 'defer'])
def test_completion_and_clear_retire_exact_owner_claims(tmp_path, operation):
    coordinator = ImportCoordinator()
    path = str(tmp_path / 'a')
    coordinator.submit(K.MODIFIED, path)
    first, = coordinator.drain(force=True)
    getattr(coordinator, operation)(first)
    second, = coordinator.drain(force=True)
    coordinator.complete(first)
    getattr(coordinator, operation)(second)
    assert coordinator.pending_count == 1
    third, = coordinator.drain(force=True)
    coordinator.clear()
    getattr(coordinator, operation)(third)
    assert coordinator.drain(force=True) == []
    coordinator.submit(K.MODIFIED, path)
    fourth, = coordinator.drain(force=True)
    coordinator.complete(fourth)
    coordinator.submit(K.CREATED, path)
    coordinator.submit(K.DELETED, path)
    assert coordinator.drain(force=True) == []


@pytest.mark.parametrize('operation', ['retry', 'defer'])
@pytest.mark.parametrize('successor', ['move', 'delete'])
@pytest.mark.parametrize('publication', ['before', 'after'])
def test_real_move_handoff_preserves_source_guid(engine, tmp_path, monkeypatch, operation, successor, publication):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    source, intermediate, target = (tmp_path / name for name in ('A.txt', 'B.txt', 'C.txt'))
    source.write_text('authored payload', encoding='utf-8')
    imported = database.import_asset(str(source))
    assert imported, imported.error
    handler = ResourceChangeHandler(engine, project_path=str(tmp_path))
    dispatch = handler._dispatch_event
    first = True
    def interleave(event):
        nonlocal first
        if first:
            first = False
            if publication == 'after':
                dispatch(event)
            if successor == 'move':
                intermediate.rename(target)
                handler.on_moved(SimpleNamespace(src_path=str(intermediate), dest_path=str(target), is_directory=False))
            else:
                intermediate.unlink()
                handler.on_deleted(SimpleNamespace(src_path=str(intermediate), is_directory=False))
            error = _AssetImportNotReady if operation == 'retry' else _AssetLocalWritePending
            raise error('controlled second watcher operation before move publication')
        return dispatch(event)
    monkeypatch.setattr(handler, '_dispatch_event', interleave)
    try:
        source.rename(intermediate)
        handler.on_moved(SimpleNamespace(src_path=str(source), dest_path=str(intermediate), is_directory=False))
        for _ in range(5):
            handler.process_pending_reloads(force=True)
        assert not database.contains_path(str(source))
        assert not database.contains_path(str(intermediate))
        if successor == 'move':
            assert database.get_guid_from_path(str(target)) == imported.guid
            assert target.read_text(encoding='utf-8') == 'authored payload'
        else:
            assert not database.get_path_from_guid(imported.guid)
        assert handler.pending_count == 0
        assert not handler._coordinator._inflight
    finally:
        handler.cleanup()
        for path in (source, intermediate, target):
            if database.contains_path(str(path)):
                database.delete_asset(str(path))


@pytest.mark.parametrize('operation', ['retry', 'defer'])
def test_created_asset_retirement_survives_post_import_wait(engine, tmp_path, monkeypatch, operation):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    source = tmp_path / 'Created.txt'
    source.write_text('new asset', encoding='utf-8')
    handler = ResourceChangeHandler(engine, project_path=str(tmp_path))
    dispatch = handler._dispatch_event
    imported = []
    event = SimpleNamespace(src_path=str(source), is_directory=False)
    def interleave(current):
        dispatch(current)
        if not imported:
            imported.append(database.get_guid_from_path(str(source)))
            assert imported[0]
            source.unlink()
            handler.on_deleted(event)
            error = _AssetImportNotReady if operation == 'retry' else _AssetLocalWritePending
            raise error('controlled post-import wait before a newer deletion')
    monkeypatch.setattr(handler, '_dispatch_event', interleave)
    try:
        handler.on_created(event)
        for _ in range(5):
            handler.process_pending_reloads(force=True)
        assert imported and not database.get_path_from_guid(imported[0])
        assert not database.contains_path(str(source))
        assert not handler.pending_count
        assert not handler._coordinator._inflight
    finally:
        handler.cleanup()
        if database.contains_path(str(source)):
            database.delete_asset(str(source))


def test_unregistered_move_continues_a_queued_identity(tmp_path):
    coordinator = ImportCoordinator()
    a, b, c = (str(tmp_path / name) for name in ('a', 'b', 'c'))
    coordinator.submit(K.MOVED, a, destination=b, guid_hint='identity')
    coordinator.submit(K.MOVED, b, destination=c, source_registered=False)
    event, = coordinator.drain(force=True)
    assert (event.kind, event.path, event.destination, event.guid_hint) == (K.MOVED, a, c, 'identity')
    coordinator.complete(event)
