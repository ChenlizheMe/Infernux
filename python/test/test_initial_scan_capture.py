"""Startup publishes all surviving sources when enumeration races filesystem edits."""
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.resources_manager import ResourcesManager
from infernux.lib import AssetRegistry


@pytest.mark.parametrize('survivors', [0, 2])
@pytest.mark.parametrize('timing', ['before_capture', 'at_open', 'after_capture', 'before_publication', 'recreated'])
def test_missing_initial_member_cannot_strand_or_discard_survivors(engine, tmp_path, monkeypatch, survivors, timing):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    assets = tmp_path / 'Assets'
    assets.mkdir()
    paths = [assets / f'Member{i}.py' for i in range(survivors + 1)]
    for i, path in enumerate(paths):
        path.write_text(f'VALUE = {i}\n', encoding='utf-8')
        result = database.import_asset(str(path))
        assert result, result.error
    missing = paths[0]
    manager = ResourcesManager(str(tmp_path), engine)
    handler = manager._ensure_event_handler()
    check = handler._check_script
    publish = handler._publish_script_transaction
    removed = False
    def remove_member():
        nonlocal removed
        if removed:
            return
        removed = True
        missing.unlink()
        handler.on_deleted(SimpleNamespace(src_path=str(missing), is_directory=False))
    def capture(path, **kwargs):
        if Path(path) == missing and not removed:
            if timing in ('before_capture', 'recreated'):
                remove_member()
                result = check(path, **kwargs)
                if timing == 'recreated':
                    missing.write_text('VALUE = 100\n', encoding='utf-8')
                return result
            if timing == 'after_capture':
                result = check(path, **kwargs)
                remove_member()
                return result
        return check(path, **kwargs)
    def publication(state, ready):
        if timing == 'before_publication':
            remove_member()
        return publish(state, ready)
    real_open = builtins.open
    def racing_open(path, mode='r', *args, **kwargs):
        if timing == 'at_open' and mode == 'rb' and Path(path) == missing and not removed:
            remove_member()
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(handler, '_check_script', capture)
    monkeypatch.setattr(handler, '_publish_script_transaction', publication)
    monkeypatch.setattr(builtins, 'open', racing_open)
    try:
        manager._initial_script_scan()
        for _ in range(6):
            manager.process_pending_reloads(force=True)
        assert removed
        assert handler._initial_scan_transaction_id is None
        assert not manager._startup_work_pending()
        assert not handler._script_transactions
        assert database.contains_path(str(missing)) == (timing == 'recreated')
        for path in paths:
            if not path.exists():
                continue
            good = handler._script_change_collector.last_known_good(str(path))
            assert good is not None, f'unchanged surviving member was dropped: {path.name}'
            assert good.source == path.read_bytes()
            record = handler.dependency_graph.module_for_path(str(path))
            assert record is not None and record.source_hash == good.content_hash
    finally:
        manager.cleanup()
        for path in paths:
            if database.contains_path(str(path)):
                database.delete_asset(str(path))


def test_repeated_scan_owns_a_fresh_complete_barrier(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(AssetManager, '_registry', AssetRegistry.instance())
    assets = tmp_path / 'Assets'
    assets.mkdir()
    source = assets / 'Stable.py'
    source.write_text('VALUE = 9\n', encoding='utf-8')
    assert database.import_asset(str(source))
    manager = ResourcesManager(str(tmp_path), engine)
    handler = manager._ensure_event_handler()
    try:
        for _ in range(2):
            manager._initial_script_scan()
            for _ in range(3):
                manager.process_pending_reloads(force=True)
            assert not manager._startup_work_pending()
            assert not handler._script_transactions
            good = handler._script_change_collector.last_known_good(str(source))
            assert good is not None and good.source == source.read_bytes()
    finally:
        manager.cleanup()
        database.delete_asset(str(source))
