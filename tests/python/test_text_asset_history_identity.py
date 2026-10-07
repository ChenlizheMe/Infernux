"""Text history follows a GUID across checkout changes, never a reused path."""
from pathlib import Path

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.undo import UndoManager
from infernux.host import EditorAutomationHost, MainThreadCommandQueue, OperationRegistry
from infernux.host.asset_operations import build_asset_operations
from model_test_support import remove_model_test_folder


@pytest.fixture
def text_host(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(UndoManager, "_instance", None)
    monkeypatch.setattr(EditorAutomationHost, "_provider", None)
    core = EditorInteractionCore()
    core.project_assets.configure(database.project_root, database)
    history = UndoManager(core.action_journal)
    EditorAutomationHost.set_provider(EditorAutomationHost())
    queue = MainThreadCommandQueue.instance()
    queue.drain()
    registry = OperationRegistry()
    for operation in build_asset_operations(database.project_root):
        registry.register(operation)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()

    def execute(name, **arguments):
        return registry.execute(name, arguments, capabilities=("asset.*",))

    try:
        yield execute, database, core, history, folder
    finally:
        core.shutdown()
        queue.release_owner("Text asset history test complete")
        remove_model_test_folder(database, folder)


def _asset(database, path, content=b"A original\n"):
    path.write_bytes(content)
    result = AssetManager.import_asset(str(path), database=database)
    assert result, result.error
    return result.guid


def _relocate(execute, path, moved):
    path.replace(moved)
    Path(str(path) + ".meta").replace(Path(str(moved) + ".meta"))
    path.write_bytes(b"B collaborator\n")
    execute("infernux.asset.refresh")


@pytest.mark.parametrize("notify", [False, True])
def test_text_history_undo_redo_follows_guid_when_old_path_is_reused(text_host, notify):
    execute, database, core, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    execute("infernux.asset.text.set", asset_guid=guid, content="A edited\n")
    moved = folder / "Moved.txt"
    _relocate(execute, path, moved)
    other_guid = database.get_guid_from_path(str(path))
    assert other_guid and other_guid != guid
    if notify:
        core.asset_mutations.publish_move(str(path), str(moved), guid=guid)
    for _ in range(2):
        history.undo()
        assert path.read_bytes() == b"B collaborator\n"
        assert moved.read_bytes() == b"A original\n"
        history.redo()
        assert path.read_bytes() == b"B collaborator\n"
        assert moved.read_bytes() == b"A edited\n"
        assert database.get_guid_from_path(str(path)) == other_guid
        assert database.get_guid_from_path(str(moved)) == guid


def test_text_history_rejects_removed_identity_even_if_replacement_has_same_content(text_host):
    execute, database, _, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    execute("infernux.asset.text.set", asset_guid=guid, content="A edited\n")
    command = history.action_journal.peek_undo().action
    assert database.delete_asset(str(path))
    path.unlink(missing_ok=True)
    Path(str(path) + ".meta").unlink(missing_ok=True)
    other_guid = _asset(database, path, b"A edited\n")
    assert other_guid != guid
    with pytest.raises(RuntimeError, match="identity"):
        command.undo()
    assert path.read_bytes() == b"A edited\n"
    assert database.get_guid_from_path(str(path)) == other_guid


@pytest.mark.parametrize("direction", ["undo", "redo"])
def test_text_history_rejects_external_revision(text_host, direction):
    execute, database, _, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    execute("infernux.asset.text.set", asset_guid=guid, content="A edited\n")
    command = history.action_journal.peek_undo().action
    if direction == "redo":
        history.undo()
    path.write_bytes(b"Collaborator revision\n")
    execute("infernux.asset.refresh")
    with pytest.raises(RuntimeError, match="external.*revision"):
        getattr(command, direction)()
    assert path.read_bytes() == b"Collaborator revision\n"


def test_text_history_preserves_original_crlf_bytes_on_undo(text_host):
    execute, database, _, history, folder = text_host
    path = folder / "Notes.txt"
    original = "你好\r\nSecond line\r\n".encode("utf-8")
    guid = _asset(database, path, original)
    read = execute("infernux.asset.text.read", asset_guid=guid)["content"]
    assert read == original.decode("utf-8")
    execute("infernux.asset.text.set", asset_guid=guid, content=read)
    assert not history.action_journal.applied_entries()
    execute("infernux.asset.text.set", asset_guid=guid, content="Edited\n")
    history.undo()
    assert path.read_bytes() == original
    history.redo()
    assert path.read_bytes() == b"Edited\n"


def test_text_history_import_failure_after_relocation_rolls_back_only_its_guid(text_host, monkeypatch):
    execute, database, core, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    moved = folder / "Moved.txt"
    reimport = AssetManager.reimport_asset
    failed = False

    def fail_first_reimport(candidate, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            _relocate(execute, path, moved)
            raise RuntimeError("injected import failure after relocation")
        return reimport(candidate, **kwargs)

    monkeypatch.setattr(AssetManager, "reimport_asset", fail_first_reimport)
    with pytest.raises(RuntimeError, match="injected import failure"):
        core.project_assets.set_text(str(path), "A edited\n")
    assert moved.read_bytes() == b"A original\n"
    assert path.read_bytes() == b"B collaborator\n"
    assert database.get_guid_from_path(str(moved)) == guid
    assert not history.action_journal.applied_entries()


def test_text_history_type_change_rejects_replay(text_host):
    execute, database, _, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path, b"value = 1\n")
    execute("infernux.asset.text.set", asset_guid=guid, content="value = 2\n")
    command = history.action_journal.peek_undo().action
    moved = folder / "Changed.py"
    _relocate(execute, path, moved)
    assert database.get_guid_from_path(str(moved)) == guid
    with pytest.raises(RuntimeError, match="resource type"):
        command.undo()
    assert moved.read_bytes() == b"value = 2\n"
    assert path.read_bytes() == b"B collaborator\n"


def test_text_history_conditional_write_failure_never_runs_rollback(text_host, monkeypatch):
    from infernux.core import document_store

    execute, database, _, history, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    execute("infernux.asset.text.set", asset_guid=guid, content="A edited\n")
    command = history.action_journal.peek_undo().action
    write = document_store.write_document_text
    writes = []

    def external_change_before_commit(target, content, **kwargs):
        writes.append(target)
        assert len(writes) == 1, "failed conditional commit must not attempt rollback"
        path.write_bytes(b"Changed after snapshot\n")
        return write(target, content, **kwargs)

    monkeypatch.setattr(document_store, "write_document_text", external_change_before_commit)
    monkeypatch.setattr(AssetManager, "reimport_asset", lambda *_args, **_kwargs: pytest.fail("uncommitted content was imported"))
    with pytest.raises(RuntimeError, match="changed outside the editor"):
        command.undo()
    assert path.read_bytes() == b"Changed after snapshot\n"
    assert len(writes) == 1


def test_text_history_import_error_does_not_rollback_new_external_content(text_host, monkeypatch):
    _, database, core, history, folder = text_host
    path = folder / "Notes.txt"
    _asset(database, path)
    imports = []

    def change_then_fail(candidate, **kwargs):
        imports.append(candidate)
        path.write_bytes(b"External edit during import\n")
        raise RuntimeError("injected import failure")

    monkeypatch.setattr(AssetManager, "reimport_asset", change_then_fail)
    with pytest.raises(RuntimeError, match="could not be rolled back") as failure:
        core.project_assets.set_text(str(path), "A edited\n")
    group = failure.value.__cause__
    assert isinstance(group, ExceptionGroup)
    assert "injected import failure" in str(group.exceptions[0])
    assert "external content revision" in str(group.exceptions[1])
    assert path.read_bytes() == b"External edit during import\n"
    assert len(imports) == 1
    assert not history.action_journal.applied_entries()


def test_text_history_reimport_resolves_identity_after_source_commit(text_host, monkeypatch):
    from infernux.core import document_store

    execute, database, core, _, folder = text_host
    path = folder / "Notes.txt"
    guid = _asset(database, path)
    moved = folder / "Moved.txt"
    write = document_store.write_document_text
    reimport = AssetManager.reimport_asset
    imports = []

    def move_after_commit(target, content, **kwargs):
        result = write(target, content, **kwargs)
        _relocate(execute, path, moved)
        return result

    def record_import(target, **kwargs):
        imports.append(Path(target))
        return reimport(target, **kwargs)

    monkeypatch.setattr(document_store, "write_document_text", move_after_commit)
    monkeypatch.setattr(AssetManager, "reimport_asset", record_import)
    result = execute("infernux.asset.text.set", asset_guid=guid, content="A edited\n")
    assert result["asset"]["guid"] == guid
    assert Path(result["asset"]["path"]) == moved
    assert imports == [moved]
    assert database.get_guid_from_path(str(moved)) == guid
    assert moved.read_bytes() == b"A edited\n"
    assert path.read_bytes() == b"B collaborator\n"
