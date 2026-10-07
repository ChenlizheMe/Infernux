"""Overwrites and history may not destroy a resident scene's backing asset."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.engine.asset_creation import AssetCreationResult
from infernux.engine.interaction import (
    DocumentKey, DocumentKind, DocumentRegistry, EditorActionJournal,
    ProjectAssetCommandService, SelectionService,
)
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.scene_authoring import encode_scene_document
from infernux.engine.undo import ProjectAssetDeleteCommand, UndoManager
from model_test_support import remove_model_test_folder


@pytest.fixture
def workspace(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    root = Path(database.assets_root) / tmp_path.name
    root.mkdir()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    manager = SimpleNamespace(current_scene_path="")
    monkeypatch.setattr(SceneFileManager, "_instance", manager)
    registry = DocumentRegistry()
    selection = SelectionService()
    journal = EditorActionJournal()
    history = UndoManager(journal)
    service = ProjectAssetCommandService(selection)
    service.configure(str(Path(database.assets_root).parent), database)

    def create(path, name):
        document = encode_scene_document(scene.serialize_document())
        document["name"] = name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        result = AssetManager.import_asset(str(path), database=database)
        assert result, result.error
        return str(result.guid)

    def open_scene(path, active=False):
        guid = str(database.get_guid_from_path(str(path)))
        document, _ = registry.open_or_create(
            DocumentKey.asset(DocumentKind.SCENE, guid), path.stem,
            resource_path=str(path),
        )
        if active:
            manager.current_scene_path = str(path)
        return document

    try:
        yield SimpleNamespace(
            root=root, database=database, create=create, open_scene=open_scene,
            service=service, history=history, journal=journal,
            registry=registry, selection=selection,
        )
    finally:
        registry.clear()
        service.shutdown()
        history.shutdown()
        remove_model_test_folder(database, root)


def disk_snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("operation", ("copy", "move", "create"))
@pytest.mark.parametrize("boundary", ("active", "open", "directory"))
def test_overwrite_rejected_before_command_or_creator(workspace, monkeypatch, operation, boundary):
    w = workspace
    source = w.root / "Replacement.scene"
    target = w.root / "Scenes" / "Target.scene" if boundary == "directory" else w.root / "Target.scene"
    w.create(source, "Replacement")
    guid = w.create(target, "Original")
    w.open_scene(target, active=boundary == "active")
    before = disk_snapshot(w.root)
    selection = w.selection.snapshot
    monkeypatch.setattr(w.service, "_execute", lambda *_: pytest.fail("preflight must precede history"))
    destination = target.parent if boundary == "directory" else target
    with pytest.raises(ValueError, match="active scene|open scene"):
        if operation == "create":
            w.service.create(str(w.root), lambda: pytest.fail("creator must not run"), replace_path=str(destination))
        else:
            getattr(w.service, operation)(str(source), str(destination), overwrite=True)
    assert disk_snapshot(w.root) == before
    assert w.database.get_guid_from_path(str(target)) == guid
    assert w.selection.snapshot == selection
    assert not w.journal.applied_entries()


@pytest.mark.parametrize("operation", ("copy", "move", "create"))
def test_redo_rechecks_scene_opened_after_undo(workspace, operation):
    w = workspace
    source, target = w.root / "Source.scene", w.root / "Target.scene"
    w.create(source, "Replacement")
    guid = w.create(target, "Original")
    original = disk_snapshot(w.root)
    if operation == "create":
        def creator():
            w.create(target, "Created")
            return AssetCreationResult(True, created_path=str(target))
        w.service.create(str(w.root), creator, replace_path=str(target))
    else:
        getattr(w.service, operation)(str(source), str(target), overwrite=True)
    assert target.read_bytes() != original["Target.scene"]
    w.history.undo()
    assert disk_snapshot(w.root) == original
    w.open_scene(target)
    w.history.redo()
    assert disk_snapshot(w.root) == original
    assert w.database.get_guid_from_path(str(target)) == guid
    assert not w.journal.applied_entries()
    assert w.history.can_redo
    w.registry.clear()
    w.history.redo()
    assert target.read_bytes() != original["Target.scene"]
    assert len(w.journal.applied_entries()) == 1
    w.history.undo()
    assert disk_snapshot(w.root) == original


@pytest.mark.parametrize("operation", ("copy", "create"))
def test_undo_cannot_delete_opened_new_scene(workspace, operation):
    w = workspace
    source, target = w.root / "Source.scene", w.root / "New.scene"
    w.create(source, "Source")
    if operation == "create":
        def creator():
            w.create(target, "Created")
            return AssetCreationResult(True, created_path=str(target))
        w.service.create(str(w.root), creator)
    else:
        w.service.copy(str(source), str(target))
    w.open_scene(target)
    before = disk_snapshot(w.root)
    w.history.undo()
    assert disk_snapshot(w.root) == before
    assert len(w.journal.applied_entries()) == 1
    assert w.history.can_undo
    w.registry.clear()
    w.history.undo()
    assert not target.exists()
    w.history.redo()
    assert disk_snapshot(w.root) == before


def test_delete_batch_checks_all_roots_before_backup_or_mutation(workspace):
    w = workspace
    first, opened = w.root / "First.scene", w.root / "Open.scene"
    w.create(first, "First")
    w.create(opened, "Open")
    w.open_scene(opened)
    before = disk_snapshot(w.root)
    backup = w.root.parent.parent / "Library" / "OpenSceneDeleteTest"
    command = ProjectAssetDeleteCommand(
        [str(first), str(opened)], project_root=str(w.root.parent.parent),
        backup_root=str(backup), asset_database=w.database,
    )
    try:
        with pytest.raises(ValueError, match="open scene"):
            command.execute()
        assert disk_snapshot(w.root) == before
        assert not backup.exists()
    finally:
        command.dispose()


def test_batch_copy_undo_preflights_before_removing_any_other_asset(workspace, monkeypatch):
    from infernux.engine.ui import project_file_ops

    w = workspace
    source, other = w.root / "First.scene", w.root / "Second.scene"
    destination = w.root / "Copies"
    destination.mkdir()
    w.create(source, "First")
    w.create(other, "Second")
    w.service.transfer_to_directory((str(source), str(other)), str(destination), cut=False)
    w.open_scene(destination / source.name)
    before = disk_snapshot(w.root)
    # Compound Undo visits the last copy first. It must not delete and restore
    # it before discovering that the earlier copy is a resident scene.
    monkeypatch.setattr(project_file_ops, "delete_item", lambda *_: pytest.fail("batch must be preflighted"))
    w.history.undo()
    assert disk_snapshot(w.root) == before
    assert len(w.journal.applied_entries()) == 1
