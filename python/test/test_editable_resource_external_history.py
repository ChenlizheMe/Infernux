"""A whole-document Undo must not overwrite an accepted collaborator revision."""
import copy
import json
from pathlib import Path

import pytest

from infernux.core.assets import AssetManager
from infernux.core.document_store import DocumentStore
from infernux.core.material import Material
from infernux.engine import project_context
from infernux.engine.interaction import (
    DocumentKind, DocumentRegistry, EditorInteractionCore, SelectionDomain,
    ensure_editable_resource_document,
)
from infernux.engine.resources_manager import ResourceChangeHandler
from infernux.engine.ui import project_file_ops
from infernux.engine.undo import UndoManager


@pytest.fixture
def material_editor(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(UndoManager, "_instance", None)
    previous_root = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    core = EditorInteractionCore()
    core.panels.register_selection_authority("inspector", (SelectionDomain.ASSET,))
    history = UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    assert project_file_ops.create_material(str(folder), "Material", database)[0]
    path = folder / "Material.mat"
    material = Material.load(str(path))
    controller = ensure_editable_resource_document(
        category="material", document_kind=DocumentKind.MATERIAL,
        file_path=str(path), resource=material, guid=material.guid, view_id="inspector",
    )
    handler = ResourceChangeHandler(engine, project_path=database.project_root)
    history.clear()
    try:
        yield controller, path, history, handler
    finally:
        DocumentStore.flush()
        core.shutdown()
        project_context.set_project_root(previous_root)


def _flush(controller):
    controller.flush_autosave(force=True)
    DocumentStore.flush()
    controller.poll_pending_writes()


def _rename(controller, name):
    value = controller.capture_document()
    value["name"] = name
    assert controller.apply_document(value, view_id="inspector", edit_key="name", description="Rename material")
    _flush(controller)


@pytest.mark.parametrize("direction", ["undo", "redo"])
def test_accepted_external_material_revision_retires_old_document_history(material_editor, direction):
    controller, path, history, handler = material_editor
    _rename(controller, "Local name")
    command = history.action_journal.peek_undo().action
    if direction == "redo":
        history.undo()
        _flush(controller)
    external = json.loads(path.read_text(encoding="utf-8"))
    external["properties"]["baseColor"]["value"] = [0.125, 0.25, 0.5, 1.0]
    path.write_text(json.dumps(external), encoding="utf-8")
    expected_bytes = path.read_bytes()
    handler._commit_modified(str(path))
    expected_live = controller.capture_document()
    assert expected_live["properties"]["baseColor"]["value"] == [0.125, 0.25, 0.5, 1.0]
    with pytest.raises(RuntimeError, match="external.*revision"):
        getattr(command, direction)()
    _flush(controller)
    assert path.read_bytes() == expected_bytes
    assert controller.capture_document() == expected_live
    assert not DocumentRegistry.instance().require(controller.document_id).is_dirty


def test_new_material_edit_undo_uses_the_accepted_external_baseline(material_editor):
    controller, path, history, handler = material_editor
    _rename(controller, "Before external")
    old_command = history.action_journal.peek_undo().action
    external = json.loads(path.read_text(encoding="utf-8"))
    external["properties"]["baseColor"]["value"] = [0.125, 0.25, 0.5, 1.0]
    path.write_text(json.dumps(external), encoding="utf-8")
    handler._commit_modified(str(path))
    _rename(controller, "After external")
    new_command = history.action_journal.peek_undo().action
    assert new_command is not old_command
    assert not old_command.can_merge(new_command)
    history.undo()
    _flush(controller)
    assert controller.capture_document()["name"] == "Before external"
    assert controller.capture_document()["properties"]["baseColor"]["value"] == [0.125, 0.25, 0.5, 1.0]
    assert json.loads(path.read_text(encoding="utf-8")) == controller.capture_document()


def test_normal_material_autosaves_do_not_retire_undo_history(material_editor):
    controller, path, history, _ = material_editor
    before = copy.deepcopy(controller.capture_document())
    _rename(controller, "Owned save")
    history.undo()
    _flush(controller)
    assert controller.capture_document() == before
    history.redo()
    _flush(controller)
    assert controller.capture_document()["name"] == "Owned save"
    assert json.loads(path.read_text(encoding="utf-8")) == controller.capture_document()
