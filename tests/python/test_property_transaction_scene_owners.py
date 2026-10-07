"""Property/history wrappers retain every actual resident Scene document owner."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.core.assets import AssetManager
from infernux.engine import project_context
from infernux.engine.interaction import DocumentRegistry, EditorInteractionCore
from infernux.engine.interaction.serialized_properties import make_attribute_property_transaction
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import CompoundCommand, SetPropertyCommand, UndoManager
from infernux.lib import SceneManager


@pytest.fixture
def resident_authors(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    monkeypatch.setattr(UndoManager, "_instance", None)
    previous_root = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    core = EditorInteractionCore()
    history = UndoManager(core.action_journal)
    files = SceneFileManager()
    files.set_asset_database(database)
    core.project_assets.configure(database.project_root, database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    native = SceneManager.instance()
    worlds = [scene, native.create_scene("B"), native.create_scene("C")]
    docs, paths, objects = [], [], []
    for index, world in enumerate(worlds):
        obj = world.create_game_object(f"Owner{index}")
        objects.append(obj)
        path = folder / f"Owner{index}.scene"
        path.write_text(world.serialize_asset(), encoding="utf-8")
        assert database.import_asset(str(path)).succeeded
        document_id = files.register_loaded_scene(world, str(path))
        docs.append(DocumentRegistry.instance().require(document_id))
        paths.append(path)
    assert files.activate_loaded_scene(worlds[0])
    history.clear()
    try:
        yield SimpleNamespace(worlds=worlds, docs=docs, paths=paths, objects=objects,
                              history=history, files=files, registry=DocumentRegistry.instance())
    finally:
        core.shutdown()
        project_context.set_project_root(previous_root)


@pytest.mark.parametrize("wrapper", ["property", "compound", "nested"])
@pytest.mark.parametrize("owners", [(1,), (1, 2)])
def test_nonactive_property_owners_survive_undo_save_and_reopen(resident_authors, wrapper, owners):
    state = resident_authors
    originals = [path.read_bytes() for path in state.paths]
    if wrapper == "property":
        edit = make_attribute_property_transaction([state.objects[i] for i in owners], "name")
        assert edit.commit("Edited").value == "applied"
    else:
        commands = [SetPropertyCommand(state.objects[i], "name", f"Owner{i}", "Edited") for i in owners]
        command = CompoundCommand(commands)
        if wrapper == "nested":
            command = CompoundCommand([CompoundCommand([command])])
        assert state.history.execute(command)
    assert [doc.is_dirty for doc in state.docs] == [i in owners for i in range(3)]
    assert [doc.revision for doc in state.docs] == [int(i in owners) for i in range(3)]
    # Focus/active-world changes must not change revision ownership during replay.
    assert state.files.activate_loaded_scene(state.worlds[2])
    state.history.undo()
    assert [obj.name for obj in state.objects] == [f"Owner{i}" for i in range(3)]
    assert not any(doc.is_dirty for doc in state.docs)
    state.history.redo()
    assert [doc.is_dirty for doc in state.docs] == [i in owners for i in range(3)]
    assert state.files.activate_loaded_scene(state.worlds[0])
    for i in owners:
        assert state.registry.request_save(state.docs[i].document_id).accepted
        assert not state.docs[i].is_dirty
        disk = json.loads(state.paths[i].read_text(encoding="utf-8"))
        assert any(obj["name"] == "Edited" for obj in disk["objects"])
        assert state.files.reload_from_resource(
            document_id=state.docs[i].document_id, resource_path=str(state.paths[i]),
        )
        loaded = state.files.scene_for_document(state.docs[i].document_id)
        assert loaded.find("Edited") is not None
    assert state.paths[0].read_bytes() == originals[0]
    assert state.worlds[0].find("Owner0") is not None


def test_asset_child_does_not_mark_the_active_scene(resident_authors):
    state = resident_authors
    asset = SimpleNamespace(value=0)
    asset_command = SetPropertyCommand(asset, "value", 0, 1)
    asset_command.marks_dirty = False
    command = CompoundCommand([asset_command, SetPropertyCommand(state.objects[1], "name", "Owner1", "Edited")])
    assert state.history.execute(command)
    assert [doc.is_dirty for doc in state.docs] == [False, True, False]
    assert asset.value == 1


def test_default_and_explicit_owner_are_only_marked_once(resident_authors):
    state = resident_authors
    default_target = SimpleNamespace(value=0)
    command = CompoundCommand([
        SetPropertyCommand(default_target, "value", 0, 1),
        SetPropertyCommand(state.objects[0], "name", "Owner0", "Edited"),
    ])
    assert state.history.execute(command)
    assert [doc.revision for doc in state.docs] == [1, 0, 0]
    state.history.undo()
    assert [doc.revision for doc in state.docs] == [0, 0, 0]


def test_document_revision_wrapper_preserves_nested_scene_owners(resident_authors):
    state = resident_authors
    assert state.history.execute(SetPropertyCommand(state.objects[1], "name", "Owner1", "Edited"))
    wrapped = state.history.action_journal.peek_undo().action
    assert wrapped.scene_world_ids() == (state.worlds[1].world_id,)
    group = CompoundCommand([wrapped])
    assert group.scene_world_ids() == (state.worlds[1].world_id,)


@pytest.mark.parametrize("kind", ["native", "python"])
def test_component_properties_mark_all_resident_owners(resident_authors, kind):
    from infernux.ui import UIImage
    from infernux.engine.interaction.serialized_properties import make_python_component_property_transaction

    state = resident_authors
    if kind == "native":
        targets = [obj.add_component("BoxCollider") for obj in state.objects[1:]]
        transaction = make_attribute_property_transaction(targets, "is_trigger")
        attribute, original, changed = "is_trigger", False, True
    else:
        targets = [obj.add_component(UIImage) for obj in state.objects[1:]]
        transaction = make_python_component_property_transaction(targets, "width")
        attribute, original, changed = "width", targets[0].width, 137.0
    assert transaction.commit(changed).value == "applied"
    assert [doc.is_dirty for doc in state.docs] == [False, True, True]
    assert [getattr(target, attribute) for target in targets] == [changed, changed]
    state.history.undo()
    assert not any(doc.is_dirty for doc in state.docs)
    assert [getattr(target, attribute) for target in targets] == [original, original]
    state.history.redo()
    assert [doc.is_dirty for doc in state.docs] == [False, True, True]


def test_user_action_group_replays_all_owned_revisions(resident_authors):
    state = resident_authors
    with state.history.user_action("Edit resident doors"):
        for i in (1, 2):
            assert make_attribute_property_transaction([state.objects[i]], "name").commit("Edited").value == "applied"
    assert len(state.history.action_journal.applied_entries()) == 1
    assert [doc.revision for doc in state.docs] == [0, 1, 1]
    state.history.undo()
    assert [doc.revision for doc in state.docs] == [0, 0, 0]
    assert [obj.name for obj in state.objects] == [f"Owner{i}" for i in range(3)]
    state.history.redo()
    assert [doc.revision for doc in state.docs] == [0, 1, 1]
