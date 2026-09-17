"""Nested source namespaces survive outer asset authoring without live templates."""
import copy

import pytest

from Infernux.components import InxComponent, FieldType, serialized_field
from Infernux.components.ref_wrappers import ComponentRef
from Infernux.engine.prefab_manager import (
    PrefabDocumentError, _read_prefab_document, _validate_prefab_document,
    instantiate_prefab as _instantiate_prefab, save_prefab, serialize_prefab_document,
)
from Infernux.engine.prefab_overrides import (
    apply_overrides_to_prefab, build_prefab_apply_command, revert_overrides,
    compute_overrides, resolve_prefab_instance_root,
    resolve_scene_prefab_documents,
)


class _NestedReferences(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.COMPONENT)


def instantiate_prefab(*, file_path, guid, scene, parent=None):
    from pathlib import Path
    from types import SimpleNamespace
    directory = Path(file_path).parent
    sources = {key + "-guid": str(directory / (key + ".prefab")) for key in ("inner", "outer", "third")}
    database = SimpleNamespace(get_path_from_guid=lambda guid: sources.get(guid, ""))
    return _instantiate_prefab(file_path=file_path, guid=guid, scene=scene, parent=parent, asset_database=database)


def _make_nested(scene, tmp_path):
    inner = scene.create_game_object("Inner")
    child = scene.create_game_object("Target")
    child.set_parent(inner)
    collider = child.add_component("BoxCollider")
    references = _NestedReferences()
    inner.add_py_component(references)
    references.target = ComponentRef(collider)
    inner_path = str(tmp_path / "inner.prefab")
    assert save_prefab(inner, inner_path)
    outer = scene.create_game_object("Outer")
    for _ in range(2):
        instance = instantiate_prefab(file_path=inner_path, guid="inner-guid", scene=scene, parent=outer)
        assert instance is not None
    outer_path = str(tmp_path / "outer.prefab")
    return outer, inner_path, outer_path


def test_repeated_nested_prefabs_have_distinct_outer_identity_namespaces(scene, tmp_path):
    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    before = outer.serialize_document()
    assert save_prefab(outer, outer_path)
    assert outer.serialize_document() == before
    document = _read_prefab_document(outer_path)
    first, second = document["root_object"]["children"]
    for node in (first, second):
        link = node["nested_prefab"]
        assert link["guid"] == "inner-guid"
        assert link["baseline"] == _read_prefab_document(inner_path)["root_object"]
        assert [node["local_id"], 1] in link["object_sources"]
    assert {pair[0] for pair in first["nested_prefab"]["object_sources"]}.isdisjoint(
        pair[0] for pair in second["nested_prefab"]["object_sources"])
    assert {pair[0] for pair in first["nested_prefab"]["component_sources"]}.isdisjoint(
        pair[0] for pair in second["nested_prefab"]["component_sources"])
    instance = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    assert instance is not None
    assert compute_overrides(instance, outer_path) == []
    for nested in instance.get_children():
        assert nested.prefab_guid == "inner-guid" and nested.prefab_root
        assert resolve_prefab_instance_root(nested.get_child(0)) is nested
        reference = nested.get_py_component(_NestedReferences).target
        assert reference.game_object.id == nested.get_child(0).id
    assert serialize_prefab_document(instance)["root_object"]["children"] == document["root_object"]["children"]


def test_outer_apply_history_and_revert_keep_nested_source_documents(scene, tmp_path):
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    first = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    nested = first.get_child(0)
    original_links = [node["nested_prefab"] for node in _read_prefab_document(path)["root_object"]["children"]]
    target_id = nested.get_child(0).id
    nested.get_child(0).name = "Edited In Outer"
    command = build_prefab_apply_command(first, path)
    for action in (command.execute, command.undo, command.redo):
        action()
        assert [node["nested_prefab"] for node in _read_prefab_document(path)["root_object"]["children"]] == original_links
        assert first.get_child(0).get_child(0).id == target_id
    second.get_child(0).get_child(0).name = "Private"
    assert revert_overrides(second, path)
    assert second.get_child(0).get_child(0).name == "Edited In Outer"
    assert [node["nested_prefab"] for node in serialize_prefab_document(second)["root_object"]["children"]] == original_links


def test_three_levels_retain_inner_source_projection(scene, tmp_path):
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    third = scene.create_game_object("Third")
    instantiate_prefab(file_path=path, guid="outer-guid", scene=scene, parent=third)
    third_path = str(tmp_path / "third.prefab")
    assert save_prefab(third, third_path)
    document = _read_prefab_document(third_path)
    middle_link = document["root_object"]["children"][0]["nested_prefab"]
    assert middle_link["guid"] == "outer-guid"
    assert [node["nested_prefab"]["guid"] for node in middle_link["baseline"]["children"]] == ["inner-guid", "inner-guid"]
    instance = instantiate_prefab(file_path=third_path, guid="third-guid", scene=scene)
    for nested in instance.get_child(0).get_children():
        assert nested.prefab_guid == "inner-guid"
        assert nested.get_py_component(_NestedReferences).target.game_object.id == nested.get_child(0).id


def test_inner_apply_keeps_outer_membership_and_other_nested_instances(scene, tmp_path):
    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, outer_path)
    instance = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    first, second = instance.get_children()
    before = [node.id for node in instance.get_children()]
    first.get_child(0).name = "Inner Source Edit"
    command = build_prefab_apply_command(first, inner_path)
    for action in (command.execute, command.undo, command.redo):
        action()
        assert [node.id for node in instance.get_children()] == before
        assert first._prefab_source_document["outer_source_id"] != second._prefab_source_document["outer_source_id"]
    assert second.get_child(0).name == "Inner Source Edit"
    assert apply_overrides_to_prefab(instance, outer_path)
    assert compute_overrides(instance, outer_path) == []


def test_private_nested_addition_survives_other_instance_apply(scene, tmp_path):
    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, outer_path)
    first = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    peer = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    private = instantiate_prefab(file_path=inner_path, guid="inner-guid", scene=scene, parent=peer)
    identity = private.id
    first.add_component("BoxCollider")
    assert apply_overrides_to_prefab(first, outer_path)
    private = scene.find_by_id(identity)
    assert private is not None and private.prefab_guid == "inner-guid" and private.prefab_root
    assert private._prefab_source_document.get("outer_source_id", 0) == 0
    assert resolve_prefab_instance_root(private.get_child(0)) is private
    assert apply_overrides_to_prefab(peer, outer_path)
    private = scene.find_by_id(identity)
    assert private._prefab_source_document["outer_source_id"] > 0
    assert len(_read_prefab_document(outer_path)["root_object"]["children"]) == 3


def test_cook_resolves_inner_source_even_when_outer_source_is_unchanged(scene, tmp_path):
    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, outer_path)
    instance = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    saved = {"objects": [instance.serialize_document()]}
    before = copy.deepcopy(saved)
    inner_source = _read_prefab_document(inner_path)["root_object"]
    inner_source["children"][0]["name"] = "Latest Inner"
    sources = {"inner-guid": inner_source, "outer-guid": _read_prefab_document(outer_path)["root_object"]}
    result = resolve_scene_prefab_documents(saved, sources.__getitem__)
    assert saved == before
    for old, new in zip(saved["objects"][0]["children"], result["objects"][0]["children"]):
        assert new["children"][0]["name"] == "Latest Inner"
        assert new["children"][0]["id"] == old["children"][0]["id"]
        assert new["prefab_source"]["outer_source_id"] == old["prefab_source"]["outer_source_id"]
    assert resolve_scene_prefab_documents(result, sources.__getitem__) == result


def test_template_cache_tracks_inner_dependency_changes(scene, tmp_path):
    from Infernux.engine.prefab_manager import save_prefab_document

    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, outer_path)
    old = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    document = _read_prefab_document(inner_path)
    document["root_object"]["children"][0]["name"] = "Changed Only In Inner Asset"
    assert save_prefab_document(document, inner_path)
    new = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    assert old.get_child(0).get_child(0).name == "Target"
    assert all(child.get_child(0).name == "Changed Only In Inner Asset" for child in new.get_children())


def test_scene_refresh_resolves_nested_source_introduced_by_outer_update(scene, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from Infernux.engine.prefab_manager import save_prefab_document
    from Infernux.engine.scene_manager import SceneFileManager

    outer, inner_path, outer_path = _make_nested(scene, tmp_path)
    for child in list(outer.get_children()):
        scene.destroy_game_object(child)
    scene.process_pending_destroys()
    assert save_prefab(outer, outer_path)
    instance = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    author = instantiate_prefab(file_path=outer_path, guid="outer-guid", scene=scene)
    instantiate_prefab(file_path=inner_path, guid="inner-guid", scene=scene, parent=author)
    assert save_prefab(author, outer_path)
    # The old scene has never contained an instance of the new inner source.
    scene.destroy_game_object(outer)
    scene.destroy_game_object(author)
    scene.process_pending_destroys()
    source = _read_prefab_document(inner_path)
    source["root_object"]["children"][0]["name"] = "Newly Introduced Latest Inner"
    assert save_prefab_document(source, inner_path)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    manager = SceneFileManager()
    manager._asset_database = SimpleNamespace(
        get_path_from_guid=lambda guid: inner_path if guid == "inner-guid" else outer_path)
    instance_id = instance.id
    assert instance.prefab_guid == "outer-guid"
    assert len(_read_prefab_document(outer_path)["root_object"]["children"]) == 1
    assert scene.find_by_id(instance_id) is not None
    manager.sync_all_prefab_instances(scene)
    instance = scene.find_by_id(instance_id)
    assert instance.get_child(0).get_child(0).name == "Newly Introduced Latest Inner"


def test_copy_nested_root_is_new_outer_instance_but_keeps_inner_identity(scene, tmp_path):
    from Infernux.engine.component_restore import clone_game_object_transactionally

    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    instance = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    copied = clone_game_object_transactionally(scene, instance.get_child(0), parent=instance)
    assert copied.prefab_guid == "inner-guid" and copied.prefab_root
    assert copied._prefab_source_document.get("outer_source_id", 0) == 0
    assert copied.get_py_component(_NestedReferences).target.game_object.id == copied.get_child(0).id
    assert apply_overrides_to_prefab(instance, path)
    children = scene.find_by_id(instance.id).get_children()
    assert len({obj._prefab_source_document["outer_source_id"] for obj in children}) == 3


def test_recursive_source_apply_is_rejected_before_saving(scene, tmp_path):
    from pathlib import Path
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    instance = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    instantiate_prefab(file_path=path, guid="outer-guid", scene=scene, parent=instance)
    before = Path(path).read_bytes()
    assert not apply_overrides_to_prefab(instance, path)
    assert Path(path).read_bytes() == before


def test_outer_unpack_keeps_nested_links_and_undo_restores_outer_anchors(scene, tmp_path):
    from Infernux.engine.undo import PrefabUnpackCommand

    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    instance = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    before = instance.serialize_document()
    command = PrefabUnpackCommand(instance.id)
    command.execute()
    assert not instance.prefab_guid
    for child in instance.get_children():
        assert child.prefab_root and child.prefab_guid == "inner-guid"
        assert "outer_source_id" not in child._prefab_source_document
        assert child.get_py_component(_NestedReferences).target.game_object.id == child.get_child(0).id
    command.undo()
    assert instance.serialize_document() == before
    command.redo()
    assert all(child.prefab_guid == "inner-guid" for child in instance.get_children())


@pytest.mark.parametrize("api", ["game_object", "transform"])
def test_moving_nested_instance_between_outer_instances_retires_only_outer_anchor(scene, tmp_path, api):
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    first = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    nested = first.get_child(0)
    identity = nested.id
    inner_guid, inner_id = nested.prefab_guid, nested.prefab_source_id
    if api == "game_object":
        nested.set_parent(second)
    else:
        nested.transform.set_parent(second.transform)
    assert "outer_source_id" not in nested._prefab_source_document
    assert (nested.prefab_guid, nested.prefab_source_id) == (inner_guid, inner_id)
    assert nested.get_py_component(_NestedReferences).target.game_object.id == nested.get_child(0).id
    assert apply_overrides_to_prefab(second, path)
    second = scene.find_by_id(second.id)
    assert len(second.get_children()) == 3
    assert len({child._prefab_source_document["outer_source_id"] for child in second.get_children()}) == 3
    assert scene.find_by_id(identity) is not None


@pytest.mark.parametrize("command_type", ["reparent", "move", "layout"])
def test_nested_reparent_history_restores_source_namespace(scene, tmp_path, command_type):
    from Infernux.engine.undo import ReparentCommand, MoveGameObjectCommand, SceneHierarchyLayoutCommand
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    first = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    nested = first.get_child(0)
    baseline = copy.deepcopy(nested._prefab_source_document)
    if command_type == "reparent":
        command = ReparentCommand(nested.id, first.id, second.id)
    elif command_type == "move":
        command = MoveGameObjectCommand(nested.id, first.id, second.id, 0, 2)
    else:
        before = {first.id: tuple(obj.id for obj in first.get_children()),
                  second.id: tuple(obj.id for obj in second.get_children())}
        after = {first.id: before[first.id][1:], second.id: (*before[second.id], nested.id)}
        command = SceneHierarchyLayoutCommand(before, after)
    command.execute()
    assert "outer_source_id" not in nested._prefab_source_document
    command.undo()
    assert nested.get_parent() is first
    assert nested._prefab_source_document == baseline
    command.redo()
    assert nested.get_parent() is second
    assert "outer_source_id" not in nested._prefab_source_document


def test_nested_move_within_same_outer_keeps_anchor(scene, tmp_path):
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    instance = instantiate_prefab(file_path=path, guid="outer-guid", scene=scene)
    nested = instance.get_child(0)
    baseline = copy.deepcopy(nested._prefab_source_document)
    folder = scene.create_game_object("Folder")
    folder.set_parent(instance)
    nested.set_parent(folder)
    assert nested._prefab_source_document == baseline


def test_regular_prefab_child_crossing_scope_becomes_private_and_undo_restores_links(scene, tmp_path):
    from Infernux.engine.undo import ReparentCommand
    _, inner_path, _ = _make_nested(scene, tmp_path)
    first = instantiate_prefab(file_path=inner_path, guid="inner-guid", scene=scene)
    second = instantiate_prefab(file_path=inner_path, guid="inner-guid", scene=scene)
    child = first.get_child(0)
    before = child.serialize_document()
    command = ReparentCommand(child.id, first.id, second.id)
    command.execute()
    assert not child.prefab_guid and child.prefab_source_id == 0
    assert all(item.get("prefab_source_id", 0) == 0 for item in child.serialize_document()["components"])
    command.undo()
    assert child.serialize_document() == before


def test_prefab_mode_repeated_save_preserves_nested_namespace(scene, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import EditorInteractionCore, SelectionDomain
    from Infernux.lib import SceneManager

    outer, inner_path, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    core = EditorInteractionCore()
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    manager = SceneFileManager()
    manager._asset_database = SimpleNamespace(get_path_from_guid=lambda guid: inner_path if guid == "inner-guid" else path)
    try:
        assert manager.open_prefab_mode(path)
        manager._asset_database = None  # Asset import bookkeeping is not this test's subject.
        root = SceneManager.instance().get_active_scene().get_root_objects()[0]
        root.name = "Edited Outer Root"
        assert manager._save_prefab()
        saved = _read_prefab_document(path)
        assert saved["root_object"]["name"] == "Edited Outer Root"
        assert manager._save_prefab()
        assert _read_prefab_document(path) == saved
        assert all(child.prefab_guid == "inner-guid" and child.prefab_root for child in root.get_children())
        assert manager._do_exit_prefab_mode()
    finally:
        core.shutdown()


@pytest.mark.parametrize("mutation", [
    lambda link: link.update(guid=""),
    lambda link: link.update(object_sources=[[True, 1]]),
    lambda link: link["object_sources"].append(link["object_sources"][0]),
    lambda link: link.update(object_sources=[]),
    lambda link: link.update(component_sources=[[-1, 2]]),
])
def test_nested_identity_contract_rejects_invalid_author_documents(scene, tmp_path, mutation):
    outer, _, path = _make_nested(scene, tmp_path)
    assert save_prefab(outer, path)
    document = copy.deepcopy(_read_prefab_document(path))
    mutation(document["root_object"]["children"][0]["nested_prefab"])
    with pytest.raises(PrefabDocumentError, match="nested_prefab"):
        _validate_prefab_document(document)
