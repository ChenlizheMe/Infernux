"""Prefab component edits use source identity across publication and cook."""
import copy
import pytest

from Infernux.engine.component_restore import deserialize_game_object_document_transactionally
from Infernux.engine.prefab_manager import (
    PrefabDocumentError, _read_prefab_document, instantiate_prefab, save_prefab,
)
from Infernux.engine.prefab_overrides import (
    apply_overrides_to_prefab, build_prefab_apply_command, compute_overrides,
    resolve_scene_prefab_documents, revert_overrides,
)


def _pair(scene, tmp_path):
    root = scene.create_game_object("Two colliders")
    root.add_component("BoxCollider")
    root.add_component("BoxCollider")
    document = root.serialize_document()
    assert len(document["components"]) == 2
    document["components"][0]["data"]["size"] = [2.0, 2.0, 2.0]
    document["components"][1]["data"]["size"] = [5.0, 5.0, 5.0]
    assert deserialize_game_object_document_transactionally(root, document)
    path = str(tmp_path / "components.prefab")
    assert save_prefab(root, path)
    return path, [instantiate_prefab(file_path=path, guid="components-guid", scene=scene) for _ in range(2)]


def _edit(obj, change):
    document = obj.serialize_document()
    change(document["components"])
    assert deserialize_game_object_document_transactionally(obj, document)


def test_same_type_delete_preserves_survivor_through_apply_undo_redo(scene, tmp_path):
    path, (first, second) = _pair(scene, tmp_path)
    original = first.serialize_document()["components"]
    peer_survivor = second.serialize_document()["components"][1]["component_id"]
    _edit(first, lambda components: components.pop(0))
    overrides = compute_overrides(first, path)
    assert len(overrides) == 1 and overrides[0].key.startswith("removed_components:")
    command = build_prefab_apply_command(first, path)
    command.execute()
    survivor = first.serialize_document()["components"][0]
    assert survivor["component_id"] == original[1]["component_id"]
    assert survivor["prefab_source_id"] == original[1]["prefab_source_id"]
    assert second.serialize_document()["components"][0]["component_id"] == peer_survivor
    assert compute_overrides(first, path) == []
    command.undo()
    assert len(second.serialize_document()["components"]) == 2
    command.redo()
    assert second.serialize_document()["components"][0]["component_id"] == peer_survivor
    retired = original[0]["prefab_source_id"]
    first.add_component("BoxCollider")
    new_runtime = first.serialize_document()["components"][-1]["component_id"]
    assert apply_overrides_to_prefab(first, path)
    new_record = first.serialize_document()["components"][-1]
    assert new_record["component_id"] == new_runtime
    assert new_record["prefab_source_id"] > max(retired, survivor["prefab_source_id"])


def test_same_type_reorder_revert_preserves_runtime_identity(scene, tmp_path):
    path, (first, _) = _pair(scene, tmp_path)
    original = first.serialize_document()["components"]
    _edit(first, lambda components: components.reverse())
    overrides = compute_overrides(first, path)
    assert [item.key for item in overrides] == ["components.order"]
    assert revert_overrides(first, path)
    assert first.serialize_document()["components"] == original


def test_private_same_type_component_does_not_replace_deleted_source(scene, tmp_path):
    path, (first, second) = _pair(scene, tmp_path)
    _edit(second, lambda components: components.clear())
    second.add_component("BoxCollider")
    private_id = second.serialize_document()["components"][0]["component_id"]
    first.add_component("BoxCollider")
    published_id = first.serialize_document()["components"][-1]["component_id"]
    assert apply_overrides_to_prefab(first, path)
    records = second.serialize_document()["components"]
    assert len(records) == 2
    private = next(record for record in records if record["component_id"] == private_id)
    assert "prefab_source_id" not in private
    assert first.serialize_document()["components"][-1]["component_id"] == published_id
    assert len({record["component_id"] for record in records}) == 2


def test_unversioned_scene_component_baseline_is_rejected(scene, tmp_path):
    path, (first, _) = _pair(scene, tmp_path)
    source = _read_prefab_document(path)["root_object"]
    document = {"objects": [first.serialize_document()]}
    root = document["objects"][0]
    root["prefab_source"] = copy.deepcopy(source)
    for record in root["components"]:
        record.pop("prefab_source_id")
    with pytest.raises(PrefabDocumentError, match="Unsupported prefab component identity baseline"):
        resolve_scene_prefab_documents(document, lambda _: source)


def test_component_source_metadata_is_not_author_data(scene, tmp_path):
    from Infernux.engine.component_restore import serialize_game_object_document_authoritatively
    path, (first, _) = _pair(scene, tmp_path)
    document = serialize_game_object_document_authoritatively(first)
    for record in document["components"]:
        assert record["prefab_source_id"] > 0
        assert "prefab_source_id" not in record["data"]
    assert apply_overrides_to_prefab(first, path)


def test_instance_reorder_survives_unrelated_source_field_change(scene, tmp_path):
    path, (first, second) = _pair(scene, tmp_path)
    _edit(second, lambda components: components.reverse())
    ids = [record["component_id"] for record in second.serialize_document()["components"]]
    _edit(first, lambda components: components[0]["data"].update(is_trigger=True))
    assert apply_overrides_to_prefab(first, path)
    records = second.serialize_document()["components"]
    assert [record["component_id"] for record in records] == ids
    assert records[1]["data"]["is_trigger"]


def test_unpack_clears_component_links_and_undo_restores_them(scene, tmp_path):
    from Infernux.engine.undo import PrefabUnpackCommand
    _, (first, _) = _pair(scene, tmp_path)
    before = first.serialize_document()["components"]
    command = PrefabUnpackCommand(first.id)
    command.execute()
    assert all("prefab_source_id" not in record for record in first.serialize_document()["components"])
    command.undo()
    assert first.serialize_document()["components"] == before


def test_deleted_highest_component_id_is_not_reused(scene, tmp_path):
    path, (first, second) = _pair(scene, tmp_path)
    watermark = _read_prefab_document(path)["next_component_id"]
    _edit(first, lambda components: components.clear())
    assert apply_overrides_to_prefab(first, path)
    assert _read_prefab_document(path)["next_component_id"] == watermark
    first.add_component("BoxCollider")
    assert apply_overrides_to_prefab(first, path)
    assert first.serialize_document()["components"][0]["prefab_source_id"] == watermark
    assert second.serialize_document()["components"][0]["prefab_source_id"] == watermark


@pytest.mark.parametrize("python_component", [False, True])
def test_copying_linked_child_clears_native_and_python_component_links(scene, tmp_path, python_component):
    from Infernux.components import InxComponent, serialized_field
    from Infernux.engine.component_restore import clone_game_object_transactionally

    class ChildProbe(InxComponent):
        value = serialized_field(default=7)

    root = scene.create_game_object("Source")
    child = scene.create_game_object("Child")
    child.set_parent(root)
    if python_component:
        child.add_py_component(ChildProbe())
    else:
        child.add_component("BoxCollider")
    path = str(tmp_path / "child.prefab")
    assert save_prefab(root, path)
    instance = instantiate_prefab(file_path=path, guid="child-guid", scene=scene)
    clone = clone_game_object_transactionally(scene, instance.get_child(0), parent=instance)
    clone_id = clone.id
    assert "prefab_source_id" not in clone.serialize_document()["components"][0]
    assert apply_overrides_to_prefab(instance, path)
    source_id = instance.get_child(0).serialize_document()["components"][0]["prefab_source_id"]
    assert scene.find_by_id(clone_id).serialize_document()["components"][0]["prefab_source_id"] != source_id
