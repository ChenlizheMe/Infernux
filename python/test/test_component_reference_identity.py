"""Exact component references survive type duplicates and document projection."""
import copy
import json
import pytest

from Infernux.components import InxComponent, serialized_field, FieldType
from Infernux.components.fields import get_raw_field_value
from Infernux.components.ref_wrappers import ComponentRef
from Infernux.engine.component_restore import (
    deserialize_game_object_document_transactionally, clone_game_object_transactionally,
    deserialize_scene_document_transactionally,
)
from Infernux.engine.prefab_manager import save_prefab, instantiate_prefab
from Infernux.engine.prefab_overrides import apply_overrides_to_prefab, compute_overrides


class _ExactReferenceProbe(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.COMPONENT)


def _source(scene):
    root = scene.create_game_object("Exact reference")
    first = root.add_component("BoxCollider")
    second = root.add_component("BoxCollider")
    first.is_trigger = False
    second.is_trigger = True
    probe = root.add_py_component(_ExactReferenceProbe())
    probe.target = second
    return root, probe, second.component_id


def test_reference_selects_second_same_type_and_never_retargets_after_delete(scene):
    root, probe, identity = _source(scene)
    reference = get_raw_field_value(probe, "target")
    assert reference.component_id == identity
    assert reference.resolve().component_id == identity
    assert copy.deepcopy(reference) == reference
    assert reference._serialize()["component_id"] == identity
    document = root.serialize_document()
    document["components"].pop(1)
    assert deserialize_game_object_document_transactionally(root, document)
    assert reference.resolve() is None
    assert root.get_py_component(_ExactReferenceProbe).target is None


def test_prefab_and_clone_remap_exact_component_reference(scene, tmp_path):
    root, _, identity = _source(scene)
    path = str(tmp_path / "exact.prefab")
    assert save_prefab(root, path)
    for created in (instantiate_prefab(file_path=path, guid="exact-guid", scene=scene),
                    clone_game_object_transactionally(scene, root)):
        assert created is not None
        probe = created.get_py_component(_ExactReferenceProbe)
        records = created.serialize_document()["components"]
        target_id = records[1]["component_id"]
        assert target_id != identity
        assert get_raw_field_value(probe, "target").component_id == target_id
        assert probe.target.component_id == target_id
        assert probe.target.is_trigger


def test_prefab_apply_preserves_component_reference_and_has_no_false_override(scene, tmp_path):
    root, _, _ = _source(scene)
    path = str(tmp_path / "exact.prefab")
    assert save_prefab(root, path)
    first = instantiate_prefab(file_path=path, guid="exact-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="exact-guid", scene=scene)
    assert compute_overrides(first, path) == []
    document = first.serialize_document()
    document["components"].pop(0)
    assert deserialize_game_object_document_transactionally(first, document)
    target_id = second.serialize_document()["components"][1]["component_id"]
    assert apply_overrides_to_prefab(first, path)
    assert second.get_py_component(_ExactReferenceProbe).target.component_id == target_id
    assert compute_overrides(second, path) == []


def test_transform_reference_uses_the_unique_transform_after_prefab_projection(scene, tmp_path):
    root, probe, _ = _source(scene)
    probe.target = root.transform
    path = str(tmp_path / "transform.prefab")
    assert save_prefab(root, path)
    created = instantiate_prefab(file_path=path, guid="transform-guid", scene=scene)
    assert created.get_py_component(_ExactReferenceProbe).target.component_id == created.transform.component_id


def test_ref_identity_participates_in_equality(scene):
    root, _, _ = _source(scene)
    first, second = root.get_cpp_components("BoxCollider")
    assert ComponentRef(first) != ComponentRef(second)


def test_direct_construction_uses_the_bound_script_asset_identity(scene):
    from Infernux.components.component_identity import bind_asset_script_guid
    class AssetBoundProbe(InxComponent):
        target = serialized_field(default=None, field_type=FieldType.COMPONENT)
    bind_asset_script_guid(AssetBoundProbe, "041-script-asset")
    root = scene.create_game_object("Asset-backed component")
    probe = root.add_py_component(AssetBoundProbe())
    assert probe._script_guid == "041-script-asset"
    assert root.serialize_document()["components"][0]["type_id"].startswith("python:041-script-asset:")


def test_python_forward_reference_survives_prefab_projection(scene, tmp_path):
    root = scene.create_game_object("Forward reference")
    probe = root.add_py_component(_ExactReferenceProbe())
    child = scene.create_game_object("Target")
    child.set_parent(root)
    target = child.add_py_component(_ExactReferenceProbe())
    probe.target = ComponentRef(go_id=child.id, component_type="_ExactReferenceProbe")
    path = str(tmp_path / "forward.prefab")
    assert save_prefab(root, path)
    created = instantiate_prefab(file_path=path, scene=scene)
    assert created.get_py_component(_ExactReferenceProbe).target is created.get_child(0).get_py_component(_ExactReferenceProbe)
    changes = compute_overrides(created, path)
    assert changes == [], [(item.instance_value, item.prefab_value) for item in changes]
    assert probe.target is target


def test_inspector_picker_offers_exact_same_type_components(scene):
    from Infernux.engine.ui._inspector_references import _picker_scene_components, _create_component_ref_from_go
    root, _, identity = _source(scene)
    choices = _picker_scene_components("Exact reference", "BoxCollider")
    assert len(choices) == 2
    assert choices[0][0] != choices[1][0]
    ref = _create_component_ref_from_go(choices[1][1], "BoxCollider")
    assert ref.component_id == identity
    assert ref.resolve().is_trigger
    assert _create_component_ref_from_go(root, "BoxCollider").component_id == choices[0][1].component_id


def test_scene_copy_remaps_native_and_python_references_without_mutating_source(scene):
    from Infernux.lib import SceneManager
    root, probe, identity = _source(scene)
    other = scene.create_game_object("Python target")
    target = other.add_py_component(_ExactReferenceProbe())
    target.target = probe
    snapshot = scene.serialize_document()
    manager = SceneManager.instance()
    copied = manager.create_scene("Reference copy")
    assert deserialize_scene_document_transactionally(copied, snapshot, clear_registries=False)
    manager.set_active_scene(copied)
    copied_root = next(obj for obj in copied.get_all_objects() if obj.name == root.name)
    copied_probe = copied_root.get_py_component(_ExactReferenceProbe)
    copied_target = next(obj for obj in copied.get_all_objects() if obj.name == other.name).get_py_component(_ExactReferenceProbe)
    assert copied_probe.target.component_id != identity
    assert copied_probe.target.game_object.id == copied_root.id
    assert copied_probe.target.is_trigger
    assert copied_target.target is copied_probe
    manager.set_active_scene(scene)
    assert probe.target.component_id == identity
    assert target.target is probe


def test_scene_round_trip_preserves_exact_component(scene):
    root, probe, identity = _source(scene)
    root_id = root.id
    snapshot = scene.serialize_document()
    assert deserialize_scene_document_transactionally(scene, snapshot)
    restored = scene.find_by_id(root_id).get_py_component(_ExactReferenceProbe)
    assert restored.target.component_id == identity
    assert restored.target.is_trigger


def test_cook_projects_exact_reference_after_source_component_reorder(scene, tmp_path):
    from Infernux.engine.prefab_overrides import resolve_scene_prefab_documents
    root, _, _ = _source(scene)
    path = str(tmp_path / "cooked.prefab")
    assert save_prefab(root, path)
    instance = instantiate_prefab(file_path=path, guid="cook-guid", scene=scene)
    before = {"objects": [instance.serialize_document()]}
    target_id = instance.get_py_component(_ExactReferenceProbe).target.component_id
    source = json.loads((tmp_path / "cooked.prefab").read_text(encoding="utf-8"))["root_object"]
    source["components"][0], source["components"][1] = source["components"][1], source["components"][0]
    cooked = resolve_scene_prefab_documents(before, lambda _: source)
    records = cooked["objects"][0]["components"]
    assert records[0]["component_id"] == target_id
    assert records[2]["data"]["target"]["component_id"] == target_id
    assert before["objects"][0]["components"][0]["component_id"] != target_id


def test_legacy_reference_serialization_does_not_change_hash(scene):
    root, _, _ = _source(scene)
    reference = ComponentRef(go_id=root.id, component_type="BoxCollider")
    entries = {reference: "legacy"}
    assert reference._serialize()["component_id"] == root.get_cpp_component("BoxCollider").component_id
    assert reference.component_id == 0
    assert entries[reference] == "legacy"


@pytest.mark.parametrize("identity", [-1, True, "42", 1.5])
def test_component_reference_document_rejects_invalid_identity(identity):
    from Infernux.components.value_document import is_component_ref_document, make_component_ref
    document = make_component_ref(1, "BoxCollider")
    document["component_id"] = identity
    assert not is_component_ref_document(document)
