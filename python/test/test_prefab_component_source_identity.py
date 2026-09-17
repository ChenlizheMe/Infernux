"""Component source identity is metadata, separate from runtime IDs and fields."""
import copy

import pytest

from Infernux.components import InxComponent, serialized_field
from Infernux.engine.component_restore import (
    deserialize_game_object_document_transactionally,
    deserialize_scene_document_transactionally,
    instantiate_game_object_document_transactionally,
    clone_game_object_transactionally,
)


class _SourceIdentityProbe(InxComponent):
    value = serialized_field(default=7)


@pytest.mark.parametrize("python_component", [False, True])
def test_component_source_identity_survives_document_restore_and_fresh_instance(scene, python_component):
    obj = scene.create_game_object("Source identity")
    if python_component:
        obj.add_py_component(_SourceIdentityProbe())
    else:
        obj.add_component("BoxCollider")
    document = obj.serialize_document()
    document["components"][0]["prefab_source_id"] = 29
    runtime_id = document["components"][0]["component_id"]
    assert deserialize_game_object_document_transactionally(obj, document)
    restored = obj.serialize_document()["components"][0]
    assert restored["component_id"] == runtime_id
    assert restored["prefab_source_id"] == 29
    assert "prefab_source_id" not in restored["data"]
    duplicate = instantiate_game_object_document_transactionally(scene, obj.serialize_document())
    cloned = duplicate.serialize_document()["components"][0]
    assert cloned["component_id"] != runtime_id
    assert cloned["prefab_source_id"] == 29
    native_clone = clone_game_object_transactionally(scene, obj)
    assert native_clone.serialize_document()["components"][0]["prefab_source_id"] == 29
    if python_component:
        previous = obj.get_py_component(_SourceIdentityProbe)
        obj.replace_py_component(previous, _SourceIdentityProbe())
        assert obj.serialize_document()["components"][0]["prefab_source_id"] == 29
    # Restoring a document without a link explicitly removes the old metadata.
    unlinked = copy.deepcopy(obj.serialize_document())
    unlinked["components"][0].pop("prefab_source_id")
    deserialize_game_object_document_transactionally(obj, unlinked)
    assert "prefab_source_id" not in obj.serialize_document()["components"][0]


@pytest.mark.parametrize("bad", [0, -1, True, "29", 1.5])
@pytest.mark.parametrize("python_component", [False, True])
def test_invalid_component_source_identity_does_not_publish(scene, bad, python_component):
    obj = scene.create_game_object("Unchanged")
    if python_component:
        obj.add_py_component(_SourceIdentityProbe())
    else:
        obj.add_component("BoxCollider")
    original = obj.serialize_document()
    bad_document = copy.deepcopy(original)
    bad_document["components"][0]["prefab_source_id"] = bad
    if python_component:
        with pytest.raises((ValueError, RuntimeError)):
            deserialize_game_object_document_transactionally(obj, bad_document)
    else:
        assert not deserialize_game_object_document_transactionally(obj, bad_document)
    assert obj.serialize_document() == original


@pytest.mark.parametrize("python_component", [False, True])
def test_scene_restore_preserves_distinct_source_ids_for_identical_components(scene, python_component):
    for name in ("First", "Second"):
        obj = scene.create_game_object(name)
        if python_component:
            obj.add_py_component(_SourceIdentityProbe())
        else:
            obj.add_component("BoxCollider")
    document = scene.serialize_document()
    roots = [node for node in document["objects"] if node["name"] in {"First", "Second"}]
    for node, source_id in zip(roots, (17, 23), strict=True):
        node["components"][0]["prefab_source_id"] = source_id
    assert deserialize_scene_document_transactionally(scene, document)
    restored = scene.serialize_document()
    actual = {node["name"]: node["components"][0]["prefab_source_id"]
              for node in restored["objects"] if node["name"] in {"First", "Second"}}
    assert actual == {"First": 17, "Second": 23}
