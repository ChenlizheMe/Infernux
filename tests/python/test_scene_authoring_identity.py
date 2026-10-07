"""Persistent Scene identities survive native authoring and transaction boundaries."""
from copy import deepcopy
import re

import pytest

from infernux.lib import GameObject, SceneManager
from infernux.components import FieldType, InxComponent, serialized_field
from infernux.components.ref_wrappers import GameObjectRef, ComponentRef
from infernux.engine.component_restore import deserialize_scene_document_transactionally


class _AuthoringRefs(InxComponent):
    target_object = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    target_component = serialized_field(default=None, field_type=FieldType.COMPONENT)


def identities(scene):
    return scene.serialize_document()["authoring_identity"]


def test_allocated_identity_is_stable_and_clone_is_independent(scene):
    owner = scene.create_game_object("Author")
    light = owner.add_component("Light")
    child = scene.create_game_object("Child")
    child.set_parent(owner, False)
    first = identities(scene)
    assert first == identities(scene)
    guids = list(first["objects"].values()) + list(first["components"].values())
    assert len(guids) == len(set(guids))
    assert all(re.fullmatch("[0-9a-f]{32}", guid) for guid in guids)
    assert str(light.component_id) in first["components"]

    clone = GameObject.instantiate(owner)
    after = identities(scene)
    assert after["objects"][str(clone.id)] != first["objects"][str(owner.id)]
    assert set(first["objects"].values()).issubset(after["objects"].values())
    assert len(after["objects"]) == 4
    assert len(set(after["components"].values())) == len(after["components"])


def test_deleted_identity_survives_snapshot_restore_and_is_not_reused(scene):
    owner = scene.create_game_object("Deleted")
    light = owner.add_component("Light")
    object_id, component_id = owner.id, light.component_id
    live = scene.serialize_document()
    before = identities(scene)
    assert owner.remove_component(light)
    scene.destroy_game_object(owner)
    scene.process_pending_destroys()
    deleted = scene.serialize_document()
    assert deleted["objects"] == []
    assert identities(scene) == before
    assert scene._commit_document(deleted)
    fresh = scene.create_game_object("Fresh")
    assert fresh.id != object_id
    assert fresh.add_component("Light").component_id != component_id
    assert scene._commit_document(live)
    assert scene.find("Deleted").id == object_id
    assert identities(scene) == before


@pytest.mark.parametrize("kind", ["object", "native", "python"])
def test_editor_delete_undo_redo_preserves_original_author_guids(scene, kind):
    from infernux.engine.undo import DeleteGameObjectCommand, RemoveNativeComponentCommand, RemovePyComponentCommand

    owner = scene.create_game_object("UndoIdentity")
    light = owner.add_component("Light")
    script = owner.add_py_component(_AuthoringRefs())
    object_id, component_id, script_id = owner.id, light.component_id, script.component_id
    before = identities(scene)
    if kind == "object":
        command = DeleteGameObjectCommand(object_id)
    elif kind == "native":
        command = RemoveNativeComponentCommand(object_id, "Light", light)
    else:
        command = RemovePyComponentCommand(object_id, script)
    for _ in range(2):
        command.execute()
        command.undo()
        restored = scene.find_by_id(object_id)
        assert restored is not None
        assert restored.get_component("Light").component_id == component_id
        assert restored.get_py_component(_AuthoringRefs).component_id == script_id
        table = identities(scene)
        assert table["objects"][str(object_id)] == before["objects"][str(object_id)]
        for key, guid in before["components"].items():
            assert table["components"][key] == guid


@pytest.mark.parametrize("finish", ["rollback", "finalize", "reject"])
def test_retained_world_transaction_owns_its_identity_tables(scene, finish):
    owner = scene.create_game_object("Retained")
    owner.add_component("Light")
    before = scene.serialize_document()
    owner_id = owner.id
    candidate = deepcopy(before)
    candidate["objects"] = []
    candidate["authoring_identity"] = {"objects": {}, "components": {}}
    if finish == "reject":
        candidate["authoring_identity"]["objects"]["1"] = "bad-guid"
        assert scene._commit_document_retaining_world(candidate) is None
        assert scene.find("Retained") is owner
        assert scene.serialize_document() == before
        return
    token = scene._commit_document_retaining_world(candidate)
    assert token is not None
    try:
        assert identities(scene) == candidate["authoring_identity"]
        scene.create_game_object("CandidateOnly")
        if finish == "rollback":
            assert token.rollback()
            assert scene.find("Retained") is owner
            assert scene.serialize_document() == before
        else:
            token.finalize()
            assert scene.find("Retained") is None
            assert before["authoring_identity"]["objects"][str(owner_id)] not in identities(scene)["objects"].values()
    finally:
        if token.is_active:
            token.rollback()


def test_additive_publication_remaps_live_and_missing_identities(scene):
    owner = scene.create_game_object("Live")
    owner.add_component("Light")
    missing = scene.create_game_object("Missing")
    missing_body = missing.add_component("Rigidbody")
    missing_object_id, missing_component_id = missing.id, missing_body.component_id
    source = scene.serialize_document()
    # The source world keeps the target alive. In the candidate it is a missing
    # target; its integer must not accidentally resolve back into that world.
    source["objects"] = [obj for obj in source["objects"] if obj["name"] != "Missing"]
    destination = SceneManager.instance().create_scene("Copy")
    token = destination._commit_document_retaining_world(source)
    assert token is not None
    try:
        object_map, component_map = token.object_id_remap, token.component_id_remap
        assert object_map[missing_object_id] != missing_object_id
        assert component_map[missing_component_id] != missing_component_id
        published = identities(destination)
        for old, guid in source["authoring_identity"]["objects"].items():
            assert published["objects"][str(object_map.get(int(old), int(old)))] == guid
        for old, guid in source["authoring_identity"]["components"].items():
            assert published["components"][str(component_map.get(int(old), int(old)))] == guid
        assert destination.find_by_id(object_map[missing_object_id]) is None
        new = destination.create_game_object("Later")
        assert new.id != object_map[missing_object_id]
        assert new.add_component("Light").component_id != component_map[missing_component_id]
        token.finalize()
    finally:
        if token.is_active:
            token.rollback()


def test_reparent_to_another_scene_preserves_author_identity_and_lookup(scene):
    owner = scene.create_game_object("Moving")
    child = scene.create_game_object("MovingChild")
    child.set_parent(owner, False)
    child.add_component("Light")
    refs = child.add_py_component(_AuthoringRefs())
    refs.target_object = GameObjectRef(owner)
    before = identities(scene)
    destination = SceneManager.instance().create_scene("Destination")
    parent = destination.create_game_object("Parent")
    owner.set_parent(parent, False)
    assert owner.scene is destination and child.scene is destination
    assert refs.game_object is child
    assert refs.target_object is owner
    for moved in (owner, child):
        assert scene.find_by_id(moved.id) is None
        assert destination.find_by_id(moved.id) is moved
        assert identities(destination)["objects"][str(moved.id)] == before["objects"][str(moved.id)]
    for component_id, guid in before["components"].items():
        assert identities(destination)["components"][component_id] == guid


@pytest.mark.parametrize("missing", [False, True])
def test_python_and_native_references_follow_publication_without_retargeting(scene, missing):
    owner = scene.create_game_object("ReferenceOwner")
    owner.add_component("Rigidbody")
    hinge = owner.add_component("HingeJoint")
    target = scene.create_game_object("ReferenceTarget")
    target_body = target.add_component("Rigidbody")
    hinge.connected_body = target_body
    refs = owner.add_py_component(_AuthoringRefs())
    refs.target_object = GameObjectRef(target)
    refs.target_component = ComponentRef(go_id=target.id, component_type="Rigidbody", component_id=target_body.component_id)
    source = scene.serialize_document()
    if missing:
        source["objects"] = [obj for obj in source["objects"] if obj["name"] != "ReferenceTarget"]
    destination = SceneManager.instance().create_scene("ReferencesCopy")
    assert deserialize_scene_document_transactionally(destination, source, clear_registries=False)
    copied_owner = destination.find("ReferenceOwner")
    copied_refs = copied_owner.get_py_component(_AuthoringRefs)
    copied_hinge = copied_owner.get_component("HingeJoint")
    published = destination.serialize_document()
    fields = copied_refs._serialize_fields_document()
    target_id = fields["target_object"]["object_id"]
    body_id = fields["target_component"]["component_id"]
    assert target_id != target.id and body_id != target_body.component_id
    assert copied_hinge.serialize_document()["connected_body_component_id"] == body_id
    assert published["authoring_identity"]["objects"][str(target_id)] == source["authoring_identity"]["objects"][str(target.id)]
    assert published["authoring_identity"]["components"][str(body_id)] == source["authoring_identity"]["components"][str(target_body.component_id)]
    if missing:
        assert copied_refs.target_object is None and copied_refs.target_component is None
        assert copied_hinge.connected_body is None
    else:
        copied_target = destination.find("ReferenceTarget")
        assert copied_refs.target_object is copied_target
        assert copied_refs.target_component.game_object is copied_target
        assert copied_hinge.connected_body.game_object is copied_target


def test_move_into_an_existing_copy_rejects_guid_alias_before_mutation(scene):
    owner = scene.create_game_object("ExistingAuthor")
    child = scene.create_game_object("ChildAuthor")
    child.set_parent(owner, False)
    before = scene.serialize_document()
    destination = SceneManager.instance().create_scene("DocumentCopy")
    assert destination._commit_document(before)
    destination_before = destination.serialize_document()
    with pytest.raises(ValueError, match="duplicate an author identity"):
        owner.set_parent(destination.find("ExistingAuthor"), False)
    assert owner.scene is scene and child.scene is scene
    assert owner.get_parent() is None
    assert scene.serialize_document() == before
    assert destination.serialize_document() == destination_before


@pytest.mark.parametrize("table", ["objects", "components"])
def test_incomplete_identity_table_rejects_without_changing_world(scene, table):
    owner = scene.create_game_object("Keep")
    before = scene.serialize_document()
    candidate = deepcopy(before)
    candidate["authoring_identity"][table] = {}
    assert not scene._commit_document(candidate)
    assert scene.find("Keep") is owner
    assert scene.serialize_document() == before
