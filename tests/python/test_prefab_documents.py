"""Strict prefab documents and typed pybind serialization boundaries."""

from __future__ import annotations

import json

import pytest

from infernux.components import FieldType, InxComponent, serialized_field
from infernux.components.builtin import BoxCollider
from infernux.components.ref_wrappers import ComponentRef, GameObjectRef
from infernux.components.value_document import make_game_object_ref
from infernux.engine.prefab_manager import (
    _PREFAB_TEMPLATE_CACHE,
    _read_prefab_document,
    instantiate_prefab,
    save_prefab,
)
from infernux.engine.prefab_overrides import (
    apply_overrides_to_prefab,
    build_prefab_apply_command,
    build_prefab_revert_command,
    compute_overrides,
    revert_overrides,
)
from infernux.engine.undo import (
    BuiltinPropertyCommand,
    UndoManager,
)
from infernux.math import Vector3
from infernux.lib import SceneManager
from infernux.engine.component_restore import (
    clone_game_object_transactionally,
    serialize_game_object_document_authoritatively,
)


class _PrefabTargetComponent(InxComponent):
    value: int = 19


class _PrefabReferenceComponent(InxComponent):
    target_object = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    target_component = serialized_field(default=None, field_type=FieldType.COMPONENT)


class _PrefabAwakeRecorder(InxComponent):
    _placements = []

    def awake(self):
        type(self)._placements.append(self.game_object.transform.position.x)


@pytest.mark.parametrize("operation", ["revert", "apply"])
def test_prefab_operations_preserve_inbound_scene_references(scene, tmp_path, operation):
    source = scene.create_game_object("ReferencedPrefab")
    child = scene.create_game_object("Target")
    child.set_parent(source)
    child.add_py_component(_PrefabTargetComponent())
    path = tmp_path / "inbound.prefab"
    assert save_prefab(source, str(path))
    instance = instantiate_prefab(file_path=str(path), guid="inbound-guid", scene=scene)
    target = instance.get_child(0)
    target_id = target.id
    watcher = scene.create_game_object("Watcher")
    references = _PrefabReferenceComponent()
    watcher.add_py_component(references)
    references.target_object = GameObjectRef(target)
    references.target_component = ComponentRef(go_id=target_id, component_type="_PrefabTargetComponent")
    assert references.target_object is target
    assert references.target_component is target.get_py_component(_PrefabTargetComponent)
    target.get_py_component(_PrefabTargetComponent).value = 28
    command = (build_prefab_revert_command if operation == "revert" else build_prefab_apply_command)(
        instance,
        str(path),
    )
    for action in (command.execute, command.undo, command.redo):
        action()
        assert instance.get_child(0).id == target_id
        assert references.target_object is instance.get_child(0)
        assert references.target_component is instance.get_child(0).get_py_component(_PrefabTargetComponent)


def test_prefab_renamed_siblings_keep_source_identity_and_inbound_references(scene, tmp_path):
    source = scene.create_game_object("Pair")
    for value in (11, 22):
        child = scene.create_game_object("Same Name")
        child.set_parent(source)
        component = _PrefabTargetComponent()
        component.value = value
        child.add_py_component(component)
    path = tmp_path / "pair.prefab"
    assert save_prefab(source, str(path))
    instance = instantiate_prefab(file_path=str(path), guid="pair-guid", scene=scene)
    first, second = instance.get_children()
    first_id, second_id = first.id, second.id
    first.name = "Renamed"
    first.transform.set_sibling_index(1)
    assert [child.id for child in instance.get_children()] == [second_id, first_id]
    changes = compute_overrides(instance, str(path))
    assert any(change.key == "name" for change in changes)
    assert not any(change.key.startswith(("added_child:", "removed_child:")) for change in changes)
    assert not any(".data" in change.key for change in changes)
    command = build_prefab_revert_command(instance, str(path))
    command.execute()
    assert [child.id for child in instance.get_children()] == [first_id, second_id]
    assert [child.get_py_component(_PrefabTargetComponent).value for child in instance.get_children()] == [11, 22]
    command.undo()
    assert [child.id for child in instance.get_children()] == [second_id, first_id]
    assert instance.get_child(1).name == "Renamed"
    command.redo()
    assert [child.id for child in instance.get_children()] == [first_id, second_id]


def test_prefab_revert_new_nodes_reserve_stable_redo_ids(scene, tmp_path):
    source = scene.create_game_object("Expandable")
    child = scene.create_game_object("Child")
    child.set_parent(source)
    child.add_py_component(_PrefabTargetComponent())
    path = tmp_path / "expandable.prefab"
    assert save_prefab(source, str(path))
    instance = instantiate_prefab(file_path=str(path), guid="expandable-guid", scene=scene)
    scene.destroy_game_object(instance.get_child(0))
    scene.process_pending_destroys()
    command = build_prefab_revert_command(instance, str(path))
    command.execute()
    restored_id = instance.get_child(0).id
    component_id = instance.get_child(0).get_py_component(_PrefabTargetComponent).component_id
    command.undo()
    assert not instance.get_children()
    command.redo()
    assert instance.get_child(0).id == restored_id
    assert instance.get_child(0).get_py_component(_PrefabTargetComponent).component_id == component_id


def test_prefab_source_identity_survives_clone_document_and_unpack(scene, tmp_path):
    from infernux.engine.component_restore import deserialize_game_object_document_transactionally
    from infernux.engine.undo import PrefabUnpackCommand

    source = scene.create_game_object("SourceIdentity")
    path = tmp_path / "identity.prefab"
    assert save_prefab(source, str(path))
    instance = instantiate_prefab(file_path=str(path), guid="identity-guid", scene=scene)
    source_id = instance.prefab_source_id
    assert source_id > 0
    clone = clone_game_object_transactionally(scene, instance)
    assert clone.id != instance.id
    assert clone.prefab_source_id == source_id
    from infernux.engine.prefab_manager import _make_prefab_baseline

    baseline = _make_prefab_baseline(_read_prefab_document(str(path))["root_object"])
    assert clone._prefab_source_document == baseline
    document = serialize_game_object_document_authoritatively(instance)
    instance.prefab_source_id = 0
    assert deserialize_game_object_document_transactionally(instance, document, preserve_document_ids=True)
    assert instance.prefab_source_id == source_id
    command = PrefabUnpackCommand(instance.id)
    command.execute()
    assert instance.prefab_source_id == 0
    assert instance._prefab_source_document is None
    command.undo()
    assert instance.prefab_source_id == source_id
    assert instance._prefab_source_document == baseline


def test_cold_and_cached_prefab_instances_awaken_after_configuration_without_template_worlds(scene, tmp_path):
    source = scene.create_game_object("CachedAwakePrefab")
    source.add_py_component(_PrefabTargetComponent())
    source.add_py_component(_PrefabAwakeRecorder())
    path = tmp_path / "cached-awake.prefab"
    assert save_prefab(source, str(path))
    manager = SceneManager.instance()
    before_worlds = [manager.get_scene_at(index).world_id for index in range(manager.scene_count)]
    _PrefabAwakeRecorder._placements.clear()

    first = instantiate_prefab(
        file_path=str(path),
        scene=scene,
        configure_created=lambda obj: setattr(obj.transform, "position", Vector3(5.0, 0.0, 0.0)),
    )
    assert first is not None and first.get_py_component(_PrefabTargetComponent).value == 19
    assert _PrefabAwakeRecorder._placements == [5.0]
    assert [manager.get_scene_at(index).world_id for index in range(manager.scene_count)] == before_worlds

    second = instantiate_prefab(
        file_path=str(path),
        scene=scene,
        configure_created=lambda obj: setattr(obj.transform, "position", Vector3(8.0, 0.0, 0.0)),
    )
    assert second is not None and second.get_py_component(_PrefabTargetComponent).value == 19
    assert _PrefabAwakeRecorder._placements == [5.0, 8.0]
    assert [manager.get_scene_at(index).world_id for index in range(manager.scene_count)] == before_worlds


def test_prefab_rejected_configuration_leaves_no_objects_or_pending_components(scene, tmp_path):
    source = scene.create_game_object("RejectedPrefab")
    source.add_py_component(_PrefabTargetComponent())
    path = tmp_path / "rejected.prefab"
    assert save_prefab(source, str(path))
    before_ids = {obj.id for obj in scene.get_all_objects()}

    def reject(_instance):
        raise RuntimeError("author rejected placement")

    assert instantiate_prefab(file_path=str(path), scene=scene, configure_created=reject) is None
    assert {obj.id for obj in scene.get_all_objects()} == before_ids
    assert not scene.has_pending_py_components()
    instance = instantiate_prefab(file_path=str(path), scene=scene)
    assert instance.get_py_component(_PrefabTargetComponent).value == 19


def test_clone_rejected_configuration_retires_pending_python_descriptors(scene):
    source = scene.create_game_object("RejectedClone")
    source.add_py_component(_PrefabTargetComponent())
    before_ids = {obj.id for obj in scene.get_all_objects()}

    def reject(_instance):
        raise RuntimeError("author rejected placement")

    with pytest.raises(RuntimeError, match="author rejected placement"):
        clone_game_object_transactionally(scene, source, configure_created=reject)
    assert not scene.has_pending_py_components()
    assert {obj.id for obj in scene.get_all_objects()} == before_ids
    instance = clone_game_object_transactionally(scene, source)
    assert instance.get_py_component(_PrefabTargetComponent).value == 19


def test_prefab_cache_survives_target_scene_unload_without_live_objects(scene, tmp_path):
    source = scene.create_game_object("CachedDocument")
    source.add_py_component(_PrefabTargetComponent())
    path = tmp_path / "unload.prefab"
    assert save_prefab(source, str(path))
    manager = SceneManager.instance()
    target_scene = manager.create_scene("TemporaryPrefabDestination")
    first = instantiate_prefab(file_path=str(path), scene=target_scene)
    assert first.get_py_component(_PrefabTargetComponent).value == 19
    manager.unload_scene(target_scene)
    second = instantiate_prefab(file_path=str(path), scene=scene)
    assert second.get_py_component(_PrefabTargetComponent).value == 19


def _assert_runtime_ids_removed(document: dict) -> None:
    assert "id" not in document
    assert type(document["local_id"]) is int and document["local_id"] > 0
    assert "component_id" not in document["transform"]
    assert "instance_guid" not in document["transform"]
    for component in document["components"]:
        assert type(component["component_id"]) is int and component["component_id"] > 0
        assert "instance_guid" not in component
    for child in document["children"]:
        _assert_runtime_ids_removed(child)


def test_typed_scene_document_bridge_and_instantiation(scene):
    root = scene.create_game_object("DocumentRoot")
    root.add_component("Rigidbody")
    child = scene.create_game_object("DocumentChild")
    child.set_parent(root)

    document = root.serialize_document()
    assert isinstance(document, dict)
    assert isinstance(document["children"], list)
    assert document["children"][0]["name"] == "DocumentChild"

    clone = scene._instantiate_document(document)
    assert clone is not None
    assert clone.id != root.id
    assert clone.get_child(0).id != child.id


def test_prefab_save_is_strict_typed_and_atomic(scene, tmp_path):
    _PREFAB_TEMPLATE_CACHE.clear()
    root = scene.create_game_object("PrefabRoot")
    root.add_component("BoxCollider")
    child = scene.create_game_object("PrefabChild")
    child.add_component("Rigidbody")
    child.set_parent(root)

    path = tmp_path / "nested" / "typed.prefab"
    assert save_prefab(root, str(path), source_canvas_name="HUD") is True

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert set(envelope) == {"root_object", "source_canvas_name", "next_local_id", "next_component_id"}
    assert envelope["source_canvas_name"] == "HUD"
    _assert_runtime_ids_removed(envelope["root_object"])
    assert list(path.parent.glob("typed.prefab.tmp.*")) == []

    instance = instantiate_prefab(file_path=str(path), scene=scene)
    assert instance is not None
    assert instance.id != root.id
    assert instance.get_child(0).id != child.id


def test_canvas_free_world_ui_uses_normal_clone_save_and_prefab_lifecycle(scene, tmp_path):
    from infernux.components.fields import get_raw_field_value
    from infernux.core.asset_ref import TextureRef
    from infernux.ui import UIButton, UICanvas, UIFrame

    root = scene.create_game_object("WorldPanel")
    root.transform.position = Vector3(2.0, 3.0, 4.0)
    frame = UIFrame()
    frame.width = 640.0
    frame.height = 360.0
    frame.clip_content = True
    root.add_py_component(frame)
    assert "world_pixels_per_unit" not in frame._serialize_fields_document()

    child = scene.create_game_object("WorldButton")
    child.set_parent(root)
    button = UIButton()
    button.width = 400.0
    button.height = 82.0
    button.label = "INTERACT WITH THE WORLD"
    button.background_texture = TextureRef(
        guid="world-button-texture-guid",
        path_hint="Assets/Textures/world-button.png",
    )
    child.add_py_component(button)

    clone = clone_game_object_transactionally(scene, root)
    assert clone is not None
    assert clone.get_py_component(UICanvas) is None
    clone_frame = clone.get_py_component(UIFrame)
    clone_button = clone.get_child(0).get_py_component(UIButton)
    assert (clone_frame.width, clone_frame.height) == (640.0, 360.0)
    assert clone_button.label == "INTERACT WITH THE WORLD"
    assert get_raw_field_value(clone_button, "background_texture").guid == ("world-button-texture-guid")
    assert [clone.transform.position[index] for index in range(3)] == pytest.approx([2.0, 3.0, 4.0])

    path = tmp_path / "world_ui.prefab"
    assert save_prefab(root, str(path)) is True
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert "source_canvas_name" not in envelope
    instance = instantiate_prefab(file_path=str(path), scene=scene)
    assert instance is not None
    assert instance.get_py_component(UICanvas) is None
    instance_frame = instance.get_py_component(UIFrame)
    instance_button = instance.get_child(0).get_py_component(UIButton)
    assert (instance_frame.width, instance_frame.height) == (640.0, 360.0)
    assert instance_button.label == "INTERACT WITH THE WORLD"
    assert get_raw_field_value(instance_button, "background_texture").guid == ("world-button-texture-guid")


def test_prefab_remaps_internal_python_references(scene, tmp_path):
    _PREFAB_TEMPLATE_CACHE.clear()
    root = scene.create_game_object("ReferencePrefabRoot")
    child = scene.create_game_object("ReferencePrefabChild")
    child.set_parent(root)
    child.add_py_component(_PrefabTargetComponent())
    references = _PrefabReferenceComponent()
    references.target_object = GameObjectRef(child)
    references.target_component = ComponentRef(
        go_id=child.id,
        component_type="_PrefabTargetComponent",
    )
    root.add_py_component(references)
    path = tmp_path / "reference.prefab"

    assert save_prefab(root, str(path)) is True
    document = json.loads(path.read_text(encoding="utf-8"))["root_object"]
    reference_record = next(
        record for record in document["components"] if record["type_id"].endswith(":_PrefabReferenceComponent")
    )
    assert reference_record["data"]["target_object"] == make_game_object_ref(document["children"][0]["local_id"])

    instance = instantiate_prefab(file_path=str(path), scene=scene)
    instance_child = instance.get_child(0)
    restored = instance.get_py_component(_PrefabReferenceComponent)
    assert restored.target_object is instance_child
    assert restored.target_component is instance_child.get_py_component(_PrefabTargetComponent)
    assert compute_overrides(instance, str(path)) == []

    external = scene.create_game_object("ExternalReference")
    restored.target_object = GameObjectRef(external)
    overrides = compute_overrides(instance, str(path))
    assert len(overrides) == 1
    assert overrides[0].instance_value["target_object"] == make_game_object_ref(external.id)
    assert overrides[0].prefab_value["target_object"] == make_game_object_ref(2)

    assert revert_overrides(instance, str(path)) is True
    assert compute_overrides(instance, str(path)) == []
    restored = instance.get_py_component(_PrefabReferenceComponent)
    restored.target_object = GameObjectRef(instance)
    assert len(compute_overrides(instance, str(path))) == 1
    assert apply_overrides_to_prefab(instance, str(path)) is True
    assert compute_overrides(instance, str(path)) == []
    next_instance = instantiate_prefab(file_path=str(path), scene=scene)
    assert next_instance.get_py_component(_PrefabReferenceComponent).target_object is next_instance
    assert compute_overrides(next_instance, str(path)) == []


def test_prefab_duplicate_siblings_keep_distinct_reference_targets(scene, tmp_path):
    root = scene.create_game_object("DuplicateChildren")
    for value in (19, 37):
        child = scene.create_game_object("Child")
        child.set_parent(root)
        target = _PrefabTargetComponent()
        target.value = value
        child.add_py_component(target)
    references = _PrefabReferenceComponent()
    references.target_object = GameObjectRef(root.get_child(0))
    references.target_component = ComponentRef(
        go_id=root.get_child(1).id,
        component_type="_PrefabTargetComponent",
    )
    root.add_py_component(references)
    path = tmp_path / "duplicate_children.prefab"
    assert save_prefab(root, str(path)) is True
    instance = instantiate_prefab(file_path=str(path), scene=scene)
    assert compute_overrides(instance, str(path)) == []
    instance.get_child(1).get_py_component(_PrefabTargetComponent).value = 53
    overrides = compute_overrides(instance, str(path))
    assert len(overrides) == 1
    assert overrides[0].prefab_value["value"] == 37
    assert overrides[0].instance_value["value"] == 53


@pytest.mark.parametrize("action", ["apply", "revert"])
def test_prefab_reference_edits_survive_undo_redo(scene, tmp_path, action):
    root = scene.create_game_object("ReferenceUndo")
    child = scene.create_game_object("Target")
    child.set_parent(root)
    child.add_py_component(_PrefabTargetComponent())
    references = _PrefabReferenceComponent()
    references.target_object = GameObjectRef(child)
    references.target_component = ComponentRef(
        go_id=child.id,
        component_type="_PrefabTargetComponent",
    )
    root.add_py_component(references)
    path = tmp_path / "reference_undo.prefab"
    assert save_prefab(root, str(path))
    instance = instantiate_prefab(file_path=str(path), scene=scene)
    instance.prefab_guid = "reference-undo-guid"
    instance.prefab_root = True
    instance.get_child(0).prefab_guid = "reference-undo-guid"
    instance.get_py_component(_PrefabReferenceComponent).target_object = GameObjectRef(instance)
    previous_manager = UndoManager._instance
    manager = UndoManager()

    def verify(root_reference, has_override):
        refs = instance.get_py_component(_PrefabReferenceComponent)
        target = instance.get_child(0)
        assert refs.target_object is (instance if root_reference else target)
        assert refs.target_component is target.get_py_component(_PrefabTargetComponent)
        assert bool(compute_overrides(instance, str(path))) is has_override

    try:
        builder = build_prefab_apply_command if action == "apply" else build_prefab_revert_command
        assert manager.execute(builder(instance, str(path)))
        verify(action == "apply", False)
        manager.undo()
        verify(True, True)
        manager.redo()
        verify(action == "apply", False)
    finally:
        UndoManager._instance = previous_manager


def test_prefab_revert_undo_captures_inspector_builtin_wrapper_edit(scene, tmp_path):
    _PREFAB_TEMPLATE_CACHE.clear()
    source = scene.create_game_object("CheckpointGate")
    source_child = scene.create_game_object("LeftPost")
    source_child.add_component("BoxCollider")
    source_child.set_parent(source)
    path = tmp_path / "checkpoint-inspector-revert-undo.prefab"
    assert save_prefab(source, str(path)) is True

    instance = instantiate_prefab(file_path=str(path), scene=scene)
    instance.prefab_guid = "checkpoint-guid"
    instance.prefab_root = True
    child = instance.get_child(0)
    child.prefab_guid = "checkpoint-guid"
    raw_collider = child.get_component("BoxCollider")
    collider = BoxCollider._get_or_create_wrapper(raw_collider, child)

    previous_manager = UndoManager._instance
    manager = UndoManager()
    try:
        assert (
            manager.execute(
                BuiltinPropertyCommand(
                    collider,
                    "is_trigger",
                    False,
                    True,
                    "Set is_trigger",
                )
            )
            is True
        )
        assert collider.is_trigger is True
        assert compute_overrides(instance, str(path))

        assert manager.execute(build_prefab_revert_command(instance, str(path))) is True
        assert instance.get_child(0).get_component("BoxCollider").is_trigger is False

        manager.undo()
        restored = instance.get_child(0).get_component("BoxCollider")
        assert restored.is_trigger is True
        assert compute_overrides(instance, str(path))
    finally:
        UndoManager._instance = previous_manager


def test_apply_propagates_to_existing_instances_and_preserves_overrides(scene, tmp_path):
    _PREFAB_TEMPLATE_CACHE.clear()
    source = scene.create_game_object("CheckpointGate")
    child = scene.create_game_object("LeftPost")
    child.add_component("BoxCollider")
    child.set_parent(source)
    path = tmp_path / "checkpoint-propagate.prefab"
    assert save_prefab(source, str(path)) is True

    first = instantiate_prefab(file_path=str(path), scene=scene)
    second = instantiate_prefab(file_path=str(path), scene=scene)
    for instance in (first, second):
        instance.prefab_guid = "checkpoint-guid"
        instance.prefab_root = True
        instance.get_child(0).prefab_guid = "checkpoint-guid"

    first.name = "CheckpointGate_A"
    first.transform.position = Vector3(1.0, 0.0, 0.0)
    first_collider = first.get_child(0).get_component("BoxCollider")
    first_collider.size = Vector3(2.0, 1.0, 1.0)

    second.name = "CheckpointGate_B"
    second.transform.position = Vector3(5.0, 0.0, 0.0)
    second.get_child(0).get_component("BoxCollider").is_trigger = True

    assert apply_overrides_to_prefab(second, str(path)) is True

    refreshed_first = first.get_child(0).get_component("BoxCollider")
    refreshed_second = second.get_child(0).get_component("BoxCollider")
    assert refreshed_first.is_trigger is True
    assert refreshed_first.size.x == pytest.approx(2.0)
    assert refreshed_second.is_trigger is True
    assert first.name == "CheckpointGate_A"
    assert first.transform.position.x == pytest.approx(1.0)
    assert second.name == "CheckpointGate_B"
    assert second.transform.position.x == pytest.approx(5.0)
    first_overrides = compute_overrides(first, str(path))
    assert any("BoxCollider" in override.key for override in first_overrides)
    assert compute_overrides(second, str(path)) == []


def test_apply_overrides_transaction_restores_asset_and_instances_on_undo(
    scene,
    tmp_path,
):
    _PREFAB_TEMPLATE_CACHE.clear()
    source = scene.create_game_object("TransactionalPrefab")
    source.add_component("BoxCollider")
    path = tmp_path / "transactional.prefab"
    assert save_prefab(source, str(path)) is True

    instance = instantiate_prefab(file_path=str(path), scene=scene)
    instance.prefab_guid = "transactional-guid"
    instance.prefab_root = True
    instance.get_component("BoxCollider").is_trigger = True
    previous_manager = UndoManager._instance
    manager = UndoManager()
    try:
        assert manager.execute(build_prefab_apply_command(instance, str(path))) is True
        saved = _read_prefab_document(str(path))["root_object"]
        assert saved["components"][0]["data"]["is_trigger"] is True

        manager.undo()
        saved = _read_prefab_document(str(path))["root_object"]
        assert saved["components"][0]["data"]["is_trigger"] is False
        assert instance.get_component("BoxCollider").is_trigger is True

        manager.redo()
        saved = _read_prefab_document(str(path))["root_object"]
        assert saved["components"][0]["data"]["is_trigger"] is True
        assert instance.get_component("BoxCollider").is_trigger is True
    finally:
        UndoManager._instance = previous_manager
