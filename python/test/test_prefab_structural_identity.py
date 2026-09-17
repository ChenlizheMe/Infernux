"""Prefab structural authoring must not alias new nodes to retired/source nodes."""
import copy
import json

import pytest

from Infernux.components import InxComponent, FieldType, serialized_field
from Infernux.components.ref_wrappers import GameObjectRef, ComponentRef
from Infernux.engine.component_restore import clone_game_object_transactionally
from Infernux.engine.prefab_manager import PrefabDocumentError, _read_prefab_document, instantiate_prefab, save_prefab
from Infernux.engine.prefab_overrides import (
    apply_overrides_to_prefab, build_prefab_apply_command, compute_overrides, revert_overrides,
)


class _StructuralReferences(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    component = serialized_field(default=None, field_type=FieldType.COMPONENT)


def _make_prefab(scene, tmp_path):
    root = scene.create_game_object("StructuralPrefab")
    child = scene.create_game_object("Original")
    child.set_parent(root)
    child.add_component("BoxCollider")
    path = str(tmp_path / "structural.prefab")
    assert save_prefab(root, path)
    first = instantiate_prefab(file_path=path, guid="structural-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="structural-guid", scene=scene)
    return path, first, second


def test_deleted_source_identity_is_not_reused_by_later_addition(scene, tmp_path):
    path, first, second = _make_prefab(scene, tmp_path)
    retired = first.get_child(0).prefab_source_id
    scene.destroy_game_object(first.get_child(0))
    scene.process_pending_destroys()
    assert apply_overrides_to_prefab(first, path)
    added = scene.create_game_object("Replacement")
    added.set_parent(first)
    runtime_id = added.id
    assert apply_overrides_to_prefab(first, path)
    assert first.get_child(0).id == runtime_id
    assert first.get_child(0).prefab_source_id != retired
    assert second.get_child(0).prefab_source_id == first.get_child(0).prefab_source_id


def test_copying_prefab_child_creates_an_added_node_not_a_second_source_node(scene, tmp_path):
    path, first, second = _make_prefab(scene, tmp_path)
    original_id = first.get_child(0).id
    copied = clone_game_object_transactionally(scene, first.get_child(0), parent=first)
    copied_id = copied.id
    assert copied.prefab_source_id == 0
    assert apply_overrides_to_prefab(first, path)
    assert [child.id for child in first.get_children()] == [original_id, copied_id]
    assert len({child.prefab_source_id for child in first.get_children()}) == 2
    assert len(second.get_children()) == 2


def test_separate_instance_additions_are_not_merged_together(scene, tmp_path):
    path, first, second = _make_prefab(scene, tmp_path)
    published = scene.create_game_object("Published")
    published.set_parent(first)
    private = scene.create_game_object("Private")
    private.set_parent(second)
    published_id, private_id = published.id, private.id
    assert apply_overrides_to_prefab(first, path)
    assert {child.name for child in second.get_children()} == {"Original", "Published", "Private"}
    assert scene.find_by_id(published_id).get_parent() is first
    assert scene.find_by_id(private_id).get_parent() is second
    assert scene.find_by_id(private_id).prefab_source_id == 0
    assert any(change.key == "added_child:Private" for change in compute_overrides(second, path))
    assert revert_overrides(second, path)
    assert {child.name for child in second.get_children()} == {"Original", "Published"}
    assert {node["name"] for node in _read_prefab_document(path)["root_object"]["children"]} == {"Original", "Published"}


def test_apply_keeps_other_instances_external_scene_reference_overrides(scene, tmp_path):
    root = scene.create_game_object("References")
    root.add_py_component(_StructuralReferences())
    path = str(tmp_path / "references.prefab")
    assert save_prefab(root, path)
    first = instantiate_prefab(file_path=path, guid="refs-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="refs-guid", scene=scene)
    external = scene.create_game_object("External")
    external.add_component("BoxCollider")
    references = second.get_py_component(_StructuralReferences)
    references.target = GameObjectRef(external)
    references.component = ComponentRef(go_id=external.id, component_type="BoxCollider")
    first.add_component("SphereCollider")
    command = build_prefab_apply_command(first, path)
    for action in (command.execute, command.undo, command.redo):
        action()
        refreshed = second.get_py_component(_StructuralReferences)
        assert refreshed.target is external
        assert refreshed.component is not None
        assert compute_overrides(second, path)
    assert _read_prefab_document(path)["root_object"]["components"][0]["data"]["target"]["object_id"] == 0


def test_instance_addition_identity_and_internal_reference_survive_apply_history(scene, tmp_path):
    root = scene.create_game_object("LocalReferences")
    root.add_py_component(_StructuralReferences())
    path = str(tmp_path / "local-refs.prefab")
    assert save_prefab(root, path)
    first = instantiate_prefab(file_path=path, guid="local-refs-guid", scene=scene)
    second = instantiate_prefab(file_path=path, guid="local-refs-guid", scene=scene)
    private = scene.create_game_object("Private")
    private.set_parent(second)
    private_id = private.id
    second.get_py_component(_StructuralReferences).target = GameObjectRef(private)
    published = scene.create_game_object("Published")
    published.set_parent(first)
    published_id = published.id
    first.get_py_component(_StructuralReferences).target = GameObjectRef(published)
    command = build_prefab_apply_command(first, path)
    for action in (command.execute, command.undo, command.redo):
        action()
        assert first.get_py_component(_StructuralReferences).target.id == published_id
        assert second.get_py_component(_StructuralReferences).target.id == private_id
        assert scene.find_by_id(private_id).prefab_source_id == 0
    assert {child.name for child in second.get_children()} == {"Private", "Published"}


@pytest.mark.parametrize("invalid", [0, 1, 2, -1, True, "4"])
def test_prefab_rejects_invalid_allocator_watermark(scene, tmp_path, invalid):
    path, _, _ = _make_prefab(scene, tmp_path)
    document = _read_prefab_document(path)
    document["next_local_id"] = invalid
    bad_path = tmp_path / "bad-watermark.prefab"
    bad_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PrefabDocumentError, match="next_local_id"):
        _read_prefab_document(str(bad_path))


def test_prefab_legacy_watermark_is_imported_from_existing_ids(scene, tmp_path):
    path, _, _ = _make_prefab(scene, tmp_path)
    document = copy.deepcopy(_read_prefab_document(path))
    document.pop("next_local_id")
    legacy = tmp_path / "legacy.prefab"
    legacy.write_text(json.dumps(document), encoding="utf-8")
    assert _read_prefab_document(str(legacy))["next_local_id"] == 3


def test_prefab_mode_save_exit_preserves_instance_overrides_and_identities(scene, tmp_path, monkeypatch):
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import EditorInteractionCore, DocumentRegistry, SelectionDomain
    from Infernux.lib import SceneManager
    from types import SimpleNamespace

    path, first, second = _make_prefab(scene, tmp_path)
    first_id, second_id = first.id, second.id
    original_id = second.get_child(0).id
    private = scene.create_game_object("Private")
    private.set_parent(second)
    private_id = private.id
    second.get_child(0).name = "Instance Rename"
    watcher = scene.create_game_object("Watcher")
    watcher_id = watcher.id
    watcher.add_py_component(_StructuralReferences())
    watcher.get_py_component(_StructuralReferences).target = GameObjectRef(second.get_child(0))

    monkeypatch.setattr(SceneFileManager, "_instance", None)
    core = EditorInteractionCore()
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    manager = SceneFileManager()
    original_document_id = manager.document_id
    original_scene_count = SceneManager.instance().scene_count
    try:
        assert manager.open_prefab_mode(path)
        mode_scene = SceneManager.instance().get_active_scene()
        mode_root = mode_scene.get_root_objects()[0]
        node = mode_scene.create_game_object("SavedChild")
        node.set_parent(mode_root)
        node.add_component("BoxCollider")
        assert manager._save_prefab()
        source_id = node.prefab_source_id
        document = _read_prefab_document(path)
        assert source_id > 0 and document["next_local_id"] > source_id
        assert manager._save_prefab()
        assert _read_prefab_document(path) == document
        assert not DocumentRegistry.instance().require(manager.document_id).is_dirty

        # Only path resolution is isolated; graph publication and file saving
        # use the real native Scene and bound document controller throughout.
        manager._asset_database = SimpleNamespace(
            get_guid_from_path=lambda _: "structural-guid", get_path_from_guid=lambda _: "",
        )
        assert manager._do_exit_prefab_mode()
        restored = SceneManager.instance().get_active_scene()
        assert restored is scene
        assert SceneManager.instance().scene_count == original_scene_count
        assert manager.scene_for_document(original_document_id) is restored
        assert manager.document_id == original_document_id
        assert DocumentRegistry.instance().require(original_document_id).is_dirty
        assert {obj.name for obj in restored.find_by_id(first_id).get_children()} == {"Original", "SavedChild"}
        assert {obj.name for obj in restored.find_by_id(second_id).get_children()} == {"Instance Rename", "SavedChild", "Private"}
        assert restored.find_by_id(original_id).name == "Instance Rename"
        assert restored.find_by_id(private_id).prefab_source_id == 0
        assert restored.find_by_id(watcher_id).get_py_component(_StructuralReferences).target.id == original_id
        # A no-edit second visit must not make a clean scene dirty, and the
        # already-saved node must retain the same source identity.
        manager._asset_database = None
        registry = DocumentRegistry.instance()
        registry.restore_saved_revision(original_document_id)
        assert manager.open_prefab_mode(path)
        reopened = SceneManager.instance().get_active_scene().get_root_objects()[0]
        assert next(obj for obj in reopened.get_children() if obj.name == "SavedChild").prefab_source_id == source_id
        assert manager._do_exit_prefab_mode()
        assert not registry.require(original_document_id).is_dirty
    finally:
        core.shutdown()
