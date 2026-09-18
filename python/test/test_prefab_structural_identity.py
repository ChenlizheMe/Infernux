"""Prefab structural authoring must not alias new nodes to retired/source nodes."""
import copy
import json

import pytest

from Infernux.components import InxComponent, FieldType, serialized_field
from Infernux.components.ref_wrappers import GameObjectRef, ComponentRef
from Infernux.engine.component_restore import clone_game_object_transactionally
from Infernux.engine.prefab_manager import PrefabDocumentError, _read_prefab_document, instantiate_prefab, save_prefab, _make_prefab_baseline
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
            get_guid_from_path=lambda _: "structural-guid", get_path_from_guid=lambda _: path,
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


def test_scene_reopen_merges_updated_source_and_keeps_instance_references(scene, tmp_path):
    from Infernux.engine._scene_prefab import ScenePrefabMixin
    from Infernux.engine.prefab_manager import save_prefab_document
    from Infernux.engine.component_restore import serialize_game_object_document_authoritatively
    from Infernux.engine.scene_document_transaction import SceneDocumentTransaction

    path, first, second = _make_prefab(scene, tmp_path)
    first_id, second_id = first.id, second.id
    child_id = second.get_child(0).id
    second.get_child(0).name = "Local Name"
    private = scene.create_game_object("Local Addition")
    private.set_parent(second)
    private_id = private.id
    watcher = scene.create_game_object("Watcher")
    watcher_id = watcher.id
    watcher.add_py_component(_StructuralReferences())
    watcher.get_py_component(_StructuralReferences).target = GameObjectRef(second.get_child(0))
    snapshot = scene.serialize_document()
    snapshot["objects"] = [serialize_game_object_document_authoritatively(obj) for obj in scene.get_root_objects()]
    scene_path = tmp_path / "instances.scene"
    scene_path.write_text(json.dumps(snapshot), encoding="utf-8")

    updated = _read_prefab_document(path)
    updated["root_object"]["children"][0]["active"] = False
    added = copy.deepcopy(updated["root_object"]["children"][0])
    added["name"] = "Source Addition"
    added["components"][0]["component_id"] = updated["next_component_id"]
    updated["next_component_id"] += 1
    added["local_id"] = updated["next_local_id"]
    updated["next_local_id"] += 1
    updated["root_object"]["children"].append(added)
    assert save_prefab_document(updated, path)
    transaction = SceneDocumentTransaction(scene, path=scene_path)
    assert transaction.run_to_completion()
    assert ScenePrefabMixin._refresh_prefab_instances(scene, "structural-guid", path)
    first, second = scene.find_by_id(first_id), scene.find_by_id(second_id)
    assert {obj.name for obj in first.get_children()} == {"Original", "Source Addition"}
    assert {obj.name for obj in second.get_children()} == {"Local Name", "Local Addition", "Source Addition"}
    assert not scene.find_by_id(child_id).active_self
    assert scene.find_by_id(private_id).prefab_source_id == 0
    assert scene.find_by_id(watcher_id).get_py_component(_StructuralReferences).target.id == child_id
    assert second._prefab_source_document == _make_prefab_baseline(updated["root_object"])
    assert not ScenePrefabMixin._refresh_prefab_instances(scene, "structural-guid", path)


def test_legacy_scene_baseline_adoption_does_not_destroy_authored_overrides(scene, tmp_path):
    from Infernux.engine._scene_prefab import ScenePrefabMixin
    path, first, _ = _make_prefab(scene, tmp_path)
    first._prefab_source_document = None
    child = first.get_child(0)
    child_id = child.id
    child.name = "Legacy Override"
    assert ScenePrefabMixin._refresh_prefab_instances(scene, "structural-guid", path)
    assert first.get_child(0).id == child_id
    assert first.get_child(0).name == "Legacy Override"
    assert first._prefab_source_document == _make_prefab_baseline(_read_prefab_document(path)["root_object"])


def test_player_cook_strips_only_objectgraph_prefab_baselines(tmp_path):
    from Infernux.engine.game_builder import GameBuilder
    builder = GameBuilder.__new__(GameBuilder)
    builder.project_path = str(tmp_path)
    source = tmp_path / "payload.scene"
    document = {"objects": [{
        "prefab_source": {"old_editor_only_data": True}, "children": [],
        "components": [{"data": {"prefab_source": "ordinary user field"}}],
    }]}
    source.write_text(json.dumps(document), encoding="utf-8")
    builder._rewrite_player_document_paths(str(source), ".scene")
    cooked = json.loads(source.read_text(encoding="utf-8"))
    assert "prefab_source" not in cooked["objects"][0]
    assert cooked["objects"][0]["components"][0]["data"]["prefab_source"] == "ordinary user field"


def test_unopened_scene_cook_merges_source_without_live_objects(scene, tmp_path):
    from Infernux.engine.prefab_overrides import resolve_scene_prefab_documents
    from Infernux.engine.component_restore import serialize_game_object_document_authoritatively

    path, first, second = _make_prefab(scene, tmp_path)
    second.get_child(0).name = "Local Override"
    private = scene.create_game_object("Private")
    private.set_parent(second)
    watcher = scene.create_game_object("Watcher")
    watcher.add_py_component(_StructuralReferences())
    watcher.get_py_component(_StructuralReferences).target = GameObjectRef(second.get_child(0))
    document = scene.serialize_document()
    document["objects"] = [serialize_game_object_document_authoritatively(obj) for obj in scene.get_root_objects()]
    before = copy.deepcopy(document)
    updated = _read_prefab_document(path)["root_object"]
    updated["children"][0]["active"] = False
    added = copy.deepcopy(updated["children"][0])
    added["local_id"] = 100
    added["name"] = "New From Source"
    added["components"][0]["component_id"] = 100
    updated["children"].append(added)
    live_ids = [obj.id for obj in scene.get_all_objects()]
    reads = []

    def load(guid):
        reads.append(guid)
        assert guid == "structural-guid"
        return updated

    cooked = resolve_scene_prefab_documents(document, load)
    assert document == before
    assert [obj.id for obj in scene.get_all_objects()] == live_ids
    assert reads == ["structural-guid"]
    assert cooked == resolve_scene_prefab_documents(document, load)
    roots = {obj["id"]: obj for obj in cooked["objects"]}
    children = roots[second.id]["children"]
    assert {obj["name"] for obj in children} == {"Local Override", "Private", "New From Source"}
    assert next(obj for obj in children if obj["name"] == "Local Override")["id"] == second.get_child(0).id
    assert not next(obj for obj in children if obj["name"] == "Local Override")["active"]
    new_first = next(obj for obj in roots[first.id]["children"] if obj["name"] == "New From Source")
    new_second = next(obj for obj in children if obj["name"] == "New From Source")
    assert new_first["id"] != new_second["id"]
    assert min(new_first["id"], new_second["id"]) > max(live_ids)
    assert roots[watcher.id] == next(obj for obj in before["objects"] if obj["id"] == watcher.id)
    component_ids = [c["component_id"] for root in cooked["objects"] for node in _all_nodes(root)
                     for c in [node["transform"], *node["components"]]]
    assert len(component_ids) == len(set(component_ids))


def _all_nodes(root):
    yield root
    for child in root["children"]:
        yield from _all_nodes(child)


def test_builder_stages_latest_prefab_for_unopened_scene(scene, tmp_path):
    from Infernux.engine.game_builder import GameBuilder
    from Infernux.engine.prefab_manager import save_prefab_document
    from Infernux.engine.scene_document_transaction import SceneDocumentTransaction

    assets = tmp_path / "Assets"
    assets.mkdir()
    path, first, second = _make_prefab(scene, assets)
    second.get_child(0).name = "Placed Override"
    scene_path = assets / "unopened.scene"
    scene_path.write_text(json.dumps(scene.serialize_document()), encoding="utf-8")
    saved_bytes = scene_path.read_bytes()
    updated = _read_prefab_document(path)
    updated["root_object"]["children"][0]["active"] = False
    assert save_prefab_document(updated, path)

    builder = GameBuilder.__new__(GameBuilder)
    builder.project_path = str(tmp_path)
    builder._cooked_asset_entries = {
        "structural-guid": {"normalized_path": path},
        "scene-guid": {"normalized_path": str(scene_path)},
    }
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data = tmp_path / "Data"
    builder._stage_library_runtime_documents(str(data))
    artifact = data / "Library/Artifacts/Document/scene-guid.scene"
    cooked = json.loads(artifact.read_text(encoding="utf-8"))
    assert all("prefab_source" not in node for root in cooked["objects"] for node in _all_nodes(root))
    assert scene_path.read_bytes() == saved_bytes
    # The Editor's live scene was deliberately not refreshed by the cook.
    assert first.get_child(0).active_self
    second_id, child_id = second.id, second.get_child(0).id
    transaction = SceneDocumentTransaction(scene, path=artifact)
    assert transaction.run_to_completion()
    assert scene.find_by_id(second_id).get_child(0).name == "Placed Override"
    assert not scene.find_by_id(child_id).active_self


def test_scene_cook_rejects_missing_prefab_source(scene, tmp_path):
    from Infernux.engine.prefab_overrides import resolve_scene_prefab_documents
    _make_prefab(scene, tmp_path)
    document = scene.serialize_document()

    def missing(guid):
        raise LookupError(f"Missing source: {guid}")

    with pytest.raises(LookupError, match="Missing source: structural-guid"):
        resolve_scene_prefab_documents(document, missing)
