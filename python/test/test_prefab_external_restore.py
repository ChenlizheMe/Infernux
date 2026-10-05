"""Restored Prefab sources must publish into every affected resident scene."""
import json
from pathlib import Path
from uuid import uuid4

import pytest

from infernux import editor
from infernux.components.ref_wrappers import ComponentRef
from infernux.core.assets import AssetManager
from infernux.engine import project_context
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.prefab_manager import instantiate_prefab, _read_resolved_prefab_document
from infernux.engine.resources_manager import ResourceChangeHandler
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.undo import UndoManager
from infernux.lib import SceneManager


@pytest.fixture
def prefab_project(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    previous_root = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
    monkeypatch.setattr(UndoManager, "_instance", None)
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    files = SceneFileManager()
    files.set_asset_database(database)
    files.register_loaded_scene(scene, "")
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    handler = ResourceChangeHandler(engine, project_path=database.project_root)
    try:
        yield database, folder, files, handler
    finally:
        core.shutdown()
        project_context.set_project_root(previous_root)


def _linked_worlds(scene, fixture, depth):
    database, folder, files, handler = fixture
    root = scene.create_game_object("Source")
    root.add_component("Rigidbody").is_kinematic = True
    child = scene.create_game_object("Door")
    child.set_parent(root)
    child.layer = 9
    child.add_component("BoxCollider")
    body = child.add_component("Rigidbody")
    body.is_kinematic = True
    root.add_component("HingeJoint").connected_body = body
    base = folder / "Base.prefab"
    editor.save_as_prefab_asset(root, base)
    source_path = base
    variant_bytes = {}
    for number in range(depth):
        variant = folder / f"Variant{number}.prefab"
        contents = editor.load_prefab_contents(source_path)
        try:
            contents.get_child(0).name = f"Variant door {number}"
            editor.save_as_prefab_asset(contents, variant)
        finally:
            editor.unload_prefab_contents(contents)
        variant_bytes[variant] = variant.read_bytes()
        source_path = variant
    other = SceneManager.instance().create_scene("Other resident world")
    files.register_loaded_scene(other, "")
    instances = [instantiate_prefab(file_path=str(source_path),
                                   guid=database.get_guid_from_path(str(source_path)),
                                   scene=world, asset_database=database)
                 for world in (scene, other)]
    instances[1].get_child(0).name = "Local override"
    refs = [ComponentRef(obj.get_child(0).get_component("BoxCollider")) for obj in instances]
    identities = [(obj.id, obj.get_child(0).id, ref.resolve().component_id)
                  for obj, ref in zip(instances, refs)]
    return base, source_path, instances, refs, identities, variant_bytes


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("event", ["modified", "restored"])
def test_source_update_reaches_resident_instances_without_losing_identity(prefab_project, scene, depth, event):
    database, _, _, handler = prefab_project
    base, variant, instances, refs, identities, variant_bytes = _linked_worlds(scene, prefab_project, depth)
    meta = Path(str(base) + ".meta")
    metadata_bytes = meta.read_bytes()
    guid = database.get_guid_from_path(str(base))
    updated = json.loads(base.read_text(encoding="utf-8"))
    updated["root_object"]["children"][0]["layer"] = 11
    if event == "restored":
        base.unlink()
        meta.unlink()
        handler._commit_deleted(str(base), guid_hint=guid)
        assert not database.get_guid_from_path(str(base))
        assert [obj.get_child(0).layer for obj in instances] == [9, 9]
        meta.write_bytes(metadata_bytes)
    base.write_text(json.dumps(updated), encoding="utf-8")
    source_bytes = base.read_bytes()
    if event == "restored":
        handler._commit_created(str(base))
    else:
        handler._commit_modified(str(base))
    assert database.get_guid_from_path(str(base)) == guid
    assert _read_resolved_prefab_document(str(variant), database)["root_object"]["children"][0]["layer"] == 11
    assert [obj.get_child(0).layer for obj in instances] == [11, 11]
    assert instances[0].get_child(0).name == (f"Variant door {depth-1}" if depth else "Door")
    assert instances[1].get_child(0).name == "Local override"
    assert [(obj.id, obj.get_child(0).id, ref.resolve().component_id)
            for obj, ref in zip(instances, refs)] == identities
    for obj in instances:
        assert obj.get_component("HingeJoint").connected_body is obj.get_child(0).get_component("Rigidbody")
    assert base.read_bytes() == source_bytes
    assert all(path.read_bytes() == payload for path, payload in variant_bytes.items())


def test_invalid_restored_prefab_preserves_worlds_until_valid_source_arrives(prefab_project, scene):
    database, _, _, handler = prefab_project
    base, _, instances, _, _, _ = _linked_worlds(scene, prefab_project, 2)
    meta = Path(str(base) + ".meta")
    original_meta = meta.read_bytes()
    valid = json.loads(base.read_text(encoding="utf-8"))
    guid = database.get_guid_from_path(str(base))
    before = [obj.serialize_document() for obj in instances]
    base.unlink()
    meta.unlink()
    handler._commit_deleted(str(base), guid_hint=guid)
    invalid = b"<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> teammate\n"
    meta.write_bytes(original_meta)
    base.write_bytes(invalid)
    with pytest.raises((RuntimeError, ValueError)):
        handler._commit_created(str(base))
    assert base.read_bytes() == invalid
    assert [obj.serialize_document() for obj in instances] == before
    valid["root_object"]["children"][0]["layer"] = 12
    base.write_text(json.dumps(valid), encoding="utf-8")
    handler._commit_modified(str(base))
    assert [obj.get_child(0).layer for obj in instances] == [12, 12]


def test_different_guid_at_old_path_does_not_replace_instance_source(prefab_project, scene):
    database, _, _, handler = prefab_project
    base, _, instances, _, _, _ = _linked_worlds(scene, prefab_project, 0)
    meta = Path(str(base) + ".meta")
    replacement_meta = json.loads(meta.read_text(encoding="utf-8"))
    replacement = json.loads(base.read_text(encoding="utf-8"))
    old_guid = database.get_guid_from_path(str(base))
    before = [obj.serialize_document() for obj in instances]
    base.unlink()
    meta.unlink()
    handler._commit_deleted(str(base), guid_hint=old_guid)
    replacement_meta["metadata"]["guid"]["value"] = uuid4().hex
    replacement["root_object"]["children"][0]["layer"] = 12
    meta.write_text(json.dumps(replacement_meta), encoding="utf-8")
    base.write_text(json.dumps(replacement), encoding="utf-8")
    handler._commit_created(str(base))
    assert database.get_guid_from_path(str(base)) != old_guid
    assert [obj.serialize_document() for obj in instances] == before
