"""Restored Prefab sources must publish into every affected resident scene."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from infernux import editor
from infernux.components.ref_wrappers import ComponentRef
from infernux.core.assets import AssetManager
from infernux.engine import project_context
from infernux.engine.interaction import EditorInteractionCore, DocumentRegistry
from infernux.engine.prefab_manager import instantiate_prefab, _read_resolved_prefab_document
from infernux.engine.resources_manager import ResourceChangeHandler
from infernux.engine.scene_manager import SceneFileManager
from infernux.engine.scene_authoring import encode_scene_document
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
    instances = [
        instantiate_prefab(
            file_path=str(source_path),
            guid=database.get_guid_from_path(str(source_path)),
            scene=world,
            asset_database=database,
        )
        for world in (scene, other)
    ]
    instances[1].get_child(0).name = "Local override"
    refs = [ComponentRef(obj.get_child(0).get_component("BoxCollider")) for obj in instances]
    identities = [(obj.id, obj.get_child(0).id, ref.resolve().component_id) for obj, ref in zip(instances, refs)]
    return base, source_path, instances, refs, identities, variant_bytes


def _publish_external_event(handler, kind, path):
    getattr(handler, f"on_{kind}")(SimpleNamespace(is_directory=False, src_path=str(path)))
    assert handler.process_pending_reloads(force=True) == 1


@pytest.mark.parametrize("event,depth", [("modified", 1), ("restored", 2)])
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
        _publish_external_event(handler, "deleted", base)
        assert not database.get_guid_from_path(str(base))
        assert [obj.get_child(0).layer for obj in instances] == [9, 9]
        meta.write_bytes(metadata_bytes)
    base.write_text(json.dumps(updated), encoding="utf-8")
    source_bytes = base.read_bytes()
    _publish_external_event(handler, "created" if event == "restored" else "modified", base)
    assert database.get_guid_from_path(str(base)) == guid
    assert _read_resolved_prefab_document(str(variant), database)["root_object"]["children"][0]["layer"] == 11
    assert [obj.get_child(0).layer for obj in instances] == [11, 11]
    assert instances[0].get_child(0).name == (f"Variant door {depth - 1}" if depth else "Door")
    assert instances[1].get_child(0).name == "Local override"
    assert [
        (obj.id, obj.get_child(0).id, ref.resolve().component_id) for obj, ref in zip(instances, refs)
    ] == identities
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
    _publish_external_event(handler, "deleted", base)
    invalid = b"<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> teammate\n"
    meta.write_bytes(original_meta)
    base.write_bytes(invalid)
    _publish_external_event(handler, "created", base)
    assert database.get_guid_from_path(str(base)) == guid
    assert database.get_import_error(guid)
    assert base.read_bytes() == invalid
    assert [obj.serialize_document() for obj in instances] == before
    valid["root_object"]["children"][0]["layer"] = 12
    base.write_text(json.dumps(valid), encoding="utf-8")
    _publish_external_event(handler, "modified", base)
    assert [obj.get_child(0).layer for obj in instances] == [12, 12]


@pytest.mark.parametrize("depth", [0, 1])
def test_failed_cached_prefab_is_unavailable_to_new_instances_and_scene_open(prefab_project, scene, engine, depth,
                                                                           monkeypatch):
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors

    database, folder, files, handler = prefab_project
    files.set_engine(engine)
    monkeypatch.setattr(AssetLoadErrors, "_instance", None)
    errors = AssetLoadErrors(database, None)
    base, source_path, instances, _refs, _identities, _variants = _linked_worlds(scene, prefab_project, depth)
    source_guid = database.get_guid_from_path(str(source_path))
    before_source = base.read_bytes()
    metadata = Path(str(base) + ".meta").read_bytes()
    before_scene = scene.serialize_document()
    before_instances = [obj.serialize_document() for obj in instances]
    target = folder / "Candidate.scene"
    disk = encode_scene_document(before_scene)
    disk["objects"][0]["name"] = "Different disk scene"
    target.write_text(json.dumps(disk), encoding="utf-8")
    before_target = target.read_bytes()
    imported_scene = database.import_asset(str(target))
    assert imported_scene, imported_scene.error
    documents = DocumentRegistry.instance()
    document = documents.require(files.document_id_for_scene(scene.world_id))
    documents.mark_changed(document.document_id)
    revision = document.revision

    base.write_bytes(b"{ damaged current Prefab source")
    rejected = AssetManager.reimport_asset(str(base), database=database)
    assert not rejected and database.get_import_error(database.get_guid_from_path(str(base)))
    assert source_guid in files._published_prefab_sources
    assert instantiate_prefab(guid=source_guid, scene=scene, asset_database=database) is None
    with pytest.raises(ValueError, match="Prefab source.*unavailable"):
        files.sync_all_prefab_instances(scene)
    assert not files._do_open_scene(str(target), record_navigation=False)
    assert files.last_scene_load["status"] == "failed"
    assert database.get_import_error(imported_scene.guid)
    assert errors.error(str(target))
    assert scene.serialize_document() == before_scene
    assert [obj.serialize_document() for obj in instances] == before_instances
    assert document.is_dirty and document.revision == revision
    assert base.read_bytes() == b"{ damaged current Prefab source"
    assert Path(str(base) + ".meta").read_bytes() == metadata
    assert target.read_bytes() == before_target

    repaired_source = json.loads(before_source)
    repaired_source["root_object"]["children"][0]["layer"] = 12
    base.write_text(json.dumps(repaired_source), encoding="utf-8")
    assert AssetManager.reimport_asset(str(base), database=database)
    files.sync_prefab_dependents(database.get_guid_from_path(str(base)))
    repaired = instantiate_prefab(guid=source_guid, scene=scene, asset_database=database)
    assert repaired is not None and repaired.get_child(0).layer == 12
    assert files._do_open_scene(str(target), record_navigation=False)
    assert files.last_scene_load["status"] == "loaded"
    assert not database.get_import_error(imported_scene.guid)
    assert not errors.error(str(target))
    assert scene.find("Different disk scene") is not None
    assert target.read_bytes() == before_target


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
    _publish_external_event(handler, "deleted", base)
    replacement_meta["metadata"]["guid"]["value"] = uuid4().hex
    replacement["root_object"]["children"][0]["layer"] = 12
    meta.write_text(json.dumps(replacement_meta), encoding="utf-8")
    base.write_text(json.dumps(replacement), encoding="utf-8")
    _publish_external_event(handler, "created", base)
    assert database.get_guid_from_path(str(base)) != old_guid
    assert [obj.serialize_document() for obj in instances] == before
