"""Offline Prefab editing uses real isolated worlds and the Project asset journal."""
import json
from pathlib import Path

import pytest

from Infernux import editor
from Infernux.components import InxComponent, FieldType, serialized_field
from Infernux.components.ref_wrappers import ComponentRef, GameObjectRef
from Infernux.components._component_lifecycle import RuntimeExecutionScheduler
from Infernux.core import AssetManager
from Infernux.engine.interaction import EditorInteractionCore
from Infernux.engine.undo import UndoManager
from Infernux.lib import SceneManager, Vector3, Physics


class ContentsProbe(InxComponent):
    target = serialized_field(default=None, field_type=FieldType.COMPONENT)
    _calls = []

    def awake(self):
        type(self)._calls.append("awake")

    def update(self, delta_time):
        type(self)._calls.append("update")


@pytest.fixture
def contents_project(engine, scene, monkeypatch, tmp_path):
    from Infernux.engine import project_context

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(project_context, "get_project_root", lambda: database.project_root)
    monkeypatch.setattr(UndoManager, "_instance", None)
    core = EditorInteractionCore()
    UndoManager(core.action_journal)
    core.project_assets.configure(database.project_root, database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    try:
        yield core, folder
    finally:
        core.shutdown()


def make_asset(scene, path):
    root = scene.create_game_object("Offline source")
    child = scene.create_game_object("Collider")
    child.set_parent(root)
    collider = child.add_component("BoxCollider")
    probe = ContentsProbe()
    root.add_py_component(probe)
    probe.target = ComponentRef(collider)
    editor.save_as_prefab_asset(root, path)
    return root


def test_load_edit_save_unload_keeps_active_world_and_identity(contents_project, scene):
    core, folder = contents_project
    path = folder / "Source.prefab"
    live = make_asset(scene, path)
    manager = SceneManager.instance()
    before = scene.serialize_document()
    count = manager.scene_count
    original = json.loads(path.read_text(encoding="utf8"))
    ContentsProbe._calls.clear()
    root = editor.load_prefab_contents(path)
    reference = GameObjectRef(root)
    world = root.scene.world_id
    try:
        assert root.name == "Offline source"
        assert root.scene.is_preview
        assert manager.scene_count == count
        assert manager.get_active_scene() is scene
        assert manager.find_runtime_object_by_id(root.id) is None
        probe = root.get_py_component(ContentsProbe)
        assert probe.target.game_object.id == root.get_child(0).id
        assert not RuntimeExecutionScheduler._owner_is_active_in_hierarchy(probe)
        scheduler = RuntimeExecutionScheduler()
        try:
            scheduler.register_component(probe)
            scheduler.execute_frame(0.02, 0.016)
        finally:
            scheduler.clear()
        assert ContentsProbe._calls == []
        with pytest.raises(ValueError, match="must be loaded"):
            manager.set_active_scene(root.scene)
        with pytest.raises(ValueError, match="preview Scene boundary"):
            root.set_parent(live)
        with pytest.raises(ValueError, match="preview Scene boundary"):
            live.set_parent(root)
        root.name = "Edited root"
        root.transform.local_position = Vector3(1, 2, 3)
        child = root.scene.create_game_object("New child")
        child.set_parent(root)
        editor.save_as_prefab_asset(root, path)
        first = json.loads(path.read_text(encoding="utf8"))
        assert first["root_object"]["name"] == "Edited root"
        assert first["root_object"]["transform"]["position"] == [1, 2, 3]
        editor.save_as_prefab_asset(root, path)
        assert json.loads(path.read_text(encoding="utf8")) == first
        assert first["root_object"]["local_id"] == original["root_object"]["local_id"]
        assert scene.serialize_document() == before
        assert live.name == "Offline source"
        editor.undo(defer=False)
        assert json.loads(path.read_text(encoding="utf8")) == original
        editor.redo(defer=False)
        assert json.loads(path.read_text(encoding="utf8")) == first
    finally:
        editor.unload_prefab_contents(root)
    assert not reference
    assert manager.get_scene_by_world_id(world) is None
    assert not core.prefabs._contents


def test_contents_never_join_physics_even_after_enable_toggle(contents_project, scene):
    _, folder = contents_project
    path = folder / "Isolated.prefab"
    make_asset(scene, path)
    root = editor.load_prefab_contents(path)
    try:
        child = root.get_child(0)
        child.transform.position = Vector3(1000, 0, 0)
        collider = child.get_component("BoxCollider")
        collider.enabled = False
        collider.enabled = True
        Physics.sync_transforms()
        assert Physics.raycast(Vector3(1000, 5, 0), Vector3(0, -1, 0), 10.0) is None
    finally:
        editor.unload_prefab_contents(root)


def test_existing_target_requires_explicit_load_and_prefab_mode_closed(contents_project, scene, monkeypatch):
    from types import SimpleNamespace
    from Infernux.engine.scene_manager import SceneFileManager

    _, folder = contents_project
    path = folder / "Guard.prefab"
    root = make_asset(scene, path)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="Load the target"):
        editor.save_as_prefab_asset(root, path)
    monkeypatch.setattr(SceneFileManager, "_instance", SimpleNamespace(
        is_prefab_mode=True, prefab_mode_path=str(path)))
    with pytest.raises(RuntimeError, match="Close this asset"):
        editor.load_prefab_contents(path)
    assert path.read_bytes() == before


def test_discard_conflict_and_session_shutdown(contents_project, scene):
    core, folder = contents_project
    path = folder / "Conflict.prefab"
    make_asset(scene, path)
    before = path.read_bytes()
    first = editor.load_prefab_contents(path)
    second = editor.load_prefab_contents(path)
    assert first.id != second.id
    first.name = "Published"
    editor.save_as_prefab_asset(first, path)
    second.name = "Stale"
    with pytest.raises(RuntimeError, match="source changed"):
        editor.save_as_prefab_asset(second, path)
    assert json.loads(path.read_text(encoding="utf8"))["root_object"]["name"] == "Published"
    editor.unload_prefab_contents(second)
    first.name = "Discard this"
    editor.unload_prefab_contents(first)
    root = editor.load_prefab_contents(path)
    reference = GameObjectRef(root)
    core.prefabs.shutdown()
    assert not reference
    assert path.read_bytes() != before
    assert json.loads(path.read_text(encoding="utf8"))["root_object"]["name"] == "Published"


def test_nested_contents_preserve_references_and_save_as(contents_project, scene):
    _, folder = contents_project
    inner = folder / "Inner.prefab"
    outer = folder / "Outer.prefab"
    copy_path = folder / "Copy.prefab"
    make_asset(scene, inner)
    root = scene.create_game_object("Outer")
    from Infernux.engine.prefab_manager import instantiate_prefab
    database = AssetManager.require_asset_database()
    for _ in range(2):
        instantiate_prefab(file_path=str(inner), guid=database.get_guid_from_path(str(inner)),
                           scene=scene, parent=root, asset_database=database)
    editor.save_as_prefab_asset(root, outer)
    loaded = editor.load_prefab_contents(outer)
    try:
        for child in loaded.get_children():
            assert child.get_py_component(ContentsProbe).target.game_object.id == child.get_child(0).id
        editor.save_as_prefab_asset(loaded, copy_path)
        first = json.loads(copy_path.read_text(encoding="utf8"))
        editor.save_as_prefab_asset(loaded, copy_path)
        assert json.loads(copy_path.read_text(encoding="utf8")) == first
        assert all(child["nested_prefab"]["guid"] == database.get_guid_from_path(str(inner))
                   for child in first["root_object"]["children"])
    finally:
        editor.unload_prefab_contents(loaded)


def test_missing_script_content_survives_offline_edit_at_safe_point(contents_project, scene):
    _, folder = contents_project
    path = folder / "Missing.prefab"
    make_asset(scene, path)
    document = json.loads(path.read_text(encoding="utf8"))
    record = document["root_object"]["components"][0]
    record["type_id"] = "python:" + "a" * 32 + ":" + "b" * 32 + ":Unavailable:LostScript"
    original_record = json.loads(json.dumps(record))
    from Infernux.engine.interaction.session import EditorInteractionCore
    EditorInteractionCore.instance().project_assets.set_text(str(path), json.dumps(document))
    root = editor.load_prefab_contents(path)
    try:
        root.name = "Preserved broken script"
        editor.save_as_prefab_asset(root, path)
        saved = json.loads(path.read_text(encoding="utf8"))["root_object"]
        assert saved["name"] == "Preserved broken script"
        assert saved["components"][0] == original_record
    finally:
        editor.unload_prefab_contents(root)


def test_deleted_source_is_not_recreated_by_stale_contents(contents_project, scene):
    core, folder = contents_project
    path = folder / "Deleted.prefab"
    make_asset(scene, path)
    root = editor.load_prefab_contents(path)
    try:
        core.project_assets.delete([str(path)])
        with pytest.raises(FileNotFoundError):
            editor.save_as_prefab_asset(root, path)
        assert not path.exists()
    finally:
        editor.unload_prefab_contents(root)
