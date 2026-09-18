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


@pytest.mark.parametrize("save_offline", [True, False])
def test_source_edit_updates_all_worlds_nested_instances_and_history(contents_project, scene, monkeypatch, save_offline):
    from types import SimpleNamespace
    from Infernux.engine.prefab_manager import instantiate_prefab
    from Infernux.engine.prefab_overrides import build_prefab_apply_command
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import DocumentRegistry, DocumentKind

    core, folder = contents_project
    database = AssetManager.require_asset_database()
    path = folder / "Shared.prefab"
    make_asset(scene, path)
    guid = database.get_guid_from_path(str(path))
    manager = SceneManager.instance()
    second_scene = manager.create_scene("Prefab second world")
    unrelated = manager.create_scene("No linked Prefab")
    root = None
    try:
        def instantiate(destination, parent=None, source=path, source_guid=guid):
            return instantiate_prefab(file_path=str(source), guid=source_guid, scene=destination,
                                      parent=parent, asset_database=database)

        first = instantiate(scene)
        outer = second_scene.create_game_object("Outer")
        nested = instantiate(second_scene, outer)
        outer_path = folder / "Outer.prefab"
        editor.save_as_prefab_asset(outer, outer_path)
        repeated = instantiate(scene, source=outer_path,
                               source_guid=database.get_guid_from_path(str(outer_path))).get_child(0)
        outer_bytes = outer_path.read_bytes()
        nested.get_child(0).name = "Private collider"
        nested.transform.local_position = Vector3(4, 5, 6)
        external = second_scene.create_game_object("External reference")
        external_collider = external.add_component("BoxCollider")
        nested.get_py_component(ContentsProbe).target = ComponentRef(external_collider)
        roots = (first, nested, repeated)
        child_ids = [obj.get_child(0).id for obj in roots]
        before = [obj.serialize_document() for obj in roots]
        source_before = path.read_bytes()

        # Real registry revisions, with only the editor's world/document lookup supplied.
        registry = DocumentRegistry.instance()
        documents = {int(world.world_id): registry.create(DocumentKind.SCENE, world.name)
                     for world in (scene, second_scene, unrelated)}
        monkeypatch.setattr(SceneFileManager, "_instance", SimpleNamespace(
            is_prefab_mode=False, document_id=documents[int(scene.world_id)].document_id,
            document_id_for_scene=lambda world: documents[int(getattr(world, "world_id", world))].document_id,
        ))
        if save_offline:
            root = editor.load_prefab_contents(path)
            root.get_child(0).name = "Updated collider"
            root.scene.create_game_object("New source child").set_parent(root)
            editor.save_as_prefab_asset(root, path)
            # History must never depend on the lifetime of the preview world.
            editor.unload_prefab_contents(root)
            root = None
        else:
            first.get_child(0).name = "Updated collider"
            scene.create_game_object("New source child").set_parent(first)
            before[0] = first.serialize_document()
            assert UndoManager.instance().execute(build_prefab_apply_command(first, str(path), database))

        def verify_updated():
            assert [obj.get_child(0).id for obj in roots] == child_ids
            assert [obj.get_child(0).name for obj in roots] == ["Updated collider", "Private collider", "Updated collider"]
            assert all([c.name for c in obj.get_children()].count("New source child") == 1 for obj in roots)
            assert nested.transform.local_position.y == 5
            assert nested.get_py_component(ContentsProbe).target.component_id == external_collider.component_id
            for obj in (first, repeated):
                assert obj.get_py_component(ContentsProbe).target.game_object.id == obj.get_child(0).id
            assert documents[int(scene.world_id)].is_dirty
            assert documents[int(second_scene.world_id)].is_dirty
            assert not documents[int(unrelated.world_id)].is_dirty
            assert outer_path.read_bytes() == outer_bytes

        verify_updated()
        after = [obj.serialize_document() for obj in roots]
        editor.undo(defer=False)
        assert [obj.serialize_document() for obj in roots] == before
        assert path.read_bytes() == source_before
        assert all(not document.is_dirty for document in documents.values())
        editor.redo(defer=False)
        assert [obj.serialize_document() for obj in roots] == after
        verify_updated()
    finally:
        if root is not None:
            editor.unload_prefab_contents(root)
        manager.unload_scene(unrelated)
        manager.unload_scene(second_scene)


def test_offline_save_rejects_other_world_parent_cycle_before_publication(contents_project, scene):
    from Infernux.engine.prefab_manager import instantiate_prefab

    _, folder = contents_project
    path = folder / "Cycle.prefab"
    source = scene.create_game_object("Source")
    for name in ("Left", "Right"):
        scene.create_game_object(name).set_parent(source)
    editor.save_as_prefab_asset(source, path)
    database = AssetManager.require_asset_database()
    manager = SceneManager.instance()
    other = manager.create_scene("Conflicting instance")
    root = editor.load_prefab_contents(path)
    try:
        peer = instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                                  scene=other, asset_database=database)
        left, right = peer.get_children()
        right.set_parent(left)
        left, right = root.get_children()
        left.set_parent(right)
        before = peer.serialize_document()
        source_before = path.read_bytes()
        with pytest.raises(RuntimeError, match="command was rejected"):
            editor.save_as_prefab_asset(root, path)
        assert peer.serialize_document() == before
        assert path.read_bytes() == source_before
    finally:
        editor.unload_prefab_contents(root)
        manager.unload_scene(other)
