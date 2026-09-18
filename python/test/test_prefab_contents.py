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
    previous_project = project_context.get_project_root()
    project_context.set_project_root(database.project_root)
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
        project_context.set_project_root(previous_project)


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


@pytest.mark.parametrize("direction", ["undo", "redo"])
def test_prefab_history_never_overwrites_an_external_source_edit(contents_project, scene, direction):
    from Infernux.engine.prefab_overrides import build_prefab_asset_edit_command

    _, folder = contents_project
    path = folder / "ExternalEdit.prefab"
    make_asset(scene, path)
    updated = json.loads(path.read_text(encoding="utf8"))
    updated["root_object"]["name"] = "Our save"
    command = build_prefab_asset_edit_command(str(path), updated, AssetManager.require_asset_database())
    command.execute()
    if direction == "redo":
        command.undo()
    outside = json.loads(path.read_text(encoding="utf8"))
    outside["root_object"]["name"] = "External author's new value"
    path.write_text(json.dumps(outside), encoding="utf8")
    external_bytes = path.read_bytes()
    scene_before = scene.serialize_document()
    with pytest.raises(RuntimeError, match="outside this history"):
        getattr(command, direction)()
    assert path.read_bytes() == external_bytes
    assert scene.serialize_document() == scene_before


@pytest.mark.parametrize("replace", [False, True])
def test_prefab_history_does_not_recreate_deleted_or_replace_new_asset(contents_project, scene, replace):
    from Infernux.engine.prefab_overrides import build_prefab_asset_edit_command

    core, folder = contents_project
    path = folder / "Replaced.prefab"
    make_asset(scene, path)
    database = AssetManager.require_asset_database()
    original_guid = database.get_guid_from_path(str(path))
    updated = json.loads(path.read_text(encoding="utf8"))
    updated["root_object"]["name"] = "Our save"
    command = build_prefab_asset_edit_command(str(path), updated, database)
    command.execute()
    core.project_assets.delete([str(path)])
    if replace:
        editor.save_as_prefab_asset(scene.create_game_object("A different asset"), path)
        assert database.get_guid_from_path(str(path)) != original_guid
        replacement = path.read_bytes()
    with pytest.raises(RuntimeError, match="identity changed"):
        command.undo()
    assert path.exists() is replace
    if replace:
        assert path.read_bytes() == replacement


def test_prefab_history_allows_formatting_only_source_changes(contents_project, scene):
    from Infernux.engine.prefab_overrides import build_prefab_asset_edit_command

    _, folder = contents_project
    path = folder / "Formatting.prefab"
    make_asset(scene, path)
    before = json.loads(path.read_text(encoding="utf8"))
    updated = json.loads(path.read_text(encoding="utf8"))
    updated["root_object"]["name"] = "Our save"
    command = build_prefab_asset_edit_command(str(path), updated, AssetManager.require_asset_database())
    command.execute()
    path.write_text(json.dumps(updated, sort_keys=True, indent=4), encoding="utf8")
    command.undo()
    assert json.loads(path.read_text(encoding="utf8")) == before
    command.redo()
    assert json.loads(path.read_text(encoding="utf8")) == updated


def test_missing_prefab_instance_world_is_resolved_before_source_write(contents_project, scene, monkeypatch):
    from Infernux.engine import prefab_overrides as overrides
    from Infernux.engine.prefab_manager import instantiate_prefab

    _, folder = contents_project
    path = folder / "ClosedWorld.prefab"
    make_asset(scene, path)
    database = AssetManager.require_asset_database()
    manager = SceneManager.instance()
    other = manager.create_scene("Unregistered closed world")
    instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                       scene=other, asset_database=database)
    updated = json.loads(path.read_text(encoding="utf8"))
    updated["root_object"]["name"] = "Our save"
    command = overrides.build_prefab_asset_edit_command(str(path), updated, database)
    try:
        command.execute()
    finally:
        manager.unload_scene(other)
    source_bytes = path.read_bytes()
    monkeypatch.setattr(overrides, "_write_prefab_apply_document",
                        lambda *args: pytest.fail("Unavailable world must be detected before writing"))
    with pytest.raises(RuntimeError, match="world .* unavailable"):
        command.undo()
    assert path.read_bytes() == source_bytes


@pytest.mark.parametrize("closed_count", [1, 2])
@pytest.mark.parametrize("interrupt_publish", [False, True])
def test_prefab_history_restores_closed_scene_owners_additively(
    contents_project, scene, monkeypatch, closed_count, interrupt_publish,
):
    from Infernux.engine.prefab_manager import instantiate_prefab
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import DocumentRegistry

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    manager = SceneManager.instance()
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    files = SceneFileManager()
    files.set_asset_database(database)
    registry = DocumentRegistry.instance()
    path = folder / "ClosedScenes.prefab"
    make_asset(scene, path)
    source_before = path.read_bytes()
    guid = database.get_guid_from_path(str(path))
    owners = []
    root = None
    active = manager.get_active_scene()
    try:
        for index in range(2):
            world = manager.create_scene(f"Owner {index}")
            doc_id = files.register_loaded_scene(world, str(folder / f"Owner{index}.scene"))
            obj = instantiate_prefab(file_path=str(path), guid=guid, scene=world, asset_database=database)
            obj.get_child(0).name = f"Private collider {index}"
            owners.append((doc_id, obj.id, obj.serialize_document(), world.world_id))
        active_before = active.serialize_document()
        root = editor.load_prefab_contents(path)
        root.name = "Updated source"
        root.scene.create_game_object("New child").set_parent(root)
        editor.save_as_prefab_asset(root, path)
        editor.unload_prefab_contents(root)
        root = None
        source_after = path.read_bytes()
        after = [files.scene_for_document(doc_id).find_by_id(obj_id).serialize_document()
                 for doc_id, obj_id, _, _ in owners]

        def close_owners():
            for doc_id, _, _, _ in owners[:closed_count]:
                world = files.scene_for_document(doc_id)
                assert files.unregister_loaded_scene(world)
                world_id = world.world_id
                manager.unload_scene(world)


                assert manager.get_scene_by_world_id(world_id) is None

        close_owners()
        if interrupt_publish:
            from Infernux.engine import prefab_overrides
            publish = prefab_overrides._publish_applied_prefab
            calls = []

            def interrupted_publish(prepared):
                calls.append(len(prepared))
                if len(calls) == 1:
                    assert publish(prepared[:1])
                    for _, _, plan in prepared[1:]:
                        plan.discard()
                    return False
                return publish(prepared)

            with monkeypatch.context() as patch:
                patch.setattr(prefab_overrides, "_publish_applied_prefab", interrupted_publish)
                editor.undo(defer=False)
            assert len(calls) == 2
            assert path.read_bytes() == source_after
            for (doc_id, obj_id, _, _), expected in zip(owners, after):
                assert files.scene_for_document(doc_id).find_by_id(obj_id).serialize_document() == expected
                assert registry.require(doc_id).is_dirty
            assert UndoManager.instance().undo_description == "Save Prefab Asset"
        editor.undo(defer=False)
        assert path.read_bytes() == source_before
        for doc_id, obj_id, before, old_world in owners:
            world = files.scene_for_document(doc_id)
            assert world is not None
            assert world.find_by_id(obj_id).serialize_document() == before
            assert not registry.require(doc_id).is_dirty
            if doc_id in [item[0] for item in owners[:closed_count]]:
                assert world.world_id != old_world
        assert manager.get_active_scene() is active
        assert active.serialize_document() == active_before
        # Redo also restores closed owners, not just worlds reopened by Undo.
        close_owners()
        editor.redo(defer=False)
        assert path.read_bytes() == source_after
        for (doc_id, obj_id, _, _), expected in zip(owners, after):
            assert files.scene_for_document(doc_id).find_by_id(obj_id).serialize_document() == expected
            assert registry.require(doc_id).is_dirty
        assert manager.get_active_scene() is active
        assert active.serialize_document() == active_before
    finally:
        if root is not None:
            editor.unload_prefab_contents(root)
        for doc_id, _, _, _ in owners:
            world = files.scene_for_document(doc_id)
            if world is not None:
                files.unregister_loaded_scene(world)
                manager.unload_scene(world)


@pytest.mark.parametrize("apply_instance", [False, True])
def test_variant_source_update_and_all_dependents_share_one_history(contents_project, scene, apply_instance):
    from Infernux.engine.prefab_manager import instantiate_prefab
    from Infernux.engine.prefab_overrides import build_prefab_apply_command

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    base_path = folder / "Base.prefab"
    first_path = folder / "Variant.prefab"
    second_path = folder / "Derived.prefab"
    make_asset(scene, base_path)
    loaded = editor.load_prefab_contents(base_path)
    try:
        loaded.name = "Variant name"
        editor.save_as_prefab_asset(loaded, first_path)
    finally:
        editor.unload_prefab_contents(loaded)
    loaded = editor.load_prefab_contents(first_path)
    try:
        loaded.get_child(0).name = "Derived collider"
        editor.save_as_prefab_asset(loaded, second_path)
    finally:
        editor.unload_prefab_contents(loaded)
    paths = [base_path, first_path, second_path]
    instances = [instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                                    scene=scene, asset_database=database) for path in paths]
    instances[-1].tag = "Private instance"
    saved_before = [path.read_bytes() for path in paths]
    before = [instance.serialize_document() for instance in instances]
    if apply_instance:
        source = instances[0]
        source.get_child(0).layer = 5
        scene.create_game_object("Added by base").set_parent(source)
        before[0] = source.serialize_document()
        assert UndoManager.instance().execute(build_prefab_apply_command(source, str(base_path), database))
    else:
        loaded = editor.load_prefab_contents(base_path)
        try:
            loaded.get_child(0).layer = 5
            loaded.scene.create_game_object("Added by base").set_parent(loaded)
            editor.save_as_prefab_asset(loaded, base_path)
        finally:
            editor.unload_prefab_contents(loaded)
    for instance in instances:
        assert instance.get_child(0).layer == 5
        assert sum(child.name == "Added by base" for child in instance.get_children()) == 1
    assert instances[1].name == "Variant name (Clone)"
    assert instances[2].get_child(0).name == "Derived collider"
    assert instances[-1].tag == "Private instance"
    assert json.loads(first_path.read_text(encoding="utf8"))["variant"]["guid"] == database.get_guid_from_path(str(base_path))
    saved_after = [path.read_bytes() for path in paths]
    after = [instance.serialize_document() for instance in instances]
    editor.undo(defer=False)
    assert [path.read_bytes() for path in paths] == saved_before
    assert [instance.serialize_document() for instance in instances] == before
    editor.redo(defer=False)
    assert [path.read_bytes() for path in paths] == saved_after
    assert [instance.serialize_document() for instance in instances] == after


@pytest.mark.parametrize("failure", ["dependent_write", "external_edit"])
def test_variant_batch_failure_does_not_leave_partial_history(contents_project, scene, monkeypatch, failure):
    from Infernux.engine import prefab_overrides
    from Infernux.engine.prefab_manager import instantiate_prefab

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    base_path, variant_path = folder / "Base.prefab", folder / "Variant.prefab"
    make_asset(scene, base_path)
    root = editor.load_prefab_contents(base_path)
    try:
        root.name = "Private variant"
        editor.save_as_prefab_asset(root, variant_path)
    finally:
        editor.unload_prefab_contents(root)
    paths = [base_path, variant_path]
    instances = [instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                                    scene=scene, asset_database=database) for path in paths]
    before_files = [path.read_bytes() for path in paths]
    before_instances = [obj.serialize_document() for obj in instances]
    history = UndoManager.instance().undo_description
    root = editor.load_prefab_contents(base_path)
    try:
        root.get_child(0).layer = 6
        if failure == "dependent_write":
            original_write = prefab_overrides._write_prefab_apply_document
            interrupted = []

            def write(path, document, asset_database=None):
                if Path(path) == variant_path and not interrupted:
                    interrupted.append(True)
                    raise OSError("Injected dependent write failure")
                return original_write(path, document, asset_database)

            with monkeypatch.context() as patch:
                patch.setattr(prefab_overrides, "_write_prefab_apply_document", write)
                with pytest.raises(RuntimeError):
                    editor.save_as_prefab_asset(root, base_path)
            assert interrupted
            assert [path.read_bytes() for path in paths] == before_files
            assert [obj.serialize_document() for obj in instances] == before_instances
            assert UndoManager.instance().undo_description == history
        else:
            editor.save_as_prefab_asset(root, base_path)
            edited = json.loads(variant_path.read_text(encoding="utf8"))
            edited["root_object"]["name"] = "External author"
            variant_path.write_text(json.dumps(edited, indent=2), encoding="utf8")
            current_files = [path.read_bytes() for path in paths]
            current_instances = [obj.serialize_document() for obj in instances]
            editor.undo(defer=False)
            assert [path.read_bytes() for path in paths] == current_files
            assert [obj.serialize_document() for obj in instances] == current_instances
            assert UndoManager.instance().undo_description == "Save Prefab Asset"
    finally:
        editor.unload_prefab_contents(root)


def test_variant_resave_preserves_equal_base_override_intent(contents_project, scene):
    _, folder = contents_project
    base_path, variant_path = folder / "Base.prefab", folder / "Variant.prefab"
    make_asset(scene, base_path)

    def save_layer(source, destination, layer):
        root = editor.load_prefab_contents(source)
        try:
            root.get_child(0).layer = layer
            editor.save_as_prefab_asset(root, destination)
        finally:
            editor.unload_prefab_contents(root)

    save_layer(base_path, variant_path, 5)
    save_layer(base_path, base_path, 5)
    root = editor.load_prefab_contents(variant_path)
    try:
        root.name = "Another edit"
        editor.save_as_prefab_asset(root, variant_path)
    finally:
        editor.unload_prefab_contents(root)
    save_layer(base_path, base_path, 7)
    result = json.loads(variant_path.read_text(encoding="utf8"))
    assert result["root_object"]["children"][0]["layer"] == 5
    assert result["root_object"]["name"] == "Another edit"
    assert result["variant"]["baseline"]["root_object"]["children"][0]["layer"] == 7


def test_external_base_change_invalidates_variant_template_without_rewriting_sources(contents_project, scene):
    from Infernux.engine.prefab_manager import instantiate_prefab, _read_resolved_prefab_document

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    base_path, variant_path = folder / "Base.prefab", folder / "Variant.prefab"
    make_asset(scene, base_path)
    root = editor.load_prefab_contents(base_path)
    try:
        root.name = "Private variant"
        editor.save_as_prefab_asset(root, variant_path)
    finally:
        editor.unload_prefab_contents(root)
    guid = database.get_guid_from_path(str(variant_path))
    first = instantiate_prefab(file_path=str(variant_path), guid=guid, scene=scene, asset_database=database)
    assert first.get_child(0).layer == 0
    stale = editor.load_prefab_contents(variant_path)
    saved_variant = variant_path.read_bytes()
    base = json.loads(base_path.read_text(encoding="utf8"))
    base["root_object"]["children"][0]["layer"] = 7
    base_path.write_text(json.dumps(base), encoding="utf8")
    saved_base = base_path.read_bytes()
    try:
        second = instantiate_prefab(file_path=str(variant_path), guid=guid, scene=scene, asset_database=database)
        assert second.get_child(0).layer == 7
        assert second.name == "Private variant (Clone)"
        assert first.get_child(0).layer == 0  # Loading is not a live-world mutation.
        assert _read_resolved_prefab_document(str(variant_path), database)["root_object"]["children"][0]["layer"] == 7
        with pytest.raises(RuntimeError, match="base changed"):
            editor.save_as_prefab_asset(stale, variant_path)
        assert variant_path.read_bytes() == saved_variant
        assert base_path.read_bytes() == saved_base
    finally:
        editor.unload_prefab_contents(stale)
    current = editor.load_prefab_contents(variant_path)
    try:
        assert current.get_child(0).layer == 7
        editor.save_as_prefab_asset(current, variant_path)
        assert first.get_child(0).layer == 7
    finally:
        editor.unload_prefab_contents(current)


def test_variant_resolver_rejects_missing_and_cyclic_bases(contents_project, scene):
    from Infernux.engine.prefab_manager import _read_resolved_prefab_document, PrefabDocumentError

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    base_path, variant_path = folder / "Base.prefab", folder / "Variant.prefab"
    make_asset(scene, base_path)
    root = editor.load_prefab_contents(base_path)
    try:
        editor.save_as_prefab_asset(root, variant_path)
    finally:
        editor.unload_prefab_contents(root)
    variant = json.loads(variant_path.read_text(encoding="utf8"))
    for guid, message in (("unavailable-source-guid", "unavailable"),
                          (database.get_guid_from_path(str(variant_path)), "cycle")):
        variant["variant"]["guid"] = guid
        variant_path.write_text(json.dumps(variant), encoding="utf8")
        before = variant_path.read_bytes()
        with pytest.raises(PrefabDocumentError, match=message):
            _read_resolved_prefab_document(str(variant_path), database)
        assert variant_path.read_bytes() == before


def test_prefab_importer_owns_base_edges_and_save_does_not_scan_unrelated_assets(contents_project, scene):
    from Infernux.lib import AssetDependencyGraph
    from Infernux.engine.prefab_variant import dependent_variant_guids

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    base_path, variant_path, other_path = [folder / name for name in ("Base.prefab", "Variant.prefab", "Other.prefab")]
    make_asset(scene, base_path)
    make_asset(scene, other_path)
    loaded = editor.load_prefab_contents(base_path)
    try:
        editor.save_as_prefab_asset(loaded, variant_path)
    finally:
        editor.unload_prefab_contents(loaded)
    base_guid = database.get_guid_from_path(str(base_path))
    variant_guid = database.get_guid_from_path(str(variant_path))
    other_guid = database.get_guid_from_path(str(other_path))
    graph = AssetDependencyGraph.instance()
    assert graph.get_dependencies(variant_guid) == {base_guid}
    assert dependent_variant_guids(base_guid, database) == (variant_guid,)
    other_original = other_path.read_bytes()
    other_path.write_text("unrelated invalid author document", encoding="utf8")
    try:
        loaded = editor.load_prefab_contents(base_path)
        try:
            loaded.get_child(0).layer = 4
            editor.save_as_prefab_asset(loaded, base_path)
        finally:
            editor.unload_prefab_contents(loaded)
    finally:
        other_path.write_bytes(other_original)
    document = json.loads(variant_path.read_text(encoding="utf8"))
    document["variant"]["guid"] = other_guid
    variant_path.write_text(json.dumps(document), encoding="utf8")
    assert AssetManager.reimport_asset(str(variant_path), database=database)
    assert graph.get_dependencies(variant_guid) == {other_guid}
    assert dependent_variant_guids(base_guid, database) == ()
    del document["variant"]
    variant_path.write_text(json.dumps(document), encoding="utf8")
    assert AssetManager.reimport_asset(str(variant_path), database=database)
    assert graph.get_dependencies(variant_guid) == set()


@pytest.mark.parametrize("interrupt_publish", [False, True])
def test_external_prefab_event_updates_loaded_variants_without_writing_their_sources(
        contents_project, scene, engine, monkeypatch, interrupt_publish):
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.resources_manager import ResourceChangeHandler
    from Infernux.engine.prefab_manager import instantiate_prefab
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.engine import prefab_overrides

    _, folder = contents_project
    database = AssetManager.require_asset_database()
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    files = SceneFileManager()
    files._asset_database = database
    registry = DocumentRegistry.instance()
    base_path, variant_path = folder / "Base.prefab", folder / "Variant.prefab"
    make_asset(scene, base_path)
    loaded = editor.load_prefab_contents(base_path)
    try:
        editor.save_as_prefab_asset(loaded, variant_path)
    finally:
        editor.unload_prefab_contents(loaded)
    guid = database.get_guid_from_path(str(variant_path))
    manager = SceneManager.instance()
    extra = manager.create_scene("External update second scene")
    files.register_loaded_scene(extra, "")
    try:
        instances = [instantiate_prefab(file_path=str(variant_path), guid=guid, scene=world,
                                        asset_database=database) for world in (scene, extra)]
        instances[-1].get_child(0).name = "Local name"
        documents = [obj.serialize_document() for obj in instances]
        owners = [files.document_id_for_scene(world) for world in (scene, extra)]
        revisions = [registry.require(owner).revision for owner in owners]
        variant_original = variant_path.read_bytes()
        base = json.loads(base_path.read_text(encoding="utf8"))
        base["root_object"]["children"][0]["layer"] = 9
        base_path.write_text(json.dumps(base), encoding="utf8")
        handler = ResourceChangeHandler(engine, project_path=database.project_root)
        if interrupt_publish:
            publish = prefab_overrides._publish_applied_prefab
            attempts = []

            def interrupted(prepared):
                if not attempts:
                    attempts.append(True)
                    assert publish(prepared[:1])
                    for _, _, plan in prepared[1:]:
                        plan.discard()
                    return False
                return publish(prepared)

            with monkeypatch.context() as patch:
                patch.setattr(prefab_overrides, "_publish_applied_prefab", interrupted)
                with pytest.raises(RuntimeError, match="synchronize"):
                    handler._commit_modified(str(base_path))
            assert [obj.serialize_document() for obj in instances] == documents
            assert [registry.require(owner).revision for owner in owners] == revisions
            files.sync_prefab_dependents(database.get_guid_from_path(str(base_path)))
        else:
            handler._commit_modified(str(base_path))
        assert [obj.get_child(0).layer for obj in instances] == [9, 9]
        assert instances[-1].get_child(0).name == "Local name"
        assert [registry.require(owner).revision for owner in owners] == [value + 1 for value in revisions]
        assert variant_path.read_bytes() == variant_original
    finally:
        files.unregister_loaded_scene(extra)
        manager.unload_scene(extra)


@pytest.mark.parametrize("interrupt_write", [False, True])
def test_prefab_mode_save_updates_variant_assets_and_suspended_and_additive_worlds(
        contents_project, scene, monkeypatch, interrupt_write):
    from Infernux.engine import prefab_overrides
    from Infernux.engine.prefab_manager import instantiate_prefab
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import DocumentRegistry, SelectionDomain

    core, folder = contents_project
    database = AssetManager.require_asset_database()
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    monkeypatch.setattr(SceneFileManager, "_instance", None)
    files = SceneFileManager()
    files._asset_database = database
    registry = DocumentRegistry.instance()
    original_owner = files.document_id
    paths = [folder / name for name in ("Base.prefab", "Variant.prefab", "Derived.prefab")]
    original = scene.create_game_object("Base")
    child = scene.create_game_object("Collider")
    child.set_parent(original)
    child.add_component("BoxCollider")
    editor.save_as_prefab_asset(original, paths[0])
    for index in (1, 2):
        root = editor.load_prefab_contents(paths[index - 1])
        try:
            root.name = f"Variant {index}"
            editor.save_as_prefab_asset(root, paths[index])
        finally:
            editor.unload_prefab_contents(root)
    manager = SceneManager.instance()
    other = manager.create_scene("Additional variant world")
    other_owner = files.register_loaded_scene(other, "")
    try:
        instances = [instantiate_prefab(file_path=str(path), guid=database.get_guid_from_path(str(path)),
                                        scene=scene, asset_database=database) for path in paths]
        extra = instantiate_prefab(file_path=str(paths[-1]), guid=database.get_guid_from_path(str(paths[-1])),
                                   scene=other, asset_database=database)
        instances[-1].get_child(0).name = "Private child"
        extra.get_child(0).name = "Other private child"
        instance_ids = [obj.id for obj in instances]
        before_files = [path.read_bytes() for path in paths]
        before_extra = extra.serialize_document()
        assert files.open_prefab_mode(str(paths[0]))
        root = manager.get_active_scene().get_root_objects()[0]
        root.get_child(0).layer = 5
        added = root.scene.create_game_object("Saved addition")
        added.set_parent(root)
        registry.mark_changed(files.document_id)
        if interrupt_write:
            original_write = prefab_overrides._write_prefab_apply_document
            attempts = []

            def write(path, document, asset_database=None):
                if Path(path) == paths[1] and not attempts:
                    attempts.append(path)
                    raise OSError("Injected Variant save failure")
                original_write(path, document, asset_database)

            with monkeypatch.context() as patch:
                patch.setattr(prefab_overrides, "_write_prefab_apply_document", write)
                assert not files._save_prefab()
            assert attempts
            assert [path.read_bytes() for path in paths] == before_files
            assert extra.serialize_document() == before_extra
            assert registry.require(files.document_id).is_dirty
            assert not registry.require(other_owner).is_dirty
        assert files._save_prefab()
        assert extra.get_child(0).layer == 5
        assert extra.get_child(0).name == "Other private child"
        assert sum(obj.name == "Saved addition" for obj in extra.get_children()) == 1
        assert registry.require(other_owner).is_dirty
        assert not registry.require(files.document_id).is_dirty
        after_files = [path.read_bytes() for path in paths]
        revision = registry.require(other_owner).revision
        assert files._save_prefab()
        assert [path.read_bytes() for path in paths] == after_files
        assert registry.require(other_owner).revision == revision
        assert files._do_exit_prefab_mode()
        assert manager.get_active_scene() is scene
        assert files.document_id == original_owner
        assert registry.require(original_owner).is_dirty
        for object_id in instance_ids:
            restored = scene.find_by_id(object_id)
            assert restored.get_child(0).layer == 5
            assert sum(obj.name == "Saved addition" for obj in restored.get_children()) == 1
        assert scene.find_by_id(instance_ids[-1]).get_child(0).name == "Private child"
        # Saving a Variant in its own Prefab Mode must preserve its base and
        # propagate new overrides through descendants, not back to the base.
        assert files.open_prefab_mode(str(paths[1]))
        manager.get_active_scene().get_root_objects()[0].get_child(0).layer = 7
        registry.mark_changed(files.document_id)
        assert files._save_prefab()
        assert files._do_exit_prefab_mode()
        assert json.loads(paths[1].read_text(encoding="utf8"))["variant"]["guid"] == database.get_guid_from_path(str(paths[0]))
        assert scene.find_by_id(instance_ids[0]).get_child(0).layer == 5
        assert scene.find_by_id(instance_ids[1]).get_child(0).layer == 7
        assert scene.find_by_id(instance_ids[2]).get_child(0).layer == 7
        assert extra.get_child(0).layer == 7
    finally:
        if files.is_prefab_mode:
            files._do_exit_prefab_mode()
        files.unregister_loaded_scene(other)
        manager.unload_scene(other)
