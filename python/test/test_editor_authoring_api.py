"""Public author tools share the Editor's native objects and action journal."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux import editor
from Infernux.engine.interaction import EditorInteractionCore, SelectionDomain
from Infernux.engine.undo import UndoManager
from Infernux.engine.play_mode import PlayModeManager
from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
from Infernux.lib import Vector3
from Infernux.components import InxComponent, FieldType, serialized_field, list_field
from Infernux.components.builtin import BoxCollider
from Infernux.core import AssetManager, DataAsset


class AuthoringLevelData(DataAsset):
    __serialized_type_id__ = "tests.editor.authoring_level"

    title: str = serialized_field(default="新关卡")
    difficulty: int = serialized_field(default=1)


class AuthoringCatalogData(DataAsset):
    __serialized_type_id__ = "tests.editor.authoring_catalog"
    levels = list_field(element_type=FieldType.ASSET, asset_type="DataAsset")
    locked: int = serialized_field(default=1, readonly=True)


def test_editor_attribute_is_available_in_a_fresh_process():
    import subprocess
    import sys
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys; import infernux as inx; "
        "assert 'Infernux.editor' not in sys.modules; "
        "assert callable(inx.editor.create_game_object); "
        "assert inx.editor is sys.modules['Infernux.editor']",
    ], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def authoring(scene, monkeypatch):
    monkeypatch.setattr(UndoManager, "_instance", None)
    monkeypatch.setattr(HierarchyCreationService, "_instance", None)
    core = EditorInteractionCore()
    core.panels.register_selection_authority("hierarchy", (SelectionDomain.SCENE_OBJECT,))
    UndoManager(core.action_journal)
    try:
        yield core
    finally:
        core.shutdown()


def test_public_create_group_undo_redo_preserves_configuration(authoring, scene):
    with editor.edit_scene("Create obstacle"):
        root = editor.create_game_object("机关")
        child = editor.create_game_object(
            "碰撞体", parent=root,
            configure=lambda obj: setattr(obj.transform, "local_position", Vector3(1, 2, 3)),
        )
    root_id, child_id = root.id, child.id
    assert UndoManager.instance().undo_description == "Create obstacle"
    assert child.get_parent().id == root_id
    editor.undo(defer=False)
    assert scene.find_by_id(root_id) is None
    assert scene.find_by_id(child_id) is None
    editor.redo(defer=False)
    assert scene.find_by_id(child_id).get_parent().id == root_id
    assert scene.find_by_id(child_id).transform.local_position.x == 1


def test_public_defer_uses_single_editor_task_and_preserves_authoring_history(authoring, scene, monkeypatch):
    from Infernux.engine.deferred_task import DeferredTaskRunner

    monkeypatch.setattr(DeferredTaskRunner, "_instance", None)
    runner = DeferredTaskRunner.instance()
    created = []
    try:
        with pytest.raises(TypeError, match="callable"):
            editor.defer(None)
        assert editor.defer(lambda: created.append(editor.create_game_object("Delayed authoring")))
        assert created == []
        assert not editor.defer(lambda: pytest.fail("Busy submission must not run"))
        runner.tick()
        assert len(created) == 1 and not runner.is_busy
        identity = created[0].id
        runner.tick()
        assert len(created) == 1
        editor.undo(defer=False)
        assert scene.find_by_id(identity) is None
        editor.redo(defer=False)
        assert scene.find_by_id(identity).name == "Delayed authoring"
    finally:
        runner.cancel()


def test_public_create_failed_initializer_leaves_no_object_or_history(authoring, scene):
    before = [obj.id for obj in scene.get_all_objects()]

    def fail(obj):
        raise ValueError("invalid author value")

    with pytest.raises(ValueError, match="invalid author value"):
        editor.create_game_object("Rejected", configure=fail)
    assert [obj.id for obj in scene.get_all_objects()] == before
    assert not UndoManager.instance().can_undo


def test_public_add_component_records_initializer_and_undo(authoring, scene):
    obj = editor.create_game_object("Collider owner")
    collider = editor.add_component(
        obj, "BoxCollider", configure=lambda item: setattr(item, "size", Vector3(2, 3, 4)),
    )
    assert collider.size.y == 3
    editor.undo(defer=False)
    assert obj.get_component(BoxCollider) is None
    editor.redo(defer=False)
    assert obj.get_component(BoxCollider).size.y == 3


def test_authoring_without_editor_does_not_create_a_session(monkeypatch):
    monkeypatch.setattr(EditorInteractionCore, "_instance", None)
    with pytest.raises(RuntimeError, match="active Editor"):
        editor.create_game_object()
    assert EditorInteractionCore.instance() is None


def test_authoring_rejects_play_mode(authoring, monkeypatch):
    monkeypatch.setattr(PlayModeManager, "_instance", SimpleNamespace(is_playing=True))
    with pytest.raises(RuntimeError, match="Play Mode"):
        editor.create_game_object()


def test_public_prefab_delegates_to_session_owner(authoring, monkeypatch, tmp_path):
    from Infernux.engine import project_context
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    obj = SimpleNamespace(id=11)
    parent = SimpleNamespace(id=7)
    calls = []
    monkeypatch.setattr(authoring.prefabs, "create_from_object", lambda *args: calls.append(args) or "created.prefab")
    monkeypatch.setattr(authoring.scene_objects, "instantiate_prefab_object", lambda *args, **kwargs: calls.append((args, kwargs)) or obj)
    assert editor.create_prefab(obj) == "created.prefab"
    assert calls[0][0] == 11
    assert calls[0][1].replace("\\", "/").endswith("/Assets")
    assert editor.instantiate_prefab("Assets/item.prefab", parent=parent) is obj
    assert calls[-1][1] == {"parent_id": 7}


def test_public_save_preserves_pending_status(authoring, monkeypatch):
    from Infernux.engine.scene_manager import SceneFileManager
    monkeypatch.setattr(SceneFileManager, "_instance", SimpleNamespace(document_id="scene-document"))
    pending = editor.DocumentActionResult(editor.DocumentActionStatus.PENDING)
    calls = []
    monkeypatch.setattr(authoring.documents, "request_save", lambda document_id: calls.append(document_id) or pending)
    assert editor.save_scene() is pending
    assert calls == ["scene-document"]


@pytest.fixture
def asset_authoring(authoring, engine, monkeypatch):
    from Infernux.engine import project_context
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(project_context, "get_project_root", lambda: database.project_root)
    authoring.project_assets.configure(database.project_root, database)
    return database


def test_public_data_asset_and_folder_share_grouped_undo(asset_authoring):
    from pathlib import Path
    source = AuthoringLevelData(title="压力板", difficulty=7)
    folder = Path(asset_authoring.assets_root) / "AuthoringDataContract"
    with editor.edit_scene("创建关卡数据"):
        assert Path(editor.create_folder(folder)).samefile(folder)
        target = Path(editor.create_data_asset(source, folder / "Level.inxdata"))
    assert not source.is_persistent
    loaded = DataAsset.load(str(target))
    assert loaded is not source
    assert (loaded.title, loaded.difficulty) == ("压力板", 7)
    guid = loaded.guid
    assert guid
    content, metadata = target.read_bytes(), Path(str(target) + ".meta").read_bytes()
    assert UndoManager.instance().undo_description == "创建关卡数据"
    editor.undo(defer=False)
    assert not folder.exists()
    editor.redo(defer=False)
    assert target.read_bytes() == content
    assert Path(str(target) + ".meta").read_bytes() == metadata
    AssetManager.invalidate(guid)
    assert DataAsset.load(str(target)).guid == guid
    editor.undo(defer=False)
    assert not folder.exists()


def test_public_data_asset_uses_published_type_after_script_refresh(asset_authoring):
    from pathlib import Path
    from Infernux.components.serializable_object import (
        _candidate_serializable_scope, _publish_serializable_types,
        _restore_serializable_types,
    )

    source = AuthoringLevelData(title="Preloaded", difficulty=7)
    candidates = {}
    with _candidate_serializable_scope(candidates, __name__):
        class PublishedLevelData(DataAsset):
            __serialized_type_id__ = AuthoringLevelData.__serialized_type_id__
            title: str = serialized_field(default="New")
            difficulty: int = serialized_field(default=1)

    before = _publish_serializable_types(candidates, {__name__})
    try:
        target = Path(asset_authoring.assets_root) / "RefreshedAuthoring.inxdata"
        editor.create_data_asset(source, target)
        loaded = DataAsset.load(str(target))
        assert type(loaded) is PublishedLevelData
        assert (loaded.title, loaded.difficulty) == ("Preloaded", 7)
        assert type(source) is AuthoringLevelData
        assert not source.is_persistent
        assert source.difficulty == 7
    finally:
        _restore_serializable_types(before)


def test_public_data_asset_never_overwrites_or_mutates_input(asset_authoring):
    from pathlib import Path
    target = Path(asset_authoring.assets_root) / "AuthoringUnique.inxdata"
    source = AuthoringLevelData(difficulty=3)
    editor.create_data_asset(source, target)
    content = target.read_bytes()
    with pytest.raises(RuntimeError, match="asset command was rejected"):
        editor.create_data_asset(AuthoringLevelData(difficulty=9), target)
    assert target.read_bytes() == content
    assert not source.is_persistent
    editor.undo(defer=False)
    assert not target.exists()
    assert not UndoManager.instance().can_undo


def test_public_data_asset_edit_uses_shared_document_and_one_undo(asset_authoring):
    from pathlib import Path
    from Infernux.engine.interaction import DocumentKind, ensure_editable_resource_document

    path = editor.create_data_asset(AuthoringLevelData(), "Assets/EditData.inxdata")
    asset = DataAsset.load(path)
    controller = ensure_editable_resource_document(
        category="data_asset", document_kind=DocumentKind.DATA_ASSET,
        file_path=path, resource=asset, guid=asset.guid, view_id="inspector.test",
    )
    UndoManager.instance().clear()
    assert editor.set_data_asset_fields(asset, title="第一关", difficulty=5)
    assert editor.load_data_asset(path) is asset
    assert editor.load_data_asset(path).title == "第一关"
    assert controller.resource is asset
    assert (asset.title, asset.difficulty) == ("第一关", 5)
    assert not editor.set_data_asset_fields(asset, title="第一关", difficulty=5)
    editor.undo(defer=False)
    assert (asset.title, asset.difficulty) == ("新关卡", 1)
    assert editor.load_data_asset(path).difficulty == 1
    assert not UndoManager.instance().can_undo
    editor.redo(defer=False)
    assert (asset.title, asset.difficulty) == ("第一关", 5)
    assert editor.save_data_asset(asset).status in (
        editor.DocumentActionStatus.APPLIED, editor.DocumentActionStatus.NO_OP,
    )
    import json
    assert json.loads(Path(path).read_text(encoding="utf-8"))["fields"]["difficulty"] == 5


def test_public_data_asset_edit_rejects_entire_invalid_batch(asset_authoring):
    path = editor.create_data_asset(AuthoringLevelData(), "Assets/InvalidEditData.inxdata")
    asset = DataAsset.load(path)
    UndoManager.instance().clear()
    with pytest.raises((KeyError, ValueError)):
        editor.set_data_asset_fields(asset, title="Must not apply", unknown_field=3)
    assert asset.title == "新关卡"
    assert not UndoManager.instance().can_undo
    with pytest.raises(ValueError, match="persistent"):
        editor.set_data_asset_fields(AuthoringLevelData(), title="Transient")


def test_public_data_asset_edit_preserves_newer_document_for_stale_handle(asset_authoring):
    from Infernux.engine.interaction import DocumentKind, ensure_editable_resource_document

    path = editor.create_data_asset(AuthoringLevelData(), "Assets/StaleEditData.inxdata")
    stale = DataAsset.load(path)
    current = stale.instantiate()
    current._bind_asset(stale.file_path, stale.guid)
    current.title = "Newer Inspector value"
    controller = ensure_editable_resource_document(
        category="data_asset", document_kind=DocumentKind.DATA_ASSET,
        file_path=path, resource=current, guid=current.guid, view_id="inspector.test",
    )
    assert editor.set_data_asset_fields(stale, difficulty=9)
    assert controller.resource is current
    assert current.title == "Newer Inspector value"
    assert current.difficulty == 9
    assert stale.difficulty == 1
    assert editor.load_data_asset(path) is current


def test_public_data_asset_save_preserves_pending_status(asset_authoring, authoring, monkeypatch):
    path = editor.create_data_asset(AuthoringLevelData(), "Assets/PendingData.inxdata")
    asset = DataAsset.load(path)
    pending = editor.DocumentActionResult(editor.DocumentActionStatus.PENDING)
    monkeypatch.setattr(authoring.documents, "request_save", lambda _: pending)
    assert editor.save_data_asset(asset) is pending


def test_public_data_asset_reference_list_and_readonly_contract(asset_authoring):
    level = DataAsset.load(editor.create_data_asset(AuthoringLevelData(), "Assets/ListLevel.inxdata"))
    catalog = DataAsset.load(editor.create_data_asset(AuthoringCatalogData(), "Assets/EditCatalog.inxdata"))
    UndoManager.instance().clear()
    with pytest.raises(ValueError, match="read-only"):
        editor.set_data_asset_fields(catalog, levels=[level], locked=2)
    assert list(catalog.levels) == []
    assert not UndoManager.instance().can_undo
    assert editor.set_data_asset_fields(catalog, levels=[level])
    assert catalog.serialize_document()["fields"]["levels"][0]["guid"] == level.guid
    editor.undo(defer=False)
    assert list(catalog.levels) == []
    editor.redo(defer=False)
    assert catalog.serialize_document()["fields"]["levels"][0]["guid"] == level.guid


@pytest.mark.parametrize("value,path,error", [
    (object(), "Assets/Invalid.inxdata", TypeError),
    (AuthoringLevelData(), "Assets/Invalid.json", ValueError),
    (AuthoringLevelData(), "Assets/MissingParent/Invalid.inxdata", RuntimeError),
])
def test_public_invalid_data_asset_does_not_enter_history(asset_authoring, value, path, error):
    with pytest.raises(error):
        editor.create_data_asset(value, path)
    assert not UndoManager.instance().can_undo


def test_public_build_scene_list_shares_settings_and_undo(authoring, monkeypatch, tmp_path):
    from Infernux.engine import project_context
    from Infernux.engine.interaction.project_settings import ensure_project_settings_document
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    assets = tmp_path / "Assets"
    assets.mkdir()
    for name in ("Start.scene", "Level.scene"):
        (assets / name).touch()
    scene_guids = {
        str((assets / "Start.scene").resolve()).casefold(): "start-guid",
        str((assets / "Level.scene").resolve()).casefold(): "level-guid",
    }
    scene_paths = {
        "start-guid": str((assets / "Start.scene").resolve()),
        "level-guid": str((assets / "Level.scene").resolve()),
    }
    database = SimpleNamespace(
        get_guid_from_path=lambda path: scene_guids.get(
            str(Path(path).resolve()).casefold(), ""
        ),
        get_path_from_guid=lambda guid: scene_paths.get(guid, ""),
    )
    authoring.project_assets.configure(str(tmp_path), database)
    controller = ensure_project_settings_document(str(tmp_path))
    before = controller.capture_document()
    assert editor.set_build_scenes([assets / "Level.scene", "Assets/Start.scene"])
    expected = ["Assets/Level.scene", "Assets/Start.scene"]
    assert editor.get_build_scenes() == expected
    detached = editor.get_build_scenes()
    detached.clear()
    assert editor.get_build_scenes() == expected
    assert not editor.set_build_scenes(expected)
    editor.undo(defer=False)
    assert controller.capture_document() == before
    editor.redo(defer=False)
    assert editor.get_build_scenes() == expected
    pending = editor.DocumentActionResult(editor.DocumentActionStatus.PENDING)
    calls = []
    monkeypatch.setattr(authoring.documents, "request_save", lambda key: calls.append(key) or pending)
    assert editor.save_project_settings() is pending
    assert calls == [controller.document_id]


@pytest.mark.parametrize("paths,error", [
    ("Assets/Start.scene", TypeError),
    (["Assets/Missing.scene"], FileNotFoundError),
    (["../Outside.scene"], ValueError),
    (["Assets/Wrong.txt"], ValueError),
])
def test_public_invalid_build_scenes_leave_settings_unchanged(authoring, monkeypatch, tmp_path, paths, error):
    from Infernux.engine import project_context
    monkeypatch.setattr(project_context, "get_project_root", lambda: str(tmp_path))
    authoring.project_assets.configure(str(tmp_path))
    before = editor.get_build_scenes()
    with pytest.raises(error):
        editor.set_build_scenes(paths)
    assert editor.get_build_scenes() == before
    assert not UndoManager.instance().can_undo


class PropertyRevertProbe(InxComponent):
    amount: float = serialized_field(default=2.0)
    label: str = serialized_field(default="Source")
    target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
    collider = serialized_field(default=None, field_type=FieldType.COMPONENT)
    locked: int = serialized_field(default=1, readonly=True)


@pytest.fixture
def property_prefab(asset_authoring, scene):
    root = scene.create_game_object("Property Source")
    root.transform.local_position = Vector3(2, 3, 4)
    child = scene.create_game_object("Referenced Child")
    child.set_parent(root)
    collider = child.add_component("BoxCollider")
    for amount in (2.0, 5.0):
        component = PropertyRevertProbe()
        component.amount = amount
        component.target = child
        component.collider = collider
        root.add_py_component(component)
    path = editor.create_prefab(root, asset_authoring.assets_root)
    instance = editor.instantiate_prefab(path)
    UndoManager.instance().clear()
    return path, instance


def test_revert_one_python_property_retains_other_fields_and_identity(property_prefab, scene):
    path, root = property_prefab
    first, second = root.get_components(PropertyRevertProbe)
    first.amount = 20.0
    second.amount = 50.0
    second.label = "Keep instance label"
    component_id = second.component_id
    child_id = root.get_child(0).id
    root.transform.local_position = Vector3(10, 11, 12)
    assert editor.revert_property_override(second, "amount")
    first, second = root.get_components(PropertyRevertProbe)
    assert (first.amount, second.amount, second.label) == (20.0, 5.0, "Keep instance label")
    assert second.component_id == component_id
    assert root.get_child(0).id == child_id
    assert root.transform.local_position.x == 10
    editor.undo(defer=False)
    assert root.get_components(PropertyRevertProbe)[1].amount == 50
    editor.redo(defer=False)
    second = root.get_components(PropertyRevertProbe)[1]
    assert second.amount == 5
    assert not editor.revert_property_override(second, "amount")


def test_property_queries_are_field_exact_detached_and_read_only(property_prefab):
    from pathlib import Path
    path, root = property_prefab
    first, second = root.get_components(PropertyRevertProbe)
    second.amount = 50
    second.label = "Instance"
    source_bytes = Path(path).read_bytes()
    before = root.serialize_document()
    assert not editor.is_property_override(first, "amount")
    assert editor.is_property_override(second, "amount")
    assert not editor.is_property_override(second, "target")
    assert not editor.is_property_override(second, "collider")
    assert not editor.is_property_override(second, "locked")
    changes = editor.get_property_modifications(root)
    fields = [item for item in changes if item.component_id == second.component_id]
    assert [(item.property_path, item.source_value, item.instance_value) for item in fields] == [
        ("data.amount", 5.0, 50.0), ("data.label", "Source", "Instance"),
    ]
    assert all(item.source_component_id > 0 and item.object_id == root.id for item in fields)
    assert root.serialize_document() == before
    assert Path(path).read_bytes() == source_bytes
    assert not UndoManager.instance().can_undo


def test_property_query_native_transform_defaults_and_snapshot(property_prefab):
    _, root = property_prefab
    root.transform.local_position = Vector3(10, 11, 12)
    collider = root.get_child(0).get_component(BoxCollider)
    collider.size = Vector3(4, 5, 6)
    assert editor.is_property_override(root.transform, "local_position")
    assert editor.is_property_override(collider, "size")
    changes = editor.get_property_modifications(root)
    position = next(item for item in changes if item.component_id == root.transform.component_id)
    assert position.property_path == "position"
    assert position.is_default_override
    size = next(item for item in changes if item.component_id == collider.component_id)
    assert size.property_path == "data.size"
    assert not size.is_default_override
    size.instance_value.clear()
    assert collider.size.x == 4
    assert editor.get_property_modifications(root) != changes


def test_property_query_tracks_revert_undo_and_exact_component_refs(property_prefab):
    _, root = property_prefab
    component = root.get_components(PropertyRevertProbe)[1]
    component.target = None
    component.collider = None
    assert editor.is_property_override(component, "target")
    assert editor.is_property_override(component, "collider")
    assert editor.revert_property_override(component, "collider")
    component = root.get_components(PropertyRevertProbe)[1]
    assert not editor.is_property_override(component, "collider")
    assert editor.is_property_override(component, "target")
    editor.undo(defer=False)
    assert editor.is_property_override(root.get_components(PropertyRevertProbe)[1], "collider")


def test_property_query_added_members_and_plain_objects_are_not_source_fields(property_prefab, scene):
    _, root = property_prefab
    added = PropertyRevertProbe()
    added.amount = 12
    root.add_py_component(added)
    assert not editor.is_property_override(added, "amount")
    assert not any(item.component_id == added.component_id for item in editor.get_property_modifications(root))
    plain = scene.create_game_object("Not a prefab")
    assert editor.get_property_modifications(plain) == ()
    assert not editor.is_property_override(plain.transform, "local_position")
    with pytest.raises((KeyError, ValueError)):
        editor.is_property_override(added, "undeclared")


def test_property_query_deleted_reference_does_not_require_remapping_missing_target(property_prefab, scene):
    _, root = property_prefab
    scene.destroy_game_object(root.get_child(0))
    scene.process_pending_destroys()
    component = root.get_components(PropertyRevertProbe)[0]
    component.target = None
    assert editor.is_property_override(component, "target")
    assert not UndoManager.instance().can_undo


@pytest.mark.parametrize("field", ["target", "collider"])
def test_revert_reference_uses_instance_object_and_exact_component(property_prefab, field):
    path, root = property_prefab
    component = root.get_components(PropertyRevertProbe)[0]
    setattr(component, field, None)
    assert editor.revert_property_override(component, field)
    component = root.get_components(PropertyRevertProbe)[0]
    target = getattr(component, field)
    if field == "target":
        assert target.id == root.get_child(0).id
    else:
        assert target.component_id == root.get_child(0).get_component("BoxCollider").component_id
    editor.undo(defer=False)
    assert getattr(root.get_components(PropertyRevertProbe)[0], field) is None


def test_revert_native_and_explicit_root_transform_properties(property_prefab):
    path, root = property_prefab
    collider = root.get_child(0).get_component(BoxCollider)
    collider.size = Vector3(4, 5, 6)
    collider.center = Vector3(1, 2, 3)
    assert editor.revert_property_override(collider, "size")
    collider = root.get_child(0).get_component(BoxCollider)
    assert collider.size.x == 1
    assert collider.center.x == 1
    root.transform.local_position = Vector3(10, 11, 12)
    assert editor.revert_property_override(root.transform, "local_position")
    assert root.transform.local_position.x == 2
    editor.undo(defer=False)
    assert root.transform.local_position.x == 10


def test_rect_gizmo_override_revert_apply_and_undo_share_one_authoring_history(
    property_prefab,
):
    """A Scene Rect gesture is a normal Prefab property edit, not a side channel."""
    from Infernux.engine.ui.scene_view_panel import SceneViewPanel, TOOL_RECT

    _path, root = property_prefab
    panel = SceneViewPanel(engine=None)
    panel._gizmo_drag_obj_id = int(root.id)
    panel._gizmo_drag_items = {
        int(root.id): panel._snapshot_gizmo_object(root),
    }

    root.transform.position = Vector3(8, 9, 10)
    root.transform.local_scale = Vector3(2, 3, 4)
    assert panel._record_gizmo_undo(TOOL_RECT)
    assert editor.is_property_override(root.transform, "local_scale")
    assert tuple(root.transform.local_scale) == (2, 3, 4)

    editor.undo(defer=False)
    assert tuple(root.transform.local_scale) == (1, 1, 1)
    assert not editor.is_property_override(root.transform, "local_scale")
    editor.redo(defer=False)
    assert tuple(root.transform.local_scale) == (2, 3, 4)
    assert editor.is_property_override(root.transform, "local_scale")

    assert editor.revert_property_override(root.transform, "local_scale")
    assert tuple(root.transform.local_scale) == (1, 1, 1)
    assert not editor.is_property_override(root.transform, "local_scale")
    editor.undo(defer=False)
    assert tuple(root.transform.local_scale) == (2, 3, 4)
    assert editor.is_property_override(root.transform, "local_scale")

    assert editor.apply_prefab(root)
    assert tuple(root.transform.local_scale) == (2, 3, 4)
    assert not editor.is_property_override(root.transform, "local_scale")
    editor.undo(defer=False)
    assert editor.is_property_override(root.transform, "local_scale")


@pytest.mark.parametrize("field", ["unknown_field", "locked"])
def test_revert_undeclared_or_readonly_field_is_not_recorded(property_prefab, field):
    path, root = property_prefab
    before = root.serialize_document()
    with pytest.raises((KeyError, ValueError)):
        editor.revert_property_override(root.get_components(PropertyRevertProbe)[0], field)
    assert root.serialize_document() == before
    assert not UndoManager.instance().can_undo


def test_revert_missing_referenced_member_does_not_fall_back_to_raw_id(property_prefab, scene):
    path, root = property_prefab
    child = root.get_child(0)
    scene.destroy_game_object(child)
    scene.process_pending_destroys()
    component = root.get_components(PropertyRevertProbe)[0]
    component.target = None
    with pytest.raises(KeyError):
        editor.revert_property_override(component, "target")
    assert component.target is None
    assert not UndoManager.instance().can_undo


def test_revert_added_component_does_not_touch_same_type_source_component(property_prefab):
    path, root = property_prefab
    added = PropertyRevertProbe()
    added.amount = 12
    root.add_py_component(added)
    with pytest.raises(ValueError, match="no source property"):
        editor.revert_property_override(added, "amount")
    assert root.get_components(PropertyRevertProbe)[-1].amount == 12
    assert not UndoManager.instance().can_undo


@pytest.mark.parametrize("already_matches", [False, True])
def test_property_revert_advances_only_selected_source_baseline(property_prefab, already_matches):
    from Infernux.engine.prefab_manager import _read_prefab_document, save_prefab_document
    path, root = property_prefab
    component = root.get_components(PropertyRevertProbe)[0]
    component.amount = 7 if already_matches else 20
    component.label = "Keep override"
    source = _read_prefab_document(path)
    source["root_object"]["components"][0]["data"]["amount"] = 7
    source["root_object"]["components"][0]["data"]["label"] = "New source label"
    assert save_prefab_document(source, path)
    before_query = root.serialize_document()
    assert editor.is_property_override(component, "amount") == (not already_matches)
    assert editor.is_property_override(component, "label")
    assert root.serialize_document() == before_query
    assert editor.revert_property_override(component, "amount")
    component = root.get_components(PropertyRevertProbe)[0]
    assert (component.amount, component.label) == (7, "Keep override")
    baseline = root.serialize_document()["prefab_source"]["root_object"]["components"][0]["data"]
    assert (baseline["amount"], baseline["label"]) == (7, "Source")
    editor.undo(defer=False)
    restored = root.serialize_document()["prefab_source"]["root_object"]["components"][0]["data"]
    assert (restored["amount"], restored["label"]) == (2, "Source")


def test_revert_nested_component_targets_nearest_prefab_source(property_prefab, asset_authoring):
    path, inner = property_prefab
    outer = editor.create_game_object("Outer Property Source")
    inner.set_parent(outer)
    outer_path = editor.create_prefab(outer, asset_authoring.assets_root)
    placed = editor.instantiate_prefab(outer_path)
    nested = placed.get_child(0)
    component = nested.get_components(PropertyRevertProbe)[1]
    component.amount = 50
    component.label = "Nested override"
    UndoManager.instance().clear()
    assert editor.is_property_override(component, "amount")
    for target in (nested, placed):
        changes = editor.get_property_modifications(target)
        amount = next(item for item in changes if item.component_id == component.component_id
                      and item.property_path == "data.amount")
        assert (amount.source_value, amount.instance_value) == (5, 50)
    assert editor.revert_property_override(component, "amount")
    component = nested.get_components(PropertyRevertProbe)[1]
    assert not editor.is_property_override(component, "amount")
    assert editor.is_property_override(component, "label")
    assert (component.amount, component.label) == (5, "Nested override")
    editor.undo(defer=False)
    assert nested.get_components(PropertyRevertProbe)[1].amount == 50
