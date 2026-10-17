"""Public author tools share the Editor's native objects and action journal."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux import editor
from infernux.engine.interaction import EditorInteractionCore, SelectionDomain
from infernux.engine.undo import UndoManager
from infernux.engine.hierarchy_creation_service import HierarchyCreationService
from infernux.lib import Vector3
from infernux.components import InxComponent, FieldType, serialized_field, list_field
from infernux.components.builtin import BoxCollider
from infernux.core import AssetManager, DataAsset


class AuthoringLevelData(DataAsset):
    __serialized_type_id__ = "tests.editor.authoring_level"

    title: str = serialized_field(default="新关卡")
    difficulty: int = serialized_field(default=1)


class AuthoringCatalogData(DataAsset):
    __serialized_type_id__ = "tests.editor.authoring_catalog"
    levels = list_field(element_type=FieldType.ASSET, asset_type="DataAsset")
    locked: int = serialized_field(default=1, readonly=True)


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


def test_public_obstacle_authoring_groups_parent_transform_and_component_configuration(authoring, scene):
    with editor.edit_scene("Create obstacle"):
        root = editor.create_game_object("机关")
        child = editor.create_game_object(
            "碰撞体", parent=root, configure=lambda obj: setattr(obj.transform, "local_position", Vector3(1, 2, 3))
        )
        editor.add_component(child, "BoxCollider", configure=lambda item: setattr(item, "size", Vector3(2, 3, 4)))
    root_id, child_id = root.id, child.id
    assert UndoManager.instance().undo_description == "Create obstacle"
    assert child.get_parent().id == root_id and child.get_component(BoxCollider).size.y == 3
    editor.undo(defer=False)
    assert scene.find_by_id(root_id) is None and scene.find_by_id(child_id) is None
    editor.redo(defer=False)
    child = scene.find_by_id(child_id)
    assert child.get_parent().id == root_id and child.transform.local_position.x == 1
    assert child.get_component(BoxCollider).size.y == 3


def test_public_create_failed_initializer_leaves_no_object_or_history(authoring, scene):
    before = [obj.id for obj in scene.get_all_objects()]

    def fail(obj):
        raise ValueError("invalid author value")

    with pytest.raises(ValueError, match="invalid author value"):
        editor.create_game_object("Rejected", configure=fail)
    assert [obj.id for obj in scene.get_all_objects()] == before
    assert not UndoManager.instance().can_undo


@pytest.fixture
def asset_authoring(authoring, engine, monkeypatch):
    from infernux.engine import project_context

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(project_context, "get_project_root", lambda: database.project_root)
    authoring.project_assets.configure(database.project_root, database)
    return database


def test_public_level_data_create_edit_save_and_history_keep_guid_bytes_and_shared_document(asset_authoring):
    from pathlib import Path
    from infernux.engine.interaction import DocumentKind, ensure_editable_resource_document
    import json

    source = AuthoringLevelData(title="压力板", difficulty=7)
    folder = Path(asset_authoring.assets_root) / "AuthoringDataContract"
    with editor.edit_scene("创建关卡数据"):
        assert Path(editor.create_folder(folder)).samefile(folder)
        target = Path(editor.create_data_asset(source, folder / "Level.inxdata"))
    assert not source.is_persistent
    asset = DataAsset.load(str(target))
    assert asset is not source and (asset.title, asset.difficulty) == ("压力板", 7)
    guid = asset.guid
    assert guid
    content, metadata = target.read_bytes(), Path(str(target) + ".meta").read_bytes()
    assert UndoManager.instance().undo_description == "创建关卡数据"
    editor.undo(defer=False)
    assert not folder.exists()
    editor.redo(defer=False)
    assert target.read_bytes() == content and Path(str(target) + ".meta").read_bytes() == metadata
    AssetManager.invalidate(guid)
    asset = DataAsset.load(str(target))
    assert asset.guid == guid
    controller = ensure_editable_resource_document(
        category="data_asset",
        document_kind=DocumentKind.DATA_ASSET,
        file_path=str(target),
        resource=asset,
        guid=guid,
        view_id="inspector.test",
    )
    before_actions = len(UndoManager.instance().action_journal.applied_entries())
    assert editor.set_data_asset_fields(asset, title="第一关", difficulty=5)
    assert editor.load_data_asset(str(target)) is asset and controller.resource is asset
    assert (asset.title, asset.difficulty) == ("第一关", 5)
    assert not editor.set_data_asset_fields(asset, title="第一关", difficulty=5)
    assert len(UndoManager.instance().action_journal.applied_entries()) == before_actions + 1
    editor.undo(defer=False)
    assert (asset.title, asset.difficulty) == ("压力板", 7)
    assert editor.load_data_asset(str(target)).difficulty == 7
    assert len(UndoManager.instance().action_journal.applied_entries()) == before_actions
    editor.redo(defer=False)
    assert (asset.title, asset.difficulty) == ("第一关", 5)
    assert editor.save_data_asset(asset).status in (
        editor.DocumentActionStatus.APPLIED,
        editor.DocumentActionStatus.NO_OP,
    )
    assert json.loads(target.read_text(encoding="utf-8"))["fields"]["difficulty"] == 5


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
    from infernux.engine.interaction import DocumentKind, ensure_editable_resource_document

    path = editor.create_data_asset(AuthoringLevelData(), "Assets/StaleEditData.inxdata")
    stale = DataAsset.load(path)
    current = stale.instantiate()
    current._bind_asset(stale.file_path, stale.guid)
    current.title = "Newer Inspector value"
    controller = ensure_editable_resource_document(
        category="data_asset",
        document_kind=DocumentKind.DATA_ASSET,
        file_path=path,
        resource=current,
        guid=current.guid,
        view_id="inspector.test",
    )
    assert editor.set_data_asset_fields(stale, difficulty=9)
    assert controller.resource is current
    assert current.title == "Newer Inspector value"
    assert current.difficulty == 9
    assert stale.difficulty == 1
    assert editor.load_data_asset(path) is current


def test_public_build_scene_list_shares_settings_and_undo(authoring, monkeypatch, tmp_path):
    from infernux.engine import project_context
    from infernux.engine.interaction.project_settings import ensure_project_settings_document

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
        get_guid_from_path=lambda path: scene_guids.get(str(Path(path).resolve()).casefold(), ""),
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


def test_prefab_scalar_queries_and_reverts_preserve_sibling_fields_identity_and_read_only_snapshots(property_prefab):
    from pathlib import Path

    path, root = property_prefab
    first, second = root.get_components(PropertyRevertProbe)
    assert not editor.is_property_override(first, "amount")
    second.amount, second.label = 50, "Keep instance label"
    component_id, child_id = second.component_id, root.get_child(0).id
    root.transform.local_position = Vector3(10, 11, 12)
    collider = root.get_child(0).get_component(BoxCollider)
    collider.size, collider.center = Vector3(4, 5, 6), Vector3(1, 2, 3)
    before, source_bytes = root.serialize_document(), Path(path).read_bytes()
    assert editor.is_property_override(second, "amount")
    assert not editor.is_property_override(second, "target")
    assert not editor.is_property_override(second, "collider")
    assert not editor.is_property_override(second, "locked")
    changes = editor.get_property_modifications(root)
    fields = [item for item in changes if item.component_id == second.component_id]
    assert [(item.property_path, item.source_value, item.instance_value) for item in fields] == [
        ("data.amount", 5.0, 50.0),
        ("data.label", "Source", "Keep instance label"),
    ]
    assert all(item.source_component_id > 0 and item.object_id == root.id for item in fields)
    assert root.serialize_document() == before and Path(path).read_bytes() == source_bytes
    assert not UndoManager.instance().can_undo
    position = next(item for item in changes if item.component_id == root.transform.component_id)
    size = next(
        item for item in changes if item.component_id == collider.component_id and item.property_path == "data.size"
    )
    assert position.property_path == "position" and position.is_default_override
    assert not size.is_default_override
    size.instance_value.clear()
    assert collider.size.x == 4 and editor.get_property_modifications(root) != changes
    first.amount = 20
    assert editor.revert_property_override(second, "amount")
    first, second = root.get_components(PropertyRevertProbe)
    assert (first.amount, second.amount, second.label) == (20.0, 5.0, "Keep instance label")
    assert second.component_id == component_id and root.get_child(0).id == child_id
    assert root.transform.local_position.x == 10
    editor.undo(defer=False)
    assert root.get_components(PropertyRevertProbe)[1].amount == 50
    editor.redo(defer=False)
    assert root.get_components(PropertyRevertProbe)[1].amount == 5
    assert not editor.revert_property_override(root.get_components(PropertyRevertProbe)[1], "amount")
    collider = root.get_child(0).get_component(BoxCollider)
    assert editor.revert_property_override(collider, "size")
    collider = root.get_child(0).get_component(BoxCollider)
    assert collider.size.x == 1 and collider.center.x == 1
    assert editor.revert_property_override(root.transform, "local_position")
    assert root.transform.local_position.x == 2
    editor.undo(defer=False)
    assert root.transform.local_position.x == 10
    assert Path(path).read_bytes() == source_bytes


def test_prefab_reference_query_revert_and_undo_target_exact_instance_members(property_prefab):
    _, root = property_prefab
    first, component = root.get_components(PropertyRevertProbe)
    first_document = first._serialize_fields_document()
    component.target, component.collider = None, None
    assert editor.is_property_override(component, "target") and editor.is_property_override(component, "collider")
    assert editor.revert_property_override(component, "target")
    component = root.get_components(PropertyRevertProbe)[1]
    assert component.target.id == root.get_child(0).id and component.collider is None
    assert first._serialize_fields_document() == first_document
    assert not editor.is_property_override(component, "target") and editor.is_property_override(component, "collider")
    assert editor.revert_property_override(component, "collider")
    component = root.get_components(PropertyRevertProbe)[1]
    assert component.collider.component_id == root.get_child(0).get_component("BoxCollider").component_id
    assert first._serialize_fields_document() == first_document
    assert not editor.is_property_override(component, "collider")
    editor.undo(defer=False)
    component = root.get_components(PropertyRevertProbe)[1]
    assert component.collider is None and component.target.id == root.get_child(0).id
    assert first._serialize_fields_document() == first_document
    assert editor.is_property_override(component, "collider")
    editor.undo(defer=False)
    component = root.get_components(PropertyRevertProbe)[1]
    assert component.target is None and component.collider is None
    assert first._serialize_fields_document() == first_document


def test_rect_gizmo_override_revert_apply_and_undo_share_one_authoring_history(
    property_prefab,
):
    """A Scene Rect gesture is a normal Prefab property edit, not a side channel."""
    from infernux.engine.ui.scene_view_panel import SceneViewPanel, TOOL_RECT

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
        amount = next(
            item
            for item in changes
            if item.component_id == component.component_id and item.property_path == "data.amount"
        )
        assert (amount.source_value, amount.instance_value) == (5, 50)
    assert editor.revert_property_override(component, "amount")
    component = nested.get_components(PropertyRevertProbe)[1]
    assert not editor.is_property_override(component, "amount")
    assert editor.is_property_override(component, "label")
    assert (component.amount, component.label) == (5, "Nested override")
    editor.undo(defer=False)
    assert nested.get_components(PropertyRevertProbe)[1].amount == 50
