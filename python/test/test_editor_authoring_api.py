"""Public author tools share the Editor's native objects and action journal."""
from types import SimpleNamespace

import pytest

from infernux import editor
from Infernux.engine.interaction import EditorInteractionCore, SelectionDomain
from Infernux.engine.undo import UndoManager
from Infernux.engine.play_mode import PlayModeManager
from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
from Infernux.lib import Vector3
from Infernux.components import serialized_field
from Infernux.core import AssetManager, DataAsset


class AuthoringLevelData(DataAsset):
    __serialized_type_id__ = "tests.editor.authoring_level"

    title: str = serialized_field(default="新关卡")
    difficulty: int = serialized_field(default=1)


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


def test_public_create_failed_initializer_leaves_no_object_or_history(authoring, scene):
    before = [obj.id for obj in scene.get_all_objects()]

    def fail(obj):
        raise ValueError("invalid author value")

    with pytest.raises(ValueError, match="invalid author value"):
        editor.create_game_object("Rejected", configure=fail)
    assert [obj.id for obj in scene.get_all_objects()] == before
    assert not UndoManager.instance().can_undo


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
        assert Path(editor.create_folder(folder)) == folder
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
    authoring.project_assets.configure(str(tmp_path))
    assets = tmp_path / "Assets"
    assets.mkdir()
    for name in ("Start.scene", "Level.scene"):
        (assets / name).touch()
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
