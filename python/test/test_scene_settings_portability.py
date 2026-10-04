from pathlib import Path
import json
from types import SimpleNamespace

from Infernux.engine import scene_manager as scene_manager_module
from Infernux.engine.scene_manager import SceneFileManager
from Infernux.engine import _scene_save as scene_save_module


class _AssetDatabase:
    def __init__(self, scene: Path, guid: str = "scene-guid") -> None:
        self.scene = str(scene)
        self.guid = guid

    def get_path_from_guid(self, guid: str) -> str:
        return self.scene if guid == self.guid else ""

    def get_guid_from_path(self, path: str) -> str:
        return self.guid if Path(path) == Path(self.scene) else ""


def _manager(database: _AssetDatabase) -> SceneFileManager:
    manager = SceneFileManager()
    manager._asset_database = database
    return manager


def test_last_scene_resolves_by_guid(tmp_path, monkeypatch):
    scene = tmp_path / "Assets" / "Scenes" / "Start.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    manager = _manager(_AssetDatabase(scene))
    opened: list[str] = []
    saved: list[dict] = []
    monkeypatch.setattr(
        scene_manager_module,
        "_load_editor_settings",
        lambda: {"lastOpenedSceneGuid": "scene-guid"},
    )
    monkeypatch.setattr(
        scene_manager_module, "_save_editor_settings", lambda value: saved.append(value)
    )
    monkeypatch.setattr(
        scene_save_module,
        "_load_editor_settings",
        lambda: {"lastOpenedSceneGuid": "scene-guid"},
    )
    monkeypatch.setattr(
        scene_save_module, "_save_editor_settings", lambda value: saved.append(value)
    )
    monkeypatch.setattr(
        manager,
        "_do_open_scene",
        lambda path, record_navigation: opened.append(path) or True,
    )

    manager.load_last_scene_or_default()

    assert opened == [str(scene)]
    assert saved == [{"lastOpenedSceneGuid": "scene-guid"}]

def test_scene_camera_state_is_keyed_by_asset_guid(tmp_path, monkeypatch):
    scene = tmp_path / "Assets" / "Scenes" / "Start.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    manager = _manager(_AssetDatabase(scene))
    manager._engine = SimpleNamespace(
        editor_camera=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=(4.0, 5.0),
            focus_point=SimpleNamespace(x=6.0, y=7.0, z=8.0),
            focus_distance=9.0,
        )
    )
    saved: list[dict] = []
    monkeypatch.setattr(
        scene_manager_module,
        "_load_editor_settings",
        lambda: {"sceneCameraStates": {}},
    )
    monkeypatch.setattr(
        scene_manager_module, "_save_editor_settings", lambda value: saved.append(value)
    )

    manager._save_camera_state(str(scene))

    assert set(saved[0]["sceneCameraStates"]) == {"scene-guid"}
    assert str(scene) not in saved[0]["sceneCameraStates"]


def test_editor_navigation_does_not_modify_shared_project_settings(tmp_path, monkeypatch):
    settings = tmp_path / "ProjectSettings" / "EditorSettings.json"
    settings.parent.mkdir()
    original = '{"lastOpenedSceneGuid": "legacy-scene"}\n'
    settings.write_text(original, encoding="utf-8")
    monkeypatch.setattr(scene_manager_module, "_effective_project_root", lambda: str(tmp_path))
    assert scene_manager_module._load_editor_settings() == {}
    scene_manager_module._save_editor_settings({"lastOpenedSceneGuid": "local-scene"})
    assert scene_manager_module._load_editor_settings()["lastOpenedSceneGuid"] == "local-scene"
    assert settings.read_text(encoding="utf-8") == original
    assert json.loads((tmp_path / "Library" / "EditorSettings.json").read_text()) == {"lastOpenedSceneGuid": "local-scene"}


def test_checkout_without_local_navigation_opens_first_authored_build_scene(tmp_path, monkeypatch):
    scene = tmp_path / "Assets" / "Scenes" / "SampleScene.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}", encoding="utf-8")
    manager = _manager(_AssetDatabase(scene))
    settings = tmp_path / "ProjectSettings" / "BuildSettings.json"
    settings.parent.mkdir()
    settings.write_text('{"scene_guids": ["scene-guid"]}\n', encoding="utf-8")
    monkeypatch.setattr(scene_manager_module, "_effective_project_root", lambda: str(tmp_path))
    monkeypatch.setattr(scene_manager_module, "_load_editor_settings", lambda: {})
    monkeypatch.setattr("Infernux.engine.build_settings.load_build_settings_for_build", lambda _: {"scene_guids": ["scene-guid"]})
    opened = []
    monkeypatch.setattr(manager, "_do_open_scene", lambda path, record_navigation: opened.append(path) or True)
    monkeypatch.setattr(manager, "_remember_last_scene", lambda _: None)
    monkeypatch.setattr(manager, "_do_new_scene", lambda: (_ for _ in ()).throw(AssertionError("initial build scene must be opened")))
    manager.load_last_scene_or_default()
    assert opened == [str(scene)]
