from types import SimpleNamespace

from Infernux.host import EditorAutomationHost


def test_editor_automation_uses_interaction_asset_database(monkeypatch):
    database = object()
    host = EditorAutomationHost()
    core = SimpleNamespace(project_assets=SimpleNamespace(asset_database=database))
    monkeypatch.setattr(host, "interaction_core", lambda: core)

    assert host.asset_database() is database


def test_editor_automation_creates_assets_through_interaction_service(monkeypatch):
    calls = []
    service = SimpleNamespace(
        create=lambda *args: calls.append(args) or "Assets/Scripts/Player.py"
    )
    host = EditorAutomationHost()
    core = SimpleNamespace(project_asset_interactions=service)
    monkeypatch.setattr(host, "interaction_core", lambda: core)

    result = host.create_project_asset(
        "script", "Assets/Scripts", "Player", ".py", "component"
    )

    assert result == "Assets/Scripts/Player.py"
    assert calls == [("script", "Assets/Scripts", "Player", ".py", "component")]


def test_editor_automation_additive_load_uses_authoring_scene_manager(monkeypatch):
    from Infernux.engine.play_mode import PlayModeManager
    from Infernux.engine.scene_manager import SceneFileManager

    calls = []
    manager = SimpleNamespace(
        load_scene_additive_immediate=lambda path: calls.append(path) or True
    )
    monkeypatch.setattr(PlayModeManager, "instance", staticmethod(lambda: None))
    monkeypatch.setattr(SceneFileManager, "instance", staticmethod(lambda: manager))

    assert EditorAutomationHost().load_additive_scene("Assets/Scenes/World.scene")
    assert calls == ["Assets/Scenes/World.scene"]
