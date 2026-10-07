"""Public Player residency must work without any editor document manager."""
import builtins
import json
import time
from types import SimpleNamespace

import pytest

from infernux.engine.player_scene import PlayerSceneService
from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.scene_authoring import encode_runtime_scene_artifact
from infernux.lib import SceneManager as NativeSceneManager
from infernux.scene import SceneManager


@pytest.fixture
def player_worlds(scene, tmp_path, monkeypatch):
    native = NativeSceneManager.instance()
    artifacts = []
    entries = []
    paths = {}
    for index, label in enumerate(("First", "Second", "Third"), start=1):
        document = json.loads(json.dumps(scene.serialize_document()))
        document["name"] = label
        path = tmp_path / "Content" / f"{index:032x}.scene"
        paths[label] = str(path)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(encode_runtime_scene_artifact(document)), encoding="utf-8")
        artifact_id = f"content:scene-{label}"
        artifacts.append(dict(runtime_artifact_id=artifact_id, runtime_path=f"Content/{path.name}",
                              package="Game_Data/Content.inxpkg", logical_type="scene",
                              asset_guid=f"guid-{label}", dependencies=[]))
        entries.append(dict(guid=f"guid-{label}", runtime_path=f"Assets/Scenes/{label}.scene",
                            primary_runtime_artifact_id=artifact_id,
                            runtime_artifact_ids=[artifact_id], dependencies=[]))
    catalog = PlayerRuntimeAssetCatalog.from_documents(
        str(tmp_path), dict(artifacts=artifacts), dict(entries=entries))
    service = PlayerSceneService()
    service.bind_runtime_catalog(catalog)
    monkeypatch.setattr(SceneManager, "_runtime_scene_service", service)
    original_import = builtins.__import__

    def reject_editor(name, *args, **kwargs):
        if name == "infernux.engine.scene_manager":
            raise ModuleNotFoundError("Player excludes infernux.engine.scene_manager")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_editor)
    assert service.load_initial("guid-First")
    first = native.get_active_scene()
    native.play()

    def load(label, *, hold=False, mode="additive"):
        assert service.request_prepared_load(f"guid-{label}", mode=mode, hold_for_activation=hold)
        deadline = time.monotonic() + 5.0
        while service.is_load_pending and not service.is_prepared and time.monotonic() < deadline:
            service.process_pending_load()
            time.sleep(0.001)
        assert service.is_prepared if hold else not service.is_load_pending, service.last_error
        return native.get_scene(label) if not hold else service._transaction_target

    try:
        second = load("Second")
        yield SimpleNamespace(native=native, service=service, first=first, second=second,
                              load=load, paths=paths)
    finally:
        service.cancel_pending_load()
        native.stop()


def test_player_selects_and_unloads_without_editor_modules(player_worlds):
    state = player_worlds
    first_world, second_world = state.first.world_id, state.second.world_id
    SceneManager.set_active_scene(state.second)
    assert SceneManager.get_active_scene() is state.second
    assert state.service.active_scene_path == state.paths["Second"]
    created = state.second.create_game_object("SelectedWorldWitness")
    assert created.scene is state.second
    SceneManager.unload_scene(state.first)
    assert state.native.get_scene_by_world_id(first_world) is None
    assert state.service.active_scene_path == state.paths["Second"]
    assert SceneManager.get_active_scene() is state.second
    SceneManager.unload_scene(state.second)
    assert state.native.get_scene_by_world_id(second_world) is None
    assert SceneManager.get_active_scene() is None
    assert state.service.active_scene_path is None


def test_player_unloads_active_scene_and_publishes_remaining_path(player_worlds):
    state = player_worlds
    SceneManager.unload_scene(state.first)
    assert SceneManager.get_active_scene() is state.second
    assert state.service.active_scene_path == state.paths["Second"]
    SceneManager.set_active_scene(None)
    assert state.service.active_scene_path is None
    SceneManager.set_active_scene(state.second)
    assert state.service.active_scene_path == state.paths["Second"]


def test_preparing_additive_target_is_not_publicly_selectable_or_unloadable(player_worlds):
    state = player_worlds
    target = state.load("Third", hold=True)
    for mutate in (SceneManager.set_active_scene, SceneManager.unload_scene):
        with pytest.raises(RuntimeError, match="prepar"):
            mutate(target)
    assert state.service.is_prepared
    SceneManager.set_active_scene(state.second)
    SceneManager.unload_scene(state.first)
    assert state.service.active_scene_path == state.paths["Second"]
    assert state.service.activate_prepared_load()
    state.service.process_pending_load()
    assert not state.service.is_load_pending
    SceneManager.set_active_scene(target)
    assert state.service.active_scene_path == state.paths["Third"]


def test_preparing_single_load_keeps_target_ownership(player_worlds):
    state = player_worlds
    state.load("Third", hold=True, mode="single")
    with pytest.raises(RuntimeError, match="prepar"):
        SceneManager.set_active_scene(state.second)
    with pytest.raises(RuntimeError, match="prepar"):
        SceneManager.unload_scene(state.first)
    assert SceneManager.get_active_scene() is state.first
    assert state.service.active_scene_path == state.paths["First"]
    state.service.cancel_pending_load()
    SceneManager.set_active_scene(state.second)
    assert state.service.active_scene_path == state.paths["Second"]


@pytest.mark.parametrize("mode", ["single", "additive"])
def test_failed_player_load_keeps_resident_origins(player_worlds, mode):
    state = player_worlds
    from pathlib import Path

    path = Path(state.paths["Third"])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["mainCameraComponentId"] = 999
    path.write_text(json.dumps(document), encoding="utf-8")
    assert state.service.request_load("guid-Third", mode=mode)
    deadline = time.monotonic() + 5.0
    while state.service.is_load_pending and time.monotonic() < deadline:
        state.service.process_pending_load()
        time.sleep(0.001)
    assert not state.service.is_load_pending
    assert "mainCameraComponentId" in state.service.last_error
    assert state.native.scene_count == 2
    assert SceneManager.get_active_scene() is state.first
    assert state.service.active_scene_path == state.paths["First"]
    SceneManager.set_active_scene(state.second)
    SceneManager.unload_scene(state.first)
    assert SceneManager.get_active_scene() is state.second
    assert state.service.active_scene_path == state.paths["Second"]
