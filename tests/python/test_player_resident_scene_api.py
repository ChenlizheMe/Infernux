"""Public Player residency must work without any editor document manager."""
import builtins
import json
import time
from types import SimpleNamespace

import pytest

from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.scene_authoring import encode_runtime_scene_artifact
from infernux.lib import SceneManager as NativeSceneManager
from infernux.scene import SceneManager


@pytest.fixture
def player_runtime(engine, tmp_path, monkeypatch):
    """Borrow the existing CI native host through its real Python companion.

    Construct no second native Engine, runtime driver or placeholder world.
    Install the production lifecycle/retirement bridge and let Player activation
    prepare, clone and publish the actual authored graph before Play.
    """
    import infernux.application as application_state
    from infernux.application import Application
    from infernux.engine.engine import Engine
    from infernux.engine.player_runtime import PlayerRuntimeSession
    from infernux.engine.project_context import using_project_root
    from infernux.lib import RuntimeMode
    from infernux.renderstack.render_stack import RenderStack

    host = Engine.__new__(Engine)
    host._engine = engine
    host._mode = RuntimeMode.Graphical
    host._application_role = "player"
    host._play_mode_manager = None
    host._world_runtimes = {}
    host._runtime_world_session = None
    host._initialize_work_runtime(engine.frame_work, accepting=True)
    session = PlayerRuntimeSession(
        asset_database=engine.get_asset_database(),
        native_engine=engine,
    )
    host._authoring_scheduler = session.execution_scheduler
    host._player_runtime = session
    native = NativeSceneManager.instance()
    with using_project_root(str(tmp_path)), monkeypatch.context() as globals_state:
        # Restore the exact prior binding, including exit code/install root;
        # rebinding the old Engine would reset that process state.
        for name in ("_engine_ref", "_runtime_kind", "_exit_code", "_player_install_root"):
            globals_state.setattr(application_state, name, getattr(application_state, name))
        globals_state.setattr(RenderStack, "_scene_instances", dict(RenderStack._scene_instances))
        Application._bind_engine(host, "player")
        try:
            host._install_pre_scene_time_callback()
            yield SimpleNamespace(host=host, session=session)
        finally:
            try:
                # These production shutdown paths remove the world preparation
                # source and any numerical work sources from the borrowed owner.
                session.shutdown()
                host._shutdown_work_runtime()
                assert native._collect_retired_runtime_worlds()
                assert host._runtime_world_session is None
                assert not host._world_runtimes
            finally:
                session.release_content()
                engine._set_runtime_frame_callbacks(None, None)
                native._set_runtime_world_retirement_callbacks(None, None)
                native.clear_runtime_lifecycle_callbacks()
                host._authoring_scheduler.end_native_frame()
                host._authoring_scheduler.unbind_native_bridge()
                host._authoring_scheduler.clear()
                host._runtime_scene_manager = None
                Application._unbind_engine(host)
                host._engine = None


def _configure_player_contract(session, catalog):
    from infernux.engine.player_service_graph import (
        RuntimeFeatureSet, RuntimeFlavor, RuntimeProductManifest,
        player_manifest_service_section, runtime_policy_for,
    )

    flavor = RuntimeFlavor.PLAYER_RELEASE
    features = RuntimeFeatureSet()
    manifest = RuntimeProductManifest.from_document({
        "$schema": "infernux.player_runtime_manifest",
        "product": {"flavor": flavor.value},
        "features": features.to_manifest(),
        "runtime_policy": runtime_policy_for(flavor).to_manifest(),
        "services": player_manifest_service_section(flavor, features),
    })
    session.configure_runtime_contract(manifest, catalog)


@pytest.fixture
def player_worlds(scene, player_runtime, tmp_path, monkeypatch):
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
    session = player_runtime.session
    _configure_player_contract(session, catalog)
    original_import = builtins.__import__

    def reject_editor(name, *args, **kwargs):
        if name == "infernux.engine.scene_manager":
            raise ModuleNotFoundError("Player excludes infernux.engine.scene_manager")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_editor)
    assert session.load_scene("guid-First")
    assert session.activate()
    identity = session._runtime_world_id
    domain = player_runtime.host._world_runtimes[identity]
    service = domain.scene_loads._runtime_scene_service

    def load(label, *, hold=False, mode="additive"):
        assert service.request_prepared_load(f"guid-{label}", mode=mode, hold_for_activation=hold)
        deadline = time.monotonic() + 5.0
        while service.is_load_pending and not service.is_prepared and time.monotonic() < deadline:
            service.process_pending_load()
            time.sleep(0.001)
        assert service.is_prepared if hold else not service.is_load_pending, service.last_error
        return native.get_scene(label) if not hold else service._transaction_target

    from infernux.lib import _Infernux
    with _Infernux._RuntimeWorldScope(identity):
        first = native.get_active_scene()
        try:
            second = load("Second")
            yield SimpleNamespace(native=native, service=service, first=first, second=second,
                                  load=load, paths=paths)
        finally:
            service.cancel_pending_load()


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


def test_player_shutdown_retires_scenes_before_asset_resolvers(scene, player_runtime, tmp_path):
    from infernux.components import InxComponent
    from infernux.engine.project_context import (
        resolve_runtime_asset_guid,
        set_runtime_asset_resolver,
    )
    from infernux.lib import _Infernux

    events = []
    retired_scene_counts = []
    native = NativeSceneManager.instance()

    class ShutdownWitness(InxComponent):
        def on_disable(self):
            events.append(("disable", resolve_runtime_asset_guid("font")))

        def on_destroy(self):
            events.append(("destroy", resolve_runtime_asset_guid("font")))
            # Destruction executes in the retiring world's own scope. The
            # authored template remains resident in the separate authoring world.
            retired_scene_counts.append(native.scene_count)
            assert native.scene_count == 0

    session = player_runtime.session
    catalog = PlayerRuntimeAssetCatalog.from_documents(
        str(tmp_path), {"artifacts": []}, {"entries": []})
    scene.create_game_object("ShutdownWitness").add_py_component(ShutdownWitness())
    authoring_scene_count = native.scene_count
    _configure_player_contract(session, catalog)
    assert session.activate()
    identity = session._runtime_world_id
    handle = native._get_runtime_world_handle(identity)
    with _Infernux._RuntimeWorldScope(identity):
        runtime_scene = native.get_active_scene()
        assert runtime_scene.find("ShutdownWitness") is not None
    set_runtime_asset_resolver(lambda guid: "cooked-font.ttf" if guid == "font" else None)
    session.shutdown()
    assert events == [("disable", "cooked-font.ttf"), ("destroy", "cooked-font.ttf")]
    assert retired_scene_counts == [0]
    assert handle.state == "destroyed"
    assert identity not in player_runtime.host._world_runtimes
    assert native.scene_count == authoring_scene_count
    assert scene.find("ShutdownWitness") is not None
    assert resolve_runtime_asset_guid("font") == "cooked-font.ttf"
    session.release_content()
    assert resolve_runtime_asset_guid("font") is None
