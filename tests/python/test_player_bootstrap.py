from __future__ import annotations

import sys
import types
import ast
from pathlib import Path

import pytest


def _runtime_contract(tmp_path):
    from infernux.engine.player_service_graph import (
        PlayerRuntimeAssetCatalog,
        RuntimeFeatureSet,
        RuntimeFlavor,
        RuntimeProductManifest,
        player_manifest_service_section,
        runtime_policy_for,
    )

    flavor = RuntimeFlavor.PLAYER_RELEASE
    features = RuntimeFeatureSet()
    document = {
        "$schema": "infernux.player_runtime_manifest",
        "product": {"flavor": flavor.value},
        "features": features.to_manifest(),
        "runtime_policy": runtime_policy_for(flavor).to_manifest(),
        "services": player_manifest_service_section(flavor, features),
    }
    return (
        RuntimeProductManifest.from_document(document),
        PlayerRuntimeAssetCatalog.from_documents(
            str(tmp_path),
            {"artifacts": []},
            {"entries": []},
        ),
    )


def _stub_engine_status(monkeypatch):
    module = types.ModuleType("infernux.engine.ui.engine_status")

    class EngineStatus:
        @classmethod
        def set(cls, *_args, **_kwargs):
            pass

        @classmethod
        def clear(cls, *_args, **_kwargs):
            pass

    module.EngineStatus = EngineStatus
    monkeypatch.setitem(sys.modules, "infernux.engine.ui.engine_status", module)


@pytest.mark.parametrize("host", ["desktop", "web"])
def test_plugin_preload_resolves_cooked_assets_before_scene_startup(monkeypatch, tmp_path, host):
    from infernux.application import Application
    from infernux.engine.player_bootstrap import PlayerBootstrap
    from infernux.engine.player_runtime import PlayerRuntimeSession
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
    from infernux.engine.project_context import set_project_root
    from infernux.plugins import PluginManager

    cooked = tmp_path / "Library/Artifacts/Blob/preload-guid.txt"
    cooked.parent.mkdir(parents=True)
    cooked.write_text("cooked preload resource", encoding="utf-8")
    # A loose file without a frozen binding must not become a Player asset.
    decoy = tmp_path / "Assets/Data/not-exported.txt"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("not exported", encoding="utf-8")
    manifest, _ = _runtime_contract(tmp_path)
    catalog = PlayerRuntimeAssetCatalog.from_documents(
        str(tmp_path),
        {
            "artifacts": [
                {
                    "runtime_artifact_id": "content:preload-guid",
                    "runtime_path": "Library/Artifacts/Blob/preload-guid.txt",
                    "asset_guid": "preload-guid",
                    "dependencies": [],
                }
            ]
        },
        {
            "entries": [
                {
                    "guid": "preload-guid",
                    "runtime_path": "Assets/Data/preload.txt",
                    "primary_runtime_artifact_id": "content:preload-guid",
                    "runtime_artifact_ids": ["content:preload-guid"],
                }
            ]
        },
    )
    from infernux.engine import player_service_graph

    assert catalog.query_asset_guids(r"assets\Data\preload.txt", exact=True) == ("preload-guid",)
    assert catalog.query_asset_guids("preload-guid", exact=True) == ("preload-guid",)
    alternate_case = ("preload-guid",) if player_service_graph.os.name == "nt" else ()
    assert catalog.query_asset_guids("Assets/data/PRELOAD.TXT", exact=True) == alternate_case
    assert catalog.query_asset_guids("Assets/data/PRELOAD.TXT") == alternate_case
    assert catalog.query_asset_guids("Assets/data") == alternate_case
    assert catalog.query_asset_guids("assets/DATA/*.TXT") == ("preload-guid",)

    resolutions = []
    resolve = player_service_graph.resolved_path
    with monkeypatch.context() as observe:
        observe.setattr(player_service_graph, "resolved_path", lambda path: resolutions.append(path) or resolve(path))
        assert catalog.resolve_guid("preload-guid") == str(cooked.resolve())
    assert len(resolutions) == 1
    session = PlayerRuntimeSession(
        asset_database=object(),
        scheduler=types.SimpleNamespace(),
        scene_service=types.SimpleNamespace(bind_runtime_catalog=lambda _catalog: None),
    )

    class PreloadObserved(Exception):
        pass

    def preload(*_args, **kwargs):
        assert kwargs["runtime"] is True
        assert not (tmp_path / "Assets/Data/preload.txt").exists()
        assert (
            Path(Application.asset_path("Assets/Data/preload.txt")).read_text(encoding="utf-8")
            == "cooked preload resource"
        )
        with pytest.raises(FileNotFoundError):
            Application.asset_path("Assets/Data/not-exported.txt")
        raise PreloadObserved

    monkeypatch.setattr(PluginManager, "startup", preload)
    set_project_root(str(tmp_path))
    try:
        if host == "desktop":
            bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
            bootstrap.engine = types.SimpleNamespace(get_player_runtime=lambda: session)
            bootstrap.project_path = str(tmp_path)
            bootstrap._runtime_manifest = manifest
            bootstrap._runtime_catalog = catalog
            for name in (
                "_force_player_mode",
                "_load_runtime_contract",
                "_init_engine",
                "_pump_startup_events",
                "_load_runtime_asset_catalog",
                "_prewarm_builtin_pipelines",
            ):
                monkeypatch.setattr(bootstrap, name, lambda: None)
            run = bootstrap.run
        else:
            template = Path(__file__).resolve().parents[2] / ("external/plugins/infernux_web/native/bootstrap.py")
            tree = ast.parse(template.read_text(encoding="utf-8"))
            function = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_prepare_player_runtime"
            )
            namespace = {
                "_player_session": None,
                "_prepare_player_asset_contract": lambda: None,
                "_runtime_data_root": str(tmp_path),
                "_player_asset_database": session.get_asset_database(),
                "_player_runtime_manifest": manifest,
                "_player_runtime_catalog": catalog,
            }
            monkeypatch.setattr(
                "infernux.engine.player_runtime.PlayerRuntimeSession",
                lambda **_kwargs: session,
            )
            exec(
                compile(ast.Module(body=[function], type_ignores=[]), str(template), "exec"),
                namespace,
            )
            run = namespace["_prepare_player_runtime"]
        with pytest.raises(PreloadObserved):
            run()
    finally:
        from infernux.core.assets import AssetManager

        AssetManager.release_engine()
        Application._unbind_engine(session)
        set_project_root(None)


def test_player_activates_initial_scene_without_editor_deferred_tasks(monkeypatch):
    from infernux.engine.player_bootstrap import PlayerBootstrap

    _stub_engine_status(monkeypatch)
    activated = []
    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap._activate_initial_scene_for_play = lambda: activated.append(True) or True

    bootstrap._enter_play_mode()
    assert activated == [True]


def test_player_runtime_session_does_not_construct_editor_managers(tmp_path):
    from infernux.engine.player_bootstrap import PlayerBootstrap

    class RuntimeSession:
        def configure_runtime_contract(self, _manifest, _catalog):
            return None

    class Engine:
        def get_player_runtime(self):
            return RuntimeSession()

    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap.engine = Engine()
    bootstrap.runtime_session = None
    bootstrap._runtime_manifest, bootstrap._runtime_catalog = _runtime_contract(tmp_path)
    bootstrap._create_managers()

    assert bootstrap.runtime_session is not None
    assert getattr(bootstrap, "scene_file_manager", None) is None


def test_player_run_prewarms_before_revealing_a_no_splash_scene():
    from infernux.engine.player_bootstrap import PlayerBootstrap

    calls = []

    class Engine:
        def prepare_startup_refresh(self):
            calls.append("prepare")

    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap.engine = Engine()
    bootstrap._force_player_mode = lambda: calls.append("force")
    bootstrap._load_runtime_contract = lambda: calls.append("contract")
    bootstrap._init_engine = lambda: calls.append("engine")
    bootstrap._prewarm_builtin_pipelines = lambda: calls.append("builtin_gpu")
    bootstrap._load_runtime_asset_catalog = lambda: calls.append("catalog")
    bootstrap._create_managers = lambda: calls.append("managers")
    bootstrap._load_plugins = lambda: calls.append("plugins")
    bootstrap._schedule_project_warmup = lambda: calls.append("project_warmup")
    bootstrap._finish_project_warmup = lambda: calls.append("finish_warmup")
    bootstrap._setup_game_camera = lambda: calls.append("camera")
    bootstrap._register_player_gui = lambda: calls.append("gui")
    bootstrap._load_initial_scene = lambda: calls.append("scene")
    bootstrap._enter_play_mode = lambda: calls.append("play")
    bootstrap._prewarm_first_frame = lambda: calls.append("first_gpu_frame")
    bootstrap._pump_startup_events = lambda: calls.append("pump")
    bootstrap.runtime_session = object()
    bootstrap.splash_items = []

    bootstrap.run()

    assert calls == [
        "force",
        "contract",
        "engine",
        "builtin_gpu",
        "catalog",
        "pump",
        "managers",
        "pump",
        "plugins",
        "project_warmup",
        "pump",
        "camera",
        "gui",
        "pump",
        "scene",
        "prepare",
        "pump",
        "play",
        "pump",
        "first_gpu_frame",
        "pump",
        "finish_warmup",
    ]


def test_player_first_frame_prewarm_builds_target_before_composition():
    from infernux.engine.player_bootstrap import PlayerBootstrap

    deltas = []

    class Engine:
        def tick(self, delta):
            deltas.append(delta)

    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap.engine = Engine()

    bootstrap._prewarm_first_frame()

    assert deltas == [0.0, 0.0]


def test_player_release_policy_rejects_debug_control_environment(monkeypatch, tmp_path):
    from infernux.engine.player_bootstrap import PlayerBootstrap

    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap.project_path = str(tmp_path)
    bootstrap.splash_items = []
    bootstrap._runtime_manifest, bootstrap._runtime_catalog = _runtime_contract(tmp_path)
    monkeypatch.setenv("_INFERNUX_PLAYER_DEBUG_BUILD", "0")
    monkeypatch.setenv("_INFERNUX_PLAYER_CONTROL_FILE", "commands.json")

    with pytest.raises(RuntimeError, match="cannot enable the debug control"):
        bootstrap._apply_runtime_policy()


def test_player_supervisor_scene_override_resolves_cooked_catalog_artifact(monkeypatch, tmp_path):
    from infernux.engine.player_bootstrap import PlayerBootstrap

    cooked = tmp_path / "Library" / "Artifacts" / "voxel.inxscene"
    cooked.parent.mkdir(parents=True)
    cooked.write_text("{}", encoding="utf-8")

    loaded = []

    class Manifest:
        @staticmethod
        def require_service(service):
            assert service == "player_scene_service"

    class RuntimeSession:
        @staticmethod
        def load_scene(path):
            loaded.append(path)
            return True

    bootstrap = PlayerBootstrap.__new__(PlayerBootstrap)
    bootstrap.project_path = str(tmp_path)
    bootstrap.scene_guids = [
        "start-guid",
        "voxel-guid",
    ]
    bootstrap._runtime_manifest = Manifest()
    bootstrap.runtime_session = RuntimeSession()
    bootstrap._resolve_runtime_scene = lambda reference: str(cooked) if reference == "voxel-guid" else None
    monkeypatch.setenv(
        "_INFERNUX_PLAYER_START_SCENE_GUID",
        "voxel-guid",
    )

    bootstrap._load_initial_scene()

    assert loaded == ["voxel-guid"]


def test_player_build_manifest_accepts_the_build_owned_contract(tmp_path):
    import json

    from infernux.engine import _load_player_build_manifest

    manifest = {
        "game_name": "StrictPlayer",
        "icon_path": "Branding/icon.png",
        "display_mode": "windowed",
        "window_width": 1280,
        "window_height": 720,
        "window_resizable": False,
        "scene_guids": ["main-scene-guid"],
        "splash_items": [{"type": "image", "path": "Splash/intro.png"}],
    }
    (tmp_path / "BuildManifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert _load_player_build_manifest(str(tmp_path)) == manifest


def test_player_init_engine_publishes_window_chrome_before_native_renderer():
    from pathlib import Path

    source = (
        (Path(__file__).resolve().parents[2] / "python")
        .joinpath("infernux", "engine", "player_bootstrap.py")
        .read_text(encoding="utf-8")
    )
    start = source.index("def _init_engine")
    body = source[start : source.index("\n    def ", start + 1)]
    assert body.index("_INFERNUX_PLAYER_FULLSCREEN") < body.index("init_renderer")
    assert body.index("_INFERNUX_PLAYER_WINDOW_RESIZABLE") < body.index("init_renderer")
    assert body.index("_INFERNUX_PLAYER_WINDOW_TITLE") < body.index("init_renderer")
