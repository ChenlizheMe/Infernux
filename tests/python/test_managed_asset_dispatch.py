"""Public managed loading dispatch with real source and binary artifact files."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["mesh", "timeline", "timelinefsm", "physicmaterial", "dataasset", "cooked-data", "cooked-mesh"])
def test_managed_asset_loading_entry_points(tmp_path, kind):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), kind],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MANAGED_DISPATCH_OK" in result.stdout


def exercise(project, kind):
    import json
    import shutil
    from infernux.components import serialized_field
    from infernux.core.assets import AssetManager
    from infernux.core.asset_ref import create_asset_ref
    from infernux.core.mesh import Mesh
    from infernux.core.animation_timeline import AnimationTimeline
    from infernux.core.anim_state_machine import AnimStateMachine
    from infernux.core.physic_material import PhysicMaterial
    from infernux.core.data_asset import DataAsset, encode_data_asset_artifact
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import RuntimeMode, LogLevel

    class PuzzleSettings(DataAsset):
        __serialized_type_id__ = "repair.managed_dispatch.settings"
        count: int = serialized_field(default=7)

    assets = project / "Assets/关卡"
    assets.mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        if kind in {"mesh", "cooked-mesh"}:
            path = assets / "Triangle.obj"
            path.write_text('o Triangle\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding="utf-8")
            source = AssetManager.import_asset(str(path))
            assert source and source.guid
            if kind == "cooked-mesh":
                artifact = project / f"Library/Artifacts/Mesh/{source.guid}.inxmesh"
                assert artifact.is_file()
                cooked = project / "cooked"
                (cooked / "Assets").mkdir(parents=True)
                (cooked / "ProjectSettings").mkdir()
                shutil.copytree(project / "Library/Artifacts", cooked / "Library/Artifacts")
                metadata = AssetManager.require_asset_database().get_meta_by_guid(source.guid)
                catalog = {"$schema": "infernux.runtime_asset_records", "entries": [{
                    "guid": source.guid,
                    "runtime_path": artifact.relative_to(project).as_posix(),
                    # Runtime catalogs retain import fingerprints used by the
                    # binary decoder; authored portable sidecars omit them.
                    "metadata": metadata.serialize_document(),
                }]}
                (cooked / "Library/RuntimeAssetRecords.json").write_text(
                    json.dumps(catalog), encoding="utf-8")
                # A fresh process installs the actual native runtime catalog.
                # Imported cache artifacts are not authored .inxmesh sources.
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
                     str(cooked), "verify-cooked-mesh", source.guid],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                return
            model, ref_type, hint = Mesh, "Mesh", Mesh
        elif kind in {"dataasset", "cooked-data"}:
            path = assets / "Settings.inxdata"
            PuzzleSettings().save_to(str(path))
            if kind == "cooked-data":
                document = json.loads(path.read_bytes())
                path = assets / "Settings.inxasset"
                path.write_bytes(encode_data_asset_artifact(document))
            model, ref_type, hint = PuzzleSettings, "DataAsset", DataAsset
        else:
            model, suffix, ref_type = {
                "timeline": (AnimationTimeline, ".animtimeline", "AnimationTimeline"),
                "timelinefsm": (AnimStateMachine, ".timelinefsm", "TimelineFSM"),
                "physicmaterial": (PhysicMaterial, ".physicmaterial", "PhysicMaterial"),
            }[kind]
            path = assets / ("Resource" + suffix)
            value = AnimStateMachine(mode="timeline") if kind == "timelinefsm" else model()
            assert value.save(str(path)) is not False
            hint = model
        handle = AssetManager.import_asset(str(path))
        assert handle and handle.guid
        guid = handle.guid
        relative = path.relative_to(project).as_posix()
        assert isinstance(model.load(str(path)), model)
        # Clear the shared cache between entrances so a previous explicit hint
        # cannot hide an absent default dispatch.
        for name, load in (
            ("managed-path", lambda: AssetManager.load(relative)),
            ("guid", lambda: AssetManager.load_by_guid(guid)),
            ("explicit-type", lambda: AssetManager.load_by_guid(guid, asset_type=hint)),
            ("asset-file", lambda: AssetManager.find_assets(relative)[0].load()),
            ("reference", lambda: create_asset_ref(ref_type, guid=guid).resolve()),
        ):
            AssetManager.flush()
            value = load()
            assert isinstance(value, model), (kind, name, type(value).__name__)
            if isinstance(value, Mesh):
                assert value.vertex_count == 3 and value.index_count == 3
            if isinstance(value, PuzzleSettings):
                assert value.count == 7 and value.guid == guid
    finally:
        engine.exit()


def exercise_cooked_mesh(project, guid):
    from infernux.core.assets import AssetManager, AssetFile
    from infernux.core.asset_ref import create_asset_ref
    from infernux.core.mesh import Mesh
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import RuntimeMode, LogLevel

    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        assert not list((project / "Assets").iterdir())
        for load in (
            lambda: AssetManager.load_by_guid(guid),
            lambda: AssetManager.load_by_guid(guid, asset_type=Mesh),
            lambda: AssetFile(guid).load(),
            lambda: create_asset_ref("Mesh", guid=guid).resolve(),
        ):
            AssetManager.flush()
            value = load()
            assert isinstance(value, Mesh)
            assert value.guid == guid and value.vertex_count == value.index_count == 3
    finally:
        engine.exit()


if __name__ == "__main__":
    if sys.argv[2] == "verify-cooked-mesh":
        exercise_cooked_mesh(Path(sys.argv[1]), sys.argv[3])
    else:
        exercise(Path(sys.argv[1]), sys.argv[2])
    print("MANAGED_DISPATCH_OK")
