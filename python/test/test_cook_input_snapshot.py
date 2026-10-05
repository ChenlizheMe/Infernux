"""Native asset import and authored saves exercise the actual Cook input boundary."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", ["assets", "package", "package-unreferenced", "package-disabled",
                                  "package-editor", "snapshot-unchanged", "snapshot-saved",
                                  "snapshot-removed", "package-snapshot-saved",
                                  "snapshot-save-during-copy", "package-save-during-copy"])
def test_native_cook_inputs(tmp_path, case):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise(project, case):
    import json
    from infernux.components import FieldType, serialized_field
    from infernux.core import DataAsset, DataAssetRef
    from infernux.core.data_asset import decode_data_asset_artifact
    from infernux.engine.engine import Engine
    from infernux.engine.game_builder import GameBuilder
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.runtime_artifact_catalog import load_asset_index
    from infernux.lib import AssetDependencyGraph, LogLevel, RuntimeMode, SceneManager

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    package = project / "Packages/test/closure"
    (package / "runtime").mkdir(parents=True)
    (package / "editor").mkdir()
    (package / "inx_package.json").write_text(
        json.dumps({"reference": "test/closure", "name": "Closure", "version": "1.0.0"}), encoding="utf-8")
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()

        class Payload(DataAsset):
            __serialized_type_id__ = "infernux.test.cook.input.payload"
            value: int = serialized_field(default=11)

        class Link(DataAsset):
            __serialized_type_id__ = "infernux.test.cook.input.link"
            target: DataAssetRef = serialized_field(default=DataAssetRef(), field_type=FieldType.ASSET,
                                                     asset_type="DataAsset")

        scene_path = project / "Assets/Main.scene"
        scene_path.write_text(json.dumps(SceneManager.instance().get_active_scene().serialize_asset_document()),
                              encoding="utf-8")
        imported = database.import_asset(str(scene_path))
        assert imported, imported.error
        (project / "ProjectSettings/BuildSettings.json").write_text(
            json.dumps({"scene_guids": [imported.guid]}), encoding="utf-8")
        package_target = case.startswith("package")
        target_path = ((package / ("editor" if case == "package-editor" else "runtime") / "Target.inxdata")
                       if package_target else project / "Assets/Target.inxdata")
        target = Payload(value=11)
        target.save_to(str(target_path), database=database)
        if case == "package-disabled":
            from infernux.plugins.package import package_control_guid
            from infernux.plugins.registry import PluginRegistry
            registry = PluginRegistry(str(project))
            document = registry.load()
            document["installed"] = [{
                "reference": "test/closure", "enabled": False, "files": [],
                "control": {"guid": package_control_guid("test/closure"), "owned": False},
            }]
            registry.save(document)
        linked = case != "package-unreferenced"
        owner = Link(target=DataAssetRef(guid=target.guid) if linked else DataAssetRef())
        owner.save_to(str(project / "Assets/Owner.inxdata"), database=database)
        assert set(AssetDependencyGraph.instance().get_dependencies(owner.guid)) == ({target.guid} if linked else set())
        database.flush_derived_index()
        entries = load_asset_index(project)
        target_entry = next(entry for entry in entries if entry["guid"] == target.guid)
        output = project / "Build"
        data = output / "Data"
        builder = GameBuilder(str(project), str(output), game_name="CookInputs")
        builder.freeze_asset_index_entries(entries)
        if case.endswith("save-during-copy"):
            import shutil
            from infernux.engine.runtime_artifact_catalog import RuntimeArtifactError
            copy = shutil.copy2
            saved = False

            def save_during_capture(source, destination, *args, **kwargs):
                nonlocal saved
                if Path(source) == target_path and not saved:
                    saved = True
                    target.value = 22
                    target.save_to(str(target_path), database=database)
                return copy(source, destination, *args, **kwargs)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(shutil, "copy2", save_during_capture)
                with pytest.raises((RuntimeError, RuntimeArtifactError), match="stale"):
                    builder._copy_game_data(str(output))
            assert saved
            assert not builder._runtime_artifact_bindings
            return
        if "snapshot" in case:
            builder._runtime_artifact_bindings = {}
            builder._runtime_artifact_source_paths = set()
            builder._stage_player_plugins(str(data))
            builder._copy_cooked_assets(str(data))
            staged = data / target_path.relative_to(project)
            assert json.loads(staged.read_text(encoding="utf-8"))["fields"]["value"] == 11
            if case.endswith("snapshot-saved"):
                target.value = 22
                target.save_to(str(target_path), database=database)
                assert json.loads(target_path.read_text(encoding="utf-8"))["fields"]["value"] == 22
            elif case == "snapshot-removed":
                target_path.unlink()
            builder._stage_library_runtime_documents(str(data))
        elif case in ("package-disabled", "package-editor"):
            with pytest.raises(RuntimeError, match="not exported by an enabled Runtime package"):
                builder._copy_game_data(str(output))
            return
        else:
            builder._copy_game_data(str(output))
        assert target.guid in builder._cooked_asset_entries
        artifact_path = f"Library/Artifacts/Data/{target.guid}.inxasset"
        decoded = decode_data_asset_artifact((data / artifact_path).read_bytes())
        assert decoded["fields"]["value"] == 11, decoded
        binding = builder._runtime_artifact_bindings[artifact_path]
        assert binding["artifact_source_hash"] == target_entry["content_hash"]
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
