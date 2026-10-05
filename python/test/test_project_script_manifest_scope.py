"""Cooked script identity follows the active project and scoped build context."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import py_compile

import pytest

from infernux.engine import project_context as context
from infernux.engine.component_restore import resolve_script_from_guid


GUID = "aabbccddaabbccddaabbccddaabbccdd"


@pytest.fixture
def projects(tmp_path, monkeypatch):
    for name in ("_project_root", "_guid_manifest", "_runtime_asset_resolver",
                 "_runtime_package_resolver", "_runtime_asset_query", "_runtime_asset_extension_resolver"):
        monkeypatch.setattr(context, name, None)
    monkeypatch.setattr(context, "_guid_manifest_loaded", False)
    result = []
    for name, script_name in (("第一位作者", "Original"), ("另一台设备", "Renamed"), ("构建项目", "Third")):
        root = tmp_path / name
        script = root / f"Assets/{script_name}.py"
        script.parent.mkdir(parents=True)
        script.write_text(f"PROJECT_MARKER = {name!r}\n", encoding="utf-8")
        compiled = script.with_suffix(".pyc")
        py_compile.compile(str(script), cfile=str(compiled), dfile=f"Assets/{script_name}.py", doraise=True)
        script.unlink()
        manifest = root / "_script_guid_map.json"
        manifest.write_text(json.dumps({GUID: compiled.relative_to(root).as_posix()}), encoding="utf-8")
        result.append((root, compiled, name))
    return result


def _assert_script(project, database=None, *, guid=GUID):
    root, expected, marker = project
    resolved = resolve_script_from_guid(guid, database)
    assert resolved is not None and Path(resolved) == expected
    loader = importlib.machinery.SourcelessFileLoader("scope_probe", resolved)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    assert module.PROJECT_MARKER == marker
    assert Path(context.get_project_root()) == root


@pytest.mark.parametrize("sequence", ["fresh", "unbound", "switch", "unbind", "missing_manifest", "unknown_guid"])
def test_direct_project_binding_owns_cooked_script_map(projects, sequence):
    first, second, third = projects
    if sequence == "unbound":
        assert resolve_script_from_guid(GUID) is None
    if sequence in {"switch", "unbind"}:
        context.set_project_root(str(first[0]))
        _assert_script(first)
    if sequence == "unbind":
        context.set_project_root(None)
        assert resolve_script_from_guid(GUID) is None
    if sequence == "missing_manifest":
        (third[0] / "_script_guid_map.json").unlink()
        context.set_project_root(str(third[0]))
        assert resolve_script_from_guid(GUID) is None
    context.set_project_root(str(second[0]))
    if sequence == "unknown_guid":
        assert resolve_script_from_guid("absent") is None
    _assert_script(second)
    context.set_project_root(str(first[0]))
    _assert_script(first)


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("scope", ["other", "unbound", "nested", "exception"])
def test_temporary_project_scope_restores_its_complete_script_map(projects, loaded, scope):
    first, second, third = projects
    context.set_project_root(str(first[0]))
    if loaded:
        _assert_script(first)
    caught = False
    try:
        with context.using_project_root(None if scope == "unbound" else str(second[0])):
            if scope == "unbound":
                assert resolve_script_from_guid(GUID) is None
            else:
                _assert_script(second)
            if scope == "nested":
                with context.using_project_root(str(third[0])):
                    _assert_script(third)
                _assert_script(second)
            if scope == "exception":
                raise ValueError("build scope ended")
    except ValueError as error:
        assert str(error) == "build scope ended"
        caught = True
    assert caught == (scope == "exception")
    _assert_script(first)


def test_returning_scope_retains_previous_immutable_map_without_reading_it_again(projects):
    first, second, _ = projects
    context.set_project_root(str(first[0]))
    _assert_script(first)
    manifest = first[0] / "_script_guid_map.json"
    # Removing the already loaded file witnesses retention of the outer
    # immutable build catalog, rather than reloading it when a scope exits.
    manifest.unlink()
    with context.using_project_root(str(second[0])):
        _assert_script(second)
    _assert_script(first)


def test_explicit_rebinding_same_root_starts_a_new_manifest_lifetime(projects):
    first, second, _ = projects
    context.set_project_root(str(first[0]))
    _assert_script(first)
    new_script = first[0] / "Assets/Replacement.pyc"
    new_script.write_bytes(second[1].read_bytes())
    (first[0] / "_script_guid_map.json").write_text(
        json.dumps({GUID: "Assets/Replacement.pyc"}), encoding="utf-8")
    context.set_project_root(str(first[0]))
    _assert_script((first[0], new_script, second[2]))


def test_asset_database_mapping_remains_authoritative(projects):
    first, second, _ = projects
    context.set_project_root(str(first[0]))
    _assert_script(first)
    context.set_project_root(str(second[0]))

    class Database:
        def get_path_from_guid(self, guid):
            assert guid == GUID
            return str(second[1])

    _assert_script(second, Database())


def test_native_cook_manifests_follow_a_guid_preserving_script_rename(tmp_path):
    import os
    import subprocess
    import sys
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def exercise_native_cook(project):
    from infernux.core import AssetManager
    from infernux.engine.engine import Engine
    from infernux.engine.game_builder import GameBuilder
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.runtime_artifact_catalog import load_asset_index
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        scene_path = project / "Assets/Main.scene"
        scene_path.write_text(json.dumps(SceneManager.instance().get_active_scene().serialize_asset_document()),
                              encoding="utf-8")
        scene = database.import_asset(str(scene_path))
        assert scene, scene.error
        (project / "ProjectSettings/BuildSettings.json").write_text(
            json.dumps({"scene_guids": [scene.guid]}), encoding="utf-8")
        source = project / "Assets/Original.py"
        source.write_text("PROJECT_MARKER = 'original'\n", encoding="utf-8")
        imported = database.import_asset(str(source))
        assert imported, imported.error
        guid = imported.guid
        cooked = []
        for revision in ("original", "renamed"):
            if revision == "renamed":
                renamed = source.with_name("Renamed.py")
                source.rename(renamed)
                moved = AssetManager.move_asset(str(source), str(renamed), database=database)
                assert moved, moved.error
                source = renamed
                source.write_text("PROJECT_MARKER = 'renamed'\n", encoding="utf-8")
                assert database.import_asset(str(source))
                assert database.get_guid_from_path(str(source)) == guid
            database.flush_derived_index()
            output = project / "Builds" / revision
            builder = GameBuilder(str(project), str(output), game_name="ManifestScope")
            builder.freeze_asset_index_entries(load_asset_index(project))
            builder._copy_game_data(str(output))
            builder._compile_user_scripts(str(output))
            data = output / "Data"
            manifest = json.loads((data / "_script_guid_map.json").read_text(encoding="utf-8"))
            expected = f"Assets/{source.stem}.pyc"
            assert manifest[guid] == expected
            assert not (data / "Assets" / source.name).exists()
            cooked.append((data, data / expected, revision))
        context.set_project_root(None)
        assert resolve_script_from_guid(guid) is None
        context.set_project_root(str(cooked[0][0]))
        _assert_script(cooked[0], guid=guid)
        with context.using_project_root(str(cooked[1][0])):
            _assert_script(cooked[1], guid=guid)
        _assert_script(cooked[0], guid=guid)
        context.set_project_root(str(cooked[1][0]))
        _assert_script(cooked[1], guid=guid)
        evidence = dict(guid=guid, outputs=[str(item[0]) for item in cooked],
                        expected_scripts=[str(item[1]) for item in cooked],
                        markers=[item[2] for item in cooked], native_cook=True, status="passed")
        (project / "manifest-scope-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence))
    finally:
        engine.exit()


if __name__ == "__main__":
    import sys
    exercise_native_cook(Path(sys.argv[1]))
