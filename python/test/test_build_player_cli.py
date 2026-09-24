from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "acceptance" / "build_player.py"


def _module():
    spec = importlib.util.spec_from_file_location("infernux_build_player_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_option_uses_json_values():
    module = _module()

    assert module._parse_option('android_artifact="apk"') == (
        "android_artifact",
        "apk",
    )
    assert module._parse_option("compress=true") == ("compress", True)
    assert module._parse_option("workers=2") == ("workers", 2)


@pytest.mark.parametrize("value", ["missing-separator", "=true", "value=nope"])
def test_build_option_rejects_invalid_syntax(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _module()._parse_option(value)


def test_invalid_project_fails_closed_and_writes_atomic_evidence(tmp_path):
    module = _module()
    project = tmp_path / "not-a-project"
    output = tmp_path / "output"
    report = tmp_path / "evidence" / "build.json"

    status = module.main(
        [str(project), "web-wasm32", str(output), "--report", str(report)]
    )

    assert status == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema"] == "infernux.build_evidence"
    assert payload["status"] == "invalid-project"
    assert payload["target"] == "web-wasm32"
    assert payload["diagnostics"][0]["code"] == "build.project.invalid"
    assert not tuple(report.parent.glob(".*.tmp"))


def test_exporter_mapping_covers_plugin_targets():
    module = _module()

    assert set(module.EXPORTERS) == {
        "android-arm64",
        "android-x64-emulator",
        "web-wasm32",
        module.DESKTOP_TARGET,
    }
    assert all(path.is_dir() for path in module.PLUGIN_EDITORS.values())


def test_supported_targets_include_the_current_desktop_host():
    module = _module()

    assert module.DESKTOP_TARGET in {"windows-x64", "linux-x64"}
    assert module.DESKTOP_TARGET in module.SUPPORTED_TARGETS
    assert set(module.SUPPORTED_TARGETS) == set(module.EXPORTERS) | {
        module.DESKTOP_TARGET
    }


def test_desktop_target_loads_the_core_exporter():
    module = _module()
    python_root = str(ROOT / "python")
    if python_root not in module.sys.path:
        module.sys.path.insert(0, python_root)

    exporter = module._load_exporter(module.DESKTOP_TARGET)

    assert exporter.exporter_id == (
        "infernux/platform-windows"
        if module.DESKTOP_TARGET == "windows-x64"
        else "infernux/platform-linux"
    )
    assert [target.id for target in exporter.targets()] == [module.DESKTOP_TARGET]


def test_installed_acceptance_rejects_the_source_checkout():
    module = _module()
    with pytest.raises(RuntimeError, match="installed wheel"):
        module._prepare_engine(installed=True)


def test_installed_acceptance_is_an_explicit_cli_mode():
    arguments = _module()._parser().parse_args(["project", "web-wasm32", "output", "--installed"])
    assert arguments.installed is True


def test_source_acceptance_can_select_an_out_of_source_plugin_editor_root(
    tmp_path, monkeypatch
):
    module = _module()
    editor_root = tmp_path / "work-package" / "editor"
    package_root = editor_root / "infernux_web"
    package_root.mkdir(parents=True)
    module_file = package_root / "__init__.py"
    module_file.write_text("", encoding="utf-8")

    class FakeExporter:
        pass

    imported = SimpleNamespace(
        __file__=str(module_file),
        WebPlatformExporter=FakeExporter,
    )
    stale = SimpleNamespace(__file__=str(tmp_path / "source" / "__init__.py"))
    monkeypatch.setitem(module.sys.modules, "infernux_web", stale)
    monkeypatch.setitem(module.sys.modules, "infernux_web.doctor", stale)

    def _import(name):
        assert "infernux_web" not in module.sys.modules
        assert "infernux_web.doctor" not in module.sys.modules
        return imported

    monkeypatch.setattr(module.importlib, "import_module", _import)

    exporter = module._load_exporter(
        "web-wasm32", editor_root_override=editor_root
    )

    assert isinstance(exporter, FakeExporter)
    assert module.sys.path[0] == str(editor_root.resolve())


def test_source_plugin_editor_override_rejects_an_import_from_another_root(
    tmp_path, monkeypatch
):
    module = _module()
    editor_root = tmp_path / "selected" / "editor"
    (editor_root / "infernux_web").mkdir(parents=True)
    escaped = tmp_path / "other" / "infernux_web" / "__init__.py"
    escaped.parent.mkdir(parents=True)
    escaped.write_text("", encoding="utf-8")
    imported = SimpleNamespace(
        __file__=str(escaped),
        WebPlatformExporter=object,
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: imported)

    with pytest.raises(RuntimeError, match="escaped the selected editor root"):
        module._load_exporter("web-wasm32", editor_root_override=editor_root)


def test_installed_preload_has_project_paths_and_mirrored_resources(tmp_path, monkeypatch):
    from Infernux.application import Application
    from Infernux.engine import library_sync, project_context
    from Infernux.engine.build import exporter_registry
    from Infernux.plugins import PluginManager

    asset = tmp_path / "Packages/probe/runtime/message.txt"
    asset.parent.mkdir(parents=True)
    asset.write_text("package preload", encoding="utf-8")
    calls = []
    monkeypatch.setattr(project_context, "_project_root", None)
    monkeypatch.setattr(project_context, "_runtime_asset_resolver", None)
    monkeypatch.setattr(library_sync, "sync_resources", lambda root: calls.append(root))

    def startup(root, *, runtime):
        assert runtime is True
        assert calls == [root]
        assert Path(Application.package_path("probe", "runtime/message.txt")) == asset

    monkeypatch.setattr(PluginManager, "startup", startup)
    assert _module()._installed_exporter_registry(tmp_path) is exporter_registry


def test_prepare_project_registry_imports_runtime_package_scripts_only(
    tmp_path, monkeypatch
):
    """Player Cook must filter package roles before importing Python files."""
    module = _module()
    package = tmp_path / "Packages" / "probe"
    runtime = package / "runtime"
    editor = package / "editor"
    runtime.mkdir(parents=True)
    editor.mkdir()
    (tmp_path / "ProjectSettings").mkdir()
    (package / "inx_package.json").write_text(
        json.dumps({"reference": "probe", "name": "probe", "engine": "*"}),
        encoding="utf-8",
    )

    runtime_marker = tmp_path / "runtime-imported.txt"
    editor_marker = tmp_path / "editor-imported.txt"
    runtime_script = runtime / "runtime_component.py"
    editor_script = editor / "editor_component.py"
    script_template = (
        "from pathlib import Path\n"
        "import Infernux as inx\n"
        "Path({marker!r}).write_text('imported', encoding='utf-8')\n"
        "class {class_name}(inx.InxComponent):\n"
        "    pass\n"
    )
    runtime_script.write_text(
        script_template.format(marker=str(runtime_marker), class_name="RuntimeComponent"),
        encoding="utf-8",
    )
    editor_script.write_text(
        script_template.format(marker=str(editor_marker), class_name="EditorComponent"),
        encoding="utf-8",
    )
    for index, script in enumerate((runtime_script, editor_script), start=1):
        (script.with_name(script.name + ".meta")).write_text(
            json.dumps(
                {"metadata": {"guid": {"type": "string", "value": f"{index:032x}"}}}
            ),
            encoding="utf-8",
        )

    startup_calls = []
    from Infernux.components import registry as component_registry
    from Infernux.engine import library_sync, project_context
    from Infernux.plugins import PluginManager

    monkeypatch.setattr(library_sync, "sync_resources", lambda _root: None)
    monkeypatch.setattr(project_context, "_project_root", None)
    monkeypatch.setattr(
        PluginManager,
        "startup",
        lambda _root, *, runtime: startup_calls.append(runtime),
    )
    monkeypatch.setattr(
        component_registry,
        "publish_component_script_types",
        lambda _path, _types: (),
    )

    module._prepare_project_registry(tmp_path)

    assert startup_calls == [True]
    assert runtime_marker.is_file()
    assert not editor_marker.exists()


def test_installed_registry_publishes_project_data_asset_types_before_cook(
    tmp_path, monkeypatch
):
    """The wheel-only build must decode authored DataAssets before staging them."""
    module = _module()
    script = tmp_path / "Assets" / "BuildConfig.py"
    script.parent.mkdir(parents=True)
    (tmp_path / "ProjectSettings").mkdir()
    script.write_text(
        textwrap.dedent(
            """
            import Infernux as inx

            class BuildConfig(inx.DataAsset):
                __serialized_type_id__ = "tests.cli.build_config"
                value = inx.serialized_field(1)

            class BuildProbe(inx.InxComponent):
                pass
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (script.with_name(script.name + ".meta")).write_text(
        json.dumps({"metadata": {"guid": {"type": "string", "value": "b" * 32}}}),
        encoding="utf-8",
    )

    from Infernux.components import registry as component_registry
    from Infernux.components.serializable_object import get_serializable_class
    from Infernux.engine import library_sync, project_context
    from Infernux.plugins import PluginManager

    monkeypatch.setattr(library_sync, "sync_resources", lambda _root: None)
    monkeypatch.setattr(project_context, "_project_root", None)
    monkeypatch.setattr(PluginManager, "startup", lambda _root, *, runtime: None)
    monkeypatch.setattr(
        component_registry,
        "publish_component_script_types",
        lambda _path, _types: (),
    )

    module._prepare_project_registry(tmp_path)

    assert get_serializable_class("tests.cli.build_config") is not None


def test_raw_build_tool_output_is_not_retained_as_phase_progress():
    module = _module()

    assert module._is_verbose_progress(
        SimpleNamespace(detail={"source": "cmake"})
    )
    assert module._is_verbose_progress(
        SimpleNamespace(detail={"source": "gradle"})
    )
    assert not module._is_verbose_progress(SimpleNamespace(detail={}))
