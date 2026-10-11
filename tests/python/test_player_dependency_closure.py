"""Dependencies must follow the source bytes actually compiled into the Player."""
from pathlib import Path
import importlib.metadata
import csv
import json
import shutil
import subprocess
import sys

import pytest

from infernux.engine.game_builder import GameBuilder
from infernux.plugins.registry import PluginRegistry
from infernux.engine.player_dependencies import resolve_runtime_dependencies, stage_runtime_dependencies
from infernux.engine.player_package_native import extract_pack, read_manifest
from infernux.engine.player_package_audit import _is_runtime_distribution_metadata_group
from test_game_builder import _make_project, _asset_index_entry, _write_asset_index


def _cook_scripts(tmp_path, *, enabled=True, pin_version=None):
    project = _make_project(tmp_path)
    sources = {
        "Assets/game.py": "import requests\n",
        "Assets/Editor/bake.py": "import unavailable_editor_tool\n",
        "Packages/vendor/service/Runtime/service.py": "import charset_normalizer\n",
        "Packages/vendor/service/Editor/panel.py": "import unavailable_panel_tool\n",
    }
    entries = []
    records = []
    for index, (relative, text) in enumerate(sources.items()):
        source = project / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(text, encoding="utf-8")
        guid = f"script-{index}"
        Path(str(source) + ".meta").write_text(
            json.dumps({"metadata": {"guid": {"type": "string", "value": guid}}}),
            encoding="utf-8",
        )
        entries.append(_asset_index_entry(project, source, guid, "", "Script"))
        if relative.startswith("Packages/"):
            logical = relative.removeprefix("Packages/vendor/service/")
            records.append(dict(logical_path=logical, path_hint=relative, guid=guid,
                                role=logical.split("/")[0].lower(), owned=True))
    _write_asset_index(project, entries)
    registry = PluginRegistry(str(project))
    registry.record_install(
        {"reference": "vendor/service", "name": "Service", "version": "1.0"},
        files=records,
        control=dict(logical_path="inx_package.json",
                     path_hint="Packages/vendor/service/inx_package.json",
                     guid="control", role="control", owned=True),
        python_requirements=(
            ({"name": "charset-normalizer", "requirement": "charset-normalizer>=1"},
             {"name": "unselected-editor-tool", "requirement": "unselected-editor-tool==99"})
            if pin_version else ()
        ),
        python_environment={"charset-normalizer": pin_version, "unselected-editor-tool": "99"},
    )
    if not enabled:
        registry.set_enabled("vendor/service", False)
    output = tmp_path / "output"
    builder = GameBuilder(str(project), str(output))
    builder._cooked_asset_entries = {"script-0": entries[0]}
    staged = output / "Data/Assets/game.py"
    staged.parent.mkdir(parents=True)
    shutil.copy2(project / "Assets/game.py", staged)
    builder._stage_player_plugins(str(output / "Data"))
    builder._compile_user_scripts(str(output))
    builder._compile_player_plugin_scripts(str(output))
    return builder, project, output


@pytest.mark.parametrize("enabled", [True, False])
def test_only_cooked_asset_and_enabled_package_imports_are_dependencies(tmp_path, enabled):
    builder, _, output = _cook_scripts(tmp_path, enabled=enabled)
    assert builder._collect_user_dependencies() == (
        ["charset_normalizer", "requests"] if enabled else ["requests"]
    )
    assert (output / "Data/Assets/game.pyc").is_file()
    assert not list((output / "Data").rglob("*.py"))


def test_dependency_collection_does_not_reread_modified_or_deleted_sources(tmp_path):
    builder, project, _ = _cook_scripts(tmp_path)
    (project / "Assets/game.py").write_text("import absent_after_cook\n", encoding="utf-8")
    (project / "Packages/vendor/service/Runtime/service.py").unlink()
    assert builder._collect_user_dependencies() == ["charset_normalizer", "requests"]


def test_dependency_collection_requires_a_completed_source_closure(tmp_path):
    project = _make_project(tmp_path)
    builder = GameBuilder(str(project), str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="before content cook"):
        builder._collect_user_dependencies()


@pytest.mark.parametrize("enabled", [True, False])
def test_only_runtime_selected_plugin_constraints_restrict_player_dependencies(tmp_path, enabled):
    builder, project, output = _cook_scripts(tmp_path, enabled=enabled, pin_version="0.1")
    registry_file = project / "ProjectSettings/InxPlugins.json"
    before = registry_file.read_bytes()
    imports = builder._collect_user_dependencies()
    if enabled:
        with pytest.raises(RuntimeError, match="version conflict: charset-normalizer==0.1"):
            builder._stage_user_dependencies(str(output), imports)
        assert not (output / ".player-dependencies").exists()
    else:
        builder._stage_user_dependencies(str(output), imports)
        assert not any("unselected" in relative for relative in builder._player_dependency_files)
    assert registry_file.read_bytes() == before
    shipped = json.loads((output / "Data/ProjectSettings/InxPlugins.json").read_text())
    assert shipped["python_dependencies"] == []


def _distribution(root, name, files, *, requires=(), entry_points=""):
    inventory = dict(files)
    info = f"{name}-1.0.dist-info"
    inventory[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
        + "".join(f"Requires-Dist: {value}\n" for value in requires)
    )
    if entry_points:
        inventory[f"{info}/entry_points.txt"] = entry_points
    # Installed metadata may contain a private build machine's local URL.
    inventory[f"{info}/direct_url.json"] = '{"url":"file:///home/author/private"}'
    for relative, value in inventory.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
    with (root / info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows((relative, "", "") for relative in [*inventory, f"{info}/RECORD"])


def _isolated_import(root, source):
    # -I -S excludes the build environment's site-packages, PYTHONPATH and cwd.
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import sys;sys.path.insert(0,sys.argv[1]);" + source, str(root)],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_real_requests_closure_native_archive_and_isolated_import(tmp_path):
    builder, _, output = _cook_scripts(tmp_path)
    builder._stage_user_dependencies(str(output), builder._collect_user_dependencies())
    builder._pack_core_runtime_archive(str(output))
    archive = output / f"{builder.project_name}_Data/Runtime.inxrt"
    names = {item["path"] for item in read_manifest(archive)["files"]}
    assert {"requests/__init__.pyc", "urllib3/__init__.pyc", "certifi/cacert.pem", "idna/__init__.pyc"} <= names
    assert not any(name.endswith(".py") or "direct_url.json" in name for name in names)
    assert not (output / ".player-dependencies").exists()
    runtime = tmp_path / "runtime"
    extract_pack(archive, runtime)
    result = _isolated_import(runtime,
        "import requests,importlib.metadata as m;"
        "r=requests.Request('GET','https://example.invalid',params={'a':1}).prepare();"
        "assert r.url.endswith('?a=1');"
        "assert m.version('requests')==requests.__version__;"
        "import certifi,ssl;assert ssl.create_default_context(cafile=certifi.where()).get_ca_certs();"
        "print('isolated-requests-ready')")
    assert "isolated-requests-ready" in result


def test_distribution_closure_single_module_extras_cycles_and_entrypoints(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    _distribution(installed, "inx_test_a", {
        "inx_test_a/__init__.py": "import inx_test_single\nVALUE=inx_test_single.VALUE\n",
        "inx_test_a/data.txt": "runtime-data",
    }, requires=("inx-test-single==1.0", 'inx-test-extra==1.0; extra == "feature"',
                 'inx-test-absent; python_version < "2"'),
        entry_points="[infernux.test]\nexample = inx_test_a:VALUE\n")
    _distribution(installed, "inx_test_single", {"inx_test_single.py": "VALUE=42\n"},
                  requires=("inx-test-a==1.0",))
    _distribution(installed, "inx_test_extra", {"inx_test_extra/__init__.py": "VALUE=99\n"})
    monkeypatch.syspath_prepend(str(installed))
    dependencies = resolve_runtime_dependencies(
        ["inx_test_a"], requirements=["inx-test-a[feature]==1.0"],
        constraints=["unselected-editor-tool==99"],
    )
    assert [item.name for item in dependencies] == ["inx-test-a", "inx-test-extra", "inx-test-single"]
    output = tmp_path / "runtime"
    written = stage_runtime_dependencies(dependencies, output)
    assert "inx_test_single.pyc" in written
    assert not any("direct_url" in relative for relative in written)
    _isolated_import(output,
        "import inx_test_a,inx_test_extra,importlib.metadata as m;"
        "assert inx_test_a.VALUE==42 and inx_test_extra.VALUE==99;"
        "e=m.entry_points(group='infernux.test');assert e['example'].load()==42;"
        "assert all(p.locate().is_file() for p in m.files('inx-test-a'));"
        "assert m.version('inx-test-single')=='1.0'")
    assert (output / "inx_test_a/data.txt").read_text() == "runtime-data"
    # Identical inputs produce identical bytecode and portable co_filename.
    second = tmp_path / "runtime2"
    assert stage_runtime_dependencies(dependencies, second) == written
    assert all((output / relative).read_bytes() == (second / relative).read_bytes() for relative in written)


@pytest.mark.parametrize("failure", ["missing", "version", "site-hook", "inventory", "jit"])
def test_dependency_errors_are_explicit_before_staging(tmp_path, monkeypatch, failure):
    installed = tmp_path / "installed"
    files = {"inx_bad.py": "VALUE=1\n"}
    requires = {"missing": ["inx-not-installed"], "version": ["inx-bad==2"],
                "jit": ["numba"]}.get(failure, ())
    if failure == "site-hook":
        files["inx_bad.pth"] = "outside-source\n"
    _distribution(installed, "inx_bad", files, requires=requires)
    if failure == "inventory":
        (installed / "inx_bad.py").unlink()
    monkeypatch.syspath_prepend(str(installed))
    with pytest.raises(RuntimeError, match={
        "missing": "not installed", "version": "version conflict", "site-hook": "site initialization",
        "inventory": "inventory file is missing", "jit": "disabled CPU JIT",
    }[failure]):
        resolve_runtime_dependencies([], requirements=["inx-bad"], forbidden=["numba", "llvmlite"])


@pytest.mark.parametrize("filename", ["WHEEL", "top_level.txt", "entry_points.txt"])
def test_separate_distribution_metadata_has_separate_runtime_ownership(filename):
    assert _is_runtime_distribution_metadata_group([
        f"Game_Data/Runtime.inxrt::first-1.0.dist-info/{filename}",
        f"Game_Data/Runtime.inxrt::second-1.0.dist-info/{filename}",
    ])


@pytest.mark.parametrize("paths", [
    ["Game_Data/Runtime.inxrt::first-1.0.dist-info/native.pyd",
     "Game_Data/Runtime.inxrt::second-1.0.dist-info/native.pyd"],
    ["Game_Data/Runtime.inxrt::first-1.0.dist-info/native.pyc",
     "Game_Data/Runtime.inxrt::second-1.0.dist-info/native.pyc"],
    ["Game_Data/Runtime.inxrt::first/WHEEL", "Game_Data/Runtime.inxrt::second/WHEEL"],
    ["Game_Data/Content.inxpkg::first-1.0.dist-info/WHEEL",
     "Game_Data/Content.inxpkg::second-1.0.dist-info/WHEEL"],
])
def test_distribution_metadata_rule_does_not_hide_other_duplicate_payloads(paths):
    assert not _is_runtime_distribution_metadata_group(paths)
