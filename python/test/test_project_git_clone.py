from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _git(project, *args):
    result = subprocess.run([
        "git", "-c", "user.name=Clone Test", "-c", "user.email=clone@example.invalid",
        "-c", "commit.gpgsign=false", "-C", str(project), *args,
    ], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_git_clone_preserves_authored_assets_without_private_runtime(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "packaging"))
    project_model = importlib.import_module("model.project_model")
    model = project_model.ProjectModel(None)
    monkeypatch.setattr(model, "_create_project_runtime", lambda *_args, **_kw: None)
    monkeypatch.setattr(model, "_install_infernux_in_runtime", lambda *_args, **_kw: None)
    monkeypatch.setattr(model, "_install_default_libraries", lambda *_args, **_kw: None)
    source = Path(model.init_project_folder("Source", str(tmp_path)))
    assert {path.name for path in (source / "Assets").iterdir()} == {"Scenes", "Settings"}
    assert "path = ." in (source / "Source.ini").read_text(encoding="utf-8")
    package = source / "Packages" / "shared" / "Ships.txt"
    package.parent.mkdir(parents=True)
    package.write_text("authored package\n", encoding="utf-8")
    project_model._write_asset_identity_meta(str(package), "1" * 32, "DefaultText", project_root=str(source))
    for relative in ("Library/cache.bin", ".runtime/python313/python.exe", "Packages/.cache/download.inxpkg"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private state")

    _git(source, "init", "-b", "main")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "initial project")
    clone = tmp_path / "DifferentDevice" / "RenamedProject"
    clone.parent.mkdir()
    _git(source, "clone", "--no-local", str(source), str(clone))
    assert not (clone / ".runtime").exists()
    assert not (clone / "Library").exists()
    assert not (clone / ".vscode").exists()
    assert not (clone / "pyrightconfig.json").exists()
    assert not (clone / "Packages" / ".cache").exists()
    assert (clone / "Packages" / "shared" / "Ships.txt").read_text() == "authored package\n"
    model._create_vscode_workspace(str(clone))
    settings = json.loads((clone / ".vscode" / "settings.json").read_text(encoding="utf-8"))
    assert all(str(source) not in path for path in settings["python.analysis.extraPaths"])
    assert settings["python.defaultInterpreterPath"].startswith("${workspaceFolder}/")

    assets = {path: path.read_bytes() for directory in (clone / "Assets", clone / "Packages")
              for path in directory.rglob("*") if path.is_file()}
    probe = '''
import json, pathlib, sys
from Infernux.engine.engine import Engine
from Infernux.lib import LogLevel, RuntimeMode
engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
try:
    engine.init_headless(sys.argv[1])
    database = engine.get_asset_database()
    database.refresh()
    for meta in pathlib.Path(sys.argv[1]).rglob("*.meta"):
        if "Library" in meta.parts: continue
        document = json.loads(meta.read_text(encoding="utf-8"))
        assert database.get_guid_from_path(str(meta)[:-5]) == document["metadata"]["guid"]["value"]
finally:
    engine.exit()
'''
    result = subprocess.run([sys.executable, "-c", probe, str(clone)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    changed = [str(path.relative_to(clone)) for path, before in assets.items() if path.read_bytes() != before]
    assert not changed, changed
    assert _git(clone, "status", "--porcelain") == ""


def test_installed_package_preloads_on_a_new_device_without_the_authors_cache(tmp_path, monkeypatch):
    from Infernux.plugins import InxPackage

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "packaging"))
    project_model = importlib.import_module("model.project_model")
    model = project_model.ProjectModel(None)
    for method in ("_create_project_runtime", "_install_infernux_in_runtime", "_install_default_libraries"):
        monkeypatch.setattr(model, method, lambda *_args, **_kwargs: None)
    project = Path(model.init_project_folder("PackageProject", str(tmp_path)))
    package_source = tmp_path / "PackageSource"
    runtime = package_source / "runtime"
    runtime.mkdir(parents=True)
    (package_source / "inx_package.json").write_text(
        '{"reference":"team/longships","version":"1.0.0"}\n', encoding="utf-8",
    )
    (runtime / "lifecycle.py").write_text(
        "from pathlib import Path\n"
        "from Infernux.lifecycle import InxPreload\n"
        "class LongshipsPreload(InxPreload):\n"
        "    def preload(self, context):\n"
        "        Path(context.project_root, 'Library', 'longships-ready.txt').write_text('ready')\n",
        encoding="utf-8",
    )
    archive = tmp_path / "longships.inxpkg"
    InxPackage.export_source(str(package_source), str(archive))
    probe = '''
import pathlib, sys
from Infernux.engine.engine import Engine
from Infernux.lib import LogLevel, RuntimeMode
from Infernux.plugins import PluginManager
project, mode = sys.argv[1:3]
engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
manager = None
try:
    engine.init_headless(project)
    database = engine.get_asset_database()
    database.refresh()
    manager = PluginManager(project, engine=engine, runtime=False)
    if mode == 'install':
        state = manager.install_package(sys.argv[3], install_dependencies=False)
        assert state.loaded, state.error
        database.refresh()
    else:
        manager._reconcile_python_requirements_for_startup()
        manager.reload_all()
        state = manager.states['team/longships']
        assert state.loaded, state.error
    assert pathlib.Path(project, 'Library', 'longships-ready.txt').read_text() == 'ready'
finally:
    if manager is not None: manager.shutdown()
    engine.exit()
'''
    for mode in ("install", "open"):
        if mode == "install":
            checkout = project
        else:
            _git(project, "init", "-b", "main")
            _git(project, "add", ".")
            _git(project, "commit", "-m", "shared project with installed package")
            checkout = tmp_path / "DifferentDevice" / "DifferentName"
            checkout.parent.mkdir()
            _git(project, "clone", "--no-local", str(project), str(checkout))
            assert not (checkout / "Library").exists()
            assert not (checkout / ".runtime").exists()
            # A package is installed from its tracked source and ownership ledger.
            # Its publisher's archive and Hub cache are unavailable on this device.
            archive.unlink()
        env = dict(os.environ, INFERNUX_PACKAGE_CACHE_ROOT=str(tmp_path / (mode + "-cache")))
        result = subprocess.run([sys.executable, "-c", probe, str(checkout), mode, str(archive)],
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert _git(checkout, "status", "--porcelain") == ""
