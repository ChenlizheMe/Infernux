"""Real extension modules use process restarts only when their files must change."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ != "__main__":
    from tests.isolated import engine_python_path


def test_native_package_load_edit_update_uninstall_and_external_change(tmp_path):
    environment = {**os.environ, "PYTHONPATH": engine_python_path(),
                   "INFERNUX_DATA_ROOT": str(tmp_path / "User"),
                   "INFERNUX_PACKAGE_CACHE_ROOT": str(tmp_path / "Cache")}
    for phase in ("install", "publish-update", "publish-uninstall"):
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), phase, str(tmp_path)],
            env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
        assert result.returncode == 0, phase + "\n" + result.stdout + result.stderr
        assert "NATIVE_PACKAGE_BOUNDARY_OK " + phase in result.stdout


REFERENCE = "review/native-extension"
SERVICE = "native-package-boundary"


def prepare_pending(project):
    """Use the real pre-import launcher and its publication subprocess."""
    from infernux_editor import prepare
    from infernux_project_lock import reserve, remove_lock
    import uuid

    token = uuid.uuid4().hex
    os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"] = token
    lease = reserve(str(project), os.getpid(), token, "editor", "preparing")
    try:
        assert prepare(str(project))
    finally:
        assert remove_lock(lease, token)


def exercise(phase, workspace):
    import importlib.util
    import json
    import shutil

    project = workspace / "project"
    if phase != "install":
        prepare_pending(project)  # The prior process has released the extension image.
    from infernux.plugins import InxPackage, PluginManager
    from infernux.runtime_services import get_runtime_service
    from infernux_editor import read_pending

    manager = None
    engine = None
    resources = None
    try:
        source = workspace / "source"
        original, current = workspace / "first.inxpkg", workspace / "second.inxpkg"
        installed = project / "Packages" / REFERENCE / "editor/owned_native"
        code = ("from infernux.lifecycle import InxPreload\n"
                "from infernux.runtime_services import install_runtime_service\n"
                "from . import _testcapi\n"
                "class NativePreload(InxPreload):\n"
                "    def preload(self, context):\n"
                "        install_runtime_service('" + SERVICE + "', {{'value': {value}, 'native': _testcapi}})\n")
        if phase == "install":
            (project / "Assets").mkdir(parents=True)
            (project / "ProjectSettings").mkdir()
            (source / "editor/owned_native").mkdir(parents=True)
            (source / "editor/owned_native/__init__.py").write_text("", encoding="utf-8")
            binary_source = Path(importlib.util.find_spec("_testcapi").origin)
            native_name = binary_source.name
            shutil.copy2(binary_source, source / "editor/owned_native" / native_name)
            (workspace / "native-name.txt").write_text(native_name, encoding="utf-8")
            metadata = {"reference": REFERENCE, "version": "1.0.0", "engine": ">=0.4.8"}
            manifest = source / "inx_package.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            (source / "editor/owned_native/lifecycle.py").write_text(code.format(value=1), encoding="utf-8")
            InxPackage.export_source(str(source), str(original))
            manager = PluginManager(str(project))
            state = manager.install_package(str(original), install_dependencies=False)
            assert state.loaded and not state.error and not state.restart_required, state.snapshot()
            native = get_runtime_service(SERVICE)["native"]
            assert get_runtime_service(SERVICE)["value"] == 1
            assert manager.preloads._reload.states[str((installed / native_name).resolve())].status == "active"
            # A delayed notification for unchanged install bytes does not invalidate first load.
            assert not manager.preloads._reload.retire_native_sources((str(installed / native_name),))
            (installed / "lifecycle.py").write_text(code.format(value=2), encoding="utf-8")
            state = manager.reload(REFERENCE)
            assert state.loaded and not state.restart_required and not state.error, state.snapshot()
            assert get_runtime_service(SERVICE)["value"] == 2
            assert get_runtime_service(SERVICE)["native"] is native
            before = (installed / native_name).read_bytes()
            (installed / "User.txt").write_text("user content", encoding="utf-8")
            record = manager.registry.installed_record(REFERENCE)
            Path(manager._installed_archive_path(record)).unlink()
            metadata["version"] = "2.0.0"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            (source / "editor/owned_native/lifecycle.py").write_text(code.format(value=3), encoding="utf-8")
            # A PE/ELF overlay changes the actual payload while retaining a valid extension.
            (source / "editor/owned_native" / native_name).write_bytes(before + b"new package payload\n")
            InxPackage.export_source(str(source), str(current))
            state = manager.install_package(str(current), update=True, install_dependencies=False)
            assert state.restart_required and not state.loaded and "restart" in state.error, state.snapshot()
            assert get_runtime_service(SERVICE) is None
            assert (installed / native_name).read_bytes() == before
            assert "'value': 2" in (installed / "lifecycle.py").read_text(encoding="utf-8")
            assert manager.registry.installed_record(REFERENCE)["version"] == "1.0.0"
            pending = read_pending(str(project))
            assert pending["request"]["kind"] == "package" and pending["native_modules"]
            assert not pending["install"] and not pending["uninstall"]
            assert not manager.reload(REFERENCE).loaded
            assert get_runtime_service(SERVICE) is None
        elif phase == "publish-update":
            assert read_pending(str(project)) is None
            native_name = (workspace / "native-name.txt").read_text(encoding="utf-8")
            manager = PluginManager(str(project))
            manager.reload_all()
            state = manager.states[REFERENCE]
            assert state.loaded and not state.error and not state.restart_required, state.snapshot()
            assert manager.registry.installed_record(REFERENCE)["version"] == "2.0.0"
            assert get_runtime_service(SERVICE)["value"] == 3
            assert (installed / native_name).read_bytes().endswith(b"new package payload\n")
            result = manager.uninstall(REFERENCE)
            assert result["restart_required"] and get_runtime_service(SERVICE) is None
            assert manager.states[REFERENCE].restart_required and not manager.states[REFERENCE].loaded
            assert (installed / native_name).is_file()
            assert read_pending(str(project))["request"]["kind"] == "uninstall"
        else:
            assert read_pending(str(project)) is None
            native_name = (workspace / "native-name.txt").read_text(encoding="utf-8")
            from infernux.engine.engine import Engine
            from infernux.engine.resources_manager import ResourcesManager
            from infernux.engine.import_coordinator import AssetFsEventKind
            from infernux.engine.ui.asset_load_errors import AssetLoadErrors
            from infernux.lib import LogLevel, RuntimeMode
            engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
            engine.init_headless(str(project))
            resources = ResourcesManager(str(project), engine.get_native_engine())
            errors = AssetLoadErrors(engine.get_asset_database(), None)
            manager = PluginManager(str(project), engine=engine.get_native_engine())
            assert not manager.registry.is_installed(REFERENCE)
            assert not (installed / native_name).exists()
            assert not Path(str(installed / native_name) + ".meta").exists()
            assert (installed / "User.txt").read_text(encoding="utf-8") == "user content"
            state = manager.install_package(str(current), install_dependencies=False)
            assert state.loaded and not state.restart_required, state.snapshot()
            native = get_runtime_service(SERVICE)["native"]
            binary = installed / native_name
            # External native rebuilds can rename the mapped image then replace its path.
            binary.rename(binary.with_name(binary.name + ".retired"))
            binary.write_bytes(b"invalid replacement extension")
            Path(str(binary) + ".meta").write_text("{ invalid metadata", encoding="utf-8")
            # A broken external build can also replace the expected file with
            # a directory. Source preflight must fail after old admission stops.
            binary.unlink()
            binary.mkdir()
            coordinator = manager.preloads._reload
            handler = resources._ensure_event_handler()
            handler._coordinator.submit(AssetFsEventKind.MODIFIED, str(binary))
            handler.process_pending_reloads(force=True)
            assert get_runtime_service(SERVICE) is None
            assert "regular file" in errors.error(str(binary))
            assert coordinator.states[str(binary.resolve())].status == "restart_required"
            assert not manager.reload(REFERENCE).loaded
            assert manager.states[REFERENCE].restart_required
            assert get_runtime_service(SERVICE) is None
            assert sys.modules[native.__name__] is native  # Retained mapping is never current execution.
            assert binary.is_dir()
        print("NATIVE_PACKAGE_BOUNDARY_OK " + phase, flush=True)
    finally:
        if manager is not None:
            manager.shutdown()
        if resources is not None:
            resources.cleanup()
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    exercise(sys.argv[1], Path(sys.argv[2]))
