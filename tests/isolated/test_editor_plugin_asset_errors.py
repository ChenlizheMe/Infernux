"""Package operations expose current asset failures without disabling valid code."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ != "__main__":
    from tests.isolated import engine_python_path


def test_package_asset_failures_repair_and_background_completion_in_editor(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": engine_python_path(),
             "INFERNUX_DATA_ROOT": str(tmp_path / "User"),
             "INFERNUX_PACKAGE_CACHE_ROOT": str(tmp_path / "PackageCache")},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EDITOR_PLUGIN_ASSET_ERRORS_REPAIRED" in result.stdout


def exercise(workspace):
    import json
    from threading import Event, Thread
    import wave

    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.i18n import t
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors
    from infernux.engine.ui.plugin_install_progress import PluginInstallProgressService
    from infernux.engine.ui.plugin_panel import PluginPanel
    from infernux.host.commands import MainThreadCommandQueue
    from infernux.host.plugin_operations import _plugins
    from infernux.lib import AssetRegistry, InxMaterial, LogLevel
    from infernux.plugins import InxPackage
    from infernux.runtime_services import get_runtime_service

    project = workspace / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings" / "requirements.txt").write_text("", encoding="utf-8")
    bootstrap = EditorBootstrap(str(project), LogLevel.Warn)
    try:
        bootstrap.run()
        manager = bootstrap.plugin_manager
        database = bootstrap.engine.get_asset_database()
        registry = AssetRegistry.instance()
        errors = AssetLoadErrors.instance()
        queue = MainThreadCommandQueue.instance()
        queue.drain()
        reference = "review/current-material"
        source = workspace / "source"
        (source / "runtime").mkdir(parents=True)
        metadata = {"reference": reference, "version": "1.0.0", "engine": ">=0.4.8"}
        manifest = source / "inx_package.json"
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        valid = InxMaterial.create_default_lit().serialize()
        material = source / "runtime/Current.mat"
        material.write_text(valid, encoding="utf-8")
        # Audio import accepts its identity; decoding is a later native operation.
        (source / "runtime/Current.wav").write_bytes(b"invalid encoded audio")
        (source / "runtime/Removed.txt").write_text("vendor member", encoding="utf-8")
        feature = source / "runtime/Feature.py"
        code = ("from infernux.runtime_services import install_runtime_service\n"
                "class Feature:\n    def marker(self): return {version!r}\n"
                "install_runtime_service('package-asset-feature', Feature())\n")
        feature.write_text(code.format(version="first"), encoding="utf-8")
        original = workspace / "first.inxpkg"
        InxPackage.export_source(str(source), str(original))
        state = manager.install_package(str(original), install_dependencies=False)
        assert state.loaded and not state.error and not state.asset_errors, state.snapshot()
        assert get_runtime_service("package-asset-feature").marker() == "first"
        root = project / "Packages" / reference / "runtime"
        installed = root / "Current.mat"
        guid = database.get_guid_from_path(str(installed))
        retained = registry.load_material_by_guid(guid)
        assert retained is not None
        extra = root / "User.txt"
        extra.write_text("user content", encoding="utf-8")
        record = manager.registry.installed_record(reference)
        Path(manager._installed_archive_path(record)).unlink()
        material.write_text("{ broken current material", encoding="utf-8")
        feature.write_text(code.format(version="second"), encoding="utf-8")
        (source / "runtime/Removed.txt").unlink()
        metadata["version"] = "2.0.0"
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        current = workspace / "second.inxpkg"
        InxPackage.export_source(str(source), str(current))
        updated = manager.install_package(str(current), update=True, install_dependencies=False)
        assert updated.loaded, updated.snapshot()  # Valid current code remains available.
        assert get_runtime_service("package-asset-feature").marker() == "second"
        assert len(updated.asset_errors) == 1 and updated.error
        failure = updated.asset_errors[0]
        assert failure["guid"] == guid and failure["logical_path"] == "runtime/Current.mat"
        assert failure["error"] == database.get_import_error(guid)
        assert failure["path"] == database.get_path_from_guid(guid)
        assert registry.load_material_by_guid(guid) is None
        assert installed.read_text(encoding="utf-8") == "{ broken current material"
        assert not (root / "Removed.txt").exists() and not (root / "Removed.txt.meta").exists()
        assert extra.read_text(encoding="utf-8") == "user content"
        assert manager.registry.installed_record(reference)["version"] == "2.0.0"
        assert PluginPanel._state_visual(updated, {"_installed": True})[0] == t("plugins.error")
        host = next(item for item in _plugins()["plugins"] if item["record"]["reference"] == reference)
        assert host["state"]["asset_errors"] == list(updated.asset_errors)
        assert host["state"]["error"] == updated.error
        # A native import repairs the result without a plugin reload or a script restart.
        errors.poll()
        service = get_runtime_service("package-asset-feature")
        installed.write_text(valid, encoding="utf-8")
        assert database.reimport_asset(str(installed)).succeeded
        errors.poll()
        repaired = manager.states[reference]
        assert repaired.loaded and not repaired.error and not repaired.asset_errors
        assert get_runtime_service("package-asset-feature") is service
        assert registry.load_material_by_guid(guid) is not None
        errors.poll()
        assert manager.states[reference] is repaired  # Same revision does not rebuild package state.
        # Loader failures also change the native error generation, without a watcher event.
        audio = root / "Current.wav"
        audio_guid = database.get_guid_from_path(str(audio))
        ticket = registry.begin_load_audio_by_guid(audio_guid)
        ticket.wait()
        try:
            registry.try_commit_asset_load(ticket)
        except RuntimeError as error:
            assert "Audio" in str(error)
        else:
            raise AssertionError("Invalid audio unexpectedly decoded")
        errors.poll()
        assert [item["guid"] for item in manager.states[reference].asset_errors] == [audio_guid]
        assert get_runtime_service("package-asset-feature") is service
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(bytes(32))
        assert database.reimport_asset(str(audio)).succeeded
        errors.poll()
        assert not manager.states[reference].asset_errors and not manager.states[reference].error
        # The actual progress UI worker must return these errors after owner-thread finalization.
        background = workspace / "background-source"
        (background / "runtime").mkdir(parents=True)
        background_reference = "review/background-material"
        (background / "inx_package.json").write_text(json.dumps({
            "reference": background_reference, "version": "1.0.0", "engine": ">=0.4.8",
        }), encoding="utf-8")
        (background / "runtime/Broken.mat").write_text("{ invalid background material", encoding="utf-8")
        archive = workspace / "background.inxpkg"
        InxPackage.export_source(str(background), str(archive))
        completed = Event()
        failures = []

        def completion(actual_reference, message):
            try:
                actual = manager.states[background_reference]
                assert actual_reference == background_reference
                assert actual.loaded and len(actual.asset_errors) == 1
                assert actual.error and actual.error == message
                assert "runtime/Broken.mat" in message
                assert PluginPanel._state_visual(actual, {"_installed": True})[0] == t("plugins.error")
                assert get_runtime_service("package-asset-feature") is service
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()
                bootstrap.engine.get_native_engine().exit()

        progress = PluginInstallProgressService.instance()
        assert progress.begin_package(manager=manager, label=str(archive), action="source", observer=completion)

        def timeout():
            if not completed.wait(40):
                failures.append(AssertionError("Background install did not complete"))
                queue.submit("plugin-asset-test-timeout", bootstrap.engine.get_native_engine().exit)

        monitor = Thread(target=timeout, daemon=True)
        monitor.start()
        bootstrap.engine.run()
        assert completed.is_set() and not failures, failures
        monitor.join()
        print("EDITOR_PLUGIN_ASSET_ERRORS_REPAIRED", flush=True)
    finally:
        if bootstrap.engine is not None:
            bootstrap.engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]))
