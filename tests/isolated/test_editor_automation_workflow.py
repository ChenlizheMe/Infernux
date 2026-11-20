"""A real editor, installed MCP service and signal-driven render capture."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ != "__main__":
    from tests.isolated import engine_python_path


def test_installed_service_loads_and_captures_in_a_real_editor(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": engine_python_path(), "INFERNUX_DATA_ROOT": str(tmp_path / "User")},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EDITOR_MCP_CAPTURE_READY" in result.stdout


def test_ordinary_plugin_is_callable_from_a_standalone_sdk_and_retires_on_uninstall(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), str(tmp_path), "--harness"],
        env={**os.environ, "PYTHONPATH": engine_python_path(), "INFERNUX_DATA_ROOT": str(tmp_path / "User")},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORDINARY_PLUGIN_STANDALONE_SDK_READY" in result.stdout


def exercise_harness(project):
    import json
    import shutil
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.lib import LogLevel
    from infernux.plugins import InxPackage
    from infernux.version import ENGINE_COMPATIBILITY_VERSION

    repo = Path(__file__).resolve().parents[2]
    sdk_source = project / "SDKSource"
    sdk_source.mkdir()
    shutil.copy2(repo / "python/pyproject.toml", sdk_source / "pyproject.toml")
    shutil.copytree(repo / "python/infernux_automation", sdk_source / "infernux_automation",
                    ignore=shutil.ignore_patterns("__pycache__"))
    wheels = project / "SDKWheel"
    installed = project / "SDKInstalled"
    for arguments in (("wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", str(wheels), str(sdk_source)),):
        result = subprocess.run([sys.executable, "-m", "pip", *arguments], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheels.glob("*.whl"))
    assert wheel.name.endswith("py3-none-any.whl")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), str(wheel)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    sdk_program = '''import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
from infernux_automation import AutomationClient, AutomationError
assert importlib.util.find_spec("infernux") is None
assert "infernux" not in sys.modules
with AutomationClient(sys.argv[2]) as engine:
    operations = engine.operations("example.harness.read")["operations"]
    if sys.argv[3] == "installed":
        assert [item["id"] for item in operations] == ["example.harness.read"]
        assert engine.schema("example.harness.read")["kind"] == "query"
        assert engine.call("example.harness.read") == {"reference": "example/harness", "origin": "ordinary-plugin"}
    else:
        assert operations == []
        try:
            engine.call("example.harness.read")
        except AutomationError as exc:
            assert exc.code == "operation.not_found", exc.code
        else:
            raise AssertionError("Uninstalled plugin operation still executed")
print("STANDALONE_SDK_" + sys.argv[3].upper())
'''
    source = project / "HarnessSource"
    (source / "editor").mkdir(parents=True)
    (source / "inx_package.json").write_text(json.dumps({
        "reference": "example/harness", "name": "Ordinary external harness", "version": "1.0.0",
        "engine": "==" + ENGINE_COMPATIBILITY_VERSION,
    }), encoding="utf-8")
    (source / "editor/startup.py").write_text('''from infernux.lifecycle import InxPreload
from infernux.host import Operation, OperationKind, OperationRegistry, OperationSchema
class Harness(InxPreload):
    def preload(self, context):
        self.reference = context.package_reference
        schema = OperationSchema(
            id="example.harness.read", kind=OperationKind.QUERY, summary="Read ordinary plugin state",
            input_schema={"type":"object", "additionalProperties":False},
            output_schema={"type":"object", "required":["reference", "origin"]},
            errors=(), thread="any", side_effects=(), reversible=False,
            capabilities=("example.read",), cost={})
        OperationRegistry.instance().register(Operation(schema, self.read, context.package_reference))
    def read(self):
        return {"reference":self.reference, "origin":"ordinary-plugin"}
''', encoding="utf-8")
    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/requirements.txt").write_text("", encoding="utf-8")
    bootstrap = EditorBootstrap(str(project), LogLevel.Warn)
    try:
        bootstrap.run()
        archive = InxPackage.export_source(str(source), str(project / "harness.inxpkg"))
        state = bootstrap.plugin_manager.install_package(archive.package_path, install_dependencies=False)
        assert state.loaded and not state.error, state.snapshot()
        for phase in ("installed", "uninstalled"):
            if phase == "uninstalled":
                bootstrap.plugin_manager.uninstall("example/harness")
            result = subprocess.run([sys.executable, "-I", "-S", "-c", sdk_program, str(installed), str(project), phase],
                env={**os.environ, "INFERNUX_NATIVE_MODULE_DIR": str(project / "missing-native")},
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            assert result.returncode == 0, result.stdout + result.stderr
            assert "STANDALONE_SDK_" + phase.upper() in result.stdout
        print("ORDINARY_PLUGIN_STANDALONE_SDK_READY", flush=True)
    finally:
        if bootstrap.engine is not None:
            bootstrap.engine.exit()


def exercise(project):
    import asyncio
    import json
    import shutil
    from threading import Event, Thread
    from fastmcp import Client
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.host import EditorAutomationHost
    from infernux.host.commands import MainThreadCommandQueue
    from infernux.lib import LogLevel, SceneManager

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/requirements.txt").write_text("", encoding="utf-8")
    bootstrap = EditorBootstrap(str(project), LogLevel.Warn)
    outcome = []
    try:
        bootstrap.run()
        print("EDITOR_BOOTED", flush=True)
        native = SceneManager.instance()
        scene = native.create_scene("自动化场景")
        scene.create_game_object("测试物体")
        scene_path = project / "Assets/Automation.scene"
        scene_path.write_text(scene.serialize_asset(), encoding="utf-8")
        native.unload_scene(scene)
        database = bootstrap.engine.get_asset_database()
        imported = database.import_asset(str(scene_path))
        assert imported.succeeded
        assert bootstrap.scene_file_manager._do_open_scene(str(scene_path))
        repo = Path(__file__).resolve().parents[2]
        plugin = repo / "external/plugins/infernux_mcp"
        archive = project / "mcp.inxpkg"
        subprocess.run([sys.executable, str(plugin / "package.py"), "build", str(archive)], check=True,
                       capture_output=True, text=True)
        descriptor = project / "Cache/Automation/mcp.json"
        arrived = Event()

        class Changes(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.event_type in {"created", "moved"} and Path(event.dest_path or event.src_path) == descriptor:
                    arrived.set()

        observer = Observer()
        observer.schedule(Changes(), str(descriptor.parent), recursive=False)
        observer.start()
        state = bootstrap.plugin_manager.install_package(str(archive), install_dependencies=False)
        assert state.loaded, state
        if descriptor.is_file():
            arrived.set()
        print("PLUGIN_INSTALLED", flush=True)
        theme_root = project / "ThemeSource"
        subprocess.run([sys.executable, str(repo / "scripts/plugins/export_theme_template.py"), str(theme_root)],
                       check=True, capture_output=True, text=True)
        theme_archive = project / "theme.inxpkg"
        subprocess.run([sys.executable, str(theme_root / "package.py"), "build", str(theme_archive)],
                       check=True, capture_output=True, text=True)
        assert bootstrap.plugin_manager.install_package(str(theme_archive), install_dependencies=False).loaded
        bootstrap.plugin_manager.themes.select("example/studio-theme/studio")
        assert bootstrap.window_manager.open_window("themes") is not None
        service = bootstrap.plugin_manager.preloads

        async def check():
            assert await asyncio.to_thread(arrived.wait, 30), "MCP child did not publish its endpoint"
            endpoint = json.loads(descriptor.read_text(encoding="utf-8"))["endpoint"]
            async with Client(endpoint, timeout=45) as client:
                print("MCP_CONNECTED", flush=True)
                tools = await client.list_tools()
                assert "load_and_capture" in {tool.name for tool in tools}
                status = await client.call_tool("engine_status")
                assert Path(status.data["project"]) == project.resolve()
                task = await client.call_tool("load_and_capture", {"asset_guid": imported.guid, "source": "editor"}, task=True)
                assert not task.returned_immediately
                result = await task.result()
                image = Path(result.data["output_path"])
                assert result.data["status"] == "completed"
                assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                previews = repo / "out/implementation-review"
                previews.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, previews / "editor-theme-plugin.png")

                def reveal_theme_category():
                    themes = bootstrap.window_manager.get_window_instance("themes")
                    selected = next(row for row in bootstrap.plugin_manager.themes.catalog()
                                    if row["id"] == "example/studio-theme/studio")
                    assert selected["reference"] == "example/studio-theme"
                    assert selected["version"] and selected["author"] and selected["description"]
                    themes._open_plugin_manager()
                    plugins = bootstrap.window_manager.get_window_instance("plugins")
                    rows = plugins._visible_rows(bootstrap.plugin_manager)
                    assert [row["reference"] for row in rows] == ["example/studio-theme"]
                    assert rows[0]["_category_key"] == "theme"
                    return EditorAutomationHost.instance().request_capture("editor", str(project / "theme-category.png"))

                capture_id = await asyncio.to_thread(
                    MainThreadCommandQueue.instance().submit("reveal-theme-category", reveal_theme_category).result)
                from infernux_automation import AutomationClient
                with AutomationClient(project) as automation:
                    observed = await asyncio.to_thread(automation.observe, "infernux.capture.status",
                                                       {"capture_id": capture_id})
                    revision = 0
                    try:
                        while True:
                            snapshot = await asyncio.to_thread(automation.observation, observed["id"],
                                                               after=revision, timeout=15)
                            assert not snapshot["error"], snapshot
                            receipt = snapshot["value"]
                            if receipt is not None and receipt["status"] in {"completed", "failed", "cancelled"}:
                                assert receipt["status"] == "completed", receipt
                                shutil.copy2(project / "theme-category.png", previews / "editor-plugin-theme-category.png")
                                break
                            revision = snapshot["revision"]
                    finally:
                        await asyncio.to_thread(automation.stop_observing, observed["id"])
                print("EDITOR_MCP_CAPTURE_READY " + str(image), flush=True)

        def client_thread():
            try:
                asyncio.run(check())
            except BaseException as exc:
                outcome.append(exc)
                import traceback
                traceback.print_exc()
            finally:
                observer.stop()
                observer.join()
                MainThreadCommandQueue.instance().submit(
                    "workflow-test-close", bootstrap.engine.get_native_engine().exit)

        thread = Thread(target=client_thread, daemon=True)
        thread.start()
        bootstrap.engine.run()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not outcome, repr(outcome)
        assert not descriptor.exists(), "Retired MCP plugin left its descriptor behind"
        assert service is not None
    finally:
        if bootstrap.engine is not None:
            bootstrap.engine.exit()


if __name__ == "__main__":
    (exercise_harness if "--harness" in sys.argv[2:] else exercise)(Path(sys.argv[1]))
