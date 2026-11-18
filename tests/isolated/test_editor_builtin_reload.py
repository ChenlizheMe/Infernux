"""Exercise observed built-in panel edits in a continuously running editor."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ != "__main__":
    from tests.isolated import engine_python_path


def test_open_builtin_panel_reconstructs_after_save_failure_and_repair(tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), str(tmp_path)],
        env={**os.environ, "PYTHONPATH": engine_python_path(), "INFERNUX_DATA_ROOT": str(tmp_path / "User")},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BUILTIN_PANEL_SAVE_REPAIR_READY" in result.stdout


def exercise(project):
    import time
    from threading import Thread
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.host import EditorAutomationHost
    from infernux.host.commands import MainThreadCommandQueue
    from infernux.engine.path_utils import resolved_path
    from infernux.lib import LogLevel
    import infernux.engine.ui.theme_panel as initial_module

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    (project / "ProjectSettings/requirements.txt").write_text("", encoding="utf-8")
    source = Path(initial_module.__file__)
    original = source.read_bytes()
    bootstrap = EditorBootstrap(str(project), LogLevel.Warn)
    failures = []
    queue = MainThreadCommandQueue.instance()

    def call(callback):
        return queue.submit("builtin-reload-experiment", callback).result(timeout=10)

    def until(callback):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = call(callback)
            if result:
                return result
            time.sleep(0.03)
        raise AssertionError("Observed editor source did not reach the expected current state")

    def experiment():
        try:
            manager = bootstrap.window_manager
            resources = bootstrap.engine._resources_manager
            coordinator = resources._event_handler.reload
            key = resolved_path(source)
            old = until(lambda: manager.get_window_instance("themes"))
            assert old._reload_owner.active
            assert old.__class__ is initial_module.ThemePanel
            assert call(lambda: bootstrap.menu_bar.execute_command("window.open.themes", "menu", ""))
            assert call(lambda: manager.is_window_open("themes"))
            assert call(lambda: manager.get_window_instance("themes")) is old
            assert old.window_id == "themes"
            assert call(lambda: bootstrap.interaction_core.commands.get("window.toggle.themes")) is None

            source.write_bytes(original + b"\n# Continuous editor save experiment.\n")
            new = until(lambda: (view if (view := manager.get_window_instance("themes"))
                                is not None and type(view) is not type(old) else None))
            assert not old._reload_owner.active
            assert new._reload_owner.active
            assert call(lambda: coordinator.states[key].status) == "active"

            source.write_bytes(original + b"\ninvalid (\n")
            until(lambda: coordinator.states[key].status == "failed")
            assert not new._reload_owner.active
            assert call(lambda: "themes" not in manager.get_registered_types())
            assert call(lambda: "plugins" in manager.get_registered_types())

            source.write_bytes(original)
            repaired = until(lambda: (view if (view := manager.get_window_instance("themes"))
                                     is not None and view._reload_owner.active
                                     and type(view) is not type(new) else None))
            assert call(lambda: coordinator.states[key].status) == "active"
            assert not old._reload_owner.active and not new._reload_owner.active
            assert repaired is not old and repaired is not new
            assert not source.with_suffix(".py.meta").exists()

            # Selection goes through the same theme service; the next frame
            # presents the full catalog/details window, rather than a dropdown.
            def choose():
                service = bootstrap.plugin_manager.themes
                service.select("graphite")
                repaired._selected = "graphite"
            call(choose)
            output = Path(__file__).resolve().parents[2] / "out/implementation-review/editor-theme-detail.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            capture_id = call(lambda: EditorAutomationHost.instance().request_capture("editor", str(output)))
            receipt = until(lambda: (state if (state := EditorAutomationHost.instance().capture_status(capture_id))
                                    ["status"] in {"completed", "failed", "cancelled"} else None))
            assert receipt["status"] == "completed", receipt
            assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            print("BUILTIN_PANEL_SAVE_REPAIR_READY " + str(output), flush=True)
        except BaseException as exc:
            failures.append(exc)
            import traceback
            traceback.print_exc()
        finally:
            source.write_bytes(original)
            queue.submit("builtin-reload-close", bootstrap.engine.get_native_engine().exit)

    try:
        bootstrap.run()
        assert bootstrap.menu_bar.execute_command("window.open.themes", "menu", "")
        thread = Thread(target=experiment, daemon=True)
        thread.start()
        bootstrap.engine.run()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not failures, repr(failures)
    finally:
        source.write_bytes(original)
        if bootstrap.engine is not None:
            bootstrap.engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]))
