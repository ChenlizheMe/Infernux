"""Runtime imports and rejected hosts must not acquire editor-only services."""

from pathlib import Path
import subprocess
import sys


def test_runtime_host_imports_do_not_load_editor_operations():
    code = """
import sys
from infernux.host import MainThreadCommandQueue, OperationRegistry
from infernux.host.queries import emit_signal
from infernux.engine.reload.owners import ReloadOwner
assert 'infernux.host.editor' not in sys.modules
assert 'infernux.host.editor_operations' not in sys.modules
assert 'infernux.engine.editor_themes' not in sys.modules
assert 'infernux.engine.ui.plugin_panel' not in sys.modules
assert 'infernux.engine.ui.theme_panel' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_rejected_host_exit_does_not_stop_current_editor_endpoint(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HOST_OWNERSHIP_READY" in result.stdout


def exercise(project):
    from infernux.engine import Engine
    from infernux.lib import RuntimeMode, LogLevel
    from infernux.host import bridge
    from infernux_automation import AutomationClient

    (project / "ProjectSettings").mkdir(parents=True)
    owner = Engine(LogLevel.Error, mode=RuntimeMode.Headless)
    owner.init_headless(str(project), gpu=False)
    endpoint = bridge.start(project)
    client = AutomationClient(project)
    try:
        before = client.request("info")["instance_id"]
        rejected = Engine(LogLevel.Error, mode=RuntimeMode.Headless)
        rejected.exit()
        assert bridge.current() is endpoint
        assert client.request("info")["instance_id"] == before
    finally:
        client.close()
        owner.exit()
    assert bridge.current() is None
    print("HOST_OWNERSHIP_READY", flush=True)


if __name__ == "__main__":
    exercise(Path(sys.argv[1]).resolve())
