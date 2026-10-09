"""The stable Windows editor host preserves project imports and subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from python_execution import editor_python_runtime
from PySide6.QtWidgets import QApplication
from splash_screen import EngineSplashScreen


@pytest.mark.skipif(sys.platform != "win32", reason="Windows managed editor process contract")
@pytest.mark.parametrize("long_home", [False, True])
def test_managed_host_uses_private_imports_and_child_interpreter(
    tmp_path, private_python_factory, long_home
):
    root = tmp_path / "project 中文"
    if long_home:
        while len(str(root)) < 275:
            root /= "Nested private runtime with spaces"
    project_python = private_python_factory(root)
    site = root / "Lib/site-packages"
    (site / "project_identity.py").write_text("VALUE = 'private-project'\n", encoding="utf-8")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "project_identity.py").write_text("VALUE = 'wrong-runtime'\n", encoding="utf-8")
    runtime = editor_python_runtime(str(project_python), sys.executable)
    child_script = (
        "import json,sys,project_identity; "
        "print(json.dumps([sys.prefix, project_identity.VALUE]))"
    )
    path_policy = Path(__file__).resolve().parents[2] / 'python/infernux/engine/path_utils.py'
    report = tmp_path / 'launched-runtime.json'
    script = f'''
import json,os,runpy,site,subprocess,sys,project_identity
from pathlib import Path
assert Path(sys.prefix).samefile({str(root)!r})
assert Path(sys.executable).samefile({str(project_python)!r})
assert Path(sys._base_executable).samefile({str(project_python)!r})
assert not site.ENABLE_USER_SITE
assert 'PYTHONHOME' not in os.environ
assert project_identity.VALUE == 'private-project'
assert {str(foreign)!r} not in sys.path
executable_path = runpy.run_path({str(path_policy)!r})['process_executable_path']
child = subprocess.run([sys.executable, '-c', {child_script!r}], executable=executable_path(sys.executable),
                       capture_output=True, text=True, check=True)
identity = json.loads(child.stdout)
assert Path(identity[0]).samefile({str(root)!r}) and identity[1] == 'private-project'
Path({str(report)!r}).write_text(json.dumps(dict(prefix=sys.prefix, child=identity)), encoding='utf-8')
'''
    environment = dict(os.environ, PYTHONPATH=str(foreign), PYTHONHOME=str(foreign), PYTHONUTF8="1")
    application = QApplication.instance() or QApplication([])
    splash = EngineSplashScreen('', 'Private runtime test')
    try:
        splash.launch(str(project_python), script, str(root), extra_env=environment, runtime=runtime)
        splash._poll_timer.stop()
        splash._stderr_thread.join(45)
        assert not splash._stderr_thread.is_alive()
        assert splash._process.returncode == 0, b''.join(splash._stderr_chunks).decode('utf-8', errors='replace')
        assert Path(json.loads(report.read_text(encoding='utf-8'))['prefix']).samefile(root)
    finally:
        splash._stop_process()
        splash._spin_timer.stop()
        splash.close()


@pytest.mark.parametrize("managed", [None, "missing-python.exe"])
def test_missing_managed_host_is_reported_without_selecting_another_interpreter(tmp_path, managed):
    project = tmp_path / "python.exe"
    project.write_bytes(b"project runtime")
    with pytest.raises(RuntimeError, match="matching managed Python"):
        editor_python_runtime(str(project), managed)


def test_only_executable_path_is_bounded_and_utf16_units_are_counted(tmp_path):
    # Astral characters occupy two Windows path units. Project data paths have
    # no MAX_PATH gate; only the Windows graphics process host has this limit.
    root = tmp_path / "host"
    while len(str(root).encode("utf-16-le")) // 2 < 270:
        root /= "runtime🚀🚀🚀🚀"
    root.mkdir(parents=True)
    host = root / "python.exe"
    host.write_bytes(b"managed runtime")
    project = tmp_path / "project-python.exe"
    project.write_bytes(b"project runtime")
    with pytest.raises(RuntimeError, match="project can keep its location"):
        editor_python_runtime(str(project), str(host))
