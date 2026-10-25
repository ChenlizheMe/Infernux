"""The stable Windows editor host preserves project imports and subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from python_execution import editor_python_runtime
from PySide6.QtWidgets import QApplication
from splash_screen import EngineSplashScreen


def test_failed_launch_preserves_complete_stderr_and_exit_code(tmp_path, monkeypatch):
    import splash_screen

    application = QApplication.instance() or QApplication([])
    splash = EngineSplashScreen('', 'Failed startup')
    failures = []
    monkeypatch.setattr(splash, '_show_failure', lambda title, detail: failures.append(detail))
    started = tmp_path / 'editor-script-started'
    script = f"from pathlib import Path; Path({str(started)!r}).touch(); " + (
        "import sys; sys.stderr.write('startup diagnostics\\n' + 'x' * 9000 + "
        "'\\nRuntimeError: final-cause-marker\\n'); sys.exit(23)"
    )
    write_lock = splash_screen.write_project_lock

    def delayed_transfer(project, pid, token, mode, state):
        if state == 'launching':
            # Give the child time to run before Hub publishes its lease.
            # Package preparation must remain blocked until that publication.
            time.sleep(0.5)
            assert not started.exists()
        return write_lock(project, pid, token, mode, state)

    monkeypatch.setattr(splash_screen, 'write_project_lock', delayed_transfer)
    try:
        splash.launch(sys.executable, script, str(tmp_path))
        splash._poll_timer.stop()
        splash._stderr_thread.join(15)
        assert not splash._stderr_thread.is_alive()
        assert started.exists()
        splash._poll_launch_state()
        assert len(failures) == 1
        assert 'startup diagnostics' in failures[0]
        assert 'RuntimeError: final-cause-marker' in failures[0]
        assert 'Raw exit code: 23' in failures[0]
    finally:
        splash._stop_process()
        splash._spin_timer.stop()
        splash.close()


def test_stop_ends_fresh_editor_as_well_as_preparing_process(tmp_path):
    import signal
    from infernux_project_lock import is_pid_running

    application = QApplication.instance() or QApplication([])
    splash = EngineSplashScreen('', 'Restart handoff')
    source = str(Path(__file__).resolve().parents[2] / 'python')
    started = tmp_path / 'fresh-editor.pid'
    child_script = f'''
import os,sys,time
from pathlib import Path
sys.path.insert(0, {source!r})
import infernux_project_lock as locks
locks.claim({str(tmp_path)!r}, os.environ['_INFERNUX_PROJECT_LOCK_TOKEN'], 'editor', require_reservation=True)
Path({str(started)!r}).write_text(str(os.getpid()))
time.sleep(120)
'''
    script = f'''
import os,subprocess,sys
sys.path.insert(0, {source!r})
import infernux_project_lock as locks
token = os.environ['_INFERNUX_PROJECT_LOCK_TOKEN']
locks.claim({str(tmp_path)!r}, token, 'editor', require_reservation=True)
locks.reserve({str(tmp_path)!r}, os.getpid(), token, 'editor', 'preparing')
child = subprocess.Popen([sys.executable, '-c', {child_script!r}])
locks.reserve({str(tmp_path)!r}, child.pid, token, 'editor', 'launching')
child.wait()
'''
    child_pid = None
    try:
        splash.launch(sys.executable, script, str(tmp_path))
        splash._poll_timer.stop()
        deadline = time.monotonic() + 15
        while not started.exists() and time.monotonic() < deadline:
            assert splash._process.poll() is None, b''.join(splash._stderr_chunks)
            time.sleep(0.01)
        child_pid = int(started.read_text())
        splash._stop_process()
        splash._stderr_thread.join(10)
        assert not splash._stderr_thread.is_alive()
        # Waiting for the actual PID also detects a surviving child when its
        # outer preparer has exited. Bound the wait outside the test process.
        subprocess.run(
            [sys.executable, '-c', f"import sys;sys.path.insert(0,{source!r});"
             f"from infernux_editor import _wait_for_process;_wait_for_process({child_pid})"],
            timeout=5, check=True,
        )
    finally:
        splash._stop_process()
        if child_pid is not None and is_pid_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        splash._spin_timer.stop()
        splash.close()


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
    project = tmp_path / "authored project"
    (project / "Assets").mkdir(parents=True)
    (project / "Assets/relative.txt").write_text("project asset", encoding="utf-8")
    site = root / "Lib/site-packages"
    (site / "project_identity.py").write_text("VALUE = 'private-project'\n", encoding="utf-8")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "project_identity.py").write_text("VALUE = 'wrong-runtime'\n", encoding="utf-8")
    runtime = editor_python_runtime(str(project_python), sys.executable)
    child_script = (
        "import json,sys,project_identity; from pathlib import Path; "
        "print(json.dumps([sys.prefix, project_identity.VALUE, "
        "Path('Assets/relative.txt').read_text(encoding='utf-8')]))"
    )
    (site / 'project_child_probe.py').write_text(child_script, encoding='utf-8')
    path_policy = Path(__file__).resolve().parents[2] / 'python/infernux/engine/path_utils.py'
    editor_entry = Path(__file__).resolve().parents[2] / 'python/infernux_editor.py'
    report = tmp_path / 'launched-runtime.json'
    script = f'''
import json,os,runpy,site,subprocess,sys,project_identity
from pathlib import Path
assert Path(sys.prefix).samefile({str(root)!r})
assert Path(sys.executable).samefile({str(project_python)!r})
assert Path(sys._base_executable).samefile({str(project_python)!r})
assert Path.cwd().samefile({str(project)!r})
assert Path('Assets/relative.txt').read_text(encoding='utf-8') == 'project asset'
assert not site.ENABLE_USER_SITE
assert 'PYTHONHOME' not in os.environ
assert project_identity.VALUE == 'private-project'
assert {str(foreign)!r} not in sys.path
executable_path = runpy.run_path({str(path_policy)!r})['process_executable_path']
child = subprocess.run([sys.executable, '-c', {child_script!r}], executable=executable_path(sys.executable),
                       capture_output=True, text=True, check=True)
identity = json.loads(child.stdout)
assert Path(identity[0]).samefile({str(root)!r}) and identity[1] == 'private-project'
assert identity[2] == 'project asset'
entry = runpy.run_path({str(editor_entry)!r})
host, restart_environment = entry['_python_process']()
assert Path(host).samefile({str(runtime.executable)!r})
assert Path(restart_environment['PYTHONHOME']).samefile({str(root)!r})
restarted_child = json.loads(entry['_run']({str(project)!r}, ['-m', 'project_child_probe']))
assert Path(restarted_child[0]).samefile({str(root)!r})
assert restarted_child[1:] == identity[1:]
Path({str(report)!r}).write_text(json.dumps(dict(prefix=sys.prefix, child=identity)), encoding='utf-8')
'''
    environment = dict(os.environ, PYTHONPATH=str(foreign), PYTHONHOME=str(foreign), PYTHONUTF8="1")
    application = QApplication.instance() or QApplication([])
    splash = EngineSplashScreen('', 'Private runtime test')
    try:
        splash.launch(str(project_python), script, str(project), extra_env=environment, runtime=runtime)
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
