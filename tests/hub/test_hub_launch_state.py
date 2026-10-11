from __future__ import annotations

import sys
import os
import time
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QMessageBox

from launcher import GameEngineLauncher
from hub_utils import HubLaunchContext
from splash_screen import EngineSplashScreen
import splash_screen
import viewmodel.control_pane_viewmodel as control_pane_viewmodel
from viewmodel.control_pane_viewmodel import LaunchPreparationWorker


class _FinishedProcess:
    returncode = 1

    @staticmethod
    def poll():
        return 1


class _RunningProcess:
    @staticmethod
    def poll():
        return None


def _app():
    return QApplication.instance() or QApplication([])


def test_hub_can_import_without_an_installed_engine():
    script = """
import importlib.abc
import sys
attempts = []
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'infernux' or fullname.startswith('infernux.'):
            attempts.append(fullname)
            raise ImportError('No engine installed')
sys.meta_path.insert(0, NoEngine())
import launcher
import hub_utils
assert not attempts, attempts
assert 'infernux_project_lock' in sys.modules
assert hasattr(launcher, 'GameEngineLauncher')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=(Path(__file__).resolve().parents[2] / "packaging"),
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stderr_reader_reaps_editor_after_successful_startup():
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stderr.buffer.write(b'editor stopped')"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    owner = SimpleNamespace(_process=process, _stderr_chunks=[])
    try:
        EngineSplashScreen._drain_stderr(owner, process, owner._stderr_chunks)
        assert process.returncode == 0
        assert owner._stderr_chunks == [b"editor stopped"]
        assert process.stderr.closed
    finally:
        process.wait(timeout=10)


def test_loading_marker_does_not_hide_process_failure(tmp_path: Path):
    _app()
    splash = EngineSplashScreen("", "Test")
    ready = tmp_path / "ready.flag"
    ready.write_text("LOADING:1/3:Loading", encoding="utf-8")
    splash._ready_file = str(ready)
    splash._process = _FinishedProcess()
    splash._launch_started_at = time.monotonic()
    failures = []
    splash._show_failure = lambda title, detail: failures.append((title, detail))

    splash._poll_launch_state()

    assert failures
    splash._spin_timer.stop()


def test_running_process_without_ready_signal_times_out(tmp_path: Path):
    _app()
    splash = EngineSplashScreen("", "Test")
    splash._ready_file = str(tmp_path / "missing.flag")
    splash._process = _RunningProcess()
    splash._launch_started_at = time.monotonic() - splash._STARTUP_TIMEOUT_SECONDS - 1
    timed_out = []
    splash._show_timeout = lambda: timed_out.append(True)

    splash._poll_launch_state()

    assert timed_out == [True]
    splash._spin_timer.stop()


def test_launch_failure_keeps_traceback_in_scrollable_details(monkeypatch):
    _app()
    splash = EngineSplashScreen("", "Test")
    detail = "The engine process exited with an error:\n\n" + "A long traceback line\n" * 80
    observed = []

    def inspect_dialog(box):
        observed.append((box.text(), box.detailedText()))
        return 0

    from view import dialogs
    monkeypatch.setattr(dialogs.HubDialog, "exec", inspect_dialog)
    monkeypatch.setattr(splash, "_fade_out_and_close", lambda: None)
    try:
        splash._show_failure("Engine Launch Failed", detail)
        assert observed == [(detail.split("\n", 1)[0], detail)]
    finally:
        splash._spin_timer.stop()
        splash.close()


def test_launch_reserves_hub_preparation_then_transfers_to_engine_pid(tmp_path: Path, monkeypatch):
    _app()
    (tmp_path / "ProjectSettings").mkdir()

    class Process:
        pid = 424242
        stderr = io.BytesIO()

        @staticmethod
        def wait():
            return 0

        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            return None

    captured = []
    monkeypatch.setattr(splash_screen.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(
        splash_screen,
        "write_project_lock",
        lambda project, pid, token, mode, state: captured.append((project, pid, state)),
    )
    monkeypatch.setattr(splash_screen, "remove_project_lock", lambda *_args, **_kwargs: None)
    splash = EngineSplashScreen("", "Test")

    splash.launch(sys.executable, "pass", str(tmp_path), detached=True)

    assert captured == [(str(tmp_path), os.getpid(), "preparing"), (str(tmp_path), 424242, "launching")]
    splash._poll_timer.stop()
    splash._spin_timer.stop()


def test_failed_handoff_reaps_child_and_closes_stderr(tmp_path, monkeypatch):
    _app()
    splash = EngineSplashScreen("", "Failed handoff")
    original_write = splash_screen.write_project_lock
    failures = []

    def transfer(project, pid, token, mode, state):
        if state == "launching":
            raise PermissionError("handoff publication failed")
        return original_write(project, pid, token, mode, state)

    monkeypatch.setattr(splash_screen, "write_project_lock", transfer)
    monkeypatch.setattr(splash_screen.QTimer, "singleShot", lambda _, callback: callback())
    monkeypatch.setattr(splash, "_show_failure", lambda *args: failures.append(args))
    try:
        splash.launch(sys.executable, "import time; time.sleep(20)", str(tmp_path))
        splash._stderr_thread.join(10)
        assert not splash._stderr_thread.is_alive()
        assert splash._process.poll() is not None
        assert splash._process.stderr.closed
        assert len(failures) == 1 and "handoff publication failed" in failures[0][1]
        assert not Path(splash_screen.get_project_lock_path(str(tmp_path))).exists()
    finally:
        if splash._process is not None and splash._process.poll() is None:
            splash._process.terminate()
            splash._process.wait(timeout=10)
        splash._spin_timer.stop()
        splash.close()


@pytest.mark.parametrize("damaged", [False, True])
def test_failed_preparation_preserves_other_owners_and_reports_errors(tmp_path, monkeypatch, damaged):
    from hub_utils import get_project_lock_path, write_project_lock, remove_project_lock
    _app()
    path = Path(get_project_lock_path(str(tmp_path)))
    if damaged:
        path.parent.mkdir()
        path.write_text("{broken", encoding="utf-8")
    else:
        write_project_lock(str(tmp_path), os.getpid(), "other-request", "editor", "preparing")
    before = path.read_bytes()
    worker = LaunchPreparationWorker(None, None, str(tmp_path), HubLaunchContext.INSTALLED)
    errors, completed = [], []
    worker.error.connect(errors.append)
    worker.finished.connect(completed.append)
    worker.run()
    assert len(errors) == 1 and not completed
    assert path.read_bytes() == before
    if not damaged:
        remove_project_lock(str(tmp_path), "other-request")


def test_preparation_owns_runtime_changes_and_retires_reservation_on_failure(tmp_path, monkeypatch):
    from hub_utils import read_project_lock
    import infernux_project_lock as locks
    _app()

    class Model:
        def _create_vscode_workspace(self, project):
            lease = read_project_lock(project)
            assert lease["pid"] == os.getpid() and lease["state"] == "preparing"
            with pytest.raises(RuntimeError, match="already open"):
                locks.claim(project, "competing", "headless")
            raise RuntimeError("workspace write failed")

    version = "same-source-version"
    versions = SimpleNamespace(read_project_version=lambda _: version)
    monkeypatch.setattr(control_pane_viewmodel, "source_engine_version", lambda: version)
    monkeypatch.setattr(control_pane_viewmodel.ProjectModel, "get_project_python_version", lambda _: "3.13")
    worker = LaunchPreparationWorker(Model(), versions, str(tmp_path), HubLaunchContext.SOURCE)
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors == ["workspace write failed"]
    assert read_project_lock(str(tmp_path)) is None


def test_retry_launch_waits_for_old_process_exit_before_reserving_again(monkeypatch):
    _app()
    splash = EngineSplashScreen("", "Test")
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
    splash._process = process
    splash._launch_args = (sys.executable, "pass", "project", True, None, None)
    pending, launches = [], []
    monkeypatch.setattr(splash_screen.QTimer, "singleShot", lambda delay, callback: pending.append(callback))
    monkeypatch.setattr(splash, "launch", lambda *args, **kwargs: launches.append(args))
    try:
        splash._retry_launch()
        assert not launches and len(pending) == 1
        process.poll = lambda: 0
        pending.pop()()
        assert len(launches) == 1
    finally:
        splash._spin_timer.stop()
        splash._poll_timer.stop()


def test_frozen_launch_preparation_does_not_cold_start_python_twice(
    tmp_path: Path,
    monkeypatch,
):
    from project_python_runtime import write_project_python_version

    write_project_python_version(tmp_path, "3.13")
    runtime_python = tmp_path / ".runtime" / "python313" / "python.exe"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"")

    class VersionManager:
        @staticmethod
        def read_project_version(_path):
            return "0.3.7"

        @staticmethod
        def is_installed(_version, _python_version=None):
            return True

    monkeypatch.setattr(control_pane_viewmodel, "is_project_open", lambda _path: False)
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "_get_project_python",
        staticmethod(lambda _path: str(runtime_python)),
    )
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "validate_python_runtime",
        staticmethod(
            lambda _path: (_ for _ in ()).throw(
                AssertionError("the editor process is the native import check")
            )
        ),
    )

    calls = []
    model = SimpleNamespace(
        runtime_manager=SimpleNamespace(get_runtime_path=lambda _version: sys.executable),
        _install_infernux_in_runtime=lambda project, version, **kw: calls.append((project, version, kw["validate_current"])),
        _create_vscode_workspace=lambda project: calls.append(("workspace", project)),
    )
    worker = LaunchPreparationWorker(
        model, VersionManager(), str(tmp_path), HubLaunchContext.INSTALLED
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert finished == [str(runtime_python)]
    if sys.platform == "win32":
        assert Path(worker.editor_runtime.executable).samefile(sys.executable)
        assert Path(worker.editor_runtime.project_executable).samefile(runtime_python)
    assert calls == [(str(tmp_path), "0.3.7", False), ("workspace", str(tmp_path))]


def test_frozen_launch_preparation_rebuilds_missing_project_runtime(
    tmp_path: Path,
    monkeypatch,
):
    runtime_python = tmp_path / ".runtime" / "python313" / "python.exe"
    calls = []

    class RuntimeManager:
        @staticmethod
        def get_runtime_path(version):
            assert version == "3.13"
            return sys.executable

        @staticmethod
        def has_runtime(version):
            return version == "3.13"

    class Model:
        runtime_manager = RuntimeManager()

        def _create_vscode_workspace(self, project):
            calls.append(("workspace", project))

        def _create_project_runtime(self, project, *, on_status=None, replace_existing=False):
            calls.append(("create", project, replace_existing))
            if on_status:
                on_status("Copying Python runtime into the project...")
            runtime_python.parent.mkdir(parents=True)
            runtime_python.write_bytes(b"runtime")

        def _install_infernux_in_runtime(
            self, project, version, *, on_status=None, validate_current=True
        ):
            calls.append(("install", project, version, validate_current))
            if on_status:
                on_status("Installing Infernux engine files...")

    class VersionManager:
        @staticmethod
        def read_project_version(_path):
            return "0.4.1"

        @staticmethod
        def is_installed(_version, _python_version=None):
            return True

    monkeypatch.setattr(control_pane_viewmodel, "is_project_open", lambda _path: False)
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "get_project_python_version",
        staticmethod(lambda _path: "3.13"),
    )
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "_get_project_python",
        staticmethod(lambda _path: str(runtime_python)),
    )

    worker = LaunchPreparationWorker(
        Model(), VersionManager(), str(tmp_path), HubLaunchContext.INSTALLED
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert finished == [str(runtime_python)]
    assert calls == [
        ("create", str(tmp_path), True),
        ("install", str(tmp_path), "0.4.1", False),
        ("workspace", str(tmp_path)),
    ]


def test_frozen_launch_preparation_blocks_when_global_runtime_is_missing(
    tmp_path: Path,
    monkeypatch,
):
    runtime_python = tmp_path / ".runtime" / "python313" / "python.exe"
    calls = []

    class RuntimeManager:
        @staticmethod
        def has_runtime(_version):
            return False

    class Model:
        runtime_manager = RuntimeManager()

        def _create_project_runtime(self, **_kwargs):
            calls.append("create")

    class VersionManager:
        @staticmethod
        def read_project_version(_path):
            return "0.4.1"

        @staticmethod
        def is_installed(_version, _python_version=None):
            return True

    monkeypatch.setattr(control_pane_viewmodel, "is_project_open", lambda _path: False)
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "get_project_python_version",
        staticmethod(lambda _path: "3.13"),
    )
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "_get_project_python",
        staticmethod(lambda _path: str(runtime_python)),
    )

    worker = LaunchPreparationWorker(
        Model(), VersionManager(), str(tmp_path), HubLaunchContext.INSTALLED
    )
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    assert len(errors) == 1
    assert "Install Python 3.13 in Hub" in errors[0]
    assert calls == []


def test_source_launch_preparation_uses_current_python_without_catalog_gate(
    tmp_path: Path, monkeypatch
):
    (tmp_path / "ProjectSettings").mkdir()

    class Model:
        @staticmethod
        def _install_infernux_in_runtime(*_args, **_kwargs):
            raise AssertionError(
                "source launches must not locate or install an Infernux wheel"
            )

        @staticmethod
        def _create_vscode_workspace(_project_path):
            pass

    class VersionManager:
        @staticmethod
        def read_project_version(_path):
            return control_pane_viewmodel.source_engine_version()

        @staticmethod
        def is_installed(*_args):
            raise AssertionError("source launches must not query installed Hub versions")

    monkeypatch.setattr(control_pane_viewmodel, "is_project_open", lambda _path: False)
    monkeypatch.setattr(
        control_pane_viewmodel.ProjectModel,
        "get_project_python_version",
        staticmethod(lambda _path: f"{sys.version_info.major}.{sys.version_info.minor}"),
    )
    worker = LaunchPreparationWorker(
        Model(), VersionManager(), str(tmp_path), HubLaunchContext.SOURCE
    )
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert finished == [sys.executable]


def test_upgraded_hub_requires_the_new_default_runtime(monkeypatch):
    _app()
    observed = {"message": "", "page": None, "finished": False}
    launcher = SimpleNamespace(
        runtime_manager=SimpleNamespace(
            default_version="3.13",
            has_runtime=lambda _version: False,
        ),
        _show_runtime_installs=lambda: observed.__setitem__("page", "runtime"),
        _finish_startup=lambda: observed.__setitem__("finished", True),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: observed.__setitem__("message", message),
    )
    monkeypatch.setattr(
        "launcher.QTimer.singleShot",
        lambda _delay, callback: callback(),
    )

    GameEngineLauncher._bootstrap_python_runtime(launcher)

    assert observed["message"] == ""
    assert observed["page"] == "runtime"
    assert observed["finished"] is True


def test_packaged_hub_preserves_explicitly_disabled_update_checks():
    observed = []
    launcher = SimpleNamespace(
        db=SimpleNamespace(get_setting=lambda _key, _default: "disabled"),
        _startup_update_pending=False,
        update_controller=SimpleNamespace(
            check=lambda **kwargs: observed.append(("update", kwargs))
        ),
        _bootstrap_python_runtime=lambda: observed.append(("runtime", {})),
    )

    GameEngineLauncher._bootstrap_hub(launcher)

    assert launcher._startup_update_pending is False
    assert observed == [("runtime", {})]


@pytest.mark.parametrize("saved", [None, "enabled"])
def test_packaged_hub_checks_by_default_and_when_explicitly_enabled(saved):
    observed = []
    launcher = SimpleNamespace(
        db=SimpleNamespace(get_setting=lambda _key, default: default if saved is None else saved),
        _startup_update_pending=False,
        update_controller=SimpleNamespace(
            check=lambda **kwargs: observed.append(("update", kwargs))
        ),
        _bootstrap_python_runtime=lambda: observed.append(("runtime", {})),
    )

    GameEngineLauncher._bootstrap_hub(launcher)

    assert launcher._startup_update_pending is True
    assert observed == [("update", {"silent": True})]

    GameEngineLauncher._on_startup_update_check_finished(launcher)

    assert launcher._startup_update_pending is False
    assert observed == [
        ("update", {"silent": True}),
        ("runtime", {}),
    ]


@pytest.mark.parametrize("saved, expected", [(None, True), ("enabled", True), ("disabled", False)])
def test_startup_release_notices_follow_the_same_update_preference(saved, expected):
    observed = []
    launcher = SimpleNamespace(
        db=SimpleNamespace(get_setting=lambda _key, default: default if saved is None else saved),
        installs_view=SimpleNamespace(refresh=lambda: observed.append("installs")),
        notification_controller=SimpleNamespace(show_pending=lambda: observed.append("notices")),
        # Seeding the installer's bundled engine is local and preference-independent.
        _seed_bundled_engines=lambda: observed.append("seed"),
    )

    GameEngineLauncher._finish_startup(launcher)

    assert observed == (["seed", "installs", "notices"] if expected else ["seed", "installs"])


def test_fresh_installer_runtime_skips_the_upgrade_requirement(monkeypatch):
    observed = {"warning": False, "finished": False}
    launcher = SimpleNamespace(
        runtime_manager=SimpleNamespace(
            default_version="3.13",
            has_runtime=lambda _version: True,
        ),
        sidebar=SimpleNamespace(select_page=lambda _page: None),
        _finish_startup=lambda: observed.__setitem__("finished", True),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args: observed.__setitem__("warning", True),
    )
    monkeypatch.setattr(
        "launcher.QTimer.singleShot",
        lambda _delay, callback: callback(),
    )

    GameEngineLauncher._bootstrap_python_runtime(launcher)

    assert observed == {"warning": False, "finished": True}
