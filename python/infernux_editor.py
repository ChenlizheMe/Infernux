"""Prepare deferred dependency changes before importing the editor."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import uuid


def pending_path(project: str) -> Path:
    return Path(project) / "Library" / "Plugins" / "pending-install.json"


def read_pending(project: str) -> dict | None:
    try:
        return json.loads(pending_path(project).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_pending(project: str, operation: dict) -> None:
    path = pending_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(operation, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_pending(project: str, operation: dict) -> None:
    if pending_path(project).exists():
        raise RuntimeError("Complete or cancel the pending dependency installation first.")
    _write_pending(project, {**operation, "prepared": False})


def resolved_arguments(install: list[dict]) -> list[str]:
    """Install the resolver's selected sources without resolving them again."""
    arguments = []
    for item in install:
        source = item["download_info"]
        url = source["url"]
        if "vcs_info" in source:
            vcs = source["vcs_info"]
            url = f"{vcs['vcs']}+{url}@{vcs['commit_id']}"
        if source.get("subdirectory"):
            url += "#subdirectory=" + source["subdirectory"]
        if source.get("dir_info", {}).get("editable"):
            arguments.append("--editable")
        arguments.append(url)
    return arguments


def _python_process() -> tuple[str, dict[str, str]]:
    environment = os.environ.copy()
    runtime = environment.get("_INFERNUX_EDITOR_RUNTIME")
    if runtime is None:
        return sys.executable, environment
    contract = json.loads(runtime)
    environment["PYTHONHOME"] = contract["home"]
    return contract["executable"], environment


def _module_command(project: str, arguments: list[str]):
    executable, environment = _python_process()
    bootstrap = (
        "import os,runpy,sys\n"
        "os.chdir(sys.argv[1])\n"
        "del sys.argv[:2]\n"
        "runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)\n"
    )
    return [executable, "-c", bootstrap, project, *arguments], environment


def _run(project: str, arguments: list[str]) -> str:
    command, environment = _module_command(project, arguments[1:])
    result = subprocess.run(
        command, cwd=os.path.dirname(command[0]), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", timeout=1800,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(result.stdout or f"Dependency installation exited with {result.returncode}")
    return result.stdout


def prepare(project: str) -> bool:
    operation = read_pending(project)
    if operation is None:
        requirements = Path(project) / "ProjectSettings" / "requirements.txt"
        registry = Path(project) / "ProjectSettings" / "InxPlugins.json"
        authored = requirements.exists() and any(
            line.strip() and not line.lstrip().startswith("#")
            for line in requirements.read_text(encoding="utf-8").splitlines()
        )
        plugins = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
        if not authored and not plugins.get("python_dependencies") and not any(
            record.get("enabled", True) and record.get("python_requirements")
            for record in plugins.get("installed", ())
        ):
            return False
    if "infernux" in sys.modules:
        raise RuntimeError("Restart with 'python -m infernux_editor <project>' to complete dependency installation.")
    from infernux_project_lock import read_lock

    owner = read_lock(project)
    token = os.environ.get("_INFERNUX_PROJECT_LOCK_TOKEN", "")
    if owner is None or owner["token"] != token or owner["pid"] not in {os.getpid(), os.getppid()}:
        raise RuntimeError("Dependency preparation requires the project's launch reservation.")
    if operation is None:
        directory = pending_path(project).parent
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, report = tempfile.mkstemp(dir=directory, suffix=".json")
        os.close(descriptor)
        try:
            _run(project, ["-m", "infernux_editor", "--plan-startup", report, project])
            operation = json.loads(Path(report).read_text(encoding="utf-8"))
        finally:
            os.unlink(report)
        if operation is None:
            return False
        save_pending(project, operation)
        operation = read_pending(project)
    if not operation["prepared"]:
        arguments = resolved_arguments(operation["install"])
        if arguments:
            command = [
                "-m", "pip", "install", "--no-deps", "--force-reinstall", *arguments,
            ]
            operation["command"] = [_python_process()[0], *command]
            operation["output"] = _run(project, command)
        if operation.get("uninstall"):
            _run(project, ["-m", "pip", "uninstall", "-y", *operation["uninstall"]])
        operation["prepared"] = True
        _write_pending(project, operation)
    _run(project, ["-m", "infernux_editor", "--publish", project])
    pending_path(project).unlink()
    return True


def launch_clean(project: str) -> int:
    """Hand the lease to an interpreter whose site hooks see the installed files."""
    from infernux_project_lock import claim, lock_path, read_lock, reserve

    token = os.environ.get("_INFERNUX_PROJECT_LOCK_TOKEN") or uuid.uuid4().hex
    owner = read_lock(project)
    if owner is not None and owner["pid"] == os.getppid():
        claim(project, token, "editor", require_reservation=True)
    reserve(project, os.getpid(), token, "editor", "preparing")
    command, environment = _module_command(project, ["infernux_editor", "--handoff", project])
    environment.update(_INFERNUX_PROJECT_LOCK_TOKEN=token, _INFERNUX_PROJECT_LOCK_PATH=lock_path(project))
    with subprocess.Popen(
        command, cwd=os.path.dirname(command[0]), env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as child:
        for line in child.stdout:
            if line.startswith("INFERNUX_EDITOR_PID="):
                pid = int(line.split("=", 1)[1])
                break
            print(line, end="", flush=True)
        else:
            raise RuntimeError(f"Editor exited before project handoff ({child.wait()}).")
        reserve(project, pid, token, "editor", "launching")
        child.stdin.write("start\n")
        child.stdin.close()
        for line in child.stdout:
            print(line, end="", flush=True)
        code = child.wait()
    if code:
        raise RuntimeError(f"Editor restart failed (exit code {code}). See the startup log for details.")
    return code


def restart_after_exit(project: str) -> None:
    """Start the replacement only after this process releases its runtime."""
    def launch() -> None:
        log_path = Path(project) / "Logs" / "editor-restart.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command, environment = _module_command(project, ["infernux_editor", "--wait-for", str(os.getpid()), project])
        environment.pop("_INFERNUX_READY_FILE", None)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                subprocess.Popen(
                    command, cwd=os.path.dirname(command[0]), env=environment,
                    stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError:
                detail = traceback.format_exc()
                log.write(detail)
                show_system_error_dialog("Editor restart failed", detail + "\n" + str(log_path))

    atexit.register(launch)


def _wait_for_process(pid: int) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    else:
        import ctypes
        import errno
        import select

        # Linux x86-64 and arm64 use 434. Conda can omit os.pidfd_open
        # when CPython was built with older headers than the host kernel.
        kernel = ctypes.CDLL(None, use_errno=True)
        kernel.syscall.restype = ctypes.c_long
        descriptor = kernel.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
        if descriptor < 0:
            error = ctypes.get_errno()
            if error == errno.ESRCH:
                return
            raise OSError(error, os.strerror(error))
        try:
            select.select([descriptor], [], [])
        finally:
            os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("--publish", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wait-for", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--handoff", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan-startup", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    project = os.path.realpath(args.project)
    if args.wait_for is not None:
        _wait_for_process(args.wait_for)
    if args.publish or args.plan_startup:
        from infernux.plugins.manager import PluginManager

        manager = PluginManager(project, installation_only=True)
        if args.plan_startup:
            Path(args.plan_startup).write_text(json.dumps(manager._plan_startup_python()), encoding="utf-8")
        else:
            manager._complete_pending_install()
        return 0
    from infernux_project_lock import lock_path, remove_lock, reserve

    if args.handoff:
        print(f"INFERNUX_EDITOR_PID={os.getpid()}", flush=True)
        if sys.stdin.readline() != "start\n":
            raise RuntimeError("The editor launcher did not complete project handoff.")
        token = os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"]
    else:
        token = uuid.uuid4().hex
        reserve(project, os.getpid(), token, "editor", "preparing")
    try:
        os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"] = token
        os.environ["_INFERNUX_PROJECT_LOCK_PATH"] = lock_path(project)
        if not args.handoff and prepare(project):
            return launch_clean(project)
        if not args.handoff:
            reserve(project, os.getpid(), token, "editor", "launching")
        os.chdir(project)
        runtime = os.environ.get("_INFERNUX_EDITOR_RUNTIME")
        if runtime is not None:
            sys.executable = json.loads(runtime)["project_executable"]
            sys._base_executable = sys.executable
            os.environ.pop("PYTHONHOME", None)
        from infernux.engine import release_engine

        release_engine(project)
        return 0
    finally:
        remove_lock(lock_path(project), token)


def show_system_error_dialog(title: str, message: str) -> None:
    """Present startup failures without importing the engine being prepared."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    else:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            messagebox.showerror(title, message, parent=root)
        finally:
            root.destroy()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        if "--wait-for" in sys.argv:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, flush=True)
            project = os.path.realpath(sys.argv[-1])
            log = Path(project) / "Logs" / "editor-restart.log"
            show_system_error_dialog("Editor restart failed", detail + "\n" + str(log))
        raise
