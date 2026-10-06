"""Shared stdlib-only project ownership protocol for Hub and engine hosts.

The stable guard file serializes short metadata transactions. The JSON lease
identifies the live owner between transactions; it is published atomically so
read-only discovery clients never observe a partially written document.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import cache


LOCK_NAME = ".infernux-engine-lock.json"
GUARD_NAME = ".infernux-engine-lock.guard"


@cache
def _windows_api():
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
    kernel.LockFileEx.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel, Overlapped


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        from ctypes import wintypes
        kernel, _ = _windows_api()
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such process.
                return False
            raise ctypes.WinError(error)
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_path(project: str) -> str:
    return os.path.join(os.path.realpath(project), "ProjectSettings", LOCK_NAME)


@contextmanager
def _guard(path: str):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    # Keep the guard outside managed roots such as ProjectSettings, which a
    # checkpoint can replace as a directory. Its filesystem identity is stable.
    descriptor = os.open(os.path.join(os.path.dirname(directory), GUARD_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt
            kernel, Overlapped = _windows_api()
            offset = Overlapped()
            # Synchronous handle: the kernel waits for this byte, with no
            # polling/backoff. It is valid to lock beyond an empty file's EOF.
            if not kernel.LockFileEx(msvcrt.get_osfhandle(descriptor), 2, 0, 1, 0, ctypes.byref(offset)):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        # Closing this non-inherited handle releases the OS lock, including
        # abnormal exits. Never unlink the guard: that could split ownership.
        os.close(descriptor)


def _read(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"project lock must contain a JSON object: {path}")
    pid, token = value.get("pid"), value.get("token")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not isinstance(token, str) or not token:
        raise ValueError(f"project lock has invalid process identity: {path}")
    return value


def _publish(path: str, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".infernux-engine-lock.", suffix=".tmp",
                                             dir=os.path.dirname(path))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _payload(project: str, pid: int, token: str, mode: str, state: str) -> dict:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not isinstance(token, str) or not token:
        raise ValueError("project lock requires a positive process ID and nonempty token")
    return dict(pid=pid, token=token, mode=mode, state=state, project_path=os.path.realpath(project))


def read_lock(project: str, *, probe=is_pid_running) -> dict | None:
    path = lock_path(project)
    if not os.path.isdir(os.path.dirname(path)):
        return None
    with _guard(path):
        current = _read(path)
        if current is not None and not probe(current["pid"]):
            os.remove(path)
            return None
        return current


def claim(project: str, token: str, mode: str, *, probe=is_pid_running, require_reservation=False) -> str:
    """Claim a free project, or consume this launcher's one-use reservation."""
    path = lock_path(project)
    payload = _payload(project, os.getpid(), token, mode, "running")
    with _guard(path):
        current = _read(path)
        handoff = (current is not None and current["token"] == token
                   and ((current.get("state") == "preparing" and current["pid"] == os.getppid())
                        or (current.get("state") == "launching" and current["pid"] == os.getpid())))
        if require_reservation and not handoff:
            raise RuntimeError(f"Project launch reservation is no longer available:\n{project}")
        if current is not None and probe(current["pid"]):
            if not handoff:
                raise RuntimeError(f"Project is already open in another Infernux process:\n{project}")
        _publish(path, payload)
    return path


def reserve(project: str, pid: int, token: str, mode: str, state: str, *, probe=is_pid_running) -> str:
    """Reserve preparation, then transfer only that reservation to its child."""
    path = lock_path(project)
    payload = _payload(project, pid, token, mode, state)
    with _guard(path):
        current = _read(path)
        if state == "launching" and (current is None or current["token"] != token):
            raise RuntimeError(f"Project launch reservation is no longer available:\n{project}")
        if current is not None and probe(current["pid"]):
            if current["token"] != token:
                raise RuntimeError(f"Project is already open in another Infernux process:\n{project}")
            # The child can claim before Popen returns to Hub. Never downgrade
            # its running state or overwrite another process with this token.
            if current["pid"] == pid and current.get("state") == "running" and state == "launching":
                return path
            if current["pid"] != os.getpid() or current.get("state") != "preparing":
                raise RuntimeError(f"Project is already open in another Infernux process:\n{project}")
        _publish(path, payload)
    return path


def remove_lock(path: str, token: str | None, *, probe=is_pid_running) -> bool:
    """Retire our lease, or our stopped child's lease, under the same guard."""
    if not path or not token or not os.path.isdir(os.path.dirname(path)):
        return False
    with _guard(path):
        current = _read(path)
        if current is None or current["token"] != token:
            return False
        if current["pid"] != os.getpid() and probe(current["pid"]):
            return False
        os.remove(path)
        return True
