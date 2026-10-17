"""One Hub per user data directory, with local activation of the existing window."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import sys
import time
import uuid

from PySide6.QtNetwork import QLocalServer, QLocalSocket


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _open = _kernel.CreateFileW
    _open.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _open.restype = wintypes.HANDLE
    _close = _kernel.CloseHandle
    _close.argtypes = [wintypes.HANDLE]
    _close.restype = wintypes.BOOL


def _read_endpoint(path: Path):
    if sys.platform != "win32":
        return json.loads(path.read_text(encoding="utf-8"))
    # A concurrent publisher may own DELETE access during atomic replacement.
    # CRT open() omits FILE_SHARE_DELETE and rejects that otherwise valid read.
    handle = _open(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _close(handle)
        raise
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        return json.load(stream)


class HubInstance:
    """Hold the process lock until the UI and its services have finished shutting down.

    The lock owns a randomly named, user-only local socket. Publishing its name
    atomically avoids platform socket path limits without hashing user paths.
    Only competing launches wait, and only for the primary's startup handshake.
    The running Hub does no polling or filesystem work per frame.
    """

    def __init__(self, data_root: str | Path):
        root = Path(data_root)
        root.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(root / "hub-instance.lock", os.O_RDWR | os.O_CREAT, 0o600)
        self._owned = False
        self._endpoint = root / "hub-instance.json"
        self._server = QLocalServer()
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._activate_clients)
        self._activate = None

    def acquire_or_activate(self, timeout_ms: int = 15000) -> bool:
        """Return True only to the owner; a successful secondary exits with False."""
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            if self._try_lock():
                try:
                    self._publish_endpoint()
                except BaseException:
                    self.close()
                    raise
                return True
            try:
                endpoint = _read_endpoint(self._endpoint)
            except FileNotFoundError:
                endpoint = None  # The owner has not published its socket yet.
            if endpoint:
                socket = QLocalSocket()
                socket.connectToServer(endpoint["server"])
                if socket.waitForConnected(max(1, min(250, int((deadline - time.monotonic()) * 1000)))):
                    if sys.platform == "win32":
                        # The newly clicked process may transfer foreground rights
                        # to the existing Hub; Windows otherwise blocks focus theft.
                        import ctypes
                        ctypes.windll.user32.AllowSetForegroundWindow(endpoint["pid"])
                    socket.write(b"activate\n")
                    socket.flush()
                    while not socket.canReadLine():
                        remaining = max(1, int((deadline - time.monotonic()) * 1000))
                        if not socket.waitForReadyRead(remaining):
                            break
                    if socket.canReadLine() and bytes(socket.readLine()) == b"activated\n":
                        socket.disconnectFromServer()
                        return False
                    raise RuntimeError("The running Infernux Hub did not respond to the activation request.")
            if time.monotonic() >= deadline:
                raise RuntimeError("Infernux Hub is already starting or shutting down. Please try again shortly.")
            time.sleep(0.05)

    def _try_lock(self) -> bool:
        # Kernel ownership ends even when a crashed process still has an open
        # process handle in its parent. PID/executable-name probes cannot ensure
        # that, and a live lock must never expire just because time has passed.
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        self._owned = True
        return True

    def _publish_endpoint(self) -> None:
        name = "InfernuxHub-" + uuid.uuid4().hex
        if not self._server.listen(name):
            raise RuntimeError(f"Cannot listen for Hub activation: {self._server.errorString()}")
        payload = json.dumps({"pid": os.getpid(), "server": name}).encode("utf-8")
        target = self._endpoint.with_name(name + ".json.tmp")
        try:
            with target.open("xb") as stream:
                stream.write(payload)
            # Close the writer before publication: Windows readers must never
            # see an endpoint still held by the publishing file handle.
            target.replace(self._endpoint)
        finally:
            target.unlink(missing_ok=True)

    def set_activation_handler(self, callback) -> None:
        self._activate = callback
        self._activate_clients()  # Includes clicks received during window construction.

    def _activate_clients(self) -> None:
        if self._activate is None:
            return
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            socket.setReadBufferSize(64)
            socket.disconnected.connect(socket.deleteLater)
            socket.readyRead.connect(lambda s=socket: self._read_activation(s))
            self._read_activation(socket)

    def _read_activation(self, socket) -> None:
        if not socket.canReadLine():
            return
        if bytes(socket.readLine()) == b"activate\n":
            self._activate()
            socket.write(b"activated\n")
            socket.disconnectFromServer()
        else:
            socket.abort()

    def close(self) -> None:
        self._server.close()
        try:
            if self._owned:
                self._endpoint.unlink(missing_ok=True)
        finally:
            self._owned = False
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            # Keep the guard file's identity stable. Unlinking it would let
            # concurrent launches lock different files at the same pathname.
