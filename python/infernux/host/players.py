"""Owned desktop Player processes and their existing engine control channel."""

from collections import deque
import json
import os
from pathlib import Path
import subprocess
from threading import Condition, Lock, Thread
import uuid

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .context import automation_context
from .queries import emit_signal


class ManagedPlayers:
    def __init__(self):
        self._condition = Condition()
        self._command = Lock()
        self._lifecycle = Lock()
        self.process = None
        self._observer = None
        self._log = None
        self._ready = ""
        self._response = {}
        self._error = ""

    def launch(self, executable, arguments=(), start_scene="", timeout=30.0):
        with self._lifecycle:
            return self._launch(executable, arguments, start_scene, timeout)

    def _launch(self, executable, arguments, start_scene, timeout):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("An owned Player is already running")
        context = automation_context()
        root = Path(context.artifact_root) / ("player-" + uuid.uuid4().hex)
        root.mkdir()
        self.root = root
        self._ready, self._response, self._error = "", {}, ""
        self.token = uuid.uuid4().hex
        self.request_path = root / "request.json"
        self.response_path = root / "response.json"
        self.ready_path = root / "ready.txt"
        controller = self

        class Changes(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory or event.event_type not in {"created", "modified", "moved"}:
                    return
                path = Path(getattr(event, "dest_path", "") or event.src_path)
                try:
                    if path == controller.ready_path:
                        value = path.read_text(encoding="utf-8").strip()
                        with controller._condition:
                            controller._ready = value
                            controller._condition.notify_all()
                    elif path == controller.response_path:
                        value = json.loads(path.read_text(encoding="utf-8"))
                        with controller._condition:
                            controller._response = value
                            controller._condition.notify_all()
                    else:
                        return
                    emit_signal("player.changed")
                except FileNotFoundError:
                    return  # A later atomic replacement is a separate filesystem event.
                except Exception as exc:
                    with controller._condition:
                        controller._error = str(exc)
                        controller._condition.notify_all()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        if self._log is not None:
            self._log.close()
        self._observer = Observer()
        self._observer.schedule(Changes(), str(root), recursive=False)
        self._observer.start()
        environment = dict(os.environ)
        environment.update(_INFERNUX_READY_FILE=str(self.ready_path),
                           _INFERNUX_PLAYER_CONTROL_FILE=str(self.request_path),
                           _INFERNUX_PLAYER_RESPONSE_FILE=str(self.response_path),
                           _INFERNUX_PLAYER_CONTROL_TOKEN=self.token,
                           _INFERNUX_PLAYER_ARTIFACT_ROOT=str(root))
        if start_scene:
            environment["_INFERNUX_PLAYER_START_SCENE_GUID"] = start_scene
        executable = str(Path(executable).resolve())
        self._log = (root / "stdout.log").open("wb")
        self.process = subprocess.Popen([executable, *arguments], cwd=str(Path(executable).parent),
                                        env=environment, stdout=self._log, stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)

        def exited(process=self.process):
            process.wait()
            with self._condition:
                self._condition.notify_all()
            emit_signal("player.changed")

        Thread(target=exited, name="InfernuxPlayerProcess", daemon=True).start()
        with self._condition:
            ready = self._condition.wait_for(lambda: self._ready or self._error or self.process.poll() is not None, timeout)
            if not ready or self._ready != "ENGINE_LOADED":
                raise RuntimeError(self._error or self._ready or f"Player did not become ready: {self.status()}")
        return self.status()

    def status(self):
        process = self.process
        running = process is not None and process.poll() is None
        return {"running": running,
                "pid": process.pid if process is not None else 0,
                "exit_code": process.poll() if process is not None else None,
                "ready": running and self._ready == "ENGINE_LOADED", "error": self._error,
                "artifact_root": str(self.root) if process is not None else ""}

    def control(self, action, arguments=None, timeout=30.0):
        with self._command:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("The owned Player is not running")
            identity = uuid.uuid4().hex
            value = {**(arguments or {}), "action": action, "command_id": identity, "token": self.token}
            temporary = self.request_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.request_path)
            with self._condition:
                complete = self._condition.wait_for(lambda: self._response.get("command_id") == identity
                                                   or self._error or self.process.poll() is not None, timeout)
                if not complete or self._response.get("command_id") != identity:
                    raise TimeoutError(f"Player command outcome is unknown: {action}. {self._error}")
                response = self._response
            if not response["ok"]:
                raise RuntimeError(str(response.get("error", "Player command failed")))
            return response["data"]

    def logs(self, limit=200):
        if self.process is None:
            return {"lines": []}
        with (self.root / "stdout.log").open(encoding="utf-8", errors="replace") as stream:
            return {"lines": list(deque(stream, maxlen=limit))}

    def stop(self, timeout=15.0):
        with self._lifecycle:
            if self.process is not None and self.process.poll() is None:
                self.control("shutdown", timeout=timeout)
                self.process.wait(timeout=timeout)
            return self.status()

    def close_service(self):
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        if self._log is not None:
            self._log.close()
            self._log = None


_players = None


def managed_players():
    global _players
    if _players is None:
        _players = ManagedPlayers()
    return _players


def close_service():
    global _players
    if _players is not None:
        _players.close_service()
        _players = None
