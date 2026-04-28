"""Main-thread command queue for the embedded MCP server."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional


class CommandFuture:
    def __init__(self, name: str):
        self.name = name
        self._event = threading.Event()
        self._result: Any = None
        self._error: Optional[BaseException] = None

    def set_result(self, value: Any) -> None:
        self._result = value
        self._event.set()

    def set_error(self, error: BaseException) -> None:
        self._error = error
        self._event.set()

    def result(self, timeout: Optional[float] = None) -> Any:
        if not self._event.wait(timeout):
            raise TimeoutError(f"MCP command timed out: {self.name}")
        if self._error is not None:
            raise self._error
        return self._result


class MainThreadCommandQueue:
    _instance: Optional["MainThreadCommandQueue"] = None

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[str, Callable[[], Any], CommandFuture]]" = queue.Queue()
        self._main_thread_id: Optional[int] = None

    @classmethod
    def instance(cls) -> "MainThreadCommandQueue":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def submit(self, name: str, fn: Callable[[], Any], *, timeout_ms: int = 30000) -> CommandFuture:
        future = CommandFuture(name)
        if self._main_thread_id == threading.get_ident():
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_error(exc)
            return future
        self._queue.put((name, fn, future))
        return future

    def run_sync(self, name: str, fn: Callable[[], Any], *, timeout_ms: int = 30000) -> Any:
        future = self.submit(name, fn, timeout_ms=timeout_ms)
        return future.result(timeout=max(timeout_ms, 1) / 1000.0)

    def drain(self, max_commands: int = 16) -> int:
        self._main_thread_id = threading.get_ident()
        processed = 0
        for _ in range(max_commands):
            try:
                _name, fn, future = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                future.set_result(fn())
            except BaseException as exc:
                future.set_error(exc)
            finally:
                processed += 1
        return processed

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._main_thread_id is not None:
                return True
            time.sleep(0.01)
        return self._main_thread_id is not None
