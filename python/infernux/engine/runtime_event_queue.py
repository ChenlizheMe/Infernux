"""Owner-thread callbacks that must run outside native GUI drawing."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable


_pending: deque[Callable[[], Any]] = deque()


def enqueue(callback: Callable[[], Any]) -> None:
    if not callable(callback):
        raise TypeError("runtime callback must be callable")
    _pending.append(callback)


def drain() -> int:
    """Run callbacks queued during the previous native frame."""
    count = 0
    while _pending:
        callback = _pending.popleft()
        callback()
        count += 1
    return count


def clear() -> None:
    _pending.clear()
