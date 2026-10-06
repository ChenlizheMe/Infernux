"""Bounded, observed read models for editor data services.

This is not a write baseline or an asset identity cache. Readers observe file
metadata on every query (once per panel submission); parsing and directory
discovery only repeat when an input changes. Write commands must still acquire
their own current document.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Hashable, TypeVar

from infernux.engine.path_utils import lexical_path

T = TypeVar("T")
_presentation = threading.local()


@contextmanager
def read_model_frame():
    """EditorPanel's internal boundary, never required from panel authors.

    A visible panel observes each model once per submission. Worker threads
    retain normal query freshness; explicit service publication clears entries
    immediately, even within the same submission.
    """
    previous = getattr(_presentation, "frame", None)
    if previous is not None:
        yield
        return
    _presentation.frame = {}
    try:
        yield
    finally:
        _presentation.frame = None


def _stamp(path: str):
    try:
        value = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


if os.name == "nt":
    from ._windows_file_observation import file_stamp as _stamp


class FileObservations:
    """Dependencies collected by a service while preparing a read model."""

    def __init__(self) -> None:
        self._files = {}

    def watch(self, path: str | os.PathLike[str]) -> None:
        path = lexical_path(path)
        if path not in self._files:
            self._files[path] = _stamp(path)

    def watch_tree(
        self, root: str | os.PathLike[str], *, suffixes: frozenset[str] | None = None,
    ) -> None:
        """Observe existing descendants and directories where new ones appear."""
        self.watch(root)

        def fail(error):
            raise error

        if not os.path.isdir(root):
            return
        for directory, directories, files in os.walk(root, onerror=fail):
            self.watch(directory)
            for name in directories:
                self.watch(os.path.join(directory, name))
            for name in files:
                if suffixes is None or os.path.splitext(name)[1].casefold() in suffixes:
                    self.watch(os.path.join(directory, name))

    def unchanged(self) -> bool:
        return all(_stamp(path) == stamp for path, stamp in self._files.items())


class FileReadCache:
    """Service-owned memoization with no TTL, content hashes or UI opt-in.

    Values are owned by the service: return copies or immutable projections to
    callers. Errors are never replaced with a previously successful value.
    A file changing during preparation makes that result ineligible for reuse.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("Read cache capacity must be positive")
        self._capacity = capacity
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._generation = 0

    def get(
        self, key: Hashable, prepare: Callable[[FileObservations], T], *, current: bool = False,
    ) -> T:
        with self._lock:
            frame = None if current else getattr(_presentation, "frame", None)
            frame_key = (self, key)
            if frame is not None:
                retained = frame.get(frame_key)
                if retained is not None and retained[0] == self._generation:
                    return retained[1]
            entry = self._entries.get(key)
            if entry is not None:
                observations, value = entry
                if observations.unchanged():
                    self._entries.move_to_end(key)
                    if frame is not None:
                        frame[frame_key] = self._generation, value
                    return value
                del self._entries[key]
            observations = FileObservations()
            value = prepare(observations)
            if observations.unchanged():
                self._entries[key] = observations, value
                if len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
            if frame is not None:
                frame[frame_key] = self._generation, value
            return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._generation += 1
