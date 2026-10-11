"""Observe Linux writes and path replacement through one shared inotify stream.

Metadata timestamps may collide even when ctime is included. File watches
observe writes through any hard link; parent watches observe path replacement.
No file contents are read or hashed. Watch ownership follows read-model entries.
"""
from __future__ import annotations

import ctypes
import errno
import os
import struct
import threading
from typing import NamedTuple
import weakref


_libc = ctypes.CDLL(None, use_errno=True)
_init = _libc.inotify_init1
_init.argtypes, _init.restype = [ctypes.c_int], ctypes.c_int
_add = _libc.inotify_add_watch
_add.argtypes, _add.restype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32], ctypes.c_int
_remove = _libc.inotify_rm_watch
_remove.argtypes, _remove.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
_MASK = (0x00000002 | 0x00000004 | 0x00000008 |  # MODIFY, ATTRIB, CLOSE_WRITE
         0x00000040 | 0x00000080 | 0x00000100 | 0x00000200 |  # MOVE, CREATE, DELETE
         0x00000400 | 0x00000800)  # DELETE_SELF, MOVE_SELF
_OVERFLOW, _IGNORED = 0x00004000, 0x00008000
_HEADER = struct.Struct("iIII")


class _Watch:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.alive = True
        self.references = 0
        self.self_revision = self.tree_revision = 0
        self.names = {}


class _Events:
    def __init__(self):
        self.lock = threading.RLock()
        self.fd = _init(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "Cannot initialize Linux file observations")
        self.watches = {}
        self.epoch = 0

    def after_fork(self):
        # A child must never drain its parent's event queue. Existing child
        # probes rearm against their own stream on the next query.
        self.lock = threading.RLock()
        os.close(self.fd)
        self.fd = _init(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "Cannot initialize child file observations")
        for watch in self.watches.values():
            watch.alive = False
        self.watches.clear()
        self.epoch += 1

    def drain(self):
        while True:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            offset = 0
            while offset < len(data):
                descriptor, mask, _, length = _HEADER.unpack_from(data, offset)
                offset += _HEADER.size
                name = data[offset:offset + length].split(b"\0", 1)[0]
                offset += length
                if mask & _OVERFLOW:
                    # Lost events invalidate every retained observation.
                    self.epoch += 1
                    continue
                watch = self.watches.get(descriptor)
                if watch is None:
                    continue
                watch.tree_revision += 1
                if name:
                    if name in watch.names:
                        watch.names[name][1] += 1
                else:
                    watch.self_revision += 1
                if mask & _IGNORED:
                    watch.alive = False
                    del self.watches[descriptor]

    def acquire(self, path, name=None):
        descriptor = _add(self.fd, os.fsencode(path), _MASK)
        if descriptor < 0:
            error = ctypes.get_errno()
            if error in (errno.ENOENT, errno.ENOTDIR):
                return None
            raise OSError(error, os.strerror(error), path)
        watch = self.watches.get(descriptor)
        if watch is None:
            watch = self.watches[descriptor] = _Watch(descriptor)
        watch.references += 1
        if name is not None:
            watch.names.setdefault(name, [0, 0])[0] += 1
        return watch, name

    def release(self, views):
        with self.lock:
            for watch, name in views:
                watch.references -= 1
                if name is not None:
                    state = watch.names[name]
                    state[0] -= 1
                    if not state[0]:
                        del watch.names[name]
                if not watch.references and watch.alive:
                    if _remove(self.fd, watch.descriptor) < 0:
                        error = ctypes.get_errno()
                        if error != errno.EINVAL:
                            raise OSError(error, os.strerror(error))
                    watch.alive = False
                    if self.watches.get(watch.descriptor) is watch:
                        del self.watches[watch.descriptor]
            # Consume IN_IGNORED before the kernel may reuse a descriptor.
            self.drain()


_events = _Events()
os.register_at_fork(after_in_child=_events.after_fork)


class FileStamp(NamedTuple):
    metadata: tuple[int, ...]
    epoch: int
    revisions: tuple


def stamp_reusable(stamp):
    return stamp is None or all(watch.alive for watch, _, _ in stamp.revisions)


class FileProbe:
    def __init__(self, path):
        self.path = path
        self.views = []
        weakref.finalize(self, _events.release, self.views)

    def __call__(self):
        with _events.lock:
            _events.drain()
            views = []
            parent, child = os.path.dirname(self.path), os.path.basename(self.path)
            try:
                while parent:
                    view = _events.acquire(parent, os.fsencode(child))
                    if view is not None:
                        views.append(view)
                        break
                    parent, child = os.path.dirname(parent), os.path.basename(parent)
                view = _events.acquire(self.path)
                if view is not None:
                    views.append(view)
            except OSError:
                _events.release(views)
                raise
            _events.release(self.views)
            self.views[:] = views
            try:
                value = os.stat(self.path)
            except (FileNotFoundError, NotADirectoryError):
                return None
            _events.drain()
            revisions = tuple((watch, watch.self_revision,
                               watch.tree_revision if name is None else watch.names[name][1])
                              for watch, name in self.views)
            return FileStamp((value.st_dev, value.st_ino, value.st_mode, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns), _events.epoch, revisions)
