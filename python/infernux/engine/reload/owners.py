"""Lifetimes of script registrations; retirement never restores old code."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
import sys
from types import FunctionType


_current = ContextVar("reload_owner", default=None)
_modules = {}


class ReloadOwner:
    def __init__(self, paths, *, parent=None, coordinator=None):
        self.paths = frozenset(paths)
        self.active = bool(parent and parent.active)
        self.closed = False
        self._cleanup = []
        self._parent = parent
        self.coordinator = parent.coordinator if parent is not None else coordinator
        self._children = set()
        self.library_roots = set()
        self._dependencies = {}
        if parent is not None:
            parent._children.add(self)

    def child(self):
        if self.closed:
            raise ReferenceError("Cannot create work under a retired source owner")
        return ReloadOwner(self.paths, parent=self)

    @property
    def source_owner(self):
        owner = self
        while owner._parent is not None:
            owner = owner._parent
        return owner

    @contextmanager
    def enter(self):
        # Cleanup (including generator.close()) retains attribution after
        # admission closes. add_cleanup() immediately retires any work it creates.
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def add_cleanup(self, callback):
        if self.closed:
            callback()
            return
        self._cleanup.append(callback)

    def bind_modules(self, names):
        for name in names:
            _modules[name] = self

    def stop_admission(self):
        """Stop callbacks immediately; structural retirement waits for a safe point."""
        self.active = False
        for child in tuple(self._children):
            child.stop_admission()

    def start_admission(self):
        if self.closed:
            raise ReferenceError("Cannot admit a retired source owner")
        self.active = True
        for child in tuple(self._children):
            child.start_admission()

    def close(self):
        if self.closed:
            return
        self.active = False
        self.closed = True
        self._dependencies.clear()
        if self._parent is not None:
            self._parent._children.discard(self)
        for name in tuple(_modules):
            if _modules[name] is self:
                del _modules[name]
        callbacks, self._cleanup = self._cleanup, []
        errors = []
        for child in tuple(self._children):
            try:
                child.close()
            except Exception as exc:
                errors.append(exc)
        for callback in reversed(callbacks):
            try:
                with self.enter():
                    callback()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Script registration cleanup failed", errors)


def owner_for(callback=None):
    from infernux.execution import _current_owner

    while isinstance(callback, partial):
        callback = callback.func
    instance = getattr(callback, "__self__", None)
    function = getattr(callback, "__func__", callback)
    if not isinstance(function, FunctionType):
        function = getattr(getattr(callback, "__call__", None), "__func__", None)
    source_owner = (
        function.__globals__.get("__infernux_reload_owner__") if isinstance(function, FunctionType)
        else getattr(callback, "__infernux_reload_owner__", None)
    )
    if source_owner is not None and source_owner.closed:
        raise ReferenceError("Callable belongs to retired source")
    owner = (
        getattr(instance, "_reload_owner", None)
        or getattr(_current_owner.get(), "_reload_owner", None)
        or _current.get()
        or source_owner
    )
    if owner is not None:
        return owner
    name = getattr(callback, "__module__", "")
    owner = _modules.get(name)
    if owner is not None and isinstance(callback, FunctionType):
        module = sys.modules.get(name)
        if module is None or callback.__globals__ is not vars(module):
            raise ReferenceError(f"Callback belongs to retired source: {name}")
    return owner


def current_owner():
    """The admitted source lifetime, including work created by a callback."""
    return _current.get()


def own_cleanup(callback, cleanup):
    owner = owner_for(callback)
    if owner is not None:
        owner.add_cleanup(cleanup)
    return owner
