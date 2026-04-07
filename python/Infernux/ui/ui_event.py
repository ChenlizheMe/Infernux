"""UIEvent — lightweight callback list, similar to Unity's ``UnityEvent``.

Allows UI components (e.g. ``UIButton``) to expose events that scripts
can subscribe to at runtime.

Usage::

    btn = go.get_component(UIButton)
    btn.on_click.add_listener(my_handler)

    # Inside UIButton:
    self.on_click.invoke()
"""

from __future__ import annotations

from typing import Any, Callable, List
from Infernux.debug import Debug


class UIEvent:
    """A multicast delegate / callback list.

    Call :meth:`invoke` to notify all registered listeners.
    Listeners are plain callables (no arguments by default).  For events
    that carry data, use :class:`UIEvent1`.
    """

    __slots__ = ("_listeners",)

    def __init__(self):
        self._listeners: List[Callable[[], Any]] = []

    def add_listener(self, callback: Callable[[], Any]) -> None:
        """Register *callback* to be called on :meth:`invoke`."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], Any]) -> None:
        """Unregister *callback*."""
        try:
            self._listeners.remove(callback)
        except ValueError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    def remove_all_listeners(self) -> None:
        """Clear every registered listener."""
        self._listeners.clear()

    def invoke(self) -> None:
        """Fire all listeners (order of registration)."""
        for cb in self._listeners:
            cb()

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def __repr__(self):
        return f"UIEvent(listeners={len(self._listeners)})"


class UIEvent1:
    """A one-argument variant: ``UIEvent1[T]``.

    Usage::

        on_value_changed = UIEvent1()          # carries new value
        on_value_changed.add_listener(lambda v: print(v))
        on_value_changed.invoke(42)
    """

    __slots__ = ("_listeners",)

    def __init__(self):
        self._listeners: List[Callable[[Any], Any]] = []

    def add_listener(self, callback: Callable[[Any], Any]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Any], Any]) -> None:
        try:
            self._listeners.remove(callback)
        except ValueError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    def remove_all_listeners(self) -> None:
        self._listeners.clear()

    def invoke(self, arg: Any) -> None:
        for cb in self._listeners:
            cb(arg)

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def __repr__(self):
        return f"UIEvent1(listeners={len(self._listeners)})"
