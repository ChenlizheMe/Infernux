"""Type stubs for infernux.ui.ui_event — lightweight callback list."""

from __future__ import annotations

from typing import Any, Callable


class UIEvent:
    """A multicast delegate / callback list (zero-argument).

    Similar to Unity's ``UnityEvent``.  UI components expose events that
    scripts can subscribe to at runtime.

    Example::

        btn.on_click.add_listener(my_handler)
        btn.on_click.invoke()
    """

    def __init__(self) -> None: ...

    def add_listener(self, callback: Callable[[], Any]) -> None:
        """Register *callback* to be called on :meth:`invoke`.

        Duplicate registrations are silently ignored.
        """
        ...

    def remove_listener(self, callback: Callable[[], Any]) -> None:
        """Unregister *callback*.  No error if not found."""
        ...

    def remove_all_listeners(self) -> None:
        """Clear every registered listener."""
        ...

    def invoke(self) -> None:
        """Fire all listeners in registration order."""
        ...

    @property
    def listener_count(self) -> int:
        """Number of currently registered listeners."""
        ...

    def __repr__(self) -> str: ...


class UIEvent1:
    """A one-argument variant of :class:`UIEvent`.

    Example::

        on_value_changed = UIEvent1()
        on_value_changed.add_listener(lambda v: print(v))
        on_value_changed.invoke(42)
    """

    def __init__(self) -> None: ...

    def add_listener(self, callback: Callable[[Any], Any]) -> None:
        """Register *callback* (receives one argument on invoke)."""
        ...

    def remove_listener(self, callback: Callable[[Any], Any]) -> None:
        """Unregister *callback*."""
        ...

    def remove_all_listeners(self) -> None:
        """Clear every registered listener."""
        ...

    def invoke(self, arg: Any) -> None:
        """Fire all listeners, passing *arg* to each."""
        ...

    @property
    def listener_count(self) -> int:
        """Number of currently registered listeners."""
        ...

    def __repr__(self) -> str: ...
