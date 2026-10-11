"""EditorIcons — centralized editor icon texture loader.

Example::

    tid = EditorIcons.get(native_engine, "plus")   # -> int texture id
"""

from __future__ import annotations


class EditorIcons:
    """GPU-cached editor icon textures."""

    @staticmethod
    def get(native_engine: object, name: str) -> int:
        """Return ImGui texture id for *name*, or ``0`` if unavailable.

        Available names: ``plus``, ``minus``, ``remove``, ``picker``,
        ``warning``, ``error``, ``ui_text``, ``ui_image``, ``ui_button``,
        ``tool_none``, ``tool_move``, ``tool_rotate``, ``tool_scale``, ``tool_rect``.

        Args:
            native_engine: The C++ native engine instance.
            name: Icon name (without extension).
        """
        ...

    @staticmethod
    def reset() -> None:
        """Clear the cache (call after engine re-init)."""
        ...
