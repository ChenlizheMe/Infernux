"""Centralized editor icon texture loader.

Lazily uploads PNG icons from ``resources/icons/`` to GPU and caches
their ImGui texture IDs.  All panels share a single cache.

Usage::

    from .editor_icons import EditorIcons
    tid = EditorIcons.get(native_engine, "plus")   # -> int texture id
"""

import os
import Infernux.resources as _resources
from Infernux.engine.texture_task_bridge import texture_stamp, query_or_schedule_texture

_cache: dict[str, int] = {}


def _ensure_loaded(native_engine) -> None:
    """Upload all known editor icons (once)."""
    if native_engine is None:
        return

    _ICONS = [
        "plus", "minus", "remove", "picker",
        "warning", "error",
        "ui_text", "ui_image", "ui_button",
        "tool_none", "tool_move", "tool_rotate", "tool_scale",
    ]
    for name in _ICONS:
        path = os.path.join(_resources.file_type_icons_dir, f"{name}.png")
        if not os.path.isfile(path):
            continue
        stamp = texture_stamp(path, "editor_icon")
        if stamp == 0:
            continue
        tid, _, _ = query_or_schedule_texture(
            native_engine,
            f"edicon|{name}",
            path,
            int(stamp),
            nearest=False,
            srgb=False,
        )
        if tid != 0:
            _cache[name] = tid


class EditorIcons:
    """Thin façade around the module-level icon cache."""

    @staticmethod
    def get(native_engine, name: str) -> int:
        """Return ImGui texture id for *name*, or 0 if unavailable."""
        _ensure_loaded(native_engine)
        return _cache.get(name, 0)

    @staticmethod
    def get_cached(name: str) -> int:
        """Return a previously loaded icon id, or 0.  No engine required."""
        return _cache.get(name, 0)

    @staticmethod
    def reset():
        """Clear the cache (e.g. after engine re-init)."""
        _cache.clear()
