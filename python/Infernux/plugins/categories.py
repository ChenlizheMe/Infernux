"""Stable official plugin category keys."""

from __future__ import annotations

OFFICIAL_PLUGIN_CATEGORIES = (
    "platform_build",
    "compute_simulation",
    "editor_tools",
    "online_services",
    "other",
)

def normalize_plugin_category(value: object) -> str:
    """Return an official key, or ``other`` for an unknown category."""

    key = str(value or "").strip()
    if key in OFFICIAL_PLUGIN_CATEGORIES:
        return key
    return "other"


__all__ = ["OFFICIAL_PLUGIN_CATEGORIES", "normalize_plugin_category"]
