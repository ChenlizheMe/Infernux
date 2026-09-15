"""Stable official plugin category keys and compatibility aliases."""

from __future__ import annotations

OFFICIAL_PLUGIN_CATEGORIES = (
    "platform_build",
    "compute_simulation",
    "editor_tools",
    "online_services",
    "other",
)

_CATEGORY_ALIASES = {
    "platform": "platform_build",
    "agent": "editor_tools",
    "tool": "editor_tools",
    "tools": "editor_tools",
    "compute": "compute_simulation",
    "simulation": "compute_simulation",
    "online": "online_services",
    "service": "online_services",
    "services": "online_services",
}


def normalize_plugin_category(value: object) -> str:
    """Return a stable key while accepting legacy catalog spellings."""

    key = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if key in OFFICIAL_PLUGIN_CATEGORIES:
        return key
    return _CATEGORY_ALIASES.get(key, "other")


__all__ = ["OFFICIAL_PLUGIN_CATEGORIES", "normalize_plugin_category"]
