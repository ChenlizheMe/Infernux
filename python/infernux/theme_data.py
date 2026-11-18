"""Standard-library-only theme declarations shared with exported packagers.

This file is copied unchanged to theme_schema.py in standalone templates.
It deliberately imports neither infernux nor the native engine.
"""
from __future__ import annotations
import math
import re

THEME_FIELDS = frozenset({"$schema", "id", "name", "author", "description", "colors", "vec2s", "floats"})


def theme_token_names(table: str) -> dict[str, list[str]]:
    kinds = {"COLOR": "colors", "VEC2": "vec2s", "FLOAT": "floats"}
    result = {kind: [] for kind in kinds.values()}
    for kind, name in re.findall(r"^INX_THEME_(COLOR|VEC2|FLOAT)\((\w+),", table, flags=re.M):
        if not name.startswith("UI_DEFAULT_"):
            result[kinds[kind]].append(name)
    return result


def validate_theme_document(document: object, tokens: dict) -> dict:
    """Normalize current declarations, rejecting unknown fields and bad metadata."""
    if not isinstance(document, dict):
        raise ValueError("Theme must be a JSON object")
    unknown = set(document) - THEME_FIELDS
    if unknown:
        raise ValueError("Unknown theme fields: " + ", ".join(sorted(unknown)))
    if "$schema" in document and document["$schema"] != "infernux.editor_theme":
        raise ValueError("Unsupported editor theme schema")
    theme_id = document.get("id")
    if not isinstance(theme_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", theme_id):
        raise ValueError("Theme id must be a nonempty package-local name using letters, numbers, '.', '_' or '-'")
    result = {"id": theme_id}
    for key, default in (("name", theme_id), ("author", ""), ("description", "")):
        value = document.get(key, default)
        if not isinstance(value, str) or (key == "name" and not value.strip()):
            raise ValueError(f"Theme {key} must be {'a nonempty string' if key == 'name' else 'a string'}")
        result[key] = value
    for kind, count in (("colors", 4), ("vec2s", 2), ("floats", 0)):
        values = document.get(kind, {})
        if not isinstance(values, dict):
            raise ValueError(f"Theme {kind} must be an object")
        supported = set(tokens[kind])
        normalized = {}
        for token, value in values.items():
            if token.startswith("UI_DEFAULT_") or token not in supported:
                raise ValueError(f"Unknown editor {kind} token: {token}")
            channels = value if count else [value]
            if count and (not isinstance(channels, (list, tuple)) or len(channels) != count):
                raise ValueError(f"Theme {token} requires {count} numbers")
            if any(isinstance(channel, bool) or not isinstance(channel, (int, float))
                   or not math.isfinite(channel) or abs(channel) > 3.402823466e38 for channel in channels):
                raise ValueError(f"Theme {token} requires finite numbers")
            if kind == "colors" and any(channel < 0 or channel > 1 for channel in channels):
                raise ValueError(f"Theme {token} requires RGBA values in [0, 1]")
            if token.startswith("GLOBAL_") and any(channel < 0 for channel in channels):
                raise ValueError(f"Theme {token} requires non-negative dimensions")
            if kind == "floats" and token.startswith(("PANEL_", "BUTTON_")) and value < 0:
                raise ValueError(f"Theme chrome dimensions must be non-negative: {token}")
            if token in ("SURFACE_GRAIN_OPACITY", "SURFACE_HEX_OPACITY", "SURFACE_PATTERN_OPACITY") and not 0 <= value <= 1:
                raise ValueError(f"Surface opacity must be in [0, 1]: {token}")
            if token in ("SURFACE_GRAIN_SCALE", "SURFACE_HEX_SCALE", "SURFACE_PATTERN_SCALE") and not 0.25 <= value <= 8:
                raise ValueError(f"Surface scale must be in [0.25, 8]: {token}")
            if token == "SURFACE_PATTERN_KIND" and value not in (0, 1, 2, 3):
                raise ValueError("Surface pattern kind must be 0, 1, 2 or 3")
            if token == "EDITOR_MONO_UI" and value not in (0, 1):
                raise ValueError("Editor monospace selection must be 0 or 1")
            normalized[token] = [float(channel) for channel in channels] if count else float(value)
        result[kind] = normalized
    return result
