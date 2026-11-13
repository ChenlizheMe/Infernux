"""Pinned editor icon textures — one loader, one cache, no text fallback.

File-type and component PNGs live under ``resources/icons/`` (and
``icons/components/``). Monochrome UI glyphs live under ``icons/glyphs/``;
legacy root filenames (plus/minus/remove/picker/warning/error) are kept as
copies of the same glyph art for existing call sites.

Unknown icon names raise. A zero texture id only means the GPU upload is
still in flight — never a cue to draw ``+`` / ``-`` text.
"""

from __future__ import annotations

import os

import infernux.resources as _resources

_cache: dict[str, int] = {}
_native = None

# Glyphs under icons/glyphs/{name}.png
_GLYPHS = (
    "plus",
    "minus",
    "close",
    "more",
    "picker",
    "grip",
    "key",
    "link",
    "reset",
    "loop",
    "drag",
    "chev_d",
    "chev_r",
    "up",
    "down",
    "arrow_r",
    "back",
    "play",
    "pause",
    "stop",
    "step_b",
    "step_f",
    "record",
    "key_add",
    "key_del",
    "folder_up",
    "search",
    "dirty",
    "check",
    "lock",
    "eye",
    "warn",
    "error",
    "info",
    "cube",
    "canvas",
    "frame",
    "text",
    "image",
    "button",
)

# Legacy EditorIcons / Theme.ICON_IMG_* names → glyph file stem
_ALIASES = {
    "remove": "close",
    "warning": "warn",
}

# Non-glyph pinned icons that still live at icons root / tools
_ROOT_ICONS = (
    "package",
    "ui_canvas",
    "ui_text",
    "ui_image",
    "ui_button",
    "tool_none",
    "tool_move",
    "tool_rotate",
    "tool_scale",
    "tool_rect",
)

_ICONS = tuple(dict.fromkeys([*_GLYPHS, *_ALIASES.keys(), *_ROOT_ICONS]))


def icon_key(name: str) -> str:
    """Canonical resource key shared by Python and native editor panels."""
    stem = _glyph_stem(name)
    return f"glyphs/{stem}" if stem in _GLYPHS else stem


def _resolve_native(native_engine=None):
    global _native
    if native_engine is not None:
        _native = native_engine
        return _native
    if _native is not None:
        return _native
    from infernux.engine.ui.editor_services import EditorServices

    svc = EditorServices.instance()
    eng = svc.native_engine if svc is not None else None
    if eng is not None:
        _native = eng
    return _native


def _glyph_stem(name: str) -> str:
    return _ALIASES.get(name, name)


def resolve_icon_path(name: str) -> str:
    """Return absolute PNG path for a known icon. Raises if the name or file is unknown."""
    if name not in _ICONS:
        raise KeyError(f"Unknown editor icon: {name!r}")

    stem = _glyph_stem(name)
    if stem in _GLYPHS:
        path = os.path.join(_resources.file_type_icons_dir, "glyphs", f"{stem}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing glyph PNG for icon {name!r}: {path}")
        return path

    path = os.path.join(_resources.file_type_icons_dir, f"{name}.png")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing icon PNG: {path}")
    return path


def _ensure_loaded(native_engine) -> None:
    native = _resolve_native(native_engine)
    for name in _ICONS:
        _cache[name] = int(native.get_editor_icon_id(icon_key(name)))


class EditorIcons:
    """Thin façade around the module-level pinned icon cache."""

    GLYPHS = _GLYPHS

    @staticmethod
    def get(native_engine, name: str) -> int:
        """Return ImGui texture id for *name*, or 0 while the GPU upload is in flight."""
        native = _resolve_native(native_engine)
        texture_id = _cache.get(name, 0)
        if not texture_id:
            texture_id = int(native.get_editor_icon_id(icon_key(name)))
            _cache[name] = texture_id
        return texture_id

    @staticmethod
    def get_cached(name: str) -> int:
        """Return live icon id via EditorServices native engine when needed."""
        native = _resolve_native()
        if native is not None:
            return EditorIcons.get(native, name)
        return _cache.get(name, 0)

    @staticmethod
    def preload(native_engine) -> None:
        """Eagerly submit all known editor icons (call once after renderer init)."""
        EditorIcons.reset()
        _ensure_loaded(native_engine)

    @staticmethod
    def reset():
        """Clear the cache (e.g. after engine re-init)."""
        global _native
        _cache.clear()
        _native = None
