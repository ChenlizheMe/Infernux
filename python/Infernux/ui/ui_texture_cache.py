"""Shared UI texture cache.

Both the UI-Editor panel (ImGui preview) and the Game-View panel
(runtime overlay) need to load project textures and convert them to
ImGui-compatible texture IDs.  This module provides a single cache
so the work is done once, regardless of which panel loads the texture
first.

Cache keys are GUIDs (resolved from paths via AssetDatabase).  This
ensures cache entries survive file renames/moves.  Falls back to the
raw path when no GUID is available.
"""

from __future__ import annotations

import os
from typing import Optional
from Infernux.debug import Debug


class UITextureCache:
    """GUID-keyed texture-path → ImGui-texture-ID cache.

    Call ``get(engine, tex_path)`` from any panel.  The cache is shared
    as a module-level singleton via ``get_shared_cache()``.
    """

    def __init__(self):
        self._cache: dict[str, int] = {}        # GUID (or path fallback) → tid
        self._path_to_key: dict[str, str] = {}  # path → cache key (GUID or path)

    # ── internal ─────────────────────────────────────────────────────

    def _resolve_key(self, tex_path: str) -> str:
        """Resolve *tex_path* to a GUID cache key; fall back to path."""
        cached = self._path_to_key.get(tex_path)
        if cached:
            return cached
        try:
            from Infernux.lib import AssetRegistry
            adb = AssetRegistry.instance().get_asset_database()
            if adb:
                guid = adb.get_guid_from_path(tex_path)
                if guid:
                    self._path_to_key[tex_path] = guid
                    return guid
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        return tex_path  # fallback: use path

    # ── public API ───────────────────────────────────────────────────

    def get(self, engine, tex_path: str) -> int:
        """Return the ImGui texture ID for *tex_path*, loading if needed."""
        if not tex_path:
            return 0
        key = self._resolve_key(tex_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if engine is None:
            return 0
        native = engine.get_native_engine()
        if native is None:
            return 0
        from Infernux.engine.project_context import get_project_root
        project_root = get_project_root()
        if not project_root:
            return 0
        abs_path = os.path.normpath(os.path.join(project_root, tex_path))
        if not os.path.isfile(abs_path):
            self._cache[key] = 0
            return 0
        cache_name = f"__ui_img__{key}"
        if native.has_imgui_texture(cache_name):
            tid = native.get_imgui_texture_id(cache_name)
            self._cache[key] = tid
            return tid
        from Infernux.lib import TextureLoader
        td = TextureLoader.load_from_file(abs_path)
        if not td or not td.is_valid():
            self._cache[key] = 0
            return 0
        pixels = td.get_pixels_list()
        tid = native.upload_texture_for_imgui(cache_name, pixels, td.width, td.height)
        self._cache[key] = tid
        return tid

    def get_bound(self, engine):
        """Return a callable ``f(tex_path) -> tid`` bound to *engine*.

        Avoids creating a fresh lambda every frame.
        """
        # Use functools.partial-like approach with a simple closure, cached per engine id
        key = id(engine)
        cached = getattr(self, '_bound_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]

        def _lookup(tex_path, _self=self, _eng=engine):
            return _self.get(_eng, tex_path)

        self._bound_cache = (key, _lookup)
        return _lookup

    def invalidate(self, identifier: Optional[str] = None):
        """Drop cached entries.  *identifier* may be a GUID or a file path."""
        if identifier is None:
            self._cache.clear()
            self._path_to_key.clear()
        else:
            # Direct removal (identifier is a GUID key)
            self._cache.pop(identifier, None)
            # Resolve path → key and remove that too
            resolved = self._path_to_key.pop(identifier, None)
            if resolved and resolved != identifier:
                self._cache.pop(resolved, None)


# ── module-level singleton ────────────────────────────────────────────

_shared: Optional[UITextureCache] = None


def get_shared_cache() -> UITextureCache:
    """Return (creating if needed) the module-level shared cache."""
    global _shared
    if _shared is None:
        _shared = UITextureCache()
    return _shared
