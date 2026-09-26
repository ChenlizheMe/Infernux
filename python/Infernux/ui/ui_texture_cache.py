"""Shared UI texture cache.

Both the UI-Editor panel (ImGui preview) and the Game-View panel
(runtime overlay) need to load project textures and convert them to
ImGui-compatible texture IDs.  This module provides a single cache
so the work is done once, regardless of which panel loads the texture
first.

Cache keys are GUIDs resolved by the project AssetDatabase. This keeps
asset identity stable across file renames and moves.
"""

from __future__ import annotations

import os
from typing import Optional

from Infernux.core.assets import AssetManager
from Infernux.engine.path_utils import resolved_path
from Infernux.engine.texture_task_bridge import texture_stamp, query_or_schedule_texture


class UITextureCache:
    """GUID-keyed texture-resource → ImGui-texture-ID cache.

    Runtime UI should pass a :class:`TextureRef`; its GUID is resolved by the
    native imported-texture path, including cooked Player artifacts.
    String paths remain an explicit editor-preview boundary and are converted
    to a GUID exactly once.  The cache is shared
    as a module-level singleton via ``get_shared_cache()``.
    """

    def __init__(self):
        self._cache: dict[str, int] = {}  # GUID → tid
        self._path_to_key: dict[str, str] = {}  # path → GUID
        self._stamp: dict[str, int] = {}        # GUID → latest stamp
        self._pending_keys: set[str] = set()
        self._generation: int = 0

    # ── internal ─────────────────────────────────────────────────────

    def _resolve_key(self, tex_path: str) -> str:
        """Resolve *tex_path* to its required project asset GUID."""
        cached = self._path_to_key.get(tex_path)
        if cached:
            return cached
        guid = str(
            AssetManager.require_asset_database().get_guid_from_path(tex_path) or ""
        )
        if not guid:
            raise KeyError(f"UI texture is not registered in AssetDatabase: {tex_path}")
        self._path_to_key[tex_path] = guid
        return guid

    def _resolve_asset(self, identifier) -> tuple[str, str]:
        """Resolve an explicit editor preview path to its project GUID."""
        tex_path = os.fspath(identifier)
        return self._resolve_key(tex_path), tex_path

    # ── public API ───────────────────────────────────────────────────

    def get(self, engine, tex_path) -> int:
        """Resolve a texture GUID reference or publish a live RenderTexture."""
        from Infernux.core.render_texture import RenderTexture
        from Infernux.core.asset_ref import TextureRef
        from Infernux.lib import _Infernux
        if isinstance(tex_path, RenderTexture):
            tex_path = tex_path._native
        if isinstance(tex_path, _Infernux._RenderTexture):
            return int(engine.get_native_engine()._get_render_texture_ui_texture_id(tex_path))
        if not tex_path:
            return 0
        if isinstance(tex_path, TextureRef):
            guid = str(tex_path.guid or "")
            if not guid:
                raise ValueError("UI TextureRef requires a non-empty GUID")
            if engine is None or engine.get_native_engine() is None:
                return 0
            native = engine.get_native_engine()
            query = getattr(native, "_get_imported_texture_ui_texture_id", None)
            if not callable(query):
                raise RuntimeError("The current engine lacks GUID-backed UI texture binding")
            key = f"ui_img|{guid}"
            tid = int(query(key, guid))
            if tid != self._cache.get(guid, 0):
                self._generation += 1
            if tid:
                self._cache[guid] = tid
                self._pending_keys.discard(guid)
            else:
                self._cache.pop(guid, None)
                self._pending_keys.add(guid)
            return tid
        key, tex_path = self._resolve_asset(tex_path)
        cached = self._cache.get(key)
        if engine is None:
            return 0
        native = engine.get_native_engine()
        if native is None:
            return 0
        from Infernux.engine.project_context import get_project_root
        project_root = get_project_root()
        if not project_root and not os.path.isabs(tex_path):
            return 0
        abs_path = resolved_path(tex_path if os.path.isabs(tex_path) else os.path.join(project_root, tex_path))
        if not os.path.isfile(abs_path):
            if self._cache.get(key, 0) != 0:
                self._generation += 1
            self._cache[key] = 0
            self._stamp[key] = 0
            self._pending_keys.discard(key)
            return 0

        stamp = texture_stamp(abs_path, "ui_cache")
        if stamp == 0:
            if self._cache.get(key, 0) != 0:
                self._generation += 1
            self._cache[key] = 0
            self._stamp[key] = 0
            self._pending_keys.discard(key)
            return 0

        resource_key = f"ui_img|{key}"
        if cached is not None and cached != 0 and self._stamp.get(key) == stamp:
            live = int(native.get_texture_preview_texture_id(resource_key))
            if live == cached:
                return cached
            if live != 0:
                self._generation += 1
                self._cache[key] = live
                return live
            self._cache.pop(key, None)
            self._stamp.pop(key, None)

        tid, _, _ = query_or_schedule_texture(
            native,
            resource_key,
            abs_path,
            int(stamp),
            nearest=False,
            srgb=False,
        )
        # Texture preview loading is asynchronous. A zero texture id means the
        # request was queued or failed for this frame; do not cache it as final,
        # or the UI will keep returning 0 forever for the same content stamp.
        if tid != 0:
            if self._cache.get(key) != tid or self._stamp.get(key) != int(stamp):
                self._generation += 1
            self._cache[key] = tid
            self._stamp[key] = int(stamp)
            self._pending_keys.discard(key)
        else:
            self._cache.pop(key, None)
            self._stamp.pop(key, None)
            self._pending_keys.add(key)
        return tid

    @property
    def generation(self) -> int:
        """Monotonic revision for native UI command-list caching."""
        return self._generation

    @property
    def has_pending(self) -> bool:
        """Whether asynchronous UI textures still need polling."""
        return bool(self._pending_keys)

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
            self._stamp.clear()
            self._pending_keys.clear()
            self._generation += 1
        else:
            # Direct removal (identifier is a GUID key)
            self._cache.pop(identifier, None)
            self._stamp.pop(identifier, None)
            self._pending_keys.discard(identifier)
            # Resolve path → key and remove that too
            resolved = self._path_to_key.pop(identifier, None)
            if resolved and resolved != identifier:
                self._cache.pop(resolved, None)
                self._stamp.pop(resolved, None)
                self._pending_keys.discard(resolved)
            self._generation += 1


# ── module-level singleton ────────────────────────────────────────────

_shared: Optional[UITextureCache] = None


def get_shared_cache() -> UITextureCache:
    """Return (creating if needed) the module-level shared cache."""
    global _shared
    if _shared is None:
        _shared = UITextureCache()
    return _shared
