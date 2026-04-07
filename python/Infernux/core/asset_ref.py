"""
Asset reference types for InxComponent serialized fields.

Provides GUID-based asset references that lazily resolve to loaded assets.
They mirror the C++ ``AssetRef<T>`` pattern and integrate with the Inspector.

All asset ref types inherit from ``AssetRefBase`` and override ``_do_resolve``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from Infernux.debug import Debug

_log = logging.getLogger("Infernux.ref")


def _get_asset_database():
    """Return the C++ AssetDatabase, trying AssetManager first then engine."""
    try:
        from Infernux.core.assets import AssetManager
        if AssetManager._asset_database is not None:
            return AssetManager._asset_database
    except ImportError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    try:
        from Infernux.engine.play_mode import PlayModeManager
        pm = PlayModeManager.instance()
        if pm and pm._asset_database is not None:
            return pm._asset_database
    except ImportError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return None


class AssetRefBase:
    """Base class for GUID-based asset references.

    Stores a GUID string and lazily resolves to the loaded asset via
    ``AssetManager``.
    """

    __slots__ = ("_guid", "_cached", "_path_hint")

    def __init__(self, guid: str = "", path_hint: str = ""):
        self._guid = guid
        self._cached = None
        self._path_hint = path_hint  # optional human-readable path for display

    # ── GUID ───────────────────────────────────────────────────────────

    @property
    def guid(self) -> str:
        return self._guid

    @guid.setter
    def guid(self, value: str):
        if value != self._guid:
            self._guid = value
            self._cached = None

    @property
    def path_hint(self) -> str:
        """Best-effort human-readable path (may be stale)."""
        return self._path_hint

    @path_hint.setter
    def path_hint(self, value: str):
        self._path_hint = value

    # ── Resolution ─────────────────────────────────────────────────────

    def resolve(self):
        """Attempt to resolve the GUID to a loaded asset.

        Returns the asset object, or ``None`` if not found.
        Subclasses override ``_do_resolve``.
        """
        if self._cached is not None:
            return self._cached
        if not self._guid:
            return None
        self._cached = self._do_resolve()
        return self._cached

    def _do_resolve(self):
        """Override in subclass to call the appropriate AssetManager loader."""
        return None

    def invalidate(self):
        """Clear the cached resolved object (GUID is kept)."""
        self._cached = None

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"guid": self._guid, "path_hint": self._path_hint}

    @classmethod
    def from_dict(cls, d: dict) -> "AssetRefBase":
        if d is None:
            return cls()
        return cls(guid=d.get("guid", ""), path_hint=d.get("path_hint", ""))

    # ── Display ────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        if self._path_hint:
            return os.path.basename(self._path_hint)
        if self._guid:
            return f"GUID:{self._guid[:8]}…"
        return "None"

    @property
    def is_missing(self) -> bool:
        """True if we have a GUID but resolution failed."""
        if not self._guid:
            return False
        return self.resolve() is None

    def __bool__(self):
        return bool(self._guid)

    def __eq__(self, other):
        if isinstance(other, AssetRefBase):
            return self._guid == other._guid
        return NotImplemented

    def __hash__(self):
        return hash(self._guid)

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(guid='{self._guid}', path_hint='{self._path_hint}')"


class TextureRef(AssetRefBase):
    """Reference to a Texture asset."""

    def _do_resolve(self):
        from Infernux.core.assets import AssetManager
        from Infernux.core.texture import Texture
        return AssetManager.load_by_guid(self._guid, asset_type=Texture)


class ShaderRef(AssetRefBase):
    """Reference to a Shader asset (resolves to ShaderAssetInfo)."""

    def _do_resolve(self):
        from Infernux.core.assets import AssetManager
        from Infernux.core.shader import Shader
        return AssetManager.load_by_guid(self._guid, asset_type=Shader)


class AudioClipRef(AssetRefBase):
    """Reference to an AudioClip asset."""

    def _do_resolve(self):
        from Infernux.core.assets import AssetManager
        from Infernux.core.audio_clip import AudioClip
        return AssetManager.load_by_guid(self._guid, asset_type=AudioClip)


class MaterialRef(AssetRefBase):
    """GUID-based reference to a Material asset.

    Lazily loads the Material through ``AssetManager.load_by_guid`` on
    first access and caches the result.  Falls back to path-hint resolution
    and direct AssetDatabase lookup when the primary GUID path fails.

    Supports attribute forwarding to the underlying Material::

        mat_ref = MaterialRef(guid="abc123")
        mat_ref.set_color("_BaseColor", (1, 0, 0, 1))  # forwards to Material
    """

    def __init__(self, material=None, *, guid: str = "", path_hint: str = ""):
        if material is not None:
            extracted_guid = self._extract_guid(material)
            native = getattr(material, "native", material)
            hint = path_hint or getattr(native, "file_path", "") or ""
            super().__init__(guid=extracted_guid, path_hint=hint)
            self._cached = material
        else:
            super().__init__(guid=guid, path_hint=path_hint)

    @staticmethod
    def _extract_guid(material) -> str:
        """Extract the GUID for a Material wrapper or native InxMaterial."""
        if hasattr(material, "guid") and material.guid:
            return material.guid
        native = getattr(material, "native", material)
        if hasattr(native, "guid") and native.guid:
            return native.guid
        file_path = getattr(native, "file_path", "") or ""
        if file_path:
            db = _get_asset_database()
            if db:
                try:
                    g = db.get_guid_from_path(file_path)
                    if g:
                        return g
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    pass
        return ""

    def resolve(self):
        """Return the loaded Material, or ``None`` if missing."""
        mat = self._cached
        if mat is not None:
            try:
                native = getattr(mat, "native", mat)
                _ = native.name
                return mat
            except (RuntimeError, AttributeError):
                self._cached = None
        return self._do_resolve()

    def _do_resolve(self):
        if not self._guid and not self._path_hint:
            self._cached = None
            return None
        # Primary: load by GUID through AssetManager
        if self._guid:
            try:
                from Infernux.core.assets import AssetManager
                from Infernux.core.material import Material
                mat = AssetManager.load_by_guid(self._guid, asset_type=Material)
                if mat is not None:
                    self._cached = mat
                    return mat
            except Exception as exc:
                _log.warning("MaterialRef.resolve by GUID failed: %s", exc)
            # Fallback: resolve GUID → path via C++ AssetDatabase
            db = _get_asset_database()
            if db:
                try:
                    path = db.get_path_from_guid(self._guid)
                    if path:
                        from Infernux.core.material import Material
                        mat = Material.load(path)
                        if mat is not None:
                            self._cached = mat
                            return mat
                except Exception as exc:
                    _log.warning("MaterialRef.resolve by path failed: %s", exc)
        # Last resort: load by path_hint
        if self._path_hint:
            try:
                from Infernux.core.material import Material
                mat = Material.load(self._path_hint)
                if mat is not None:
                    self._cached = mat
                    if not self._guid:
                        g = self._extract_guid(mat)
                        if g:
                            self._guid = g
                    return mat
            except Exception as exc:
                _log.warning("MaterialRef.resolve by path_hint failed: %s", exc)
        self._cached = None
        return None

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying Material."""
        if name.startswith("_"):
            raise AttributeError(name)
        mat = self.resolve()
        if mat is None:
            return None
        return getattr(mat, name)

    def __copy__(self):
        return type(self)(guid=self._guid, path_hint=self._path_hint)

    def __deepcopy__(self, memo):
        copied = type(self)(guid=self._guid, path_hint=self._path_hint)
        memo[id(self)] = copied
        return copied

    def __eq__(self, other):
        if other is None:
            return not self._guid and not self._path_hint
        if isinstance(other, AssetRefBase):
            return self._guid == other._guid
        return NotImplemented
