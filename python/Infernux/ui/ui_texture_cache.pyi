"""Type stubs for Infernux.ui.ui_texture_cache — shared UI texture cache."""

from __future__ import annotations

from typing import Any, Callable, Optional
from Infernux.core.asset_ref import TextureRef
from Infernux.core.render_texture import RenderTexture


class UITextureCache:
    """GUID-keyed texture-resource -> ImGui-texture-ID cache.

    Runtime UI passes ``TextureRef``; string paths are reserved for explicit
    editor preview boundaries.  The cache is shared
    as a module-level singleton via ``get_shared_cache()``.
    """

    def __init__(self) -> None: ...

    def get(self, engine: Any, texture: TextureRef | RenderTexture | str) -> int:
        """Return the ImGui texture ID for *texture*, loading if needed.

        Args:
            engine: The Engine instance for native texture upload.
            texture: GUID-backed texture reference, live render texture, or an
                explicit editor-preview path.

        Returns:
            An ImGui texture ID (> 0 on success, 0 on failure).
        """
        ...

    def get_bound(self, engine: Any) -> Callable[[TextureRef | RenderTexture | str], int]:
        """Return a callable ``f(tex_path) -> tid`` bound to *engine*.

        Avoids creating a fresh lambda every frame.
        """
        ...

    def invalidate(self, identifier: Optional[str] = ...) -> None:
        """Drop cached entries.

        Args:
            identifier: A GUID or file path to invalidate, or ``None``
                to clear the entire cache.
        """
        ...


def get_shared_cache() -> UITextureCache:
    """Return (creating if needed) the module-level shared cache singleton."""
    ...
