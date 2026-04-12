"""
Shader management helpers.

Provides a clean API for shader loading, hot-reload, and cache management.
Wraps the C++ ShaderCache and pipeline reload functionality.

Usage::

    # Reload a shader after editing .spv files
    Shader.reload("pbr_lit")

    # Check if a shader is loaded
    if Shader.is_loaded("pbr_lit", "vertex"):
        print("Vertex shader ready")

    # Invalidate cache before loading new code
    Shader.invalidate("pbr_lit")

    # Refresh all materials using a given shader
    Shader.refresh_materials("pbr_lit", engine)
"""

from __future__ import annotations

from typing import Optional


class Shader:
    """Static utility class for shader management.

    Shader loading and compilation happens in C++ (SPIR-V).
    This class provides the Python-side control API for:
    - Cache invalidation (hot-reload)
    - Loading status queries
    - Material pipeline refresh after shader changes
    """

    # Reference to the native engine (set during Engine initialization)
    _engine = None

    @classmethod
    def _set_engine(cls, engine):
        """Internal: bind to the native Infernux instance."""
        cls._engine = engine

    @classmethod
    def is_loaded(cls, name: str, shader_type: str = "vertex") -> bool:
        """Check if a shader is loaded in the cache.

        Args:
            name: Shader identifier (e.g. "pbr_lit")
            shader_type: "vertex" or "fragment"

        Returns:
            True if the shader module exists in the cache.
        """
        if cls._engine is None:
            return False
        return cls._engine.has_shader(name, shader_type)

    @classmethod
    def invalidate(cls, shader_id: str):
        """Invalidate the shader program cache for hot-reload.

        Must be called BEFORE loading updated shader code to force
        pipeline recreation.

        Args:
            shader_id: The shader identifier to invalidate.
        """
        if cls._engine is None:
            return
        cls._engine.invalidate_shader_cache(shader_id)

    @classmethod
    def reload(cls, shader_id: str) -> bool:
        """Convenience: invalidate cache and refresh all materials using this shader.

        This is the one-call solution for shader hot-reload:
        1. Invalidates the cached ShaderProgram
        2. Refreshes all material pipelines that reference this shader

        Args:
            shader_id: The shader identifier to reload.

        Returns:
            True if at least one material was refreshed.
        """
        if cls._engine is None:
            return False
        cls._engine.invalidate_shader_cache(shader_id)
        return cls._engine.refresh_materials_using_shader(shader_id)

    @classmethod
    def refresh_materials(cls, shader_id: str, engine=None) -> bool:
        """Refresh all material pipelines that use a given shader.

        Args:
            shader_id: The shader identifier.
            engine: Optional Engine instance (uses bound engine if None).

        Returns:
            True if at least one material was refreshed.
        """
        eng = engine or cls._engine
        if eng is None:
            return False
        native = getattr(eng, '_engine', eng)
        if hasattr(native, 'refresh_materials_using_shader'):
            return native.refresh_materials_using_shader(shader_id)
        return False

    @classmethod
    def load_spirv(cls, name: str, spirv_code: bytes, shader_type: str = "vertex"):
        """Load a SPIR-V shader module into the engine.

        Args:
            name: Shader identifier.
            spirv_code: Raw SPIR-V binary data.
            shader_type: "vertex" or "fragment".
        """
        if cls._engine is None:
            raise RuntimeError("Shader system not initialized — Engine not bound")
        cls._engine.load_shader(name, spirv_code, shader_type)
