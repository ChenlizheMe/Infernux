"""Explicit renderer membership and captured parameters for a graph pass."""
from __future__ import annotations

from Infernux.lib import DrawParameterBlock, RendererSelection as _NativeSelection
from Infernux.core.material import Material
from Infernux.components.builtin.mesh_renderer import MeshRenderer


class RendererSelection:
    """Draw selected live MeshRenderers using one material.

    The graph retains this object. Updating membership or parameters does not
    rebuild its topology. Geometry, skinning, transforms and GPU vertex storage
    come from the current camera-visible render publication, not a mesh copy.
    A destroyed/replaced renderer is not selected merely by reusing its ID.
    Mutate in the engine update phase, before rendering.
    """

    def __init__(self, material: Material):
        if not isinstance(material, Material):
            raise TypeError("RendererSelection requires a Material")
        self._native = _NativeSelection(material.native)

    def set(self, renderer: MeshRenderer, *, submesh: int = -1,
            parameters: DrawParameterBlock | None = None) -> None:
        """Select all submeshes (-1) or one index and capture parameter values.

        An exact submesh entry overrides an all-submeshes entry. Later changes
        to the supplied block do not alter an entry until ``set`` is called.
        Shader parameters are validated against this selection's material.
        """
        if not isinstance(renderer, MeshRenderer):
            raise TypeError("RendererSelection requires a MeshRenderer")
        self._native.set(renderer._require_cpp_component(), submesh, parameters)

    def remove(self, renderer: MeshRenderer, *, submesh: int = -1) -> bool:
        """Remove the specified entry; -1 removes the all-submeshes entry."""
        if not isinstance(renderer, MeshRenderer):
            raise TypeError("RendererSelection requires a MeshRenderer")
        return self._native.remove(renderer._require_cpp_component(), submesh)

    def clear(self) -> None:
        """Remove every entry, including references to retired renderers."""
        self._native.clear()

    @property
    def size(self) -> int:
        """Number of authored entries, not the visible draw count."""
        return self._native.size

    @property
    def revision(self) -> int:
        return self._native.revision
