"""
ResourceBus — Transient resource handle dictionary for graph construction.

Created by ``RenderStack.build_graph()``, passed into
``Pipeline.define_topology()`` and each ``Pass.inject()``.
Carries graph-local texture and buffer handles between pipeline stages and passes.

Lifecycle::

    RenderStack creates bus
        → Pipeline initialises base resources (color, depth)
        → injection point callbacks pass bus to each Pass
        → RenderStack reads final output from bus

Resource name conventions::

    "color"       — scene color
    "depth"       — scene depth
    custom names  — introduced by Pass ``creates`` declarations
"""

from __future__ import annotations

from typing import Dict, Optional, Set, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from Infernux.rendergraph.graph import BufferHandle, RenderGraph, TextureHandle

    ResourceHandle = Union[TextureHandle, BufferHandle]


class ResourceBus:
    """Dictionary that carries resource handles during graph construction.

    Passes interact with the bus through their ``requires``, ``modifies``,
    and ``creates`` declarations. Undeclared resources pass through to later
    passes automatically.

    .. note::
        ``modifies`` implies ``requires``. A pass that declares
        ``modifies={"color"}`` both reads and writes ``color``.
    """

    def __init__(
        self, initial: Optional[Dict[str, "ResourceHandle"]] = None,
        *, graph: "RenderGraph | None" = None,
    ) -> None:
        self._graph = graph
        self._resources: Dict[str, "ResourceHandle"] = {}
        for name, handle in (initial or {}).items():
            self.set(name, handle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional["ResourceHandle"]:
        """Return a resource handle, or ``None`` if it is missing."""
        return self._resources.get(name)

    def set(self, name: str, handle: "ResourceHandle") -> None:
        """Publish a graph-local handle for the current View build."""
        if self._graph is not None:
            from Infernux.rendergraph.graph import BufferHandle, TextureHandle

            owned = (
                isinstance(handle, TextureHandle) and self._graph._owns_texture(handle)
                or isinstance(handle, BufferHandle) and self._graph._owns_buffer(handle)
            )
            if not owned:
                raise ValueError(f"Resource '{name}' does not belong to this RenderGraph")
        self._resources[name] = handle

    def require_buffer(self, name: str) -> "BufferHandle":
        """Resolve a declared buffer at this injection stage."""
        from Infernux.rendergraph.graph import BufferHandle

        handle = self._resources.get(name)
        if handle is None:
            raise ValueError(f"Buffer resource '{name}' is not available at this stage")
        if not isinstance(handle, BufferHandle):
            raise TypeError(f"Resource '{name}' is not a graph BufferHandle")
        return handle

    def require_texture(self, name: str) -> "TextureHandle":
        """Resolve a declared texture at this injection stage."""
        from Infernux.rendergraph.graph import TextureHandle

        handle = self._resources.get(name)
        if handle is None:
            raise ValueError(f"Texture resource '{name}' is not available at this stage")
        if not isinstance(handle, TextureHandle):
            raise TypeError(f"Resource '{name}' is not a graph TextureHandle")
        return handle

    def has(self, name: str) -> bool:
        """Return whether a resource exists."""
        return name in self._resources

    @property
    def available_resources(self) -> Set[str]:
        """Return the set of currently available resource names."""
        return set(self._resources.keys())

    def snapshot(self) -> Dict[str, "ResourceHandle"]:
        """Return a shallow snapshot of the current resources for debugging."""
        return dict(self._resources)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._resources.keys()))
        return f"<ResourceBus [{keys}]>"
