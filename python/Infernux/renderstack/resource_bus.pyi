from __future__ import annotations

from typing import Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from Infernux.rendergraph.graph import BufferHandle, RenderGraph, TextureHandle

ResourceHandle = TextureHandle | BufferHandle


class ResourceBus:
    """View-local bus for graph texture and buffer handles."""

    def __init__(self, initial: Optional[Dict[str, ResourceHandle]] = ..., *, graph: RenderGraph | None = ...) -> None: ...
    def get(self, name: str) -> Optional[ResourceHandle]:
        """Get a resource handle by name, or None."""
        ...
    def set(self, name: str, handle: ResourceHandle) -> None:
        """Publish a graph-local resource handle."""
        ...
    def require_buffer(self, name: str) -> BufferHandle: ...
    def require_texture(self, name: str) -> TextureHandle: ...
    def has(self, name: str) -> bool:
        """Check if a resource name is registered."""
        ...
    @property
    def available_resources(self) -> Set[str]:
        """Set of all registered resource names."""
        ...
    def snapshot(self) -> Dict[str, ResourceHandle]:
        """Return a copy of the current resource state."""
        ...
    def __repr__(self) -> str: ...
