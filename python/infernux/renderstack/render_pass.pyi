from __future__ import annotations

from typing import ClassVar, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class RenderPass:
    """Base class for custom render passes injected into the render stack."""

    name: ClassVar[str]
    """The unique name of this render pass."""
    injection_point: ClassVar[str]
    """The injection point where this pass is inserted."""
    default_order: ClassVar[int]
    """Default execution order within the injection point."""
    requires: ClassVar[Set[str]]
    """Resource names this pass reads from."""
    modifies: ClassVar[Set[str]]
    """Resource names this pass writes to."""
    creates: ClassVar[Set[str]]
    """Resource names this pass creates."""
    enabled: bool
    """Whether this pass is currently enabled."""

    def __init__(self, enabled: bool = ...) -> None: ...
    def inject(self, graph: RenderGraph, bus: ResourceBus) -> None:
        """Inject render commands into the graph using the resource bus."""
        ...
    def validate(self, available_resources: Set[str]) -> List[str]:
        """Validate that all required resources are available. Returns error messages."""
        ...
    def __repr__(self) -> str: ...
