from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from infernux.renderstack.render_pass import RenderPass

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class GeometryPass(RenderPass):
    """A render pass that draws scene geometry filtered by queue range."""

    queue_range: Tuple[int, int]
    """Min/max render queue range for filtering renderers."""
    sort_mode: str
    """Sorting mode for draw calls."""

    def inject(self, graph: RenderGraph, bus: ResourceBus) -> None:
        """Inject geometry drawing commands into the render graph."""
        ...
