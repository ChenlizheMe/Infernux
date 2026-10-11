"""
GeometryPass — Scene geometry drawing pass.

Used for passes that need to draw scene objects (e.g. Outline mask, Decal).
Subclasses configure ``queue_range`` and ``sort_mode``.

**Important**: The base ``inject()`` raises ``NotImplementedError``.
GeometryPass subclasses **must** override ``inject()`` because MRT slot
assignment depends on the specific shader layout and cannot be handled
generically.

Example::

    class OutlineMaskPass(GeometryPass):
        name = "OutlineMask"
        injection_point = "after_opaque"
        default_order = 50
        requires = {"depth"}
        creates = {"outline_mask"}
        queue_range = (0, 2500)

        def inject(self, graph, bus):
            if not self.enabled:
                return
            depth = bus.get("depth")
            mask = graph.create_texture("_OutlineMask", format=Format.R8_UNORM)
            p = graph.add_pass(self.name)
            p.read(depth)
            p.write_color(mask)
            p.set_clear(color=(0, 0, 0, 0))
            p.draw_renderers(queue_range=self.queue_range, sort_mode="none")
            bus.set("outline_mask", mask)
"""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from infernux.renderstack.render_pass import RenderPass

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class GeometryPass(RenderPass):
    """Base pass for scene-geometry drawing.

    Use this for passes that render scene objects such as outline masks or
    decals.

    **Important**: the base ``inject()`` raises ``NotImplementedError``.
    Subclasses must override it and explicitly control every
    ``write_color()`` slot because MRT assignment depends on the shader
    layout.
    """

    queue_range: Tuple[int, int] = (0, 5000)
    sort_mode: str = "none"

    def inject(self, graph: "RenderGraph", bus: "ResourceBus") -> None:
        """GeometryPass subclasses must override this method.

        The base class does not provide a default implementation because MRT
        slot assignment depends on the concrete shader layout.

        Raises:
            NotImplementedError: Always. Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override inject(). "
            f"GeometryPass base class does not provide a default "
            f"implementation because MRT slot assignment must be "
            f"explicit. See design doc Section 9 for examples."
        )
