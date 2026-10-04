"""
RenderPass — Base class for all mountable render passes.

A RenderPass represents a single rendering step that can be mounted to
a RenderStack at a named injection point. Subclasses declare their
resource requirements via class-level sets and implement ``inject()``
to add passes to the RenderGraph.

Subclass hierarchy::

    RenderPass          (abstract base)
    └── GeometryPass    (scene geometry drawing)

Resource declaration rules:
    - ``requires``: read-only — resource passes through unchanged
    - ``modifies``: read + write — modified handle replaces bus entry
    - ``creates``:  new resource — added to bus for subsequent passes
    - ``modifies`` implicitly includes ``requires`` (no need to declare both)
    - Undeclared resources auto-pass-through via ResourceBus
"""

from __future__ import annotations

from typing import ClassVar, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph
    from infernux.renderstack.resource_bus import ResourceBus


class RenderPass:
    """Base class for render steps mountable on a RenderStack.

    Subclasses must define:
        - ``name``: unique identifier used for serialization
        - ``injection_point``: target injection point name
        - ``default_order``: default ordering within the injection point

    Subclasses declare resource usage with class attributes:
        - ``requires``: ``Set[str]`` of read-only resources
        - ``modifies``: ``Set[str]`` of read/write resources
        - ``creates``: ``Set[str]`` of newly created resources
    """

    # ---- Required subclass metadata ----
    name: str = ""
    injection_point: str = ""
    default_order: int = 0

    # ---- Resource declarations ----
    requires: ClassVar[Set[str]] = set()
    modifies: ClassVar[Set[str]] = set()
    creates: ClassVar[Set[str]] = set()

    # ---- Runtime state ----
    enabled: bool = True

    def __init__(self, enabled: bool = True) -> None:
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must define 'name'"
            )
        if not self.injection_point:
            raise ValueError(
                f"{type(self).__name__} must define 'injection_point'"
            )
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def inject(self, graph: "RenderGraph", bus: "ResourceBus") -> None:
        """Inject this pass into the RenderGraph.

        Subclasses should:
        1. Read the required resource handles from ``bus``
        2. Add one or more render passes to ``graph``
        3. Write modified or created resources back to ``bus``

        Args:
            graph: RenderGraph currently being built.
            bus: Resource bus used for inputs and outputs.
        """
        raise NotImplementedError

    def validate(self, available_resources: Set[str]) -> List[str]:
        """Validate that the pass resource requirements are satisfied.

        ``modifies`` is treated as a superset of ``requires``:
        ``effective_requires = requires ∪ modifies``.

        Args:
            available_resources: Resource names currently available on the bus.

        Returns:
            A list of validation errors. An empty list means success.
        """
        errors: List[str] = []
        for r in self.requires | self.modifies:
            if r not in available_resources:
                errors.append(
                    f"Pass '{self.name}' requires resource '{r}' "
                    f"but it is not available at injection point "
                    f"'{self.injection_point}'"
                )
        return errors

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name='{self.name}' "
            f"point='{self.injection_point}' "
            f"enabled={self.enabled}>"
        )
