"""
InjectionPoint — Named injection slot in a RenderPipeline topology.

Pipeline authors declare injection points inline via
``graph.injection_point()``.  RenderStack uses them to determine where
user-mounted Passes are injected into the render graph.

Each injection point has:
- A unique ``name`` (e.g. "after_opaque")
- A ``resource_state`` describing the **minimum guaranteed** resources
  available from the ResourceBus at that point in the topology

Example::

    graph.injection_point(
        "after_opaque",
        display_name="After Opaque",
        resources={"color", "depth"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class InjectionPoint:
    """Named injection location inside a pipeline topology.

    Pipeline authors declare injection points through
    ``graph.injection_point()``.

    Attributes:
        name: Unique identifier such as ``"after_opaque"``.
        display_name: Editor-facing display name such as ``"After Opaque"``.
            Generated from ``name`` by default.
        description: Human-readable description of the slot.
        resource_state: Minimum guaranteed resource names available at this
            point. The runtime bus may contain more resources because earlier
            passes can add them through ``creates`` declarations. RenderStack
            validates against ``Pass.requires ⊆ bus.keys()`` rather than
            ``Pass.requires ⊆ resource_state``.
        removable: Whether the injection point can be removed in the editor.
    """

    name: str
    display_name: str = ""
    description: str = ""
    resource_state: Set[str] = field(
        default_factory=lambda: {"color", "depth"}
    )
    removable: bool = True

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
