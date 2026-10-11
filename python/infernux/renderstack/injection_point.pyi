from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set


@dataclass
class InjectionPoint:
    """A named point in the render graph where external passes can be inserted."""

    name: str
    """Unique identifier for the injection point."""
    display_name: str = ...
    """Human-readable display name."""
    description: str = ...
    """Description of when this injection point runs."""
    resource_state: Set[str] = ...
    """Set of resources available at this point."""
    removable: bool = ...
    """Whether this injection point can be removed."""

    def __post_init__(self) -> None: ...
