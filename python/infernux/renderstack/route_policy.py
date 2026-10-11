"""Queue-route image ownership policies used by RenderEffect features."""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class RoutePolicy(str, Enum):
    """How a route contributes its geometry and effect result to its parent."""

    INLINE = "inline"
    MASK_AND_MODIFY = "mask_and_modify"
    ISOLATE_AND_COMPOSITE = "isolate_and_composite"
    ADDITIVE_EXTRACT = "additive_extract"
    ORDERED_COMPOSITE = "ordered_composite"
    CUSTOM_FEATURE = "custom_feature"


def merge_route_policies(policies: Iterable[RoutePolicy]) -> RoutePolicy:
    """Choose storage/composition for the whole ordered route effect chain."""
    normalized = {
        value if isinstance(value, RoutePolicy) else RoutePolicy(value)
        for value in policies
    }
    normalized.discard(RoutePolicy.INLINE)
    if not normalized:
        return RoutePolicy.INLINE
    if RoutePolicy.CUSTOM_FEATURE in normalized:
        if len(normalized) != 1:
            raise ValueError("custom route policy cannot be mixed with built-in policies")
        return RoutePolicy.CUSTOM_FEATURE
    if RoutePolicy.ORDERED_COMPOSITE in normalized:
        return RoutePolicy.ORDERED_COMPOSITE
    if RoutePolicy.ADDITIVE_EXTRACT in normalized:
        if len(normalized) != 1:
            return RoutePolicy.ORDERED_COMPOSITE
        return RoutePolicy.ADDITIVE_EXTRACT
    if RoutePolicy.ISOLATE_AND_COMPOSITE in normalized:
        return RoutePolicy.ISOLATE_AND_COMPOSITE
    return RoutePolicy.MASK_AND_MODIFY


__all__ = ["RoutePolicy", "merge_route_policies"]
