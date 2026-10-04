from enum import Enum
from typing import Iterable

class RoutePolicy(str, Enum):
    INLINE = "inline"
    MASK_AND_MODIFY = "mask_and_modify"
    ISOLATE_AND_COMPOSITE = "isolate_and_composite"
    ADDITIVE_EXTRACT = "additive_extract"
    ORDERED_COMPOSITE = "ordered_composite"
    CUSTOM_FEATURE = "custom_feature"

def merge_route_policies(policies: Iterable[RoutePolicy]) -> RoutePolicy: ...
