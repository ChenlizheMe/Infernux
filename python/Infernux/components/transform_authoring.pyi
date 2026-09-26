from enum import IntFlag
from typing import Callable, TypeVar

_T = TypeVar("_T")

class DrivenTransformProperties(IntFlag):
    NONE: DrivenTransformProperties
    POSITION: DrivenTransformProperties
    ROTATION: DrivenTransformProperties
    SCALE: DrivenTransformProperties
    ALL: DrivenTransformProperties

def drives_transform(properties: DrivenTransformProperties) -> Callable[[type[_T]], type[_T]]: ...
def component_driven_transform_properties(component: object) -> DrivenTransformProperties: ...
def driven_transform_properties(game_object: object) -> DrivenTransformProperties: ...
def is_transform_property_driven(game_object: object, property_mask: DrivenTransformProperties) -> bool: ...
DrivesTransform = drives_transform
