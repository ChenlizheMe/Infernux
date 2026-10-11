"""Authoring ownership for Transform properties driven by components."""
from __future__ import annotations

from enum import IntFlag
from typing import Callable, Type


class DrivenTransformProperties(IntFlag):
    """Transform fields a component owns during editor authoring."""

    NONE = 0
    POSITION = 1 << 0
    ROTATION = 1 << 1
    SCALE = 1 << 2
    ALL = POSITION | ROTATION | SCALE


def drives_transform(
    properties: DrivenTransformProperties,
) -> Callable[[Type], Type]:
    """Declare Transform fields that editor tools must not author directly."""
    value = DrivenTransformProperties(properties)
    if int(value) & ~int(DrivenTransformProperties.ALL):
        raise ValueError(f"unsupported driven Transform properties: {int(value)}")

    def decorator(component_type: Type) -> Type:
        inherited = DrivenTransformProperties(
            getattr(component_type, "_driven_transform_properties_", 0)
        )
        component_type._driven_transform_properties_ = inherited | value
        return component_type

    return decorator


def component_driven_transform_properties(component) -> DrivenTransformProperties:
    """Return the current authoring ownership declared by one component."""
    declared = getattr(component, "driven_transform_properties", None)
    if callable(declared):
        declared = declared()
    if declared is None:
        declared = getattr(type(component), "_driven_transform_properties_", 0)
    value = DrivenTransformProperties(declared)
    if int(value) & ~int(DrivenTransformProperties.ALL):
        raise ValueError(
            f"{type(component).__name__} declares unsupported driven Transform properties: "
            f"{int(value)}"
        )
    return value


def driven_transform_properties(game_object) -> DrivenTransformProperties:
    """Combine every live component's Transform authoring ownership."""
    if game_object is None:
        return DrivenTransformProperties.NONE
    result = DrivenTransformProperties.NONE
    components = getattr(game_object, "get_py_components", lambda: ())() or ()
    for component in components:
        if component is not None and bool(getattr(component, "enabled", True)):
            result |= component_driven_transform_properties(component)
    return result


def is_transform_property_driven(game_object, property_mask) -> bool:
    return bool(
        driven_transform_properties(game_object)
        & DrivenTransformProperties(property_mask)
    )


# Unity-style spellings remain aliases, not a second implementation.
DrivesTransform = drives_transform
