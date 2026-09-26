"""Authoritative native transform dependencies for UI publication."""

from __future__ import annotations


def create_ui_transform_dependencies(screen_objects, world_objects):
    """Create the native dependency tracker required by the current host."""
    from Infernux.lib import _Infernux as native

    dependency_type = getattr(native, "_UITransformDependencies", None)
    if dependency_type is None:
        raise RuntimeError(
            "The current Infernux host does not provide _UITransformDependencies; "
            "rebuild or reinstall the current engine runtime"
        )
    return dependency_type(screen_objects, world_objects)


__all__ = ["create_ui_transform_dependencies"]
