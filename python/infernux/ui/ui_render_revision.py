"""Independent revisions for UI draw content and input eligibility."""

from __future__ import annotations

from enum import Enum


_runtime_ui_revision = 1
_shared_visual_revision = 1
_resource_binding_revision = 1
_hit_policy_revision = 1


def _mark_hit_policy_dirty() -> None:
    """Eligibility/clipping changed; tint animations do not enter this domain."""
    global _hit_policy_revision
    _hit_policy_revision += 1


def _get_hit_policy_revision() -> int:
    return _hit_policy_revision


def is_unchanged_ui_value(instance, name: str, value) -> bool:
    """Compare scalar and declared color/vector values before publishing UI edits."""
    if name.startswith("_"):
        return False
    from infernux.components.fields import (
        FieldType, get_raw_field_value, get_serialized_fields, normalize_runtime_field_value,
    )

    try:
        previous = get_raw_field_value(instance, name)
    except (AttributeError, KeyError):
        return False
    if isinstance(value, (str, bytes, int, float, bool, type(None), Enum)):
        if type(previous) in (int, float) and type(value) in (int, float):
            return previous == value
        return type(previous) is type(value) and previous == value
    # An in-place edit may be written back through the same mutable object.
    if previous is value:
        return False
    metadata = get_serialized_fields(type(instance)).get(name)
    if metadata is None:
        return False
    if metadata.field_type == FieldType.COLOR:
        return previous == normalize_runtime_field_value(value, metadata)
    axes = {FieldType.VEC2: "xy", FieldType.VEC3: "xyz", FieldType.VEC4: "xyzw"}.get(metadata.field_type)
    if axes is None:
        return False
    normalized = normalize_runtime_field_value(value, metadata)
    return all(getattr(previous, axis) == getattr(normalized, axis) for axis in axes)


def mark_runtime_ui_dirty(element=None, *, binding: bool = False, subtree: bool = False) -> int:
    """Publish an element edit, or an inherited/global visual change.

    An element edit preserves other elements' extracted draw parameters.
    Callers without an element (Canvas, Group, resource/tool invalidation)
    invalidate shared parameters as well.
    """
    global _runtime_ui_revision, _shared_visual_revision, _resource_binding_revision
    _runtime_ui_revision += 1
    if binding or element is None:
        _resource_binding_revision += 1
    if element is None:
        _shared_visual_revision += 1
    else:
        _invalidate_ui_command_groups(element, subtree=subtree)
    return _runtime_ui_revision


def _invalidate_ui_command_groups(element, *, subtree: bool = False) -> None:
    """Notify existing submission owners, without retaining an edit history."""
    for group in element.__dict__.get('_ui_command_groups', ()):
        group.invalidate(element)
    if subtree:
        from .ui_canvas import UICanvas

        pending = list(element.game_object.get_children())
        while pending:
            owner = pending.pop()
            components = owner.get_py_components()
            if any(isinstance(component, UICanvas) for component in components):
                continue
            for component in components:
                _invalidate_ui_command_groups(component)
            pending.extend(owner.get_children())


def _get_shared_visual_revision() -> int:
    return _shared_visual_revision


def _get_resource_binding_revision() -> int:
    """Only resource assignments/global edits change binding membership."""
    return _resource_binding_revision


def get_runtime_ui_revision() -> int:
    """Return the current process-wide runtime UI revision."""
    return _runtime_ui_revision
