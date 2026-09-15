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


def is_unchanged_ui_scalar(instance, name: str, value) -> bool:
    """Return whether an authored scalar assignment leaves UI state unchanged."""
    if name.startswith("_") or not isinstance(
        value, (str, bytes, int, float, bool, type(None), Enum)
    ):
        return False
    try:
        previous = object.__getattribute__(instance, name)
    except (AttributeError, KeyError):
        return False
    return type(previous) is type(value) and previous == value


def mark_runtime_ui_dirty(element=None, *, binding: bool = False) -> int:
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
        _invalidate_ui_command_groups(element)
    return _runtime_ui_revision


def _invalidate_ui_command_groups(element) -> None:
    """Notify existing submission owners, without retaining an edit history."""
    for group in element.__dict__.get('_ui_command_groups', ()):
        group.invalidate(element)


def _get_shared_visual_revision() -> int:
    return _shared_visual_revision


def _get_resource_binding_revision() -> int:
    """Only resource assignments/global edits change binding membership."""
    return _resource_binding_revision


def get_runtime_ui_revision() -> int:
    """Return the current process-wide runtime UI revision."""
    return _runtime_ui_revision
