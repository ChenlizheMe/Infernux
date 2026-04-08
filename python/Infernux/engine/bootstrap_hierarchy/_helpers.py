"""Standalone utility functions for hierarchy wiring."""
from __future__ import annotations

from Infernux.debug import Debug


def _safe_sequence(values):
    if values is None:
        return []
    if isinstance(values, list):
        return values
    try:
        return list(values)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []


def _get_py_components_safe(obj):
    if obj is None or not hasattr(obj, 'get_py_components'):
        return []
    try:
        return _safe_sequence(obj.get_py_components())
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []


def _get_children_safe(obj):
    if obj is None or not hasattr(obj, 'get_children'):
        return []
    try:
        return _safe_sequence(obj.get_children())
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []
