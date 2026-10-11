"""Revision-driven runtime Canvas discovery for editor UI consumers."""

from __future__ import annotations

from infernux.ui.ui_canvas_utils import (
    canvas_membership_revision,
    collect_runtime_canvases_with_go as _collect_runtime_canvases_with_go,
    collect_sorted_runtime_canvases as _collect_sorted_runtime_canvases,
    scene_canvas_cache_key as _scene_canvas_cache_key,
)


def _runtime_scene_key(scenes) -> tuple:
    scenes = tuple({id(scene): scene for scene in scenes if scene is not None}.values())
    return tuple(_scene_canvas_cache_key(scene) for scene in scenes)


def runtime_canvas_snapshot_token(*scenes) -> tuple:
    """Return the scene epoch plus the dedicated Canvas membership revision."""
    return (
        _runtime_scene_key(scenes),
        canvas_membership_revision(),
    )


def collect_sorted_runtime_canvas_snapshot(*scenes):
    """Return the current sorted Canvas snapshot without unrelated scene scans."""
    return _collect_sorted_runtime_canvases(
        *scenes,
        allow_stale_empty=False,
    )


def collect_runtime_canvas_snapshot_with_go(*scenes):
    """Return the shared runtime Canvas snapshot with owning GameObjects."""
    return _collect_runtime_canvases_with_go(
        *scenes,
        allow_stale_empty=False,
    )
