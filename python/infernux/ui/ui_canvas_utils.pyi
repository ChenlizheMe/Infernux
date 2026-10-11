"""Type stubs for infernux.ui.ui_canvas_utils — shared canvas-discovery utilities."""

from __future__ import annotations

from typing import Any, List, Tuple

from infernux.ui.ui_canvas import UICanvas


def scene_canvas_cache_key(scene: Any) -> tuple[int, int] | None:
    """Return the cache identity for one native Scene lifetime."""
    ...


def invalidate_canvas_cache() -> None:
    """Force cache invalidation (e.g. on scene load)."""
    ...


def collect_canvases_with_go(scene: Any) -> List[Tuple[Any, UICanvas]]:
    """Return ``[(GameObject, UICanvas), ...]`` for every canvas in *scene*.

    Walks the full scene hierarchy.

    Args:
        scene: The active Scene object.
    """
    ...


def collect_canvases(scene: Any, *, allow_stale_empty: bool = False) -> List[UICanvas]:
    """Return ``[UICanvas, ...]`` for every canvas in *scene*."""
    ...


def collect_sorted_canvases(scene: Any, *, allow_stale_empty: bool = False) -> List[UICanvas]:
    """Return ``[UICanvas, ...]`` sorted by ``sort_order`` (cached)."""
    ...


def runtime_ui_scenes(scene_manager: Any) -> tuple[Any, ...]:
    """Return loaded worlds in native order, then the persistent world."""
    ...


def collect_sorted_runtime_canvases(*scenes: Any, allow_stale_empty: bool = False) -> List[UICanvas]:
    """Return sorted canvases from the supplied resident worlds."""
    ...


def collect_runtime_canvases_with_go(*scenes: Any, allow_stale_empty: bool = False) -> List[Tuple[Any, UICanvas]]:
    """Return the resident Canvas snapshot with owning GameObjects."""
    ...


def canvas_membership_revision() -> int:
    """Return the Canvas membership and ordering publication revision."""
    ...
