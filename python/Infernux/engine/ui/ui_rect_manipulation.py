"""Authoritative layout semantics shared by the UI Editor and Scene Rect tool."""
from __future__ import annotations

import math


def screen_ui_component(game_object, *, world_only: bool = False):
    """Return the object's authored UI rectangle component, when present."""
    if game_object is None or not hasattr(game_object, "get_py_components"):
        return None
    from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent

    for component in game_object.get_py_components() or ():
        if isinstance(component, InxUIScreenComponent):
            if world_only and not component.is_world_space():
                return None
            return component
    return None


def layout_snapshot(component) -> dict:
    """Capture every layout field a direct rectangle gesture may change."""
    fields = (
        "x", "y", "width", "height", "layout_position",
        "width_sizing", "height_sizing", "resize_mode",
    )
    snapshot = {
        field: getattr(component, field)
        for field in fields
        if hasattr(component, field)
    }
    return snapshot


def prepare_layout_resize(component, *, width: bool = True, height: bool = True) -> None:
    """Make directly resized axes authoritative instead of layout-derived."""
    from Infernux.ui.enums import TextResizeMode, UILayoutSizing

    if width and hasattr(component, "width_sizing"):
        component.width_sizing = UILayoutSizing.Fixed
    if height and hasattr(component, "height_sizing"):
        component.height_sizing = UILayoutSizing.Fixed
    if hasattr(component, "resize_mode"):
        component.resize_mode = TextResizeMode.FixedSize


def apply_layout_size(
    component,
    *,
    width: float | None = None,
    height: float | None = None,
) -> None:
    """Commit a direct rectangle resize through the shared authoring rule.

    Views own pointer projection and handle visuals.  They do not independently
    decide how a direct size edit changes layout authority.
    """
    touches_width = width is not None
    touches_height = height is not None
    if not touches_width and not touches_height:
        return
    prepare_layout_resize(
        component,
        width=touches_width,
        height=touches_height,
    )
    if touches_width:
        component.width = float(width)
    if touches_height:
        component.height = float(height)


def resolve_world_ui_frame(component):
    """Resolve one free world UI element into a Scene-tool world frame."""
    if component is None or not component.is_world_space():
        return None
    game_object = getattr(component, "game_object", None)
    if game_object is None:
        return None

    width, height = (max(1.0, value) for value in component.get_resolved_size())
    from Infernux.lib import Vector3
    from Infernux.ui.inx_ui_screen_component import WORLD_UI_PIXELS_PER_UNIT

    pixels_per_unit = WORLD_UI_PIXELS_PER_UNIT
    transform = game_object.transform
    center = transform.position
    world_u = transform.transform_direction(Vector3(1.0, 0.0, 0.0))
    # UI Y grows downward; the Rect frame follows that authored direction.
    world_v = transform.transform_direction(Vector3(0.0, -1.0, 0.0))
    scale_u = max(math.sqrt(sum(float(world_u[i]) ** 2 for i in range(3))), 1.0e-6)
    scale_v = max(math.sqrt(sum(float(world_v[i]) ** 2 for i in range(3))), 1.0e-6)
    axis_u = tuple(float(world_u[i]) / scale_u for i in range(3))
    axis_v = tuple(float(world_v[i]) / scale_v for i in range(3))
    return {
        "object_id": int(component.game_object.id),
        "center": tuple(float(center[i]) for i in range(3)),
        "axis_u": axis_u,
        "axis_v": axis_v,
        "half_size": (
            max(float(width) * scale_u / pixels_per_unit * 0.5, 0.001),
            max(float(height) * scale_v / pixels_per_unit * 0.5, 0.001),
        ),
        "axis_indices": (0, 1),
        "pixels_per_unit": pixels_per_unit,
        "scale_u": scale_u,
        "scale_v": scale_v,
    }


def world_delta_to_layout(frame: dict, delta_u: float, delta_v: float) -> tuple[float, float]:
    """Convert Rect-plane world distances to authored UI pixel distances."""
    return (
        float(delta_u) * float(frame["pixels_per_unit"]) / float(frame["scale_u"]),
        float(delta_v) * float(frame["pixels_per_unit"]) / float(frame["scale_v"]),
    )
