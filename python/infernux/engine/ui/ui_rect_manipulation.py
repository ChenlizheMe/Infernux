"""Authoritative layout semantics shared by the UI Editor and Scene Rect tool."""
from __future__ import annotations

import math


def screen_ui_component(game_object, *, world_only: bool = False):
    """Return the object's authored UI rectangle component, when present."""
    if game_object is None or not hasattr(game_object, "get_py_components"):
        return None
    from infernux.ui.inx_ui_screen_component import InxUIScreenComponent

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
    from infernux.ui.enums import TextResizeMode, UILayoutSizing

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


def _screen_layout_context(component, render_size):
    from infernux.ui import UIFrame
    from infernux.ui.enums import UILayoutDirection, UILayoutPosition

    canvas = component.get_canvas()
    if canvas is None:
        return None, None, None, None
    width, height = (canvas.compute_logical_size(*render_size) if render_size is not None
                     else (canvas.reference_width, canvas.reference_height))
    parent = next(component._layout_ancestors(), None)
    frame = parent.get_component(UIFrame) if parent is not None else None
    if frame is not None and (frame.layout_direction == UILayoutDirection.None_
                              or component.layout_position != UILayoutPosition.Flow):
        frame = None
    return canvas, width, height, frame


def describe_ui_layout(component, *, render_size=None) -> dict:
    """Read the same resolved geometry used by the renderer and UI Editor."""
    canvas, width, height, flow = _screen_layout_context(component, render_size)
    owner = component.game_object
    rect = component.get_rect(width, height)
    render_scale = canvas.compute_scale(*render_size)[:2] if canvas is not None and render_size is not None else None
    return {
        "object_id": int(owner.id),
        "component_id": int(component.component_id),
        "coordinates": {
            "space": "canvas" if canvas is not None else "world_ui_local",
            "units": "logical_pixels",
            "basis": ("world_geometry" if canvas is None else
                      "game_render_target" if render_size is not None else "design_reference"),
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
            "transform_position": "parent_center_x_right_y_up" if canvas is not None else "world_transform",
            "render_rect_space": "render_target_pixels" if render_scale is not None else None,
        },
        "canvas": None if canvas is None else {
            "object_id": int(canvas.game_object.id),
            "component_id": int(canvas.component_id),
            "reference_size": [canvas.reference_width, canvas.reference_height],
            "logical_size": [width, height],
            "render_size": list(render_size) if render_size is not None else None,
            "render_mode": canvas.render_mode.name,
            "scale_mode": canvas.ui_scale_mode.name,
            "pixel_perfect": canvas.pixel_perfect,
        },
        "rect": list(rect),
        "render_scale": list(render_scale) if render_scale is not None else None,
        "render_rect": list(canvas.pixel_adjust_rect(rect, *render_scale)) if render_scale is not None else None,
        "visual_rect": list(component.get_visual_rect(width, height)),
        "corners": [list(point) for point in component.get_rotated_corners(width, height)],
        "clip_bounds": component.get_effective_clip_rect(width, height),
        "parent_rect": None if canvas is None else list(component._get_parent_world_rect(width, height)),
        "local_position": list(owner.transform.local_position),
        "rotation_degrees": component.get_layout_rotation(),
        "authored_size": [component.width, component.height],
        "layout_position": component.layout_position.name,
        "width_sizing": component.width_sizing.name,
        "height_sizing": component.height_sizing.name,
        "flow": None if flow is None else {
            "object_id": int(flow.game_object.id),
            "component_id": int(flow.component_id),
            "direction": flow.layout_direction.name,
        },
    }


def edit_layout_rect(component, rect, commands, *, render_size=None, detach_from_layout=False):
    """Apply one UI rect edit through the same history as a UI Editor gesture.

    The rectangle is unrotated canvas geometry, not its rotated visual AABB.
    Flow membership changes only when the author explicitly requests it.
    """
    from infernux.ui.enums import TextResizeMode, UILayoutPosition, UILayoutSizing

    canvas, cw, ch, flow = _screen_layout_context(component, render_size)
    if canvas is None:
        raise ValueError("Canvas-free UI is positioned through its world Transform")
    if flow is not None and not detach_from_layout:
        raise ValueError("Parent auto layout controls this rectangle; set detach_from_layout=true to author an absolute rect")
    x, y, width, height = rect
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("UI rectangle values must be finite")
    if width < 0 or height < 0:
        raise ValueError("UI rectangle extents must be nonnegative")
    for extent, minimum, maximum in (
        (width, component.min_width, component.max_width),
        (height, component.min_height, component.max_height),
    ):
        if extent < minimum or (maximum > 0 and extent > maximum):
            raise ValueError("UI rectangle exceeds its authored min/max size constraints")
    values = {}
    for axis, extent, resolved in zip(("width", "height"), (width, height), component.get_rect(cw, ch)[2:]):
        if extent != resolved or (detach_from_layout and flow is not None):
            values[axis] = extent
            values[axis + "_sizing"] = UILayoutSizing.Fixed
    if detach_from_layout:
        values["layout_position"] = UILayoutPosition.Absolute
    if ("width" in values or "height" in values) and hasattr(component, "resize_mode"):
        values["width"], values["height"] = width, height
        values["resize_mode"] = TextResizeMode.FixedSize
    description = "Set UI Rect"
    changes = [
        (component, field, getattr(component, field), value, description)
        for field, value in values.items()
    ]
    transform = component.game_object.transform
    before_position = transform.local_position
    if not commands.can_record():
        return False
    with commands.suppress_replay():
        try:
            for target, field, _old, value, _description in changes:
                setattr(target, field, value)
            # Resolve after sizing/Flow changes: a Hug parent can change its
            # own rect when this child leaves auto layout.
            component._set_layout_rect_origin(x, y, cw, ch)
        except Exception:
            for target, field, old, _value, _description in changes:
                setattr(target, field, old)
            transform.local_position = before_position
            raise
    changes.append((transform, "local_position", before_position, transform.local_position, description))
    if not any(old != new for _, _, old, new, _ in changes):
        return True
    return commands.record_applied_property_changes(changes, description=description)


def resolve_world_ui_frame(component):
    """Resolve one free world UI element into the frame used by its renderer.

    UI editing remains strictly local for authored transforms. Billboard
    rectangles use the Game Camera plane because that is the plane rendered
    for gameplay-facing text, while the editor camera remains the Scene ray
    and picking camera.
    """
    if component is None or not component.is_world_space():
        return None
    game_object = getattr(component, "game_object", None)
    if game_object is None:
        return None

    width, height = (max(1.0, value) for value in component.get_resolved_size())
    from infernux.lib import Vector3
    from infernux.ui.inx_ui_screen_component import WORLD_UI_PIXELS_PER_UNIT

    pixels_per_unit = WORLD_UI_PIXELS_PER_UNIT
    transform = game_object.transform
    center = transform.position
    world_u = transform.transform_direction(Vector3(1.0, 0.0, 0.0))
    world_v = transform.transform_direction(Vector3(0.0, -1.0, 0.0))
    if bool(getattr(component, "world_billboard", False)):
        from infernux.lib import SceneManager

        scene = SceneManager.instance().get_active_scene()
        camera = getattr(scene, "effective_game_camera", None) if scene is not None else None
        camera_to_world = getattr(camera, "camera_to_world_matrix", None) if camera is not None else None
        if camera_to_world is not None:
            world_u = Vector3(float(camera_to_world[0][0]), float(camera_to_world[1][0]),
                              float(camera_to_world[2][0]))
            world_v = Vector3(-float(camera_to_world[0][1]), -float(camera_to_world[1][1]),
                              -float(camera_to_world[2][1]))
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
