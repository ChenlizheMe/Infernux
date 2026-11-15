"""Shared visual curve field and popup editor for Inspector-style surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from infernux._animation_curve import Keyframe, WeightedMode, _decode_number

from infernux.graph.ramp import CURVE_WRAP_MODES, AnimationCurve

from .dpi import editor_dpi_scale
from .theme import Theme, ImGuiStyleVar
from .igui import IGUI
from infernux.engine.i18n import t


@dataclass
class _CurveEditorState:
    selected: int = 0
    dragging: str = ""
    opening_curve: dict | None = None


_STATES: dict[str, _CurveEditorState] = {}


def _edit_key_number(ctx, widget_id, value, speed, minimum, maximum):
    """Keep authored precision until the native float widget is actually edited."""
    number = _decode_number(value)
    if not isfinite(number):
        ctx.label(str(number))
        ctx.same_line()
        return 0.0 if IGUI.icon_button(ctx, f"##reset_{widget_id}", "reset", tooltip=t("curve.reset")) else value
    updated = float(ctx.drag_float(widget_id, number, speed, minimum, maximum))
    return updated if ctx.is_item_edited() else value


def _edit_tangent(ctx, widget_id, value):
    number = _decode_number(value)
    if not isfinite(number):
        ctx.label("NaN" if number != number else t("curve.step"))
        ctx.same_line()
        return 0.0 if ctx.button(f"{t('curve.slope')}##{widget_id}") else value
    return _edit_key_number(ctx, f"##{widget_id}", value, .02, -1.0e7, 1.0e7)


def _evaluate_keys(keys: list[dict], time: float) -> float:
    """Use runtime semantics, including weighted and stepped segments."""
    return AnimationCurve.from_dict(dict(keys=keys, pre_wrap="clamp", post_wrap="clamp")).evaluate(time)


def _view_bounds(keys: list[dict], *, non_negative: bool) -> tuple[float, float, float, float]:
    curve = AnimationCurve.from_dict(dict(keys=keys, pre_wrap="clamp", post_wrap="clamp"))
    points = curve.keys
    times = [key.time for key in points if isfinite(key.time)]
    time_min = min([0.0, *times])
    time_max = max([1.0, *times])
    values = [key.value for key in points if isfinite(key.value)]
    sample_count = max(32, len(keys) * 24)
    for index in range(sample_count + 1):
        sample_time = time_min + (time_max - time_min) * index / sample_count
        value = curve.evaluate(sample_time)
        if isfinite(value):
            values.append(value)
    value_min = min([0.0, *values])
    value_max = max([1.0, *values])
    if non_negative:
        value_min = min(0.0, value_min)
    padding = (
        max(0.5, abs(value_max) * 0.25)
        if value_max - value_min < 1.0e-4
        else (value_max - value_min) * 0.12
    )
    value_min = max(0.0, value_min - padding) if non_negative else value_min - padding
    value_max += padding
    return time_min, time_max, value_min, value_max


def _draw_curve(
    ctx,
    keys: list[dict],
    rect: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    *,
    selected: int = -1,
    show_handles: bool = False,
    dpi_scale: float = 1.0,
) -> dict[str, tuple[float, float]]:
    x0, y0, x1, y1 = rect
    time_min, time_max, value_min, value_max = bounds
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    time_span = max(1.0e-6, time_max - time_min)
    value_span = max(1.0e-6, value_max - value_min)

    def to_screen(time: float, value: float) -> tuple[float, float]:
        return (
            x0 + (time - time_min) / time_span * width,
            y1 - (value - value_min) / value_span * height,
        )

    ctx.draw_filled_rect(x0, y0, x1, y1, *Theme.CURVE_EDITOR_BG, 3.0 * dpi_scale)
    for index in range(1, 5):
        gx = x0 + width * index / 5.0
        gy = y0 + height * index / 5.0
        ctx.draw_line(gx, y0, gx, y1, *Theme.CURVE_EDITOR_GRID, 1.0 * dpi_scale)
        ctx.draw_line(x0, gy, x1, gy, *Theme.CURVE_EDITOR_GRID, 1.0 * dpi_scale)
    if time_min <= 0.0 <= time_max:
        axis_x, _ = to_screen(0.0, value_min)
        ctx.draw_line(axis_x, y0, axis_x, y1, *Theme.CURVE_EDITOR_AXIS, 1.0 * dpi_scale)
    if value_min <= 0.0 <= value_max:
        _, axis_y = to_screen(time_min, 0.0)
        ctx.draw_line(x0, axis_y, x1, axis_y, *Theme.CURVE_EDITOR_AXIS, 1.0 * dpi_scale)

    curve = AnimationCurve.from_dict(dict(keys=keys, pre_wrap="clamp", post_wrap="clamp"))
    previous = None
    steps = max(64, int(width * 0.5))
    for index in range(steps + 1):
        time = time_min + time_span * index / steps
        value = curve.evaluate(time)
        if not isfinite(value):
            previous = None
            continue
        point = to_screen(time, value)
        if previous is not None:
            ctx.draw_line(*previous, *point, *Theme.CURVE_EDITOR_LINE, 2.0 * dpi_scale)
        previous = point

    handles: dict[str, tuple[float, float]] = {}
    for index, item in enumerate(keys):
        time, value = _decode_number(item["time"]), _decode_number(item["value"])
        if not isfinite(time) or not isfinite(value):
            continue
        point = to_screen(time, value)
        color = Theme.CURVE_EDITOR_KEY_SELECTED if index == selected else Theme.CURVE_EDITOR_KEY
        radius = (5.0 if index == selected else 4.0) * dpi_scale
        px, py = point
        top, right = (px, py - radius), (px + radius, py)
        bottom, left = (px, py + radius), (px - radius, py)
        ctx.draw_triangle_filled(top, right, bottom, color)
        ctx.draw_triangle_filled(top, bottom, left, color)
        if index == selected:
            handles["key"] = point

    if show_handles and 0 <= selected < len(keys):
        item = keys[selected]
        key_time, key_value = _decode_number(item["time"]), _decode_number(item["value"])
        if not isfinite(key_time) or not isfinite(key_value):
            return handles
        key_x, key_y = to_screen(key_time, key_value)
        time_per_pixel = time_span / width
        for name, direction, tangent_name in (
            ("in", -1.0, "in_tangent"),
            ("out", 1.0, "out_tangent"),
        ):
            key = Keyframe.from_dict(item)
            tangent = getattr(key, tangent_name)
            if not isfinite(tangent):
                continue
            neighbor = selected - 1 if direction < 0 else selected + 1
            weighted = key.weighted_mode & (WeightedMode.IN if direction < 0 else WeightedMode.OUT)
            handle_span = 48.0 * dpi_scale * time_per_pixel
            if weighted and 0 <= neighbor < len(keys):
                handle_span = abs(_decode_number(keys[neighbor]["time"]) - key.time) * getattr(key, name + "_weight")
            handle_time = key.time + direction * handle_span
            handle_value = _decode_number(item["value"]) + (
                handle_time - _decode_number(item["time"])
            ) * tangent
            handle = to_screen(handle_time, handle_value)
            ctx.draw_line(
                key_x, key_y, *handle, *Theme.CURVE_EDITOR_TANGENT, 1.0 * dpi_scale
            )
            ctx.draw_filled_circle(
                *handle, 3.5 * dpi_scale, *Theme.CURVE_EDITOR_TANGENT, 12
            )
            handles[name] = handle

    ctx.draw_rect(
        x0,
        y0,
        x1,
        y1,
        *Theme.INSPECTOR_LIST_BODY_BORDER,
        1.0 * dpi_scale,
        3.0 * dpi_scale,
    )
    return handles


def _distance_squared(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def render_curve_property(
    ctx,
    widget_id: str,
    value,
    *,
    semantic_prefix: str = "",
    non_negative: bool = False,
):
    """Render a compact curve field; clicking it opens a visual curve editor."""
    dpi = editor_dpi_scale(ctx)
    curve = AnimationCurve.from_dict(value)
    keys = [item.to_dict() for item in curve.keys]
    state = _STATES.setdefault(widget_id, _CurveEditorState())
    state.selected = min(max(state.selected, 0), len(keys) - 1)

    preview_width = max(120.0 * dpi, float(ctx.get_content_region_avail_width()))
    clicked = ctx.invisible_button(
        f"##{widget_id}_curve_preview",
        preview_width,
        Theme.CURVE_EDITOR_PREVIEW_H * dpi,
    )
    preview_rect = (
        float(ctx.get_item_rect_min_x()),
        float(ctx.get_item_rect_min_y()),
        float(ctx.get_item_rect_max_x()),
        float(ctx.get_item_rect_max_y()),
    )
    _draw_curve(
        ctx,
        keys,
        preview_rect,
        _view_bounds(keys, non_negative=non_negative),
        dpi_scale=dpi,
    )
    if ctx.is_item_hovered():
        ctx.set_mouse_cursor(7)
        ctx.set_tooltip(t("curve.click_to_edit"))
    ctx.record_semantic_rect(
        "curve",
        "AnimationCurve",
        preview_rect[0],
        preview_rect[1],
        preview_rect[2] - preview_rect[0],
        preview_rect[3] - preview_rect[1],
        True,
        semantic_prefix,
    )

    popup_id = f"##{widget_id}_curve_editor"
    if clicked:
        state.opening_curve = curve.to_dict()
        ctx.open_popup(popup_id)
    ctx.set_next_window_size(440.0 * dpi, 570.0 * dpi, Theme.COND_FIRST_USE_EVER)
    ctx.push_style_var_vec2(ImGuiStyleVar.WindowMinSize, 440.0 * dpi, 220.0 * dpi)
    opened = ctx.begin_popup(popup_id, resizable=True)
    ctx.pop_style_var(1)
    if not opened:
        return curve.to_dict()

    try:
        ctx.set_next_item_width(220.0 * dpi)
        preset = ctx.combo(f"##{widget_id}_preset", 0,
                           [t(f"curve.preset_{name}") for name in ("choose", "linear", "smooth", "zero", "one")], -1)
        ctx.same_line()
        reset = IGUI.icon_button(ctx, f"##{widget_id}_reset", "reset", tooltip=t("curve.reset"))
        if preset:
            curve = (AnimationCurve.linear(0, 0, 1, 1) if preset == 1 else
                     AnimationCurve.ease_in_out(0, 0, 1, 1) if preset == 2 else
                     AnimationCurve.constant(0, 1, 0 if preset == 3 else 1))
        elif reset:
            curve = AnimationCurve.from_dict(state.opening_curve)
        if preset or reset:
            keys = [item.to_dict() for item in curve.keys]
            state.selected = 0
            state.dragging = ""
        ctx.separator()
        ctx.align_text_to_frame_padding()
        ctx.label(t("curve.pre_wrap"))
        ctx.same_line(72.0 * dpi)
        ctx.set_next_item_width(120.0 * dpi)
        pre_index = ctx.combo(
            f"##{widget_id}_pre",
            CURVE_WRAP_MODES.index(curve.pre_wrap),
            [t(f"curve.wrap_{mode}") for mode in CURVE_WRAP_MODES],
            -1,
        )
        ctx.same_line(218.0 * dpi)
        ctx.align_text_to_frame_padding()
        ctx.label(t("curve.post_wrap"))
        ctx.same_line(294.0 * dpi)
        ctx.set_next_item_width(-1.0)
        post_index = ctx.combo(
            f"##{widget_id}_post",
            CURVE_WRAP_MODES.index(curve.post_wrap),
            [t(f"curve.wrap_{mode}") for mode in CURVE_WRAP_MODES],
            -1,
        )

        if not keys:
            ctx.label(t("curve.empty"))
            if IGUI.icon_button(ctx, f"##{widget_id}_empty_add", "plus", tooltip=t("curve.add_key")):
                keys.append(Keyframe().to_dict())
            else:
                return AnimationCurve(keys=keys, pre_wrap=CURVE_WRAP_MODES[pre_index],
                                      post_wrap=CURVE_WRAP_MODES[post_index]).to_dict()
            state.selected = 0

        canvas_width = max(280.0 * dpi, float(ctx.get_content_region_avail_width()))
        ctx.invisible_button(
            f"##{widget_id}_curve_canvas",
            canvas_width,
            Theme.CURVE_EDITOR_CANVAS_H * dpi,
        )
        canvas_rect = (
            float(ctx.get_item_rect_min_x()),
            float(ctx.get_item_rect_min_y()),
            float(ctx.get_item_rect_max_x()),
            float(ctx.get_item_rect_max_y()),
        )
        bounds = _view_bounds(keys, non_negative=non_negative)
        handles = _draw_curve(
            ctx,
            keys,
            canvas_rect,
            bounds,
            selected=state.selected,
            show_handles=True,
            dpi_scale=dpi,
        )
        hovered = bool(ctx.is_item_hovered())
        mouse = (float(ctx.get_mouse_pos_x()), float(ctx.get_mouse_pos_y()))

        time_min, time_max, value_min, value_max = bounds
        x0, y0, x1, y1 = canvas_rect

        def from_screen(position: tuple[float, float]) -> tuple[float, float]:
            time = time_min + (position[0] - x0) / max(1.0, x1 - x0) * (time_max - time_min)
            result = value_max - (position[1] - y0) / max(1.0, y1 - y0) * (value_max - value_min)
            return time, result

        if hovered and ctx.is_mouse_double_clicked(0) :
            new_time, new_value = from_screen(mouse)
            if non_negative:
                new_value = max(0.0, new_value)
            if any(abs(float(k["time"]) - new_time) < 1.0e-6 for k in keys):
                new_time += 1.0e-4
            keys.append(
                {"time": new_time, "value": new_value, "in_tangent": 0.0, "out_tangent": 0.0}
            )
            keys.sort(key=lambda item: _decode_number(item["time"]))
            state.selected = min(
                range(len(keys)), key=lambda index: abs(_decode_number(keys[index]["time"]) - new_time)
            )
            state.dragging = "key"
        elif hovered and ctx.is_mouse_button_clicked(0):
            candidates = [(name, _distance_squared(mouse, point)) for name, point in handles.items()]
            if candidates:
                name, distance = min(candidates, key=lambda item: item[1])
                if distance <= (12.0 * dpi) ** 2:
                    state.dragging = name
            for index, item in enumerate(keys):
                key_point = (
                    x0 + (_decode_number(item["time"]) - time_min) / max(1.0e-6, time_max - time_min) * (x1 - x0),
                    y1 - (_decode_number(item["value"]) - value_min) / max(1.0e-6, value_max - value_min) * (y1 - y0),
                )
                if _distance_squared(mouse, key_point) <= (10.0 * dpi) ** 2:
                    state.selected = index
                    state.dragging = "key"
                    break

        if state.dragging and ctx.is_mouse_button_down(0):
            selected = keys[state.selected]
            mouse_time, mouse_value = from_screen(mouse)
            if state.dragging == "key":
                minimum = _decode_number(keys[state.selected - 1]["time"]) + 1.0e-4 if state.selected else -1.0e7
                maximum = (
                    _decode_number(keys[state.selected + 1]["time"]) - 1.0e-4
                    if state.selected + 1 < len(keys)
                    else 1.0e7
                )
                selected["time"] = min(max(mouse_time, minimum), maximum)
                selected["value"] = max(0.0, mouse_value) if non_negative else mouse_value
            elif state.dragging in {"in", "out"}:
                delta_time = mouse_time - _decode_number(selected["time"])
                if abs(delta_time) > 1.0e-6:
                    selected[f"{state.dragging}_tangent"] = (
                        mouse_value - _decode_number(selected["value"])
                    ) / delta_time
                    key = Keyframe.from_dict(selected)
                    side = state.dragging
                    bit = WeightedMode.IN if side == "in" else WeightedMode.OUT
                    neighbor = state.selected + (-1 if side == "in" else 1)
                    if key.weighted_mode & bit and 0 <= neighbor < len(keys):
                        span = abs(_decode_number(keys[neighbor]["time"]) - key.time)
                        if span:
                            selected[f"{side}_weight"] = min(1.0, max(0.0, abs(delta_time) / span))
        elif state.dragging:
            state.dragging = ""

        selected = keys[state.selected]
        for field, default in (("in_weight", 0.0), ("out_weight", 0.0), ("weighted_mode", 0)):
            selected.setdefault(field, default)
        minimum = _decode_number(keys[state.selected - 1]["time"]) + 1.0e-4 if state.selected else -1.0e7
        maximum = (
            _decode_number(keys[state.selected + 1]["time"]) - 1.0e-4
            if state.selected + 1 < len(keys)
            else 1.0e7
        )
        first_input_x = 92.0 * dpi
        second_label_x = 218.0 * dpi
        second_input_x = 294.0 * dpi
        ctx.align_text_to_frame_padding()
        ctx.label(t("curve.time"))
        ctx.same_line(first_input_x)
        ctx.set_next_item_width(110.0 * dpi)
        selected["time"] = _edit_key_number(
            ctx, f"##{widget_id}_time", selected["time"], 0.01, minimum, maximum)
        ctx.same_line(second_label_x)
        ctx.label(t("curve.value"))
        ctx.same_line(second_input_x)
        ctx.set_next_item_width(-1.0)
        selected["value"] = _edit_key_number(
            ctx, f"##{widget_id}_value", selected["value"], 0.01,
            0.0 if non_negative else -1.0e7, 1.0e7)
        ctx.align_text_to_frame_padding()
        ctx.label(t("curve.in_tangent"))
        ctx.same_line(first_input_x)
        ctx.set_next_item_width(110.0 * dpi)
        selected["in_tangent"] = _edit_tangent(ctx, f"{widget_id}_in", selected["in_tangent"])
        ctx.same_line(second_label_x)
        ctx.label(t("curve.out_tangent"))
        ctx.same_line(second_input_x)
        ctx.set_next_item_width(-1.0)
        selected["out_tangent"] = _edit_tangent(ctx, f"{widget_id}_out", selected["out_tangent"])
        ctx.label(t("curve.weighted_mode"))
        ctx.same_line(140.0 * dpi)
        ctx.set_next_item_width(-1.0)
        original_mode = int(selected["weighted_mode"])
        mode = ctx.combo(f"##{widget_id}_weighted", original_mode & 3,
                         [t(f"curve.weight_{name}") for name in ("none", "in", "out", "both")], -1)
        selected["weighted_mode"] = (original_mode & ~3) | mode
        for side in ("in", "out"):
            ctx.label(t(f"curve.{side}_weight"))
            ctx.same_line(140.0 * dpi)
            ctx.set_next_item_width(-1.0)
            selected[f"{side}_weight"] = _edit_key_number(
                ctx, f"##{widget_id}_{side}_weight", selected[f"{side}_weight"], .005, 0.0, 1.0)
        for side in ("in", "out"):
            if side == "out":
                ctx.same_line()
            if ctx.button(f"{t(f'curve.step_{side}')}##{widget_id}_step_{side}"):
                selected[f"{side}_tangent"] = "infinity"

        if IGUI.icon_button(ctx, f"##{widget_id}_add", "plus", tooltip=t("curve.add_key")):
            if len(keys) == 1:
                insertion_index = 1
                new_time = _decode_number(keys[0]["time"]) + 1.0
            else:
                largest_gap = max(
                    range(len(keys) - 1),
                    key=lambda index: _decode_number(keys[index + 1]["time"])
                    - _decode_number(keys[index]["time"]),
                )
                insertion_index = largest_gap + 1
                new_time = (
                    float(keys[largest_gap]["time"])
                    + float(keys[insertion_index]["time"])
                ) * 0.5
            keys.insert(
                insertion_index,
                {
                    "time": new_time,
                    "value": _evaluate_keys(keys, new_time),
                    "in_tangent": 0.0,
                    "out_tangent": 0.0,
                },
            )
            state.selected = insertion_index
        if keys:
            ctx.same_line(0, 8.0 * dpi)
            if IGUI.icon_button(ctx, f"##{widget_id}_remove", "minus", tooltip=t("curve.remove_key")):
                del keys[state.selected]
                state.selected = min(state.selected, len(keys) - 1)

        return AnimationCurve.from_dict(
            {
                "keys": keys,
                "pre_wrap": CURVE_WRAP_MODES[pre_index],
                "post_wrap": CURVE_WRAP_MODES[post_index],
            }
        ).to_dict()
    finally:
        ctx.end_popup()
