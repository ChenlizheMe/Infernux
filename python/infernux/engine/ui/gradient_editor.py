"""Shared gradient field editor for Inspector-style surfaces."""

from __future__ import annotations

from functools import lru_cache
from math import cbrt
from infernux.engine.i18n import t
from infernux.graph.ramp import GRADIENT_MODES, MAX_RAMP_KEYS, Gradient

from .dpi import editor_dpi_scale
from .igui import IGUI
from .theme import ImGuiStyleVar

_SELECTED_STOPS: dict[str, int] = {}


def _matrix_vector(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


@lru_cache(maxsize=128)
def _perceptual_samples(left, right, segments):
    """Match the particle backend's linear RGB / Oklab interpolation."""
    to_lms = ((0.4122214708, 0.5363325363, 0.0514459929),
              (0.2119034982, 0.6806995451, 0.1073969566),
              (0.0883024619, 0.2817188376, 0.6299787005))
    to_lab = ((0.2104542553, 0.7936177850, -0.0040720468),
              (1.9779984951, -2.4285922050, 0.4505937099),
              (0.0259040371, 0.7827717662, -0.8086757660))
    from_lab = ((1.0, 0.3963377774, 0.2158037573),
                (1.0, -0.1055613458, -0.0638541728),
                (1.0, -0.0894841775, -1.2914855480))
    from_lms = ((4.0767416621, -3.3077115913, 0.2309699292),
                (-1.2684380046, 2.6097574011, -0.3413193965),
                (-0.0041960863, -0.7034186147, 1.7076147010))
    start = _matrix_vector(to_lab, tuple(cbrt(x) for x in _matrix_vector(to_lms, left[:3])))
    end = _matrix_vector(to_lab, tuple(cbrt(x) for x in _matrix_vector(to_lms, right[:3])))
    colors = [left]
    for index in range(1, segments):
        factor = index / segments
        lab = tuple(a + (b - a) * factor for a, b in zip(start, end))
        roots = _matrix_vector(from_lab, lab)
        rgb = _matrix_vector(from_lms, tuple(x ** 3 for x in roots))
        colors.append((*rgb, left[3] + (right[3] - left[3]) * factor))
    colors.append(right)
    return tuple(colors)


def _gradient_strip(ctx, widget_id, keys, mode, *, editable=False):
    """Draw the authored stops directly; dragging a stop preserves key order."""
    from .theme import Theme
    dpi = editor_dpi_scale(ctx)
    width = ctx.get_content_region_avail_width()
    if editable:
        ctx.dummy(width, 42.0 * dpi)
        clicked = False
    else:
        ctx.invisible_button(f"##{widget_id}_strip", width, 22.0 * dpi)
        clicked = ctx.is_item_clicked(0)
    x0, y0 = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
    x1, y1 = ctx.get_item_rect_max_x(), ctx.get_item_rect_max_y()
    inset = 9.0 * dpi if editable else 0.0
    x0, x1 = x0 + inset, x1 - inset
    width = x1 - x0
    top, bottom = y0 + inset, y1 - inset
    stops = [{"time": 0.0, "color": keys[0]["color"]}, *keys,
             {"time": 1.0, "color": keys[-1]["color"]}]
    for left, right in zip(stops, stops[1:]):
        a, b = x0 + width * left["time"], x0 + width * right["time"]
        if mode == "perceptual_blend":
            segments = max(1, round((right["time"] - left["time"]) * 24))
            colors = _perceptual_samples(tuple(left["color"]), tuple(right["color"]), segments)
            for index in range(segments):
                ctx.draw_gradient_rect(a + (b - a) * index / segments, top,
                                       a + (b - a) * (index + 1) / segments, bottom,
                                       colors[index], colors[index + 1])
        else:
            ctx.draw_gradient_rect(a, top, b, bottom, left["color"],
                                   left["color"] if mode == "fixed" else right["color"])
    if editable:
        restore_x, restore_y = ctx.get_cursor_pos_x(), ctx.get_cursor_pos_y()
        for index, key in enumerate(keys):
            x = x0 + width * key["time"]
            active = False
            for side, y in (("color", bottom), ("alpha", y0)):
                ctx.set_cursor_screen_pos(x - 6.0 * dpi, y)
                ctx.invisible_button(f"##{widget_id}_{side}_stop_{index}", 12.0 * dpi, 9.0 * dpi)
                if ctx.is_item_clicked(0):
                    _SELECTED_STOPS[widget_id] = index
                if ctx.is_item_active():
                    active = True
            color = Theme.ROLE_ACCENT if _SELECTED_STOPS.get(widget_id, 0) == index else key["color"]
            ctx.draw_triangle_filled((x, bottom), (x - 5.0 * dpi, y1), (x + 5.0 * dpi, y1), color)
            ctx.draw_triangle_filled((x, top), (x - 5.0 * dpi, y0), (x + 5.0 * dpi, y0), Theme.TEXT_DISABLED)
            if active:
                lo = keys[index - 1]["time"] + 0.0001 if index else 0.0
                hi = keys[index + 1]["time"] - 0.0001 if index + 1 < len(keys) else 1.0
                key["time"] = min(hi, max(lo, (ctx.get_mouse_pos_x() - x0) / max(width, 1.0)))
        ctx.set_cursor_pos_x(restore_x)
        ctx.set_cursor_pos_y(restore_y)
    return clicked


def render_gradient_field(ctx, widget_id, value, *, hdr=False):
    """Compact field preview and an independently scrollable popup editor."""
    dpi = editor_dpi_scale(ctx)
    gradient = Gradient.from_dict(value)
    keys = [key.to_dict() for key in gradient.keys]
    popup = f"##{widget_id}_gradient_popup"
    if _gradient_strip(ctx, widget_id, keys, gradient.mode):
        ctx.open_popup(popup)
    ctx.set_next_window_size(480.0 * dpi, 390.0 * dpi, 4)
    ctx.push_style_var_vec2(ImGuiStyleVar.WindowMinSize, 320.0 * dpi, 180.0 * dpi)
    opened = ctx.begin_popup(popup, resizable=True)
    ctx.pop_style_var(1)
    if opened:
        try:
            ctx.begin_child(f"##{widget_id}_gradient_keys", 0.0, 0.0, False)
            try:
                return render_gradient_property(ctx, widget_id, value, hdr=hdr)
            finally:
                ctx.end_child()
        finally:
            ctx.end_popup()
    return value


def render_gradient_property(
    ctx,
    widget_id: str,
    value,
    *,
    semantic_prefix: str = "",
    hdr: bool = False,
):
    """Render an authored Gradient document and return its edited document."""
    from infernux.engine.ui.inspector_utils import render_color_value_bar

    dpi = editor_dpi_scale(ctx)
    gradient = Gradient.from_dict(value)
    keys = [item.to_dict() for item in gradient.keys]
    _gradient_strip(ctx, widget_id, keys, gradient.mode, editable=True)
    mode_index = ctx.combo(
        f"{t('particle_graph_editor.gradient_mode')}##{widget_id}_mode",
        GRADIENT_MODES.index(gradient.mode),
        [t(f"gradient.mode_{mode}") for mode in GRADIENT_MODES],
        -1,
    )
    remove_index = -1
    selected = min(_SELECTED_STOPS.get(widget_id, 0), len(keys) - 1)
    for index in (selected,):
        item = keys[index]
        ctx.separator()
        ctx.label(f"{t('particle_graph_editor.key')} {index + 1}")
        minimum = keys[index - 1]["time"] + 1.0e-4 if index else 0.0
        maximum = (
            keys[index + 1]["time"] - 1.0e-4
            if index + 1 < len(keys)
            else 1.0
        )
        edited_time = float(
            ctx.drag_float(
                f"{t('particle_graph_editor.time')}##{widget_id}_{index}_time",
                item["time"],
                0.01,
                minimum,
                maximum,
            )
        )
        if ctx.is_item_edited():
            item["time"] = edited_time
        prefix = semantic_prefix or f"gradient.{widget_id}"
        ctx.record_semantic_item(
            "drag_float",
            t("particle_graph_editor.time"),
            True,
            f"{prefix}.key.{index}.time",
            numeric_value=item["time"],
        )
        ctx.align_text_to_frame_padding()
        ctx.label(t("particle_graph_editor.color"))
        ctx.same_line(92.0 * dpi)
        ctx.set_next_item_width(-1.0)
        color = render_color_value_bar(
            ctx,
            f"##{widget_id}_{index}_color",
            item["color"],
            allow_hdr=hdr,
            default_hdr_enabled=hdr,
        )
        for channel_index, channel in enumerate(("r", "g", "b", "a")):
            ctx.record_semantic_item(
                "color_channel",
                channel.upper(),
                True,
                f"{prefix}.key.{index}.color.{channel}",
                numeric_value=color[channel_index],
            )
        item["color"] = color
        if len(keys) > 1 and IGUI.icon_button(
            ctx, f"##{widget_id}_{index}_remove", "close", tooltip=t('particle_graph_editor.remove_key')
        ):
            remove_index = index
    if remove_index >= 0:
        del keys[remove_index]
        _SELECTED_STOPS[widget_id] = max(0, remove_index - 1)
    if len(keys) < MAX_RAMP_KEYS and IGUI.icon_button(
        ctx, f"##{widget_id}_add", "plus", tooltip=t('particle_graph_editor.add_key')
    ):
        if len(keys) == 1:
            new_time = 1.0 if keys[0]["time"] < 1.0 else 0.0
            keys.append({"time": new_time, "color": list(keys[0]["color"])})
            keys.sort(key=lambda item: item["time"])
            _SELECTED_STOPS[widget_id] = next(index for index, key in enumerate(keys) if key["time"] == new_time)
        else:
            gap_index = max(
                range(len(keys) - 1),
                key=lambda index: keys[index + 1]["time"] - keys[index]["time"],
            )
            left = keys[gap_index]
            right = keys[gap_index + 1]
            keys.insert(
                gap_index + 1,
                {
                    "time": (left["time"] + right["time"]) * 0.5,
                    "color": [
                        a + (b - a) * 0.5
                        for a, b in zip(left["color"], right["color"])
                    ],
                },
            )
            _SELECTED_STOPS[widget_id] = gap_index + 1
    return Gradient.from_dict(
        {"keys": keys, "mode": GRADIENT_MODES[mode_index]}
    ).to_dict()


__all__ = ["render_gradient_property", "render_gradient_field"]
