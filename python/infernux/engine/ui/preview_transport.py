"""Shared transport chrome; preview owners keep their existing playback state."""

from __future__ import annotations

from .dpi import editor_dpi_scale
from .igui import IGUI
from .theme import Theme, ImGuiCol
from infernux.engine.i18n import t


def render_preview_transport(
    ctx, key, *, playing, seconds, duration, on_toggle, on_stop,
    on_step=None, on_seek=None, frame=None, frame_count=None, enabled=True,
    toggle_icon=None, toggle_label=None,
    actions=(),
):
    """Draw one seek bar and segmented transport without owning playback logic."""
    dpi = editor_dpi_scale(ctx)
    wide = ctx.get_content_region_avail_width() >= 320.0 * dpi
    ctx.push_id_str(key)
    ctx.begin_group()
    if duration > 0.0:
        if on_seek is None:
            ctx.dummy(ctx.get_content_region_avail_width(), 12.0 * dpi)
        else:
            ctx.invisible_button("##seek", ctx.get_content_region_avail_width(), 12.0 * dpi)
        x0, y0 = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
        x1, y1 = ctx.get_item_rect_max_x(), ctx.get_item_rect_max_y()
        progress = min(1.0, max(0.0, seconds / duration)) if seconds is not None else 0.0
        y = (y0 + y1) * 0.5
        ctx.draw_line(x0, y, x1, y, *Theme.BORDER, 2.0 * dpi)
        head = x0 + (x1 - x0) * progress
        ctx.draw_line(x0, y, head, y, *Theme.ROLE_ACCENT, 2.0 * dpi)
        if seconds is not None:
            ctx.draw_line(head, y0 + dpi, head, y1 - dpi, *Theme.TEXT, 2.0 * dpi)
        if enabled and on_seek is not None and ctx.is_item_active():
            on_seek(min(duration, max(0.0, (ctx.get_mouse_pos_x() - x0) / max(x1 - x0, 1.0) * duration)))
        ctx.record_semantic_item("timeline_scrubber", t("transport.seek"), enabled and on_seek is not None,
                                 f"{key}.time", numeric_value=seconds or 0.0)

    ctx.begin_disabled(not enabled)
    buttons = []
    if wide and on_step is not None:
        buttons.append(("previous", "step_b", t("transport.previous"), lambda: on_step(-1)))
    buttons.extend((
        ("play_pause", toggle_icon or ("pause" if playing else "play"),
         toggle_label or t("transport.pause" if playing else "transport.play"), on_toggle),
        ("stop", "stop", t("transport.stop"), on_stop),
    ))
    if wide and on_step is not None:
        buttons.append(("next", "step_f", t("transport.next"), lambda: on_step(1)))
    for index, (action, icon, label, callback) in enumerate(buttons):
        if index:
            ctx.same_line(0.0, dpi)
        if IGUI.icon_button(ctx, f"##{action}", icon, tooltip=label, width=26.0, height=22.0,
                            kind="primary" if playing and action == "play_pause" else "ghost"):
            callback()
        ctx.record_semantic_item("button", label, enabled, f"{key}.{action}")
    ctx.end_disabled()
    ctx.same_line(0.0, 8.0 * dpi)
    with IGUI.mono_font(ctx):
        ctx.align_text_to_frame_padding()
        ctx.push_style_color(ImGuiCol.Text, *(Theme.ROLE_ACCENT if playing else Theme.TEXT_DISABLED))
        current = _time_label(seconds) if seconds is not None else "—"
        ctx.label(f"{current} / {_time_label(duration)}")
        ctx.pop_style_color(1)
        if wide and frame is not None:
            ctx.same_line(0.0, 8.0 * dpi)
            ctx.label(f"F {frame} / {frame_count}")
    for action, icon, label, available, callback in actions:
        ctx.same_line(0.0, 8.0 * dpi)
        ctx.begin_disabled(not available)
        if IGUI.icon_button(ctx, f"##{action}", icon, tooltip=label, kind="ghost", width=26.0, height=22.0):
            callback()
        ctx.record_semantic_item("button", label, available, f"{key}.{action}")
        ctx.end_disabled()
    ctx.end_group()
    ctx.pop_id()


def _time_label(seconds):
    minutes, remainder = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{remainder:05.2f}"
