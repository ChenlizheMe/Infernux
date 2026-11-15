"""Completion notifications for infrequent editor operations."""

from __future__ import annotations

import time
from .dpi import editor_dpi_scale
from .editor_icons import EditorIcons
from .engine_status import EngineStatus
from .igui import IGUI
from .theme import Theme, ImGuiCol, ImGuiStyleVar, ImGuiWindowFlags
from infernux.engine.i18n import t


class EditorToasts:
    def __init__(self):
        self._messages = []

    def render(self, ctx):
        now = time.monotonic()
        self._messages.extend((sequence, text, kind, source, now)
                              for sequence, text, kind, source in EngineStatus.take_notifications())
        self._messages = [item for item in self._messages if item[2] == "error" or now - item[4] < 3.0][-3:]
        if not self._messages:
            return
        dpi = editor_dpi_scale(ctx)
        vx, vy, vw, vh = ctx.get_main_viewport_bounds()
        width = min(360.0 * dpi, vw - 32.0 * dpi)
        flags = (ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.NoDocking | ImGuiWindowFlags.NoSavedSettings
                 | ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoMove)
        for index, item in enumerate(reversed(tuple(self._messages))):
            sequence, text, kind, source, started = item
            alpha = 1.0 if kind == "error" else min(1.0, (3.0 - now + started) / 0.4)
            ctx.set_next_window_pos(vx + vw - 12.0 * dpi, vy + vh - (38.0 + index * 84.0) * dpi,
                                    Theme.COND_ALWAYS, 1.0, 1.0)
            ctx.set_next_window_size(width, 76.0 * dpi, Theme.COND_ALWAYS)
            ctx.push_style_var_float(ImGuiStyleVar.Alpha, alpha)
            ctx.push_style_color(ImGuiCol.WindowBg, *Theme.INSPECTOR_HEADER_PRIMARY)
            visible = ctx.begin_window(f"##editor_toast_{sequence}", True, flags)
            try:
                if visible:
                    color = Theme.LOG_ERROR if kind == "error" else Theme.SUCCESS_TEXT
                    x, y = ctx.get_window_pos_x(), ctx.get_window_pos_y()
                    ctx.draw_filled_rect(x, y, x + 3.0 * dpi, y + 76.0 * dpi, *color)
                    icon = EditorIcons.get_cached("error" if kind == "error" else "check")
                    if icon:
                        ctx.draw_image_rect(icon, x + 12.0 * dpi, y + 12.0 * dpi,
                                            x + 26.0 * dpi, y + 26.0 * dpi)
                    ctx.set_cursor_pos_x(34.0 * dpi)
                    ctx.set_cursor_pos_y(10.0 * dpi)
                    ctx.invisible_button(f"##message_{sequence}", width - 70.0 * dpi, 24.0 * dpi)
                    ctx.draw_text_aligned(x + 34.0 * dpi, y + 10.0 * dpi,
                                          x + width - 36.0 * dpi, y + 34.0 * dpi,
                                          text.splitlines()[0], *Theme.TEXT, 0.0, 0.5, 0.0, True)
                    if ctx.is_item_hovered():
                        ctx.set_tooltip(text)
                    ctx.set_cursor_pos_x(34.0 * dpi)
                    ctx.set_cursor_pos_y(42.0 * dpi)
                    with IGUI.mono_font(ctx):
                        ctx.draw_text_aligned(x + 34.0 * dpi, y + 42.0 * dpi,
                                              x + width - 12.0 * dpi, y + 64.0 * dpi,
                                              source, *Theme.TEXT_DISABLED, 0.0, 0.5, 0.0, True)
                    ctx.set_cursor_pos_x(width - 30.0 * dpi)
                    ctx.set_cursor_pos_y(8.0 * dpi)
                    if IGUI.icon_button(ctx, "##dismiss", "close", tooltip=t("editor.toast.dismiss"), kind="ghost"):
                        self._messages.remove(item)
            finally:
                ctx.end_window()
                ctx.pop_style_color(1)
                ctx.pop_style_var(1)
