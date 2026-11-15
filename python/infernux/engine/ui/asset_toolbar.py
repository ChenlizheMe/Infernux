"""Shared document toolbar for asset editors; commands stay with the owner."""

from __future__ import annotations

import os

from infernux.engine.i18n import t
from .igui import IGUI
from .theme import Theme


def _caption(ctx, owner, text, width):
    key = (text, round(width, 1), ctx.get_dpi_scale())
    cached = getattr(owner, "_asset_toolbar_caption", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    fitted = text
    if ctx.calc_text_width(text) > width:
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if ctx.calc_text_width(text[:middle] + "…") <= width:
                low = middle
            else:
                high = middle - 1
        fitted = text[:low] + "…"
    owner._asset_toolbar_caption = (key, fitted)
    return fitted


def render_asset_toolbar(ctx, owner, *, icon, name, path, dirty, on_save,
                         actions, semantic_prefix, can_save=True):
    """Actions are (id, localized label, enabled, callback) tuples."""
    dpi = ctx.get_dpi_scale()
    selected = None
    ctx.push_id_str(semantic_prefix)
    try:
        ctx.set_next_item_allow_overlap()
        ctx.invisible_button("##document_header", ctx.get_content_region_avail_width(), 34 * dpi)
        left, top = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
        right, bottom = ctx.get_item_rect_max_x(), ctx.get_item_rect_max_y()
        cursor_x, cursor_y = ctx.get_cursor_pos_x(), ctx.get_cursor_pos_y()
        if path and ctx.is_item_hovered():
            ctx.set_tooltip(path)
        texture = ctx.editor_icon(icon)
        if texture:
            ctx.draw_image_rect(texture, left + 4 * dpi, top + 8 * dpi, left + 22 * dpi, top + 26 * dpi)
        save_x, more_x = right - 106 * dpi, right - 30 * dpi
        caption_x = left + 30 * dpi
        caption_width = max(0, save_x - 12 * dpi - caption_x - (12 * dpi if dirty else 0))
        with IGUI.mono_font(ctx):
            text = _caption(ctx, owner, os.path.basename(path) if path else name, caption_width) if caption_width else ""
            ctx.draw_text_aligned(caption_x, top, save_x - 12 * dpi, bottom, text, *Theme.TEXT, 0, 0.5, 0, True)
            if dirty:
                x = min(save_x - 18 * dpi, caption_x + ctx.calc_text_width(text) + 8 * dpi)
                dot = ctx.editor_icon("glyphs/dirty")
                if dot:
                    ctx.draw_image_rect(dot, x - 3 * dpi, top + 12 * dpi, x + 3 * dpi, top + 18 * dpi,
                                        0, 0, 1, 1, *Theme.ROLE_ACCENT)
        ctx.set_cursor_screen_pos(save_x, top + 3 * dpi)
        style = Theme.push_button_kind_style(ctx, "primary")
        ctx.begin_disabled(not can_save)
        try:
            if ctx.button(t("asset_toolbar.save") + "##save", width=72 * dpi, height=28 * dpi):
                selected = ("save", on_save)
            ctx.record_semantic_item("button", t("asset_toolbar.save"), can_save, f"{semantic_prefix}.save")
        finally:
            ctx.end_disabled()
            Theme.pop_button_kind_style(ctx, style)
        ctx.set_cursor_screen_pos(more_x, top + 3 * dpi)
        if IGUI.icon_button(ctx, "##actions", "more", tooltip=t("asset_toolbar.actions"),
                            kind="ghost", width=26, height=28):
            ctx.open_popup("##document_actions")
        ctx.record_semantic_item("button", t("asset_toolbar.actions"), True, f"{semantic_prefix}.actions")
        ctx.set_cursor_pos_x(cursor_x)
        ctx.set_cursor_pos_y(cursor_y)
        if ctx.begin_popup("##document_actions"):
            try:
                for action, label, enabled, callback in actions:
                    if ctx.menu_item(label, "", False, enabled):
                        selected = (action, callback)
                    ctx.record_semantic_item("menu_item", label, enabled, f"{semantic_prefix}.{action}")
            finally:
                ctx.end_popup()
    finally:
        ctx.pop_id()
    if selected is not None:
        selected[1]()
        return selected[0]
    return ""
