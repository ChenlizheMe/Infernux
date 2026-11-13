"""A compact, single-window settings workspace shared by separate groups."""
from __future__ import annotations

from infernux.engine.i18n import t
from .theme import Theme, ImGuiCol, ImGuiStyleVar


class SettingsWorkspaceMixin:
    """Page navigation is a child context of one real editor window."""
    SETTINGS_PAGES = ()

    def current_child_context_id(self):
        return self._settings_page

    def restore_child_context(self, context_id):
        if not context_id:
            return True
        if context_id not in {page[0] for page in self.SETTINGS_PAGES}:
            return False
        self._settings_page = context_id
        return True

    def select_settings_page(self, page):
        if page != self._settings_page:
            self._settings_page = page
            self.publish_child_context(page, reason="settings_page")


def render_settings_workspace(ctx, panel, render_page):
    dpi = ctx.get_dpi_scale()
    pages = panel.SETTINGS_PAGES
    ctx.set_next_item_width(-1)
    query = ctx.input_text_with_hint("##settings_search", t("settings.search"), getattr(panel, "_settings_query", ""), 256)
    panel._settings_query = query
    ctx.dummy(0, 4 * dpi)
    height = ctx.get_content_region_avail_height()
    ctx.push_style_var_vec2(ImGuiStyleVar.CellPadding, 0, 0)
    try:
        if not ctx.begin_table("##settings_workspace", 2, (1 << 9) | (3 << 13), 0):
            return
        try:
            ctx.table_setup_column("##navigation", 1 << 4, 180 * dpi, 0)
            ctx.table_setup_column("##content", 1 << 3, 1, 1)
            ctx.table_next_row()
            ctx.table_set_column_index(0)
            ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, 8 * dpi, 8 * dpi)
            visible = ctx.begin_child("##settings_navigation", 0, height, padding=True)
            ctx.pop_style_var(1)
            try:
                if visible:
                    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, 4 * dpi, 0)
                    try:
                        terms = query.casefold().split()
                        for key, title, description, keywords in pages:
                            label = t(title)
                            if not all(term in f"{label} {keywords}".casefold() for term in terms):
                                continue
                            if ctx.selectable(f"{label}##{key}", key == panel._settings_page, height=Theme.EDITOR_ROW_HEIGHT * dpi):
                                panel.select_settings_page(key)
                            ctx.record_semantic_item("settings_page", label, True, f"{panel.panel_type_id}.page.{key}")
                    finally:
                        ctx.pop_style_var(1)
            finally:
                ctx.end_child()
            ctx.table_set_column_index(1)
            px, py = Theme.WORKSPACE_PADDING
            ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding, px * dpi, py * dpi)
            visible = ctx.begin_child("##settings_content", 0, height, padding=True)
            ctx.pop_style_var(1)
            try:
                if visible:
                    active = next(page for page in pages if page[0] == panel._settings_page)
                    ctx.set_window_font_scale(Theme.EDITOR_TITLE_FONT_SIZE / Theme.EDITOR_FONT_SIZE)
                    ctx.label(t(active[1]))
                    ctx.set_window_font_scale(1.0)
                    ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DISABLED)
                    ctx.text_wrapped(t(active[2]))
                    ctx.pop_style_color(1)
                    ctx.dummy(0, 8 * dpi)
                    render_page(ctx, active[0])
            finally:
                ctx.end_child()
        finally:
            ctx.end_table()
    finally:
        ctx.pop_style_var(1)
