"""Theme catalog using the same list/detail structure as the plugin browser."""
from infernux.engine.editor_themes import EditorThemes
from infernux.engine.i18n import t
from infernux.engine.interaction import PanelInteractionDescriptor
from .catalog_browser import CatalogBrowserPanel, metric
from .panel_registry import editor_panel
from .theme import ImGuiCol, Theme, active_editor_theme


@editor_panel("Themes", type_id="themes", title_key="themes.title", menu_path="",
              interaction=PanelInteractionDescriptor())
class ThemePanel(CatalogBrowserPanel):
    def __init__(self):
        super().__init__(t("themes.title"), "themes")
        self._search = ""
        self._scope = 0
        self._selected = ""
        self._rows_key = None
        self._rows = ()
        self._requested = ""

    def on_render_content(self, ctx):
        service = EditorThemes.instance()
        if service is None:
            ctx.text_wrapped(t("themes.unavailable"))
            return
        self._render_toolbar(ctx)
        key = (service.revision, self._search, self._scope)
        if key != self._rows_key:
            query = self._search.strip().casefold()
            self._rows = tuple(row for row in service.catalog()
                               if (self._scope != 1 or row["builtin"])
                               and (self._scope != 2 or not row["builtin"])
                               and (not query or query in " ".join((row["name"], row["author"],
                                                                    row["reference"], row["description"])).casefold()))
            self._rows_key = key
        if not any(row["id"] == self._selected for row in self._rows):
            self._selected = next((row["id"] for row in self._rows if row["id"] == service.selection),
                                  self._rows[0]["id"] if self._rows else "")
        selected = next((row for row in self._rows if row["id"] == self._selected), None)
        self._render_catalog_columns(ctx, service, self._rows, selected)

    def _render_toolbar(self, ctx):
        ctx.set_next_item_width(metric(ctx, 180.0))
        self._scope = ctx.combo("##theme_scope", self._scope,
                                [t("themes.scope.all"), t("themes.scope.builtin"), t("themes.scope.plugins")])
        ctx.same_line()
        if ctx.button(t("themes.manage_plugins") + "##theme_manage_plugins"):
            self._open_plugin_manager()
        ctx.same_line()
        ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
        ctx.label(t("themes.immediate"))
        ctx.pop_style_color()
        ctx.dummy(0.0, metric(ctx, Theme.INSPECTOR_SECTION_GAP))
        ctx.set_next_item_width(-1.0)
        self._search = ctx.input_text_with_hint("##theme_search", t("themes.search"), self._search, 512)

    def _render_catalog_list(self, ctx, service, rows):
        ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
        ctx.label(t("themes.count").format(count=len(rows)))
        ctx.pop_style_color()
        ctx.separator()
        ctx.dummy(0.0, metric(ctx, Theme.INSPECTOR_SECTION_GAP))
        if ctx.begin_child("##theme_rows", 0.0, 0.0, False):
            group = None
            active = active_editor_theme()
            for row in rows:
                category = "builtin" if row["builtin"] else "plugins"
                if category != group:
                    if group is not None:
                        ctx.dummy(0.0, metric(ctx, Theme.INSPECTOR_SECTION_GAP))
                    ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
                    ctx.label(t("themes.scope." + category))
                    ctx.pop_style_color()
                    ctx.separator()
                    group = category
                status = t("themes.status.current") if row["id"] == active else (
                    t("themes.status.available") if row["available"] else t("themes.status.unavailable"))
                color = Theme.PREFAB_TEXT if row["id"] == active else (
                    Theme.TEXT_DIM if row["available"] else Theme.ERROR_TEXT)
                padding = metric(ctx, 10.0)
                width = max(0.0, ctx.get_content_region_avail_width() - ctx.calc_text_width(status) - padding * 3)
                label = self._fit_text(ctx, row["name"], width)
                if ctx.selectable_row(f"{label}###theme_{row['id']}", row["id"] == self._selected,
                                      status, color, metric(ctx, 28.0), padding):
                    self._selected = row["id"]
                    if row["available"]:
                        self._apply(service, row["id"])
            if not rows:
                ctx.text_wrapped(t("themes.empty"))
        ctx.end_child()

    def _render_catalog_details(self, ctx, service, row):
        if row is None:
            ctx.text_wrapped(t("themes.select"))
            return
        gap = metric(ctx, Theme.INSPECTOR_SECTION_GAP)
        line_height = ctx.calc_text_size("M")[1] + metric(
            ctx, 2 * Theme.GLOBAL_FRAME_PAD[1] + Theme.TOOLBAR_ITEM_SPC[1])
        summary_h = line_height * 3 + metric(ctx, 2 * Theme.PROJECT_PANEL_PAD[1])
        footer_h = metric(ctx, 46.0)
        start_y, height = ctx.get_cursor_pos_y(), ctx.get_content_region_avail_height()
        footer_y = start_y + max(0.0, height - footer_h)
        ctx.push_style_color(ImGuiCol.ChildBg, *Theme.FRAME_BG)
        if ctx.begin_child("##theme_summary", 0.0, summary_h, True, Theme.WINDOW_FLAGS_NO_SCROLL):
            ctx.label(row["name"])
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
            ctx.label(row["author"] or (t("themes.engine_author") if row["builtin"] else "—"))
            ctx.label(t("themes.source.builtin") if row["builtin"] else row["reference"] + "  " + row["version"])
            ctx.pop_style_color()
        ctx.end_child()
        ctx.pop_style_color()
        ctx.set_cursor_pos_y(start_y + summary_h + gap)
        if ctx.begin_child("##theme_detail_content", 0.0,
                           max(1.0, footer_y - start_y - summary_h - gap * 2), True):
            ctx.text_wrapped(row["description"] or t("themes.no_description"))
            ctx.dummy(0.0, gap)
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
            ctx.label(t("themes.identity"))
            ctx.pop_style_color()
            ctx.text_wrapped(row["id"])
            ctx.dummy(0.0, gap)
            if row["error"]:
                ctx.push_style_color(ImGuiCol.Text, *Theme.ERROR_TEXT)
                ctx.text_wrapped(row["error"])
                ctx.pop_style_color()
            if row["path"]:
                ctx.dummy(0.0, gap)
                ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
                ctx.label(t("themes.source.file"))
                ctx.text_wrapped(row["path"])
                ctx.pop_style_color()
            ctx.dummy(0.0, gap)
            ctx.text_wrapped(t("themes.authoring_hint"))
            ctx.dummy(0.0, gap)
            ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DIM)
            ctx.text_wrapped(t("themes.editor_only"))
            ctx.pop_style_color()
        ctx.end_child()
        ctx.set_cursor_pos_y(footer_y)
        self._render_footer(ctx, service, row, footer_h)

    def _render_footer(self, ctx, service, row, height):
        ctx.push_style_color(ImGuiCol.ChildBg, *Theme.FRAME_BG)
        if ctx.begin_child("##theme_detail_footer", 0.0, height, True, Theme.WINDOW_FLAGS_NO_SCROLL):
            if row["reference"]:
                if ctx.button(t("themes.manage_plugin") + "##theme_plugin"):
                    self._open_plugin_manager(row["reference"])
                ctx.same_line()
            if row["path"]:
                if ctx.button(t("plugins.open_location") + "##theme_location"):
                    from .project_utils import reveal_in_file_explorer
                    reveal_in_file_explorer(row["path"])
                ctx.same_line()
            current = row["id"] == active_editor_theme()
            ctx.begin_disabled(not row["available"] or current)
            if ctx.button(t("themes.status.current") if current else t("themes.apply"), width=metric(ctx, 104.0)):
                self._apply(service, row["id"])
            ctx.end_disabled()
        ctx.end_child()
        ctx.pop_style_color()

    def _apply(self, service, name):
        if self._requested == name:
            return
        self._requested = name
        from infernux.engine.runtime_event_queue import enqueue
        def apply():
            try:
                service.select(name)
            finally:
                self._requested = ""
        enqueue(apply)

    def _open_plugin_manager(self, reference=""):
        panel = self.services.window_manager.open_window_from_user("plugins")
        if reference:
            panel.select_reference(reference)
        else:
            panel.select_category("theme")
