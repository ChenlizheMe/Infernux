"""Shared spacing and list/detail layout for editor catalog windows."""

from .dpi import scaled_editor_metric
from .editor_panel import EditorPanel
from .theme import ImGuiStyleVar, Theme


def metric(ctx, value):
    return scaled_editor_metric(ctx, value)


class CatalogBrowserPanel(EditorPanel):
    def __init__(self, title, window_id):
        super().__init__(title, window_id)
        self._catalog_id = window_id

    def _initial_size(self):
        return 1180.0, 740.0

    def _push_window_style(self, ctx):
        ctx.push_style_var_vec2(ImGuiStyleVar.WindowPadding,
                               metric(ctx, Theme.PROJECT_PANEL_PAD[0]), metric(ctx, Theme.PROJECT_PANEL_PAD[1]))
        ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing,
                               metric(ctx, Theme.TOOLBAR_ITEM_SPC[0]), metric(ctx, Theme.TOOLBAR_ITEM_SPC[1]))

    def _pop_window_style(self, ctx):
        ctx.pop_style_var(2)

    @staticmethod
    def _fit_text(ctx, value, max_width):
        if ctx.calc_text_width(value) <= max_width:
            return value
        low, high = 0, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if ctx.calc_text_width(value[:middle] + "…") <= max_width:
                low = middle
            else:
                high = middle - 1
        return value[:low] + "…"

    def _render_catalog_columns(self, ctx, model, rows, selected):
        ctx.dummy(0.0, metric(ctx, Theme.INSPECTOR_SECTION_GAP))
        ctx.separator()
        ctx.dummy(0.0, metric(ctx, Theme.INSPECTOR_SECTION_GAP))
        width = max(1.0, ctx.get_content_region_avail_width())
        height = max(1.0, ctx.get_content_region_avail_height())
        gap = metric(ctx, Theme.INSPECTOR_TITLE_GAP)
        left = min(metric(ctx, 360.0), max(metric(ctx, 220.0), width * 0.31), width * 0.44)
        right = max(1.0, width - left - gap)
        if ctx.begin_child(f"##{self._catalog_id}_list", left, height, True, Theme.WINDOW_FLAGS_NO_SCROLL):
            self._render_catalog_list(ctx, model, rows)
        ctx.end_child()
        ctx.same_line(0.0, gap)
        if ctx.begin_child(f"##{self._catalog_id}_details", right, height, False, Theme.WINDOW_FLAGS_NO_SCROLL):
            self._render_catalog_details(ctx, model, selected)
        ctx.end_child()
