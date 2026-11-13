"""Dockable read-only view of the global editor action journal."""

from __future__ import annotations

from infernux.engine.i18n import t
from infernux.engine.interaction import EditorInteractionCore, PanelInteractionDescriptor

from .editor_panel import EditorPanel
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol
from .igui import IGUI
from .editor_icons import EditorIcons


@editor_panel(
    "History",
    type_id="history",
    title_key="panel.history",
    menu_path="Window",
    interaction=PanelInteractionDescriptor(),
)
class HistoryPanel(EditorPanel):
    """Search journal evidence without creating a second history cursor."""

    def __init__(self) -> None:
        super().__init__(t("panel.history"), "history")

    def _initial_size(self) -> tuple[float, float]:
        return 560.0, 420.0

    def on_render_content(self, ctx) -> None:
        core = EditorInteractionCore.instance()
        if core is None:
            ctx.text_wrapped(t("history.unavailable"))
            return

        model = core.history
        query = ctx.text_input(
            t("history.search") + "##history_search",
            model.query,
            256,
        )
        model.set_query(query)
        ctx.record_semantic_item(
            "text_input",
            t("history.search"),
            True,
            "history.search",
        )

        snapshot = model.snapshot
        ctx.label(
            t("history.summary").format(
                applied=snapshot.cursor,
                total=snapshot.total,
                visible=len(snapshot.entries),
            )
        )
        ctx.separator()

        if not snapshot.entries:
            ctx.text_wrapped(t("history.empty"))
            return

        visible = ctx.begin_child("##history_entries", 0.0, 0.0, False)
        if visible:
            for row in snapshot.entries:
                state = (
                    t("history.state.applied")
                    if row.state == "applied"
                    else t("history.state.redo")
                )
                current = row.is_next_undo or row.is_next_redo
                dpi = ctx.get_dpi_scale()
                width = ctx.get_content_region_avail_width()
                if row.state != "applied":
                    ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DISABLED)
                ctx.selectable(f"##history_{row.operation_id}", current, width=0, height=32 * dpi)
                x, y = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
                icon_name = "plus" if ".create" in row.command_id else "close" if ".delete" in row.command_id else "back" if row.state != "applied" else "key"
                icon = EditorIcons.get_cached(icon_name)
                if icon:
                    ctx.draw_image_rect(icon, x + 6 * dpi, y + 9 * dpi, x + 20 * dpi, y + 23 * dpi)
                with IGUI.mono_font(ctx):
                    ctx.draw_text(x + 30 * dpi, y + 9 * dpi, f"{row.sequence:03d}", *Theme.TEXT_DISABLED)
                color = Theme.TEXT if row.state == "applied" else Theme.TEXT_DISABLED
                ctx.draw_text_aligned(x + 72 * dpi, y, x + width - 8 * dpi, y + 32 * dpi,
                                      row.description, *color, 0, 0.5, 0, True)
                if row.state != "applied":
                    ctx.pop_style_color(1)
                ctx.record_semantic_item(
                    "history_entry",
                    row.description,
                    True,
                    f"history.entry.{row.operation_id}",
                )
                if ctx.is_item_hovered():
                    details = [
                        state,
                        f"{t('history.command')}: {row.command_id}",
                        f"{t('history.target')}: {row.target or '-'}",
                        f"{t('history.document')}: {row.document or '-'}",
                        f"{t('history.origin')}: {row.origin}",
                    ]
                    ctx.set_tooltip("\n".join(details))
        ctx.end_child()


__all__ = ["HistoryPanel"]
