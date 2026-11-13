"""Top-level modal presentation for the global command palette."""

from __future__ import annotations

from infernux.engine.i18n import t
from infernux.engine.interaction.command_palette import (
    COMMAND_PALETTE_MODAL_ID,
    CommandPaletteService,
)

from .editor_modal import begin_editor_modal, end_editor_modal
from .dpi import scaled_editor_metric
from .theme import Theme, ImGuiStyleVar
from .igui import IGUI
from .editor_icons import EditorIcons
from .inspector_utils import render_section_kicker


class CommandPalettePresenter:
    def __init__(self, service: CommandPaletteService, modal_service) -> None:
        if not isinstance(service, CommandPaletteService):
            raise TypeError("command palette presenter requires CommandPaletteService")
        self._service = service
        modal_service.register(
            COMMAND_PALETTE_MODAL_ID,
            is_active=lambda: self._service.is_active,
            render=self.render,
            cancel=self._service.close,
        )

    def render(self, ctx) -> None:
        request_focus = self._service.request_search_focus
        if not begin_editor_modal(
            ctx,
            popup_id=t("command_palette.title") + "###editor_command_palette",
            title=t("command_palette.title"),
            semantic_id="editor.command_palette",
            request_open=request_focus,
            width=560.0,
            height=520.0,
        ):
            return

        if request_focus:
            ctx.set_keyboard_focus_here()
        dpi = ctx.get_dpi_scale()
        ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, 8 * dpi, 8 * dpi)
        ctx.set_next_item_width(-1.0)
        query = ctx.input_text_with_hint(
            "##command_palette_search", t("command_palette.search"),
            self._service.query,
            256,
        )
        ctx.pop_style_var(1)
        self._service.set_query(query)
        ctx.record_semantic_item(
            "text_input",
            t("command_palette.search"),
            True,
            "command_palette.search",
        )
        ctx.spacing()

        entries = self._service.entries
        if not entries:
            ctx.text_wrapped(t("command_palette.empty"))
        else:
            child_visible = ctx.begin_child(
                "##command_palette_results", 0.0, 0.0, False, padding=True,
            )
            if child_visible:
                previous_category = ""
                for index, entry in enumerate(entries):
                    category = entry.category or t("command_palette.uncategorized")
                    if category != previous_category:
                        render_section_kicker(ctx, category)
                        previous_category = category
                    if not entry.enabled:
                        ctx.begin_disabled(True)
                    width = ctx.get_content_region_avail_width()
                    clicked = ctx.selectable(
                        f"##palette_{entry.command_id}",
                        index == self._service.selected_index,
                        width=0.0,
                        height=scaled_editor_metric(ctx, 30.0),
                    )
                    x, y = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
                    ctx.record_semantic_item(
                        "command",
                        entry.display_name,
                        entry.enabled,
                        f"command_palette.command.{entry.command_id}",
                    )
                    icon = EditorIcons.get_cached("key")
                    if icon:
                        ctx.draw_image_rect(icon, x + 6 * dpi, y + 8 * dpi, x + 20 * dpi, y + 22 * dpi)
                    shortcut_width = ctx.calc_text_width(entry.shortcut) + 16 * dpi if entry.shortcut else 0
                    color = Theme.TEXT if entry.enabled else Theme.TEXT_DISABLED
                    ctx.draw_text_aligned(x + 30 * dpi, y, x + width - shortcut_width, y + 30 * dpi,
                                          entry.display_name, *color, 0, 0.5, 0, True)
                    if entry.shortcut:
                        with IGUI.mono_font(ctx):
                            ctx.draw_text_aligned(x + width - shortcut_width, y, x + width - 6 * dpi, y + 30 * dpi,
                                                  entry.shortcut, *Theme.TEXT_DISABLED, 1, 0.5, 0, True)
                    if not entry.enabled:
                        ctx.end_disabled()
                        if entry.disabled_reason and ctx.is_item_hovered():
                            ctx.set_tooltip(entry.disabled_reason)
                    if clicked:
                        self._service.select(index)
                        self._service.execute_selected()
                        break
            ctx.end_child()

        end_editor_modal(ctx)


__all__ = ["CommandPalettePresenter"]
