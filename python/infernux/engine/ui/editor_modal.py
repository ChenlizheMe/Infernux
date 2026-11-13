"""Shared presentation primitives for Editor-owned modal dialogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .dpi import editor_dpi_scale
from .theme import Theme


@dataclass(frozen=True)
class EditorModalAction:
    """One dialog command. *kind* is a :meth:`Theme.push_button_kind_style`
    kind: ``danger`` actions sit at the left edge, the rest are right-aligned
    in the given order (the ``primary`` action belongs last)."""

    label: str
    semantic_id: str
    callback: Callable[[], None]
    enabled: bool = True
    kind: str = "secondary"


DEFAULT_MODAL_WIDTH = 560.0
DEFAULT_MODAL_HEIGHT = 220.0
ACTION_MIN_WIDTH = 80.0
ACTION_HEIGHT = 28.0
ACTION_PAD_X = 16.0
ACTION_GAP = 8.0


def begin_editor_modal(
    ctx,
    *,
    popup_id: str,
    title: str,
    semantic_id: str,
    request_open: bool = False,
    width: float = DEFAULT_MODAL_WIDTH,
    height: float = DEFAULT_MODAL_HEIGHT,
) -> bool:
    """Open and begin a consistently centred Editor modal."""
    dpi = editor_dpi_scale(ctx)
    window_id = popup_id.rsplit("###", 1)[-1]
    window_label = f"{title}###{window_id}"
    if request_open:
        ctx.open_popup(window_label)

    if request_open:
        viewport_x, viewport_y, viewport_w, viewport_h = ctx.get_main_viewport_bounds()
        ctx.set_next_window_pos(
            viewport_x + viewport_w * 0.5,
            viewport_y + viewport_h * 0.5,
            Theme.COND_ALWAYS,
            0.5,
            0.5,
        )
        ctx.set_next_window_size(float(width) * dpi, float(height) * dpi, Theme.COND_ALWAYS)
    if not ctx.begin_popup_modal(window_label, 0):
        return False
    ctx.record_semantic_window("modal", title, semantic_id)
    return True


def render_editor_modal_actions(
    ctx,
    actions: Iterable[EditorModalAction],
    *,
    semantic_prefix: str,
) -> None:
    """Render the standard bottom-anchored command-button row."""
    dpi = editor_dpi_scale(ctx)
    action_list = list(actions)
    button_h = ACTION_HEIGHT * dpi
    gap = ACTION_GAP * dpi

    # Anchor actual button bounds. Spacing()/Separator()/Spacing() used to
    # add inherited ItemSpacing three times, overflowing the reserved footer.
    separator_gap = 8.0 * dpi
    action_block_h = button_h + separator_gap + 1.0 * dpi
    remaining = float(ctx.get_content_region_avail_height())
    footer_y = float(ctx.get_cursor_pos_y()) + max(0.0, remaining - action_block_h)
    ctx.set_cursor_pos_y(footer_y)
    ctx.separator()
    ctx.set_cursor_pos_y(footer_y + separator_gap + 1.0 * dpi)

    leading = [action for action in action_list if action.kind == "danger"]
    trailing = [action for action in action_list if action.kind != "danger"]
    widths = {
        id(action): max(ACTION_MIN_WIDTH * dpi, float(ctx.calc_text_width(action.label)) + 2.0 * ACTION_PAD_X * dpi)
        for action in action_list
    }
    row_x = float(ctx.get_cursor_pos_x())
    available = float(ctx.get_content_region_avail_width())

    for index, action in enumerate(leading):
        if index:
            ctx.same_line(0.0, gap)
        _render_action(ctx, action, widths[id(action)], button_h, semantic_prefix)

    trailing_w = sum(widths[id(action)] for action in trailing) + gap * max(0, len(trailing) - 1)
    for index, action in enumerate(trailing):
        if index:
            ctx.same_line(0.0, gap)
        else:
            if leading:
                ctx.same_line(0.0, gap)
            ctx.set_cursor_pos_x(max(float(ctx.get_cursor_pos_x()), row_x + available - trailing_w))
        _render_action(ctx, action, widths[id(action)], button_h, semantic_prefix)


def _render_action(ctx, action: EditorModalAction, width: float, height: float, semantic_prefix: str) -> None:
    if not action.enabled:
        ctx.begin_disabled(True)
    pushed = Theme.push_button_kind_style(ctx, action.kind)
    suffix = action.semantic_id.replace(".", "_")
    ctx.button(f"{action.label}##{suffix}", action.callback, width=width, height=height)
    Theme.pop_button_kind_style(ctx, pushed)
    ctx.record_semantic_item(
        "button", action.label, action.enabled,
        f"{semantic_prefix}.{action.semantic_id.rsplit('.', 1)[-1]}",
    )
    if not action.enabled:
        ctx.end_disabled()


def end_editor_modal(ctx) -> None:
    ctx.end_popup()
