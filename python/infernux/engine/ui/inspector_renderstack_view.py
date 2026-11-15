"""RenderStack presentation: pipeline controls, sources and flat effect cards."""

from __future__ import annotations

import copy
import json
import os

from infernux.engine.i18n import t
from .dpi import editor_dpi_scale
from .igui import IGUI
from .inspector_declarative import (
    InspectorList, InspectorMessages, _render_messages, _render_serialized_target,
)
from .inspector_utils import field_label, render_compact_section_header, render_section_kicker
from .theme import Theme

_SLOT_DRAG = "INX_RENDERSTACK_SLOT"


def render_renderstack_inspector(ctx, stack, model):
    """Consume the existing data model and its journal-backed edit callbacks."""
    pipeline_controls = model.sections[0].controls
    choice, parameters = pipeline_controls[:2]
    field_label(ctx, choice.label, ctx.inspector_label_width())
    current = choice.current_index()
    selected = ctx.combo(f"##{model.key}_pipeline", current, list(choice.options()))
    if selected != current:
        choice.on_change(selected)
        return True

    if render_compact_section_header(
        ctx, t("renderstack.pipeline_settings"), default_open=False,
        header_id=f"{model.key}.pipeline_parameters", level="secondary",
    ):
        _render_serialized_target(ctx, parameters)
    for control in pipeline_controls[2:]:
        _render_messages(ctx, control, model.key)

    controls = model.sections[1].controls
    stages = [(control, list(control.value())) for control in controls if isinstance(control, InspectorList)]
    render_section_kicker(ctx, t("renderstack.effect_sequence"))
    for index, (control, items) in enumerate(stages):
        ctx.push_id_str(control.key)
        try:
            _stage_title(ctx, control, items, index)
            if items:
                _stage_items(ctx, control, items)
        finally:
            ctx.pop_id()
    for control in controls:
        if isinstance(control, InspectorMessages):
            _render_messages(ctx, control, model.key)
    return False


def _stage_title(ctx, control, items, index):
    dpi = editor_dpi_scale(ctx)
    ctx.set_next_item_allow_overlap()
    ctx.invisible_button("##stage", ctx.get_content_region_avail_width(), 28 * dpi)
    left, top = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
    right, bottom = ctx.get_item_rect_max_x(), ctx.get_item_rect_max_y()
    saved_x, saved_y = ctx.get_cursor_pos_x(), ctx.get_cursor_pos_y()
    with IGUI.mono_font(ctx):
        ctx.draw_text_aligned(left, top, left + 24 * dpi, bottom, f"{index + 1:02d}", *Theme.TEXT_DISABLED,
                              0, 0.5, Theme.EDITOR_SMALL_FONT_SIZE * dpi, True)
    ctx.draw_text_aligned(left + 28 * dpi, top, right - 30 * dpi, bottom, control.label, *Theme.TEXT_DIM,
                          0, 0.5, 0, True)
    ctx.record_semantic_item("effect_stage", control.label, True, f"renderstack.{control.key}",
                              numeric_value=float(index))
    if control.accept_drop:
        def append(payload):
            updated = items + [control.drop_factory(payload)]
            control.on_change(control.target, control.field_name, items, updated)
        IGUI.drop_target(ctx, control.accept_drop, append)
    ctx.set_cursor_screen_pos(right - 22 * dpi, top + 3 * dpi)
    if IGUI.icon_button(ctx, "##add_effect", "plus", tooltip=t("renderstack.add_effect"),
                        kind="ghost", width=22, height=22):
        ctx.open_popup("##add_effect_popup")
    ctx.record_semantic_item("button", t("renderstack.add_effect"), True,
                              f"renderstack.{control.key}.add")
    ctx.set_cursor_pos_x(saved_x)
    ctx.set_cursor_pos_y(saved_y)
    if ctx.begin_popup("##add_effect_popup"):
        try:
            ctx.label(t("renderstack.add_effect"))
            def add(path):
                control.on_change(control.target, control.field_name, items, items + [control.drop_factory(path)])
                ctx.close_current_popup()
            IGUI.asset_reference_field(
                ctx, "add_effect_asset", t("igui.none"), "RenderEffect", asset_type="RenderEffect",
                on_assign=add, has_value=False, fixed_width=280 * dpi,
            )
        finally:
            ctx.end_popup()


def _stage_items(ctx, control, items):
    def insert_at(target):
        def move(payload):
            owner, field, source = json.loads(payload)
            if owner != id(control.target) or field != control.field_name:
                return
            updated = list(items)
            item = updated.pop(source)
            updated.insert(target - int(source < target), item)
            if updated != items:
                control.on_change(control.target, control.field_name, items, updated)
        return move

    IGUI.reorder_separator(ctx, "##insert_0", _SLOT_DRAG, insert_at(0))
    for index, slot in enumerate(items):
        ctx.push_id_str(slot.slot_id)
        try:
            if _source_row(ctx, control, items, slot, index):
                return
            if slot.effect_ref:
                if not slot.enabled:
                    ctx.begin_disabled(True)
                try:
                    _effect_cards(ctx, slot, f"{control.key}.{slot.slot_id}")
                finally:
                    if not slot.enabled:
                        ctx.end_disabled()
            else:
                ctx.text_wrapped(t("renderstack.choose_effect"))
        finally:
            ctx.pop_id()
        IGUI.reorder_separator(ctx, f"##insert_{index + 1}", _SLOT_DRAG, insert_at(index + 1))


def _source_row(ctx, control, items, slot, index):
    dpi = editor_dpi_scale(ctx)
    width = ctx.get_content_region_avail_width()
    changed = False

    def commit(updated_slot):
        nonlocal changed
        updated = list(items)
        updated[index] = updated_slot
        control.on_change(control.target, control.field_name, items, updated)
        changed = True

    def replace(path):
        edited = copy.deepcopy(slot)
        edited.effect = control.drop_factory(path).effect_ref
        commit(edited)

    IGUI.icon_button(ctx, "##source_grip", "grip", tooltip=t("igui.reorder"), kind="ghost", width=20)
    if ctx.begin_drag_drop_source(0):
        try:
            ctx.set_drag_drop_payload_str(_SLOT_DRAG, json.dumps((id(control.target), control.field_name, index)))
            ctx.label(control.item_label(slot, index))
        finally:
            ctx.end_drag_drop_source()
    ctx.same_line(0, 4 * dpi)
    path = str(slot.effect_ref.path_hint or "")
    display = os.path.splitext(os.path.basename(path))[0] if path else t("renderstack.choose_effect")
    IGUI.asset_reference_field(
        ctx, "source", display, "RenderEffect", asset_type="RenderEffect", on_assign=replace,
        reference_value=slot.effect_ref, has_value=bool(slot.effect_ref), ping_path=path or None,
        fixed_width=max(60 * dpi, width - 76 * dpi),
        semantic_id=f"renderstack.slot.{slot.slot_id}.source",
    )
    ctx.same_line(0, 8 * dpi)
    enabled = ctx.checkbox_field("##source_enabled", slot.enabled)
    ctx.record_semantic_item("checkbox", t("plugins.enabled"), True,
                              f"renderstack.slot.{slot.slot_id}.enabled", bool_value=enabled)
    if enabled != slot.enabled:
        edited = copy.deepcopy(slot)
        edited.enabled = enabled
        commit(edited)
    ctx.same_line(0, 8 * dpi)
    if IGUI.icon_button(ctx, "##source_more", "more", tooltip=t("renderstack.source_options"), kind="ghost", width=20):
        ctx.open_popup("##source_actions")
    if ctx.begin_popup("##source_actions"):
        try:
            if ctx.menu_item(t("renderstack.remove")):
                control.on_change(control.target, control.field_name, items, items[:index] + items[index + 1:])
                changed = True
        finally:
            ctx.end_popup()
    ctx.dummy(0, 4 * dpi)
    return changed


def _effect_cards(ctx, slot, widget_prefix):
    from infernux.renderstack.render_effect_compiler import expand_render_effect_reference
    from .inspector_renderstack import _resolve_effect_document_controller, _resolve_effect_group_document_controller
    from .render_effect_inspector import render_render_effect_card

    for index, effect in enumerate(expand_render_effect_reference(slot.effect_ref)):
        group = _resolve_effect_group_document_controller(effect)
        if group is not None:
            effect.bind_group_document_controller(group)
        render_render_effect_card(
            ctx, effect, widget_prefix=f"{widget_prefix}.{index}",
            resource_controller=group or _resolve_effect_document_controller(effect),
        )
