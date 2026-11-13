"""Shared Inspector controls for material-like RenderEffect assets."""

from __future__ import annotations

import copy

from infernux.components.fields import get_serialized_fields
from infernux.engine.i18n import t
from infernux.engine.ui.inspector_utils import (
    field_label,
    has_field_changed,
    max_label_w,
    pretty_field_name,
    render_serialized_field,
    serialized_field_label,
)
from infernux.engine.ui.theme import Theme

_effect_foldouts: dict[str, bool] = {}


def render_render_effect_header(ctx, effect, widget_id, *, label=None):
    """Resource title, deliberately distinct from a scene component header."""
    from .dpi import editor_dpi_scale

    if label is None:
        builtin = effect.feature_type.startswith("infernux.post.")
        label = t(f"effect.{effect.feature_type.rsplit('.', 1)[-1]}") if builtin else effect.name
    dpi = editor_dpi_scale(ctx)
    opened = _effect_foldouts.get(widget_id, True)
    if ctx.invisible_button(f"##effect_resource_{widget_id}", ctx.get_content_region_avail_width(), 26.0 * dpi):
        opened = not opened
        _effect_foldouts[widget_id] = opened
    left, top = ctx.get_item_rect_min_x(), ctx.get_item_rect_min_y()
    right, bottom = ctx.get_item_rect_max_x(), ctx.get_item_rect_max_y()
    icon_key = ("effect_" + effect.feature_type.rsplit('.', 1)[-1]
                if effect.feature_type.startswith("infernux.post.") else "render_effect")
    icon = ctx.editor_icon(icon_key)
    if icon:
        ctx.draw_image_rect(icon, left, top + 5 * dpi, left + 16 * dpi, top + 21 * dpi)
    ctx.draw_text_aligned(left + 24 * dpi, top, right - 30 * dpi, bottom, label, *Theme.TEXT, 0, 0.5, 0, True)
    chevron = ctx.editor_icon("glyphs/chev_d" if opened else "glyphs/chev_r")
    if chevron:
        ctx.draw_image_rect(chevron, right - 16 * dpi, top + 7 * dpi, right - 4 * dpi, top + 19 * dpi)
    ctx.record_semantic_item("effect_resource", label, True, f"renderstack.effect.{widget_id}.fold", bool_value=opened)
    if ctx.is_item_hovered():
        ctx.set_tooltip(effect.feature_type)
    return opened


def render_render_effect_card(ctx, effect, *, widget_prefix, resource_controller):
    """A padded asset parameter surface; no component chrome or nested window."""
    from .dpi import editor_dpi_scale
    from .theme import ImGuiCol, ImGuiStyleVar

    dpi = editor_dpi_scale(ctx)
    value_column = ctx.inspector_label_width()
    ctx.push_style_var_vec2(ImGuiStyleVar.CellPadding, 10 * dpi, 6 * dpi)
    ctx.push_style_color(ImGuiCol.TableRowBg, *Theme.ROLE_BG_SURFACE)
    ctx.push_style_color(ImGuiCol.TableRowBgAlt, *Theme.ROLE_BG_SURFACE)
    try:
        # Borderless tables default to NoPadOuterX. PadOuterX is required for
        # CellPadding to inset both outside edges of this single-column card.
        if ctx.begin_table(f"##effect_card_{widget_prefix}", 1, (1 << 6) | (1 << 21), 0.0):
            try:
                ctx.table_next_row()
                ctx.table_set_column_index(0)
                if render_render_effect_header(ctx, effect, widget_prefix):
                    ctx.dummy(0, 4 * dpi)
                    render_render_effect_parameters(
                        ctx, effect, widget_prefix=widget_prefix,
                        resource_controller=resource_controller, label_width=value_column,
                    )
                    ctx.dummy(0, 4 * dpi)
            finally:
                ctx.end_table()
    finally:
        ctx.pop_style_color(2)
        ctx.pop_style_var(1)
    ctx.dummy(0, 6 * dpi)


def _inspector_parameter_instance(effect, feature):
    """Return a typed parameter view rebuilt only after live asset changes."""
    cache_key = (feature.type_id, id(feature.effect_class), effect.revision)
    cached = getattr(effect, "_inspector_parameter_cache", None)
    if isinstance(cached, tuple) and cached[0] == cache_key:
        return cached[1]
    instance = feature.instantiate(effect)
    effect._inspector_parameter_cache = (cache_key, instance)
    return instance


def render_render_effect_feature(ctx, effect, *, widget_prefix: str = "effect") -> float:
    """Show the feature's source through the shared Inspector reference field."""
    from infernux.renderstack.render_effect_compiler import get_render_effect_feature
    from ._inspector_references import (
        _is_project_asset_path,
        open_asset_reference,
        ping_asset_in_project,
        render_object_field,
    )

    feature = get_render_effect_feature(effect.feature_type)
    fields = get_serialized_fields(feature.effect_class)
    feature_label = t("render_effect.feature")
    labels = [feature_label] + [serialized_field_label(name, metadata) for name, metadata in fields.items() if not metadata.hidden]
    label_width = max_label_w(ctx, labels)
    source_path = feature.source_identity[0]

    def locate_source():
        if _is_project_asset_path(source_path):
            ping_asset_in_project(source_path)
        else:
            open_asset_reference(source_path)

    field_label(ctx, feature_label, label_width)
    render_object_field(
        ctx,
        f"##{widget_prefix}_feature",
        effect.feature_type,
        "Script",
        clickable=bool(source_path),
        has_value=bool(source_path),
        on_open=locate_source if source_path else None,
        semantic_id=f"render_effect.{widget_prefix}.feature",
    )
    if source_path and ctx.is_item_hovered():
        ctx.set_tooltip(source_path)
    return label_width


def apply_render_effect_parameter_edit(
    effect,
    field_name: str,
    value,
    *,
    resource_controller=None,
    origin=None,
) -> bool:
    """Apply one typed parameter edit to the shared asset with Undo support."""
    from infernux.renderstack.render_effect_compiler import get_render_effect_feature

    feature = get_render_effect_feature(effect.feature_type)
    fields = get_serialized_fields(feature.effect_class)
    metadata = fields.get(field_name)
    if metadata is None or metadata.readonly:
        return False

    instance = _inspector_parameter_instance(effect, feature)
    current_value = getattr(instance, field_name, metadata.default)
    if not has_field_changed(metadata.field_type, current_value, value):
        return False
    # The revision-keyed view represents the published resource. Normalize an
    # actual edit separately so rejected/failed submissions cannot change it.
    candidate = feature.instantiate(effect)
    setattr(candidate, field_name, value)

    old_document = effect.to_dict()
    new_document = copy.deepcopy(old_document)
    new_document["parameters"][field_name] = candidate.get_params_dict()[field_name]

    group_edit = getattr(effect, "apply_group_parameter_edit", None)
    if callable(group_edit):
        return bool(
            group_edit(
                field_name,
                new_document["parameters"][field_name],
                view_id="inspector",
                description=f"Set RenderEffect {pretty_field_name(field_name)}",
                origin=origin,
            )
        )

    if resource_controller is not None:
        return bool(
            resource_controller.apply_document(
                new_document,
                view_id="inspector",
                edit_key=field_name,
                description=f"Set RenderEffect {pretty_field_name(field_name)}",
                origin=origin,
            )
        )

    # Inspector mutations must always belong to a formal resource Document.
    # Runtime scripts may still use RenderEffect.set_* directly.
    return False


def render_render_effect_parameters(
    ctx,
    effect,
    *,
    widget_prefix: str = "effect",
    resource_controller=None,
    label_width: float = 0.0,
) -> bool:
    """Render and edit parameters from the feature's serialized schema."""
    from infernux.renderstack.render_effect_compiler import get_render_effect_feature

    feature = get_render_effect_feature(effect.feature_type)
    fields = get_serialized_fields(feature.effect_class)
    if not fields:
        return False
    instance = _inspector_parameter_instance(effect, feature)
    labels = [serialized_field_label(name, metadata) for name, metadata in fields.items()]
    label_width = label_width or max_label_w(ctx, labels)
    changed = False
    current_header = ""
    for field_name, metadata in fields.items():
        if metadata.hidden:
            continue
        if metadata.visible_when is not None and not metadata.visible_when(instance):
            continue
        current_value = getattr(instance, field_name, metadata.default)

        def commit(value, *, _name=field_name):
            nonlocal changed
            changed = apply_render_effect_parameter_edit(
                effect, _name, value, resource_controller=resource_controller,
            ) or changed

        from .field_draw import draw_field
        draw_field(ctx, metadata, current_value, label_width=label_width,
                   widget_id=f"##{widget_prefix}_{field_name}", on_change=commit,
                   show_header=metadata.header != current_header)
        if metadata.header:
            current_header = metadata.header
    return changed
