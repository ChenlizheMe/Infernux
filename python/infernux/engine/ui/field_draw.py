"""``draw_field`` — the one row renderer for a serialized field.

Row layout: ``label column │ value │ trailing slot``. The trailing slot holds
the lock glyph for readonly fields (units/pickers go there later). Spacing and
``header`` are drawn before the row (space first, then header); ``info_text``
is always drawn *below* the value row.

Readonly fields never reach an editable widget: they are drawn as a dashed,
transparent box with disabled text plus the lock glyph. Double-click copies
the displayed value.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from infernux.components.fields import FieldMetadata, FieldType
from infernux.engine.i18n import t
from infernux.lib import InxGUIContext

from ._inspector_references import (
    _get_asset_ref_config,
    _get_reference_display_name,
    _render_asset_reference_field,
    _render_component_ref_inline,
    _render_gameobject_ref_inline,
)
from .dpi import editor_dpi_scale
from .igui import IGUI
from .inspector_utils import (
    DRAG_SPEED_DEFAULT,
    DRAG_SPEED_INT,
    _render_animation_curve_sf,
    _render_color_sf,
    _render_enum_sf,
    _render_gradient_sf,
    field_label,
    find_enum_index,
    float_close,
    get_enum_members,
    has_field_changed,
    record_inspector_component_item,
    render_info_text,
    resolve_enum_display_labels,
    serialized_field_label,
)

_UNBOUNDED_FLOAT = 0.0
_UNBOUNDED_INT = 0
_VEC_ARITY = {FieldType.VEC2: 2, FieldType.VEC3: 3, FieldType.VEC4: 4}
_VEC_ATTRS = ("x", "y", "z", "w")
_OBJECT_REF_TYPES = (FieldType.GAME_OBJECT, FieldType.COMPONENT)


def is_reference_field(field_type) -> bool:
    return (
        field_type in _OBJECT_REF_TYPES
        or field_type == FieldType.ASSET
        or field_type in _get_asset_ref_config()
    )


def draw_field(
    ctx: InxGUIContext,
    metadata: FieldMetadata,
    value: Any,
    *,
    label_width: float,
    on_change: Callable[[Any], None],
    component: Optional[Any] = None,
    widget_id: str | None = None,
    display_name: str | None = None,
    show_metadata: bool = True,
    show_header: bool = True,
) -> bool:
    """Draw one serialized field row and return True when *on_change* fired.

    Value fields call ``on_change(new_value)`` once per edit. Reference fields
    (GameObject / Component / asset refs) commit through the owning
    *component*'s reference transaction instead, so *component* is required
    for them and ``on_change`` is not invoked.
    """
    ft = metadata.field_type
    label = display_name if display_name is not None else serialized_field_label(metadata.name, metadata)
    wid = widget_id or f"##{metadata.name}"
    reference = is_reference_field(ft)
    if ft in _OBJECT_REF_TYPES and component is None and not metadata.readonly:
        raise ValueError(
            f"Reference field {metadata.name!r} requires its owning component"
        )

    if show_metadata:
        if metadata.space and metadata.space > 0:
            ctx.dummy(0, metadata.space * editor_dpi_scale(ctx))
        if show_header and metadata.header:
            from .inspector_utils import render_section_kicker
            render_section_kicker(ctx, t(metadata.header))

    changed = False
    ctx.begin_group()
    try:
        if reference and component is None and not metadata.readonly:
            from ._inspector_list_field import _render_serializable_asset_reference
            new_value = _render_serializable_asset_reference(ctx, wid, metadata.name, metadata, value, label_width)
            if has_field_changed(ft, value, new_value):
                on_change(new_value)
                changed = True
        elif reference:
            _draw_reference(ctx, metadata, value, label, label_width, component)
        elif metadata.readonly:
            if label:
                field_label(ctx, label, label_width)
            _draw_readonly_value(ctx, wid, format_field_value(metadata, value))
        else:
            new_value = _draw_value_widget(ctx, metadata, value, label, label_width, wid)
            if has_field_changed(ft, value, new_value):
                on_change(new_value)
                changed = True
    finally:
        ctx.end_group()

    if metadata.tooltip and ctx.is_item_hovered():
        ctx.set_tooltip(t(metadata.tooltip))
    if component is not None and not reference:
        record_inspector_component_item(
            ctx, component, metadata.name, "inspector_field", label,
            enabled=not metadata.readonly,
        )
    if show_metadata and metadata.info_text:
        render_info_text(ctx, t(metadata.info_text), label_width)
    return changed


# ── Value widgets ─────────────────────────────────────────────────────────

def draw_field_value(ctx, widget_id, label, metadata, value, label_width):
    """Value-returning adapter for resource and legacy Inspector call sites."""
    result = [value]
    draw_field(
        ctx, metadata, value, widget_id=widget_id, display_name=label,
        label_width=label_width, on_change=lambda updated: result.__setitem__(0, updated),
        show_metadata=False,
    )
    return result[0]


def _draw_value_widget(ctx, metadata, value, label, label_width, wid):
    ft = metadata.field_type
    if ft == FieldType.GRADIENT:
        if label:
            field_label(ctx, label, label_width)
        return _render_gradient_sf(ctx, wid, label, metadata, value)

    if ft in _VEC_ARITY:
        return _draw_vector(ctx, wid, metadata, value, _VEC_ARITY[ft], label, label_width)

    if label:
        field_label(ctx, label, label_width)
    else:
        ctx.set_next_item_width(-1)
    if ft == FieldType.BOOL:
        return bool(ctx.checkbox_field(wid, bool(value)))
    if ft == FieldType.FLOAT:
        return _draw_float(ctx, wid, metadata, value)
    if ft == FieldType.INT:
        return _draw_int(ctx, wid, metadata, value)
    if ft == FieldType.STRING:
        text = str(value) if value else ""
        if metadata.multiline:
            return ctx.input_text_multiline(wid, text, buffer_size=4096, width=-1,
                                            height=4 * ctx.get_text_line_height_with_spacing())
        return ctx.text_input(wid, text, 256)
    if ft == FieldType.ENUM:
        if not _resolve_enum_members(metadata):
            _draw_readonly_value(ctx, wid, format_field_value(metadata, value))
            return value
        return _render_enum_sf(ctx, wid, label, metadata, value, label_width, False)
    if ft == FieldType.COLOR:
        return _render_color_sf(ctx, wid, label, metadata, value, label_width, False)
    if ft == FieldType.ANIMATION_CURVE:
        return _render_animation_curve_sf(ctx, wid, label, metadata, value, label_width, False)
    raise TypeError(f"draw_field cannot render {ft!r} field {metadata.name!r}")


def _draw_float(ctx, wid, metadata, value) -> float:
    current = float(value)
    speed = metadata.drag_speed or DRAG_SPEED_DEFAULT
    with IGUI.mono_font(ctx):
        if metadata.range:
            lo, hi = float(metadata.range[0]), float(metadata.range[1])
            if metadata.slider:
                updated = IGUI.slider_float(ctx, wid, current, lo, hi)
            else:
                updated = ctx.drag_float(wid, current, speed, lo, hi)
        else:
            updated = ctx.drag_float(wid, current, speed, -_UNBOUNDED_FLOAT, _UNBOUNDED_FLOAT)
    # The native control uses float32. Merely drawing a Python float must not
    # publish its round-trip rounding as a user edit.
    return updated if ctx.is_item_edited() else value


def _draw_int(ctx, wid, metadata, value) -> int:
    current = int(value)
    speed = metadata.drag_speed or DRAG_SPEED_INT
    with IGUI.mono_font(ctx):
        if metadata.range:
            lo, hi = int(metadata.range[0]), int(metadata.range[1])
            if metadata.slider:
                return int(IGUI.slider_int(ctx, wid, current, lo, hi))
            return int(ctx.drag_int(wid, current, speed, lo, hi))
        return int(ctx.drag_int(wid, current, speed, -_UNBOUNDED_INT, _UNBOUNDED_INT))


def _vector_components(value, arity: int) -> list[float]:
    if value is None:
        return [0.0] * arity
    if isinstance(value, (tuple, list)):
        return list(map(float, value[:arity]))
    return [float(getattr(value, attr)) for attr in _VEC_ATTRS[:arity]]


def _make_vector(arity: int, components: list[float]):
    if arity == 2:
        from infernux.lib import Vector2
        return Vector2(*components)
    if arity == 3:
        from infernux.lib import Vector3
        return Vector3(*components)
    from infernux.lib import vec4f
    return vec4f(*components)


def _draw_vector(ctx, wid, metadata, value, arity: int, label, label_width):
    """N equal-width drag boxes; no axis tags or axis colours."""
    current = _vector_components(value, arity)
    speed = metadata.drag_speed or DRAG_SPEED_DEFAULT
    ctx.push_id_str(wid)
    try:
        edited = getattr(ctx, f"vector{arity}")(label or wid, *current, speed, label_width)
    finally:
        ctx.pop_id()
    if all(float_close(a, b) for a, b in zip(edited, current)):
        return value
    return _make_vector(arity, edited)


def _resolve_enum_members(metadata) -> list:
    enum_cls = metadata.enum_type
    if isinstance(enum_cls, str):
        import infernux.lib as _lib
        enum_cls = getattr(_lib, enum_cls, None)
    return get_enum_members(enum_cls)


# ── Reference fields ──────────────────────────────────────────────────────

def _draw_reference(ctx, metadata, value, label, label_width, component):
    ft = metadata.field_type
    name = metadata.name
    if metadata.readonly:
        field_label(ctx, label, label_width)
        _draw_readonly_value(ctx, f"##{name}", _get_reference_display_name(ft, value))
        return
    if ft == FieldType.COMPONENT:
        _render_component_ref_inline(ctx, component, name, metadata, label_width)
    elif ft == FieldType.GAME_OBJECT:
        _render_gameobject_ref_inline(ctx, component, name, metadata, value, label_width)
    else:
        _render_asset_reference_field(ctx, component, name, metadata, value, ft, label_width)


# ── Readonly presentation ─────────────────────────────────────────────────

def format_field_value(metadata, value) -> str:
    """Display text for a field value in readonly / non-editable rows."""
    ft = metadata.field_type
    if value is None:
        return t("igui.none")
    if is_reference_field(ft):
        return _get_reference_display_name(ft, value)
    if ft == FieldType.FLOAT:
        return _format_float(float(value))
    if ft in _VEC_ARITY:
        return "  ".join(_format_float(c) for c in _vector_components(value, _VEC_ARITY[ft]))
    if ft == FieldType.COLOR:
        rgba = [max(0.0, min(1.0, float(value[i]))) for i in range(4)]
        return "#" + "".join(f"{round(c * 255):02X}" for c in rgba)
    if ft == FieldType.ENUM:
        members = _resolve_enum_members(metadata)
        if members:
            labels = resolve_enum_display_labels(metadata, members)
            return labels[find_enum_index(members, value)]
        return str(getattr(value, "name", value))
    if ft in (FieldType.ANIMATION_CURVE, FieldType.GRADIENT):
        return type(value).__name__
    return str(value)


def _format_float(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _draw_readonly_value(ctx, wid: str, text: str) -> None:
    """The same native readonly control is used by Python and batch fields."""
    ctx.readonly_field(f"{wid}_readonly", text)
