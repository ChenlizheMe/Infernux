"""
Unified layout utilities and field renderers for Inspector panel sub-modules.

All inspector code (components, render-stack, materials) **must** use the
functions in this module for a consistent look and feel.  DO NOT duplicate
field-rendering logic in individual inspector modules.

Hierarchy of abstractions
~~~~~~~~~~~~~~~~~~~~~~~~~
1.  **Constants** — drag-speeds, padding, minimum widths.
2.  **Layout helpers** — ``field_label``, ``max_label_w``, ``pretty_field_name``.
3.  **Enum helpers** — robust Python-Enum / pybind11-enum compat.
4.  **Unified field renderers** — ``render_serialized_field`` (FieldType-based),
    ``render_material_property`` (Material JSON type-system).
5.  **Change-detection** — ``has_field_changed``, ``float_close``.
6.  **Misc UI widgets** — ``render_info_text``, ``render_apply_revert``.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
from enum import Flag
from dataclasses import fields as dataclass_fields, is_dataclass, replace
from functools import lru_cache
from typing import Any

from infernux.lib import InxGUIContext
from infernux.engine.i18n import t
from .dpi import editor_dpi_scale
from .theme import Theme, ImGuiCol, ImGuiStyleVar


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

DRAG_SPEED_DEFAULT: float = 0.1

DRAG_SPEED_FINE: float = 0.01

DRAG_SPEED_INT: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Float comparison
# ═══════════════════════════════════════════════════════════════════════════

def float_close(a: float, b: float, rel_tol: float = 1e-5,
                abs_tol: float = 1e-7) -> bool:
    """Return True if *a* and *b* are close enough to be treated as equal.

    Avoids phantom change-detection caused by float32-float64 round-trips
    through JSON & pybind11.
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def preserve_ui_float_precision(candidate, original):
    """Restore Python values when a UI widget only introduced float32 noise."""
    if type(candidate) is float and type(original) is float:
        if math.isclose(candidate, original, rel_tol=6.0e-8, abs_tol=1.0e-8):
            return original
        return candidate
    if (
        type(candidate) is tuple
        and type(original) is tuple
        and len(candidate) == len(original)
    ):
        return tuple(
            preserve_ui_float_precision(value, previous)
            for value, previous in zip(candidate, original)
        )
    if (
        type(candidate) is list
        and type(original) is list
        and len(candidate) == len(original)
    ):
        return [
            preserve_ui_float_precision(value, previous)
            for value, previous in zip(candidate, original)
        ]
    if (
        type(candidate) is dict
        and type(original) is dict
        and set(candidate) == set(original)
    ):
        return {
            key: preserve_ui_float_precision(value, original[key])
            for key, value in candidate.items()
        }
    if (
        type(candidate) is type(original)
        and is_dataclass(candidate)
        and not isinstance(candidate, type)
    ):
        return replace(
            candidate,
            **{
                field.name: preserve_ui_float_precision(
                    getattr(candidate, field.name), getattr(original, field.name)
                )
                for field in dataclass_fields(candidate)
            },
        )
    return candidate


# ═══════════════════════════════════════════════════════════════════════════
#  Naming / formatting
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def format_display_name(name: str, *, title_case: bool = False) -> str:
    """Format identifiers for UI display.

    Rules:
    - one or more leading underscores become a single ``[Inner]`` prefix
    - remaining underscores become spaces
    - optional title-casing only affects lowercase words so CamelCase survives
    """
    if not name:
        return ""

    stripped = name.lstrip("_")
    has_inner_prefix = stripped != name
    body = " ".join(stripped.replace("_", " ").split())

    if title_case:
        words = []
        for word in body.split(" "):
            if word.islower():
                words.append(word.title())
            else:
                words.append(word)
        body = " ".join(words)

    if has_inner_prefix:
        return f"[Inner] {body}" if body else "[Inner]"
    return body


def pretty_field_name(name: str) -> str:
    """Convert identifiers to a readable field label."""
    return format_display_name(name, title_case=True)


def serialized_field_label(name: str, metadata) -> str:
    key = metadata.display_name_key or f"field.{name}"
    label = t(key) if key else ""
    return label if label and label != key else pretty_field_name(name)


def semantic_capture_enabled(ctx: InxGUIContext) -> bool:
    """Return whether this frame is collecting an MCP semantic snapshot.

    Test doubles created before request-only capture do not expose the native
    property, so they retain the historical capture-on behavior.
    """
    return bool(getattr(ctx, "semantic_capture_enabled", True))


def inspector_component_semantic_id(component, field_name: str) -> str:
    """Return the stable semantic identity for an Inspector field.

    ImGui widget IDs are intentionally local and often hidden (``##...``).
    MCP needs an identity that survives relayouts and remains unambiguous when
    different components expose fields with the same display name.
    """
    try:
        game_object = getattr(component, "game_object", None)
        object_id = int(getattr(game_object, "id", 0) or 0)
        component_id = int(getattr(component, "component_id", 0) or 0)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ""
    if object_id <= 0 or component_id <= 0:
        return ""
    suffix = str(field_name or "").strip(".")
    base = f"inspector.object.{object_id}.component.{component_id}"
    return f"{base}.{suffix}" if suffix else base


def record_inspector_component_item(
    ctx: InxGUIContext,
    component,
    field_name: str,
    kind: str,
    label: str,
    *,
    enabled: bool = True,
) -> str:
    """Record the previously rendered widget under the Inspector contract."""
    if not semantic_capture_enabled(ctx):
        return ""
    semantic_id = inspector_component_semantic_id(component, field_name)
    recorder = getattr(ctx, "record_semantic_item", None)
    if semantic_id and callable(recorder):
        recorder(kind, label, bool(enabled), semantic_id)
    return semantic_id


# ═══════════════════════════════════════════════════════════════════════════
#  Label / layout primitives
# ═══════════════════════════════════════════════════════════════════════════



def max_label_w(ctx: InxGUIContext, labels, *, min_width: float = 0.0) -> float:
    """The common DPI-aware label column used by native and Python fields."""
    return max(ctx.inspector_label_width(), min_width)


def field_label(ctx: InxGUIContext, label: str, width: float = 0.0):
    """Use the native label column, including inside grouped fields."""
    ctx.inspector_field_label(label, width)


# ═══════════════════════════════════════════════════════════════════════════
#  Enum helpers (Python Enum + pybind11 enum compat)
# ═══════════════════════════════════════════════════════════════════════════

def get_enum_members(enum_cls):
    """Return member list for a Python Enum **or** pybind11 enum-like type."""
    if enum_cls is None:
        return []
    members_dict = getattr(enum_cls, "__members__", None)
    if isinstance(members_dict, dict):
        return list(members_dict.values())
    try:
        return list(enum_cls)
    except TypeError:
        pass
    return []


def get_enum_member_name(member) -> str:
    """Human-readable name of an enum member."""
    name = getattr(member, "name", None)
    return str(name) if name else str(member)


def resolve_enum_display_labels(metadata, members) -> list[str]:
    """Return localized enum labels for Inspector combo widgets."""
    from infernux.engine.i18n import t

    labels = getattr(metadata, "enum_labels", None)
    if labels and len(labels) == len(members):
        return [t(label) for label in labels]
    return [get_enum_member_name(member) for member in members]


def get_enum_member_value(member):
    """Raw value of an enum member."""
    return member.value if hasattr(member, "value") else member


def find_enum_index(members, current_value) -> int:
    """Find the best matching member index for the given *current_value*."""
    if not members:
        return 0
    # 1. Direct equality
    for idx, member in enumerate(members):
        if member == current_value:
            return idx
    # 2. Compare raw values
    current_raw = get_enum_member_value(current_value)
    for idx, member in enumerate(members):
        if get_enum_member_value(member) == current_raw:
            return idx
    # 3. Compare as ints
    try:
        current_int = int(current_raw)
    except (ValueError, TypeError):
        current_int = None
    if current_int is not None:
        for idx, member in enumerate(members):
            try:
                if int(get_enum_member_value(member)) == current_int:
                    return idx
            except (ValueError, TypeError):
                continue
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Unified serialized-field renderer
# ═══════════════════════════════════════════════════════════════════════════

def _label_or_fullwidth(ctx, display_name, lw, has_visible_label):
    """Render field label or set full-width for the next item."""
    if has_visible_label:
        field_label(ctx, display_name, lw)
    else:
        ctx.set_next_item_width(-1)






def _render_enum_sf(ctx, wid, display_name, metadata, current_value, lw, has_visible_label):
    """Render an ENUM inspector field."""
    enum_cls = metadata.enum_type
    if isinstance(enum_cls, str):
        import infernux.lib as _lib
        enum_cls = getattr(_lib, enum_cls, None)
    if enum_cls is not None:
        if isinstance(enum_cls, type) and issubclass(enum_cls, Flag):
            members = list(enum_cls.__members__.values())
            names = resolve_enum_display_labels(metadata, members)
            labels = {member.name: name for member, name in zip(members, names)}
            current = (
                current_value
                if isinstance(current_value, enum_cls)
                else enum_cls(current_value or 0)
            )
            if any(
                member.value < -(1 << 63) or member.value >= (1 << 64)
                for member in [current, *members]
            ):
                ctx.label(f"{display_name}: {current}")
                return current
            summary = ", ".join(
                labels.get(part, part) for part in (current.name or "0").split("|")
            )
            _label_or_fullwidth(ctx, display_name, lw, has_visible_label)
            mask64 = (1 << 64) - 1
            updated = ctx.flags_combo(
                wid,
                current.value & mask64,
                names,
                [member.value & mask64 for member in members],
                summary,
            )
            if updated & (1 << 63) and any(member.value < 0 for member in members):
                updated -= 1 << 64
            return enum_cls(updated)
        members = get_enum_members(enum_cls)
        if not members:
            ctx.label(f"{display_name}: {current_value}")
            return current_value
        if (
            hasattr(metadata, "enum_labels")
            and metadata.enum_labels
            and len(metadata.enum_labels) == len(members)
        ):
            member_names = resolve_enum_display_labels(metadata, members)
        else:
            member_names = [get_enum_member_name(m) for m in members]
        current_idx = find_enum_index(members, current_value)
        _label_or_fullwidth(ctx, display_name, lw, has_visible_label)
        new_idx = ctx.combo(wid, current_idx, member_names, -1)
        if new_idx != current_idx:
            return members[new_idx]
    else:
        ctx.label(f"{display_name}: {current_value}")
    return current_value


def _render_color_sf(ctx, wid, display_name, metadata, current_value, lw, has_visible_label):
    """Render a COLOR inspector field."""
    if current_value is not None:
        r, g, b, a = current_value[0], current_value[1], current_value[2], current_value[3]
    else:
        r, g, b, a = 1.0, 1.0, 1.0, 1.0
    _label_or_fullwidth(ctx, display_name, lw, has_visible_label)
    allow_hdr = getattr(metadata, "hdr", False)
    nr, ng, nb, na = render_color_value_bar(
        ctx,
        wid,
        (r, g, b, a),
        allow_hdr=allow_hdr,
        default_hdr_enabled=allow_hdr,
    )
    if (nr, ng, nb, na) != (r, g, b, a):
        return [nr, ng, nb, na]
    return current_value


def _render_animation_curve_sf(
    ctx, wid, display_name, metadata, current_value, lw, has_visible_label
):
    """Render a reusable AnimationCurve preview and popup editor."""
    from infernux.graph.ramp import AnimationCurve
    from .curve_editor import render_curve_property

    curve = (
        current_value
        if isinstance(current_value, AnimationCurve)
        else AnimationCurve.from_dict(current_value)
    )
    _label_or_fullwidth(ctx, display_name, lw, has_visible_label)
    document = render_curve_property(
        ctx,
        str(wid),
        curve.to_dict(),
        semantic_prefix=f"serialized_field.{wid}",
        non_negative=bool(getattr(metadata, "curve_non_negative", False)),
    )
    return AnimationCurve.from_dict(document)


def _render_gradient_sf(ctx, wid, display_name, metadata, current_value):
    """Gradients and curves share a compact preview with popup editing."""
    from infernux.graph.ramp import Gradient
    from .gradient_editor import render_gradient_field
    gradient = current_value if isinstance(current_value, Gradient) else Gradient.from_dict(current_value)
    document = render_gradient_field(ctx, str(wid), gradient.to_dict(), hdr=metadata.hdr)
    return Gradient.from_dict(document)


def render_serialized_field(ctx, wid, display_name, metadata, current_value, label_width):
    """Compatibility entry point; all field rows use the common renderer."""
    from .field_draw import draw_field_value
    return draw_field_value(ctx, wid, display_name, metadata, current_value, label_width)


# ═══════════════════════════════════════════════════════════════════════════
#  Change detection
# ═══════════════════════════════════════════════════════════════════════════

def has_field_changed(field_type, old_value, new_value) -> bool:
    """Check whether a serialized field value has actually changed.

    Uses tolerance comparison for floats and identity check for vec
    types (``render_serialized_field`` already creates a new vec object
    only when the value differs).
    """
    from infernux.components.fields import FieldType

    if field_type == FieldType.FLOAT:
        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            return not float_close(float(new_value), float(old_value))
    elif field_type in (FieldType.VEC2, FieldType.VEC3, FieldType.VEC4):
        return new_value is not old_value
    elif field_type == FieldType.COLOR:
        from infernux.components.fields import rgba_equal
        return not rgba_equal(old_value, new_value)
    return new_value != old_value


# ═══════════════════════════════════════════════════════════════════════════
#  Info-text / tooltip helper
# ═══════════════════════════════════════════════════════════════════════════

def render_section_kicker(ctx, label: str) -> None:
    """Section labels use the same quiet typography and divider in every Inspector."""
    from .igui import IGUI
    from .dpi import editor_dpi_scale
    ctx.spacing()
    ctx.push_style_color(ImGuiCol.Text, *Theme.TEXT_DISABLED)
    ctx.push_ui_font(Theme.EDITOR_SMALL_FONT_SIZE)
    with IGUI.mono_font(ctx):
        ctx.label(label)
    ctx.pop_font()
    ctx.pop_style_color(1)
    ctx.separator()
    ctx.spacing()


def render_info_text(ctx: InxGUIContext, text: str, label_width: float = 0.0):
    """The common below-field information row."""
    ctx.inspector_field_info(text, label_width)


# ═══════════════════════════════════════════════════════════════════════════
#  Unity-style colour bar (shared by FieldType.COLOR and material ptype 7)
# ═══════════════════════════════════════════════════════════════════════════

# Per-widget HDR state: wid -> {"enabled": bool, "intensity": float}
_hdr_state: dict = {}
# Live base (pre-HDR) colour while the popup is open: wid -> [r, g, b, a]
_color_popup_live: dict = {}
# First-frame guard to prevent click-through on popup open
_color_popup_guard: set = set()


def _render_color_popup(
    ctx,
    wid,
    popup_id,
    r,
    g,
    b,
    a,
    allow_hdr,
    first_frame_set,
    default_hdr_enabled=False,
):
    """Render the colour-edit popup contents. Return *(nr, ng, nb, na)*."""
    nr, ng, nb, na = r, g, b, a
    if ctx.begin_popup(popup_id):
        try:
            first_frame = wid in first_frame_set
            if first_frame:
                first_frame_set.discard(wid)

            state = _hdr_state.setdefault(
                wid,
                {
                    "enabled": bool(allow_hdr and default_hdr_enabled),
                    "intensity": 1.0,
                },
            )
            live = _color_popup_live.setdefault(wid, [r, g, b, a])

            # Undo/redo is allowed while this non-modal popup remains open. Pull an
            # externally restored value back into the popup once no picker control
            # owns the pointer; otherwise the stale live cache would visually mask
            # the undo and could reapply the discarded colour on the next drag.
            if not ctx.is_any_item_active():
                incoming = [r, g, b, a]
                if allow_hdr and state["enabled"] and state["intensity"] not in (0.0, 0):
                    inverse_intensity = 1.0 / state["intensity"]
                    incoming[0] *= inverse_intensity
                    incoming[1] *= inverse_intensity
                    incoming[2] *= inverse_intensity
                if any(abs(float(live[index]) - float(incoming[index])) > 1e-6 for index in range(4)):
                    live[:] = incoming
            lr, lg, lb, la = live[0], live[1], live[2], live[3]

            # AlphaBar = 1 << 18
            changed_pick, pick_r, pick_g, pick_b, pick_a = ctx.color_picker(
                f"{wid}_cpicker", float(lr), float(lg), float(lb), float(la), 1 << 18,
            )
            interaction = False
            if changed_pick and not first_frame:
                lr, lg, lb, la = pick_r, pick_g, pick_b, pick_a
                interaction = True

            if allow_hdr:
                ctx.separator()
                old_en = state["enabled"]
                field_label(ctx, "HDR", ctx.inspector_label_width())
                state["enabled"] = ctx.checkbox_field("##_hdr_toggle", state["enabled"])
                if state["enabled"]:
                    old_int = state["intensity"]
                    field_label(ctx, t("field.intensity"), ctx.inspector_label_width())
                    state["intensity"] = ctx.drag_float(
                        "##_hdr_val", state["intensity"], 0.01, 0.0, 0.0,
                    )
                    if state["intensity"] != old_int:
                        interaction = True
                else:
                    state["intensity"] = 1.0
                if state["enabled"] != old_en:
                    interaction = True

            _color_popup_live[wid] = [lr, lg, lb, la]

            if interaction:
                nr, ng, nb, na = lr, lg, lb, la
                if allow_hdr and state["enabled"]:
                    m = state["intensity"]
                    nr, ng, nb = lr * m, lg * m, lb * m

        finally:
            ctx.end_popup()
    else:
        _color_popup_live.pop(wid, None)
        first_frame_set.discard(wid)

    return nr, ng, nb, na


def _render_color_bar(
    ctx: InxGUIContext,
    wid: str,
    r: float, g: float, b: float, a: float,
    *,
    allow_hdr: bool = False,
    default_hdr_enabled: bool = False,
    width: float | None = None,
    height: float | None = None,
) -> tuple:
    """Render the shared native colour swatch and return ``(nr, ng, nb, na)``.

    Opaque RGB, checkerboard alpha and the hexadecimal readout share one row.

    Clicking the bar opens a colour-edit popup directly.

    When *allow_hdr* is True the popup also shows an HDR checkbox and
    intensity slider.  The returned RGBA values are multiplied by the
    HDR intensity when HDR is enabled.
    """
    bar_width = (
        ctx.get_content_region_avail_width()
        if width is None
        else max(1.0, float(width))
    )
    clicked = ctx.color_swatch(f"{wid}_bar", (r, g, b, a), bar_width,
                               0.0 if height is None else float(height), allow_hdr)

    popup_id = f"{wid}_cpop"
    if clicked:
        # Compute base (pre-HDR) colour for live state
        hdr = _hdr_state.get(wid)
        if hdr and hdr["enabled"] and hdr["intensity"] not in (0.0, 0):
            inv = 1.0 / hdr["intensity"]
            _color_popup_live[wid] = [r * inv, g * inv, b * inv, a]
        else:
            _color_popup_live[wid] = [r, g, b, a]
        _color_popup_guard.add(wid)
        ctx.open_popup(popup_id)

    nr, ng, nb, na = _render_color_popup(
        ctx,
        wid,
        popup_id,
        r,
        g,
        b,
        a,
        allow_hdr,
        _color_popup_guard,
        default_hdr_enabled,
    )

    return nr, ng, nb, na


def render_color_value_bar(
    ctx: InxGUIContext,
    wid: str,
    value,
    *,
    allow_hdr: bool = False,
    default_hdr_enabled: bool = False,
    width: float | None = None,
    height: float | None = None,
) -> list[float]:
    """Render the shared Inspector-style RGBA bar for any editor surface."""
    components = list(value) if isinstance(value, (list, tuple)) else []
    defaults = (1.0, 1.0, 1.0, 1.0)
    rgba = [
        float(components[index]) if index < len(components) else defaults[index]
        for index in range(4)
    ]
    return list(
        _render_color_bar(
            ctx,
            wid,
            *rgba,
            allow_hdr=allow_hdr,
            default_hdr_enabled=default_hdr_enabled,
            width=width,
            height=height,
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Material property renderer — moved to inspector_material.py
#  Re-exported here for import convenience.
# ═══════════════════════════════════════════════════════════════════════════

def render_material_property(*args, **kwargs):
    from .inspector_material import render_material_property as _rmp
    return _rmp(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Component header (icon + collapsing header + enabled checkbox)
# ═══════════════════════════════════════════════════════════════════════════



def render_inspector_checkbox(
    ctx: InxGUIContext, label: str, value: bool, label_width: float = 0.0,
) -> bool:
    """Render a field label on the left and a compact square in the value column."""
    checkbox_field = getattr(ctx, "checkbox_field", None)
    if not callable(checkbox_field):
        raise RuntimeError(
            "Inspector rendering requires the native checkbox_field binding"
        )
    visible, separator, suffix = str(label).partition("##")
    if visible:
        field_label(ctx, visible, label_width)
    widget_id = f"##{suffix}" if separator else f"##bool_{label}"
    return bool(checkbox_field(widget_id, value))


def render_compact_section_header(
    ctx: InxGUIContext,
    label: str,
    *,
    icon_id: int = 0,
    default_open: bool = True,
    text_color=None,
    level: str = "secondary",
    allow_overlap: bool = False,
    header_id: str = "",
) -> bool:
    """Render a compact framed tree header shared by inspector-style panels."""
    if level == "primary":
        frame_pad = Theme.INSPECTOR_HEADER_PRIMARY_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_PRIMARY_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_PRIMARY
        hover_color = Theme.INSPECTOR_HEADER_PRIMARY_HOVERED
        active_color = Theme.INSPECTOR_HEADER_PRIMARY_ACTIVE
    elif level == "list":
        frame_pad = Theme.INSPECTOR_HEADER_LIST_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_LIST_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_LIST
        hover_color = Theme.INSPECTOR_HEADER_LIST_HOVERED
        active_color = Theme.INSPECTOR_HEADER_LIST_ACTIVE
    elif level == "tertiary":
        frame_pad = Theme.INSPECTOR_HEADER_TERTIARY_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_TERTIARY_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_TERTIARY
        hover_color = Theme.INSPECTOR_HEADER_TERTIARY_HOVERED
        active_color = Theme.INSPECTOR_HEADER_TERTIARY_ACTIVE
    else:
        frame_pad = Theme.INSPECTOR_HEADER_SECONDARY_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_SECONDARY_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_SECONDARY
        hover_color = Theme.INSPECTOR_HEADER_SECONDARY_HOVERED
        active_color = Theme.INSPECTOR_HEADER_SECONDARY_ACTIVE

    native_header = getattr(ctx, "render_compact_section_header", None)
    if not callable(native_header):
        raise RuntimeError(
            "Inspector rendering requires the native render_compact_section_header binding"
        )

    dpi = editor_dpi_scale(ctx)
    kicker = level in ("primary", "secondary")
    if kicker:
        frame_pad = (0.0, 5.0)
        font_scale = 0.85
        base_color = (*Theme.ROLE_BG_BASE[:3], 0.0)
        hover_color = Theme.BTN_GHOST_HOVERED
        active_color = Theme.BTN_GHOST_ACTIVE
    effective_text_color = text_color if text_color is not None else (Theme.TEXT_DIM if kicker else Theme.TEXT)
    from .igui import IGUI
    with IGUI.mono_font(ctx) if kicker else nullcontext():
        opened = bool(native_header(
        f"{label}###{header_id}" if header_id else label,
        int(icon_id or 0),
        bool(default_open),
        Theme.COND_FIRST_USE_EVER,
        bool(allow_overlap),
        frame_pad[0] * dpi,
        frame_pad[1] * dpi,
        Theme.INSPECTOR_HEADER_ITEM_SPC[0] * dpi,
        Theme.INSPECTOR_HEADER_ITEM_SPC[1] * dpi,
        Theme.INSPECTOR_HEADER_BORDER_SIZE * dpi,
        level in ("secondary", "list", "tertiary"),
        font_scale,
        Theme.INSPECTOR_HEADER_RIGHT_MARGIN * dpi,
        Theme.COMPONENT_ICON_SIZE * dpi,
        base_color,
        hover_color,
        active_color,
        True,
        effective_text_color,
        ))
    if kicker:
        x1 = ctx.get_item_rect_min_x()
        x2 = ctx.get_item_rect_max_x()
        y = ctx.get_item_rect_max_y()
        ctx.draw_line(x1, y, x2, y, *Theme.ROLE_BORDER, dpi)
    return opened


def render_compact_section_title(
    ctx: InxGUIContext,
    label: str,
    *,
    level: str = "secondary",
    text_color=None,
) -> None:
    """Render a non-collapsible inspector subsection title."""
    if level == "primary":
        font_scale = Theme.INSPECTOR_HEADER_PRIMARY_FONT_SCALE
        default_color = Theme.TEXT
    else:
        font_scale = Theme.INSPECTOR_HEADER_SECONDARY_FONT_SCALE
        default_color = Theme.TEXT_DIM

    color = text_color if text_color is not None else default_color
    ctx.dummy(0.0, Theme.INSPECTOR_SECTION_GAP * 0.5 * editor_dpi_scale(ctx))
    ctx.set_window_font_scale(font_scale)
    ctx.push_style_color(ImGuiCol.Text, *color)
    ctx.label(label)
    ctx.pop_style_color(1)
    ctx.set_window_font_scale(1.0)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Apply / Revert buttons for asset inspectors
# ═══════════════════════════════════════════════════════════════════════════

_COL_BUTTON = 21  # ImGuiCol.Button


def render_apply_revert(
    ctx: InxGUIContext,
    is_dirty: bool,
    on_apply,
    on_revert,
    semantic_prefix: str = "",
) -> None:
    """Render a unified Apply / Revert button bar for import-settings editors."""
    ctx.separator()
    if is_dirty:
        ctx.push_style_color(_COL_BUTTON, *Theme.APPLY_BUTTON)
        ctx.button("Apply", on_apply)
        if semantic_prefix and semantic_capture_enabled(ctx):
            ctx.record_semantic_item("button", "Apply", True, f"{semantic_prefix}.apply")
        ctx.pop_style_color(1)
        ctx.same_line()
        ctx.button("Revert", on_revert)
        if semantic_prefix and semantic_capture_enabled(ctx):
            ctx.record_semantic_item("button", "Revert", True, f"{semantic_prefix}.revert")
    else:
        ctx.begin_disabled(True)
        ctx.button("Apply", None)
        if semantic_prefix and semantic_capture_enabled(ctx):
            ctx.record_semantic_item("button", "Apply", False, f"{semantic_prefix}.apply")
        ctx.same_line()
        ctx.button("Revert", None)
        if semantic_prefix and semantic_capture_enabled(ctx):
            ctx.record_semantic_item("button", "Revert", False, f"{semantic_prefix}.revert")
        ctx.end_disabled()
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Batch property descriptor builder (for C++ RenderPropertyBatch)
# ═══════════════════════════════════════════════════════════════════════════

# Must match PropertyDesc::Type enum in InxGUIContext.h
PROP_FLOAT  = 0
PROP_INT    = 1
PROP_BOOL   = 2
PROP_STRING = 3
PROP_VEC2   = 4
PROP_VEC3   = 5
PROP_VEC4   = 6
PROP_ENUM   = 7
PROP_COLOR  = 8

_FIELD_TYPE_TO_PROP = None  # lazy init


def _ensure_prop_map():
    global _FIELD_TYPE_TO_PROP
    if _FIELD_TYPE_TO_PROP is not None:
        return
    from infernux.components.fields import FieldType
    _FIELD_TYPE_TO_PROP = {
        FieldType.FLOAT:  PROP_FLOAT,
        FieldType.INT:    PROP_INT,
        FieldType.BOOL:   PROP_BOOL,
        FieldType.STRING: PROP_STRING,
        FieldType.VEC2:   PROP_VEC2,
        FieldType.VEC3:   PROP_VEC3,
        FieldType.VEC4:   PROP_VEC4,
        FieldType.ENUM:   PROP_ENUM,
        # COLOR intentionally omitted — rendered via Python _render_color_bar
        # (Unity long-bar style, same as material ptype-7).  C++ PROP_COLOR uses
        # ImGui::ColorEdit4 (numeric fields + square swatch) and must not be
        # used for component serialized fields.
    }


def is_batch_renderable(field_type) -> bool:
    """Return True if this FieldType can be handled by C++ batch renderer."""
    from infernux.components.fields import FieldType
    _ensure_prop_map()
    if field_type == FieldType.COLOR:
        return False
    return field_type in _FIELD_TYPE_TO_PROP


_VEC_KEYS = ("f", "f2", "f3", "f4")
_VEC_ATTRS = ("x", "y", "z", "w")


def _pack_vec_components(desc, current_value, n, default=0.0):
    """Pack *n* vector components into *desc* using standard keys."""
    for i in range(n):
        if current_value is not None:
            attribute = _VEC_ATTRS[i]
            if hasattr(current_value, attribute):
                component = getattr(current_value, attribute)
            else:
                component = current_value[i]
            desc[_VEC_KEYS[i]] = float(component)
        else:
            desc[_VEC_KEYS[i]] = default


def build_scalar_desc(
    wid: str,
    display_name: str,
    metadata,
    current_value,
    *,
    header_text: str = "",
    space_before: float = 0,
    mixed: bool = False,
    semantic_id: str = "",
) -> dict | None:
    """Build a single property descriptor dict for the C++ batch renderer.

    Returns None if the field type is not batch-renderable (references, lists, etc.).
    """
    _ensure_prop_map()
    prop_type = _FIELD_TYPE_TO_PROP.get(metadata.field_type)
    if prop_type is None:
        return None

    desc = {"t": prop_type, "w": wid, "n": display_name}
    if metadata.readonly:
        from .field_draw import format_field_value
        desc["ro"] = True
        desc["rt"] = format_field_value(metadata, current_value)
    if metadata.info_text:
        from infernux.engine.i18n import t
        desc["info"] = t(metadata.info_text)

    if semantic_id:
        desc["sid"] = semantic_id
    if mixed:
        desc["mix"] = True

    # --- Value ---
    if prop_type == PROP_FLOAT:
        desc["f"] = float(current_value) if current_value is not None else 0.0
    elif prop_type == PROP_INT:
        desc["i"] = int(current_value) if current_value is not None else 0
    elif prop_type == PROP_BOOL:
        desc["b"] = bool(current_value) if current_value is not None else False
        desc["fl"] = True
    elif prop_type == PROP_STRING:
        desc["s"] = str(current_value) if current_value else ""
    elif prop_type == PROP_VEC2:
        _pack_vec_components(desc, current_value, 2)
    elif prop_type == PROP_VEC3:
        _pack_vec_components(desc, current_value, 3)
    elif prop_type == PROP_VEC4:
        _pack_vec_components(desc, current_value, 4)
    elif prop_type == PROP_ENUM:
        enum_cls = metadata.enum_type
        if isinstance(enum_cls, str):
            import infernux.lib as _lib

            enum_cls = getattr(_lib, enum_cls, None)
        if enum_cls is not None:
            if isinstance(enum_cls, type) and issubclass(enum_cls, Flag):
                return None  # A single enum index cannot represent a flag mask.
            members = get_enum_members(enum_cls)
            if (
                hasattr(metadata, "enum_labels")
                and metadata.enum_labels
                and len(metadata.enum_labels) == len(members)
            ):
                desc["en"] = resolve_enum_display_labels(metadata, members)
            else:
                desc["en"] = [get_enum_member_name(m) for m in members]
            desc["ei"] = find_enum_index(members, current_value)
        else:
            return None  # can't render without enum info
    # PROP_COLOR: only if COLOR is re-added to _FIELD_TYPE_TO_PROP (not used today).

    # --- Metadata ---
    if metadata.range:
        if prop_type == PROP_INT:
            desc["mn"] = int(metadata.range[0])
            desc["mx"] = int(metadata.range[1])
        else:
            desc["mn"] = float(metadata.range[0])
            desc["mx"] = float(metadata.range[1])
    speed = getattr(metadata, "drag_speed", None)
    if speed is not None:
        desc["sp"] = float(speed)
    elif prop_type == PROP_INT:
        desc["sp"] = DRAG_SPEED_INT
    elif prop_type == PROP_FLOAT:
        desc["sp"] = DRAG_SPEED_DEFAULT
    if getattr(metadata, "slider", False):
        desc["sl"] = True
    if getattr(metadata, "multiline", False):
        desc["ml"] = True

    # --- Layout ---
    if header_text:
        from infernux.engine.i18n import t
        desc["hdr"] = t(header_text)
    if space_before > 0:
        desc["spc"] = space_before

    # --- Tooltip ---
    if metadata.tooltip:
        from infernux.engine.i18n import t
        desc["tt"] = t(metadata.tooltip)

    return desc
