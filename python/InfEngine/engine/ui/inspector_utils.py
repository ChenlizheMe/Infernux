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

import math
import os
from typing import Any

from InfEngine.lib import InfGUIContext
from .theme import Theme, ImGuiCol, ImGuiStyleVar, ImGuiTreeNodeFlags


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

LABEL_PAD: float = Theme.INSPECTOR_LABEL_PAD
"""Extra pixels added after the widest label text."""

MIN_LABEL_WIDTH: float = Theme.INSPECTOR_MIN_LABEL_WIDTH
"""Absolute lower bound for the label column so the inspector never looks cramped."""

DRAG_SPEED_DEFAULT: float = 0.1
"""Default drag-float speed — matches Unity feel."""

DRAG_SPEED_FINE: float = 0.01
"""Fine drag speed for scale / small-precision fields."""

DRAG_SPEED_INT: float = 1.0
"""Drag speed for integer fields."""


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


# ═══════════════════════════════════════════════════════════════════════════
#  Naming / formatting
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Label / layout primitives
# ═══════════════════════════════════════════════════════════════════════════

_LABEL_W_CACHE: dict = {}  # (tuple_of_labels,) -> float


def max_label_w(ctx: InfGUIContext, labels, *, min_width: float = 0.0) -> float:
    """Return the pixel width for the widest label + padding.

    Results are cached per unique label set (font is constant at runtime).
    """
    key = tuple(labels)
    cached = _LABEL_W_CACHE.get(key)
    if cached is not None:
        return cached
    _min = min_width if min_width > 0 else MIN_LABEL_WIDTH
    w = 0.0
    for lb in labels:
        tw = ctx.calc_text_width(lb)
        if tw > w:
            w = tw
    result = max(w + LABEL_PAD, _min)
    _LABEL_W_CACHE[key] = result
    return result


def field_label(ctx: InfGUIContext, label: str, width: float = 0.0):
    """Label on the left; the next widget fills the remaining row width.

    If *width* is 0 the column is auto-sized to this label (but never
    narrower than ``MIN_LABEL_WIDTH``).
    """
    if width <= 0.0:
        width = max(ctx.calc_text_width(label) + LABEL_PAD, MIN_LABEL_WIDTH)
    ctx.align_text_to_frame_padding()
    ctx.label(label)
    ctx.same_line(width)
    ctx.set_next_item_width(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Enum helpers (Python Enum + pybind11 enum compat)
# ═══════════════════════════════════════════════════════════════════════════

def get_enum_members(enum_cls):
    """Return member list for a Python Enum **or** pybind11 enum-like type."""
    if enum_cls is None:
        return []
    try:
        return list(enum_cls)
    except TypeError:
        pass
    members_dict = getattr(enum_cls, "__members__", None)
    if isinstance(members_dict, dict):
        return list(members_dict.values())
    return []


def get_enum_member_name(member) -> str:
    """Human-readable name of an enum member."""
    name = getattr(member, "name", None)
    return str(name) if name else str(member)


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

def render_serialized_field(
    ctx: InfGUIContext,
    wid: str,
    display_name: str,
    metadata,
    current_value: Any,
    lw: float,
) -> Any:
    """Render an inspector field based on its ``FieldType`` metadata.

    Handles **scalar / value types** only:
    ``FLOAT``, ``INT``, ``BOOL``, ``STRING``, ``VEC2``, ``VEC3``, ``VEC4``,
    ``ENUM``, ``COLOR``.

    Reference types (``GAME_OBJECT``, ``MATERIAL``, ``TEXTURE``,
    ``SHADER``, ``ASSET``) are **not** handled — they require
    context-specific callbacks and should be rendered by the caller.

    Returns
    -------
    The (possibly changed) value.  Callers should compare the return
    value with *current_value* using :func:`has_field_changed` and
    apply changes + undo as appropriate.
    """
    from InfEngine.components.serialized_field import FieldType

    ft = metadata.field_type
    new_value = current_value
    has_visible_label = bool(display_name and str(display_name).strip())
    vector_label = display_name if has_visible_label else " "

    # ── FLOAT ──────────────────────────────────────────────────────────
    if ft == FieldType.FLOAT:
        if has_visible_label:
            field_label(ctx, display_name, lw)
        else:
            ctx.set_next_item_width(-1)
        speed = getattr(metadata, "drag_speed", None) or DRAG_SPEED_DEFAULT
        slider = getattr(metadata, "slider", False)
        if metadata.range:
            if slider:
                new_value = ctx.float_slider(
                    wid, float(current_value),
                    metadata.range[0], metadata.range[1],
                )
            else:
                new_value = ctx.drag_float(
                    wid, float(current_value), speed,
                    metadata.range[0], metadata.range[1],
                )
        else:
            new_value = ctx.drag_float(
                wid, float(current_value), speed, -1e6, 1e6,
            )

    # ── INT ────────────────────────────────────────────────────────────
    elif ft == FieldType.INT:
        if has_visible_label:
            field_label(ctx, display_name, lw)
        else:
            ctx.set_next_item_width(-1)
        speed = getattr(metadata, "drag_speed", None) or DRAG_SPEED_INT
        slider = getattr(metadata, "slider", False)
        if metadata.range:
            if slider:
                new_value = int(ctx.int_slider(
                    wid, int(current_value),
                    metadata.range[0], metadata.range[1],
                ))
            else:
                new_value = int(ctx.drag_int(
                    wid, int(current_value), speed,
                    metadata.range[0], metadata.range[1],
                ))
        else:
            new_value = int(ctx.drag_int(
                wid, int(current_value), speed, -1000000, 1000000,
            ))

    # ── BOOL ───────────────────────────────────────────────────────────
    elif ft == FieldType.BOOL:
        new_value = render_inspector_checkbox(ctx, display_name if has_visible_label else wid, bool(current_value))

    # ── STRING ─────────────────────────────────────────────────────────
    elif ft == FieldType.STRING:
        if has_visible_label:
            field_label(ctx, display_name, lw)
        else:
            ctx.set_next_item_width(-1)
        multiline = getattr(metadata, "multiline", False)
        if multiline:
            new_value = ctx.input_text_multiline(
                wid,
                str(current_value) if current_value else "",
                buffer_size=4096, width=-1, height=80,
            )
        else:
            new_value = ctx.text_input(
                wid, str(current_value) if current_value else "", 256,
            )

    # ── VEC2 ───────────────────────────────────────────────────────────
    elif ft == FieldType.VEC2:
        if current_value is not None:
            x, y = float(current_value.x), float(current_value.y)
        else:
            x, y = 0.0, 0.0
        vec_lw = lw if has_visible_label else 1.0
        nx, ny = ctx.vector2(vector_label, x, y, DRAG_SPEED_DEFAULT, vec_lw)
        if any(not float_close(a, b) for a, b in [(nx, x), (ny, y)]):
            from InfEngine.lib import Vector2
            new_value = Vector2(nx, ny)

    # ── VEC3 ───────────────────────────────────────────────────────────
    elif ft == FieldType.VEC3:
        if current_value is not None:
            x, y, z = (
                float(current_value.x),
                float(current_value.y),
                float(current_value.z),
            )
        else:
            x, y, z = 0.0, 0.0, 0.0
        vec_lw = lw if has_visible_label else 1.0
        nx, ny, nz = ctx.vector3(vector_label, x, y, z, DRAG_SPEED_DEFAULT, vec_lw)
        if any(not float_close(a, b) for a, b in [(nx, x), (ny, y), (nz, z)]):
            from InfEngine.lib import Vector3
            new_value = Vector3(nx, ny, nz)

    # ── VEC4 ───────────────────────────────────────────────────────────
    elif ft == FieldType.VEC4:
        if current_value is not None:
            x = float(current_value.x)
            y = float(current_value.y)
            z = float(current_value.z)
            w = float(current_value.w)
        else:
            x, y, z, w = 0.0, 0.0, 0.0, 0.0
        vec_lw = lw if has_visible_label else 1.0
        nx, ny, nz, nw = ctx.vector4(
            vector_label, x, y, z, w, DRAG_SPEED_DEFAULT, vec_lw,
        )
        if any(not float_close(a, b) for a, b in [(nx, x), (ny, y), (nz, z), (nw, w)]):
            from InfEngine.lib import vec4f
            new_value = vec4f(nx, ny, nz, nw)

    # ── ENUM ───────────────────────────────────────────────────────────
    elif ft == FieldType.ENUM:
        enum_cls = metadata.enum_type
        if isinstance(enum_cls, str):
            import InfEngine.lib as _lib
            enum_cls = getattr(_lib, enum_cls, None)
        if enum_cls is not None:
            members = get_enum_members(enum_cls)
            if not members:
                ctx.label(f"{display_name}: {current_value}")
                return new_value
            if (hasattr(metadata, "enum_labels")
                    and metadata.enum_labels
                    and len(metadata.enum_labels) == len(members)):
                member_names = metadata.enum_labels
            else:
                member_names = [get_enum_member_name(m) for m in members]
            current_idx = find_enum_index(members, current_value)
            if has_visible_label:
                field_label(ctx, display_name, lw)
            else:
                ctx.set_next_item_width(-1)
            new_idx = ctx.combo(wid, current_idx, member_names, -1)
            if new_idx != current_idx:
                new_value = members[new_idx]
        else:
            ctx.label(f"{display_name}: {current_value}")

    # ── COLOR ──────────────────────────────────────────────────────────
    elif ft == FieldType.COLOR:
        if current_value is not None:
            r, g, b, a = (
                current_value[0], current_value[1],
                current_value[2], current_value[3],
            )
        else:
            r, g, b, a = 1.0, 1.0, 1.0, 1.0
        if has_visible_label:
            field_label(ctx, display_name, lw)
        else:
            ctx.set_next_item_width(-1)
        allow_hdr = getattr(metadata, 'hdr', False)
        nr, ng, nb, na = _render_color_bar(ctx, wid, r, g, b, a, allow_hdr=allow_hdr)
        if (nr, ng, nb, na) != (r, g, b, a):
            new_value = [nr, ng, nb, na]

    # ── Fallback ───────────────────────────────────────────────────────
    else:
        ctx.label(f"{display_name}: {current_value}")

    return new_value


# ═══════════════════════════════════════════════════════════════════════════
#  Change detection
# ═══════════════════════════════════════════════════════════════════════════

def has_field_changed(field_type, old_value, new_value) -> bool:
    """Check whether a serialized field value has actually changed.

    Uses tolerance comparison for floats and identity check for vec
    types (``render_serialized_field`` already creates a new vec object
    only when the value differs).
    """
    from InfEngine.components.serialized_field import FieldType

    if field_type == FieldType.FLOAT:
        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            return not float_close(float(new_value), float(old_value))
    elif field_type in (FieldType.VEC2, FieldType.VEC3, FieldType.VEC4):
        return new_value is not old_value
    return new_value != old_value


# ═══════════════════════════════════════════════════════════════════════════
#  Info-text / tooltip helper
# ═══════════════════════════════════════════════════════════════════════════

def render_info_text(ctx: InfGUIContext, text: str):
    """Render a dimmed, non-editable info line below a field."""
    ctx.push_style_color(ImGuiCol.Text, *Theme.META_TEXT)
    ctx.label(f"  {text}")
    ctx.pop_style_color(1)


# ═══════════════════════════════════════════════════════════════════════════
#  Unity-style colour bar (shared by FieldType.COLOR and material ptype 7)
# ═══════════════════════════════════════════════════════════════════════════

_COLOR_BAR_H: float = 20.0

# Per-widget HDR state: wid -> {"enabled": bool, "intensity": float}
_hdr_state: dict = {}
# Live base (pre-HDR) colour while the popup is open: wid -> [r, g, b, a]
_color_popup_live: dict = {}
# First-frame guard to prevent click-through on popup open
_color_popup_guard: set = set()


def _render_color_bar(
    ctx: InfGUIContext,
    wid: str,
    r: float, g: float, b: float, a: float,
    *,
    allow_hdr: bool = False,
) -> tuple:
    """Render a Unity-style colour bar and return ``(nr, ng, nb, na)``.

    The bar is a horizontal rectangle:
    * Top 3/4 — solid RGB colour (full opacity).
    * Bottom 1/4 — grey-scale representing alpha.

    Clicking the bar opens a colour-edit popup directly.

    When *allow_hdr* is True the popup also shows an HDR checkbox and
    intensity slider.  The returned RGBA values are multiplied by the
    HDR intensity when HDR is enabled.
    """
    avail_w = ctx.get_content_region_avail_width()
    clicked = ctx.invisible_button(f"{wid}_bar", avail_w, _COLOR_BAR_H)

    min_x = ctx.get_item_rect_min_x()
    min_y = ctx.get_item_rect_min_y()
    max_x = ctx.get_item_rect_max_x()
    max_y = ctx.get_item_rect_max_y()

    split_y = min_y + (max_y - min_y) * 0.75

    # Top 3/4: RGB at full opacity
    ctx.draw_filled_rect(min_x, min_y, max_x, split_y, r, g, b, 1.0)
    # Bottom 1/4: alpha as grey
    ctx.draw_filled_rect(min_x, split_y, max_x, max_y, a, a, a, 1.0)
    # Thin border
    ctx.draw_rect(min_x, min_y, max_x, max_y, *Theme.COLOR_SWATCH_BORDER, 1.0)

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

    nr, ng, nb, na = r, g, b, a
    if ctx.begin_popup(popup_id):
        first_frame = wid in _color_popup_guard
        if first_frame:
            _color_popup_guard.discard(wid)

        state = _hdr_state.setdefault(wid, {"enabled": False, "intensity": 1.0})
        live = _color_popup_live.setdefault(wid, [r, g, b, a])
        lr, lg, lb, la = live[0], live[1], live[2], live[3]

        # AlphaBar = 1 << 18
        changed_pick, pick_r, pick_g, pick_b, pick_a = ctx.color_picker(
            f"{wid}_cpicker", float(lr), float(lg), float(lb), float(la), 1 << 18,
        )
        interaction = False
        if changed_pick and not first_frame:
            lr, lg, lb, la = pick_r, pick_g, pick_b, pick_a
            interaction = True

        ctx.separator()
        new_r = float(ctx.drag_float(f"R##{wid}_r", float(lr), 0.01, 0.0, 0.0))
        new_g = float(ctx.drag_float(f"G##{wid}_g", float(lg), 0.01, 0.0, 0.0))
        new_b = float(ctx.drag_float(f"B##{wid}_b", float(lb), 0.01, 0.0, 0.0))
        new_a = float(ctx.drag_float(f"A##{wid}_a", float(la), 0.01, 0.0, 1.0))

        if new_r != lr:
            lr = new_r; interaction = True
        if new_g != lg:
            lg = new_g; interaction = True
        if new_b != lb:
            lb = new_b; interaction = True
        if new_a != la:
            la = new_a; interaction = True

        if allow_hdr:
            old_en = state["enabled"]
            state["enabled"] = ctx.checkbox("HDR##_hdr_toggle", state["enabled"])
            if state["enabled"]:
                old_int = state["intensity"]
                state["intensity"] = ctx.drag_float(
                    "Intensity##_hdr_val", state["intensity"], 0.01, 0.0, 0.0,
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

        ctx.end_popup()
    else:
        _color_popup_live.pop(wid, None)
        _color_popup_guard.discard(wid)

    return nr, ng, nb, na


# ═══════════════════════════════════════════════════════════════════════════
#  Material property renderer — moved to inspector_material.py
#  Re-exported here for backward compatibility.
# ═══════════════════════════════════════════════════════════════════════════

def render_material_property(*args, **kwargs):
    from .inspector_material import render_material_property as _rmp
    return _rmp(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Component header (icon + collapsing header + enabled checkbox)
# ═══════════════════════════════════════════════════════════════════════════

def render_component_header(
    ctx: InfGUIContext,
    type_name: str,
    *,
    icon_id: int = 0,
    show_enabled: bool = True,
    is_enabled: bool = True,
    suffix: str = "",
    default_open: bool = True,
    force_open: bool = False,
) -> tuple:
    """Render a standard component / section header.

    Layout:  [▶ full-width bar:  icon → checkbox → display_name]

    The collapsing header spans the entire row, and the icon, checkbox,
    and label are overlaid via ``SameLine``.

    Returns ``(header_open: bool, new_enabled: bool)``.
    """
    from .theme import Theme

    display_name = format_display_name(f"{type_name}{suffix}")
    new_enabled = is_enabled

    # ── styling ──
    ctx.push_style_color(ImGuiCol.Header, *Theme.INSPECTOR_HEADER_PRIMARY)
    ctx.push_style_color(ImGuiCol.HeaderHovered, *Theme.INSPECTOR_HEADER_PRIMARY_HOVERED)
    ctx.push_style_color(ImGuiCol.HeaderActive, *Theme.INSPECTOR_HEADER_PRIMARY_ACTIVE)
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_HEADER_PRIMARY_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_HEADER_ITEM_SPC)
    ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, Theme.INSPECTOR_HEADER_BORDER_SIZE)
    ctx.set_window_font_scale(Theme.INSPECTOR_HEADER_PRIMARY_FONT_SCALE)

    # ── full-width collapsing header (arrow + hidden label) ──
    if force_open:
        ctx.set_next_item_open(True, 0)          # 0 = ImGuiCond_Always
    elif default_open:
        ctx.set_next_item_open(True, Theme.COND_FIRST_USE_EVER)
    ctx.set_next_item_allow_overlap()
    header_open = ctx.collapsing_header(f"##comp_{type_name}")

    # ── overlay icon / checkbox / label on the same row ──
    indent = Theme.INSPECTOR_HEADER_CONTENT_INDENT
    ctx.same_line(indent, 0)

    if icon_id:
        ctx.image(icon_id, Theme.COMPONENT_ICON_SIZE, Theme.COMPONENT_ICON_SIZE)
        ctx.same_line(0, Theme.INSPECTOR_HEADER_ITEM_SPC[0])

    if show_enabled:
        new_enabled = render_inspector_checkbox(ctx, "##hdr_en", is_enabled)
        ctx.same_line(0, Theme.INSPECTOR_HEADER_ITEM_SPC[0])

    ctx.label(display_name)

    # ── cleanup ──
    ctx.set_window_font_scale(1.0)
    ctx.pop_style_color(3)
    ctx.pop_style_var(3)

    return header_open, new_enabled


def render_inspector_checkbox(ctx: InfGUIContext, label: str, value: bool) -> bool:
    """Render a compact checkbox with the shared inspector sizing."""
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_CHECKBOX_FRAME_PAD)
    ctx.set_window_font_scale(Theme.INSPECTOR_CHECKBOX_FONT_SCALE)
    new_value = ctx.checkbox(label, value)
    ctx.set_window_font_scale(1.0)
    ctx.pop_style_var(1)
    return new_value


def render_compact_section_header(
    ctx: InfGUIContext,
    label: str,
    *,
    icon_id: int = 0,
    default_open: bool = True,
    text_color=None,
    level: str = "secondary",
    allow_overlap: bool = False,
) -> bool:
    """Render a compact framed tree header shared by inspector-style panels."""
    if default_open:
        ctx.set_next_item_open(True, Theme.COND_FIRST_USE_EVER)
    if allow_overlap:
        ctx.set_next_item_allow_overlap()

    if level == "primary":
        frame_pad = Theme.INSPECTOR_HEADER_PRIMARY_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_PRIMARY_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_PRIMARY
        hover_color = Theme.INSPECTOR_HEADER_PRIMARY_HOVERED
        active_color = Theme.INSPECTOR_HEADER_PRIMARY_ACTIVE
    else:
        frame_pad = Theme.INSPECTOR_HEADER_SECONDARY_FRAME_PAD
        font_scale = Theme.INSPECTOR_HEADER_SECONDARY_FONT_SCALE
        base_color = Theme.INSPECTOR_HEADER_SECONDARY
        hover_color = Theme.INSPECTOR_HEADER_SECONDARY_HOVERED
        active_color = Theme.INSPECTOR_HEADER_SECONDARY_ACTIVE

    ctx.push_style_color(ImGuiCol.Header, *base_color)
    ctx.push_style_color(ImGuiCol.HeaderHovered, *hover_color)
    ctx.push_style_color(ImGuiCol.HeaderActive, *active_color)
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *frame_pad)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_HEADER_ITEM_SPC)
    ctx.push_style_var_float(ImGuiStyleVar.FrameBorderSize, Theme.INSPECTOR_HEADER_BORDER_SIZE)
    if level == "secondary":
        ctx.push_style_var_float(ImGuiStyleVar.IndentSpacing, 0.0)
    ctx.set_window_font_scale(font_scale)
    if text_color is not None:
        ctx.push_style_color(ImGuiCol.Text, *text_color)

    if icon_id:
        ctx.image(icon_id, Theme.COMPONENT_ICON_SIZE, Theme.COMPONENT_ICON_SIZE)
        ctx.same_line()

    header_open = ctx.collapsing_header(label)

    if text_color is not None:
        ctx.pop_style_color(1)
    ctx.set_window_font_scale(1.0)
    ctx.pop_style_color(3)
    ctx.pop_style_var(4 if level == "secondary" else 3)
    return header_open


# ═══════════════════════════════════════════════════════════════════════════
#  Apply / Revert buttons for asset inspectors
# ═══════════════════════════════════════════════════════════════════════════

_COL_BUTTON = 21  # ImGuiCol.Button


def render_apply_revert(
    ctx: InfGUIContext,
    is_dirty: bool,
    on_apply,
    on_revert,
) -> None:
    """Render a unified Apply / Revert button bar for import-settings editors."""
    ctx.separator()
    if is_dirty:
        ctx.push_style_color(_COL_BUTTON, *Theme.APPLY_BUTTON)
        ctx.button("Apply", on_apply)
        ctx.pop_style_color(1)
        ctx.same_line()
        ctx.button("Revert", on_revert)
    else:
        ctx.begin_disabled(True)
        ctx.button("Apply", None)
        ctx.same_line()
        ctx.button("Revert", None)
        ctx.end_disabled()
    return changed
