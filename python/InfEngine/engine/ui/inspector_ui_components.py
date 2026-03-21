"""Custom Inspector renderers for screen-space UI components."""

from __future__ import annotations

import math
import os
import copy

import time

from InfEngine.ui import UICanvas, UIText, UIImage, UIButton
from InfEngine.ui.enums import TextResizeMode
from InfEngine.ui.enums import RenderMode, TextAlignH, TextAlignV
from InfEngine.engine.project_context import get_project_root

from .inspector_components import _record_property, register_py_component_renderer
from InfEngine.engine.i18n import t
from .inspector_utils import field_label, max_label_w, render_compact_section_header, _render_color_bar, render_inspector_checkbox
from .theme import Theme


def _igui():
    """Lazy import to avoid circular dependency with inspector_components."""
    from .igui import IGUI
    return IGUI


def _apply_if_changed(comp, field_name: str, current, new_value):
    if new_value == current:
        return
    _record_property(comp, field_name, current, new_value, f"Set {field_name}")
    if hasattr(comp, "_call_on_validate"):
        comp._call_on_validate()


_font_cache = None
_font_cache_time = 0.0
_FONT_CACHE_TTL = 2.0  # seconds

_FONT_EXCLUDE_DIRS = {'.venv', '__pycache__', 'build', '.git', 'node_modules', 'external'}


def _get_project_font_options():
    global _font_cache, _font_cache_time
    now = time.monotonic()
    if _font_cache is not None and (now - _font_cache_time) < _FONT_CACHE_TTL:
        return _font_cache

    root = get_project_root()
    options = [(t("ui_comp.default_font"), "")]
    if not root or not os.path.isdir(root):
        _font_cache = options
        _font_cache_time = now
        return options

    seen_names = set()
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _FONT_EXCLUDE_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in (".ttf", ".otf"):
                continue
            # Deduplicate by filename (case-insensitive)
            name_lower = filename.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root).replace("\\", "/")
            found.append((filename, rel_path))

    found.sort(key=lambda item: item[0].lower())
    options.extend(found)
    _font_cache = options
    _font_cache_time = now
    return options


def _find_canvas(comp):
    go = getattr(comp, "game_object", None)
    while go is not None:
        for py_comp in go.get_py_components():
            if isinstance(py_comp, UICanvas):
                return py_comp
        go = go.get_parent()
    return None


def _get_parent_rect(comp):
    """Return (canvas, parent_visual_rect) for alignment purposes."""
    from InfEngine.ui import InfUIScreenComponent

    canvas = _find_canvas(comp)
    if canvas is None:
        return None, None

    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)
    parent_go = getattr(comp.game_object, "get_parent", lambda: None)()
    while parent_go is not None:
        for py_comp in parent_go.get_py_components():
            if isinstance(py_comp, InfUIScreenComponent):
                rect = py_comp.get_visual_rect(cw, ch)
                return canvas, rect
        for py_comp in parent_go.get_py_components():
            if isinstance(py_comp, UICanvas):
                return canvas, (0.0, 0.0, cw, ch)
        parent_go = parent_go.get_parent()

    return canvas, (0.0, 0.0, cw, ch)


def _canvas_dims(comp):
    """Return (canvas, cw, ch) or (None, 0, 0)."""
    canvas = _find_canvas(comp)
    if canvas is None:
        return None, 0.0, 0.0
    return canvas, float(canvas.reference_width), float(canvas.reference_height)


def _apply_visual_position(comp, vis_x, vis_y, canvas):
    """Move element so its visual AABB top-left is at (vis_x, vis_y), with undo."""
    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)
    # Temporarily apply to compute parent-aware target values, then restore & record undo
    old_x = comp.x
    old_y = comp.y
    comp.set_visual_position(vis_x, vis_y, cw, ch)
    new_x = comp.x
    new_y = comp.y
    comp.x = old_x
    comp.y = old_y
    _apply_if_changed(comp, "x", old_x, new_x)
    _apply_if_changed(comp, "y", old_y, new_y)


def _apply_size_preserve_visual_position(comp, width, height, canvas):
    """Update width/height while keeping the current visual top-left fixed."""
    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)
    old_x = comp.x
    old_y = comp.y
    old_w = comp.width
    old_h = comp.height
    comp.set_size_preserve_visual_position(float(width), float(height), cw, ch)
    new_x = comp.x
    new_y = comp.y
    new_w = comp.width
    new_h = comp.height
    comp.x = old_x
    comp.y = old_y
    comp.width = old_w
    comp.height = old_h
    _apply_if_changed(comp, "x", old_x, new_x)
    _apply_if_changed(comp, "y", old_y, new_y)
    _apply_if_changed(comp, "width", old_w, new_w)
    _apply_if_changed(comp, "height", old_h, new_h)


def _apply_size_preserve_top_left(comp, width, height, canvas):
    """Update width/height while keeping the rotated top-left corner fixed."""
    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)
    old_x = comp.x
    old_y = comp.y
    old_w = comp.width
    old_h = comp.height
    comp.set_size_preserve_corner(float(width), float(height), cw, ch, "top_left")
    new_x = comp.x
    new_y = comp.y
    new_w = comp.width
    new_h = comp.height
    comp.x = old_x
    comp.y = old_y
    comp.width = old_w
    comp.height = old_h
    _apply_if_changed(comp, "x", old_x, new_x)
    _apply_if_changed(comp, "y", old_y, new_y)
    _apply_if_changed(comp, "width", old_w, new_w)
    _apply_if_changed(comp, "height", old_h, new_h)


def _set_native_size(comp):
    """Resize UIImage/UIButton to the native pixel dimensions of its texture."""
    tex_path = getattr(comp, "texture_path", "") or ""
    if not tex_path:
        return
    project_root = get_project_root()
    if not project_root:
        return
    abs_path = os.path.normpath(os.path.join(project_root, tex_path))
    if not os.path.isfile(abs_path):
        return
    from InfEngine.lib import TextureLoader
    td = TextureLoader.load_from_file(abs_path)
    if not td or not td.is_valid():
        return
    canvas, _, _ = _canvas_dims(comp)
    w, h = float(td.width), float(td.height)
    if canvas is not None:
        _apply_size_preserve_top_left(comp, w, h, canvas)
    else:
        _apply_if_changed(comp, "width", comp.width, w)
        _apply_if_changed(comp, "height", comp.height, h)


def _align_component(comp, axis: str, mode: str):
    """Align the visual bounding box to the parent rect edge/center."""
    canvas, parent_rect = _get_parent_rect(comp)
    if canvas is None or parent_rect is None:
        return

    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)
    vis = comp.get_visual_rect(cw, ch)
    if vis is None:
        return

    vis_x, vis_y, vis_w, vis_h = vis
    parent_x, parent_y, parent_w, parent_h = parent_rect

    if axis == "x":
        if mode == "left":
            vis_x = parent_x
        elif mode == "center":
            vis_x = parent_x + (parent_w - vis_w) * 0.5
        elif mode == "right":
            vis_x = parent_x + parent_w - vis_w
    elif axis == "y":
        if mode == "top":
            vis_y = parent_y
        elif mode == "middle":
            vis_y = parent_y + (parent_h - vis_h) * 0.5
        elif mode == "bottom":
            vis_y = parent_y + parent_h - vis_h

    _apply_visual_position(comp, vis_x, vis_y, canvas)


def _render_common_position(ctx, comp):
    if not render_compact_section_header(ctx, t("ui_comp.location"), level="primary"):
        return

    section_lw = max_label_w(ctx, [t("ui_comp.alignment"), t("ui_comp.position"), t("ui_comp.rotation")])

    # ── Alignment ──
    if render_compact_section_header(ctx, t("ui_comp.alignment"), level="secondary"):
        field_label(ctx, t("ui_comp.alignment"), section_lw)
        clicked = Theme.render_inline_button_row(
            ctx,
            "ui_align_row",
            [
                ("left", t("ui_comp.align_left")),
                ("center_x", t("ui_comp.align_cx")),
                ("right", t("ui_comp.align_right")),
                ("top", t("ui_comp.align_top")),
                ("middle", t("ui_comp.align_mid")),
                ("bottom", t("ui_comp.align_bot")),
            ],
        )
        if clicked == "left":
            _align_component(comp, "x", "left")
        elif clicked == "center_x":
            _align_component(comp, "x", "center")
        elif clicked == "right":
            _align_component(comp, "x", "right")
        elif clicked == "top":
            _align_component(comp, "y", "top")
        elif clicked == "middle":
            _align_component(comp, "y", "middle")
        elif clicked == "bottom":
            _align_component(comp, "y", "bottom")

    # ── Position (shows visual AABB position) ──
    if render_compact_section_header(ctx, t("ui_comp.position"), level="secondary"):
        canvas, cw, ch = _canvas_dims(comp)
        if canvas is None:
            ctx.label(t("ui_comp.no_canvas_context"))
        else:
            vis_x, vis_y, vis_w, vis_h = comp.get_visual_rect(cw, ch)
            new_x, new_y = ctx.vector2("Position", float(vis_x), float(vis_y), 1.0, section_lw)
            if float(new_x) != float(vis_x) or float(new_y) != float(vis_y):
                _apply_visual_position(comp, float(new_x), float(new_y), canvas)

    # ── Rotation ──
    if render_compact_section_header(ctx, t("ui_comp.rotation"), level="secondary"):
        field_label(ctx, t("ui_comp.rotation"), section_lw)
        new_rot = ctx.drag_float("##ui_rotation", float(comp.rotation), 1.0, -3600.0, 3600.0)
        if float(new_rot) != float(comp.rotation):
            _apply_if_changed(comp, "rotation", comp.rotation, float(new_rot))

        clicked = Theme.render_inline_button_row(
            ctx,
            "ui_rotation_row",
            [
                ("rotate_90", t("ui_comp.rotate_90")),
                ("mirror_x", t("ui_comp.mirror_h")),
                ("mirror_y", t("ui_comp.mirror_v")),
            ],
            active_items=[
                item_id for item_id, enabled in (
                    ("mirror_x", bool(getattr(comp, "mirror_x", False))),
                    ("mirror_y", bool(getattr(comp, "mirror_y", False))),
                ) if enabled
            ],
        )
        if clicked == "rotate_90":
            new_rot = float(comp.rotation) + 90.0
            _apply_if_changed(comp, "rotation", comp.rotation, new_rot)
        elif clicked == "mirror_x":
            _apply_if_changed(comp, "mirror_x", comp.mirror_x, not bool(comp.mirror_x))
        elif clicked == "mirror_y":
            _apply_if_changed(comp, "mirror_y", comp.mirror_y, not bool(comp.mirror_y))


def _sync_text_layout_from_ctx(ctx, text_comp: UIText):
    text = getattr(text_comp, "text", "")
    font_size = max(1.0, float(getattr(text_comp, "font_size", 24.0)))
    wrap_width = float(text_comp.get_editor_wrap_width()) if hasattr(text_comp, "get_editor_wrap_width") else float(text_comp.get_wrap_width())
    font_path = str(getattr(text_comp, "font_path", "") or "")
    line_height = float(getattr(text_comp, "line_height", 1.2))
    letter_spacing = float(getattr(text_comp, "letter_spacing", 0.0))
    pad_x, pad_y = getattr(text_comp, "get_auto_size_padding", lambda: (0.0, 0.0))()
    canvas = _find_canvas(text_comp)
    if wrap_width > 0.0:
        measured_w, measured_h = ctx.calc_text_size_wrapped(
            text, font_size, wrap_width, font_path, line_height, letter_spacing
        )
    else:
        measured_w, measured_h = ctx.calc_text_size(text, font_size, font_path, line_height, letter_spacing)

    if canvas is None:
        if text_comp.is_auto_width():
            _apply_if_changed(text_comp, "width", text_comp.width, max(1.0, float(measured_w) + float(pad_x)))
        elif text_comp.is_auto_height():
            _apply_if_changed(text_comp, "height", text_comp.height, max(1.0, float(measured_h) + float(pad_y)))
        return

    if text_comp.is_auto_width():
        _apply_size_preserve_top_left(text_comp, max(1.0, float(measured_w) + float(pad_x)), float(text_comp.height), canvas)
    elif text_comp.is_auto_height():
        _apply_size_preserve_top_left(text_comp, float(text_comp.width), max(1.0, float(measured_h) + float(pad_y)), canvas)


def _set_text_resize_mode(ctx, text_comp: UIText, mode: TextResizeMode):
    if text_comp.resize_mode == mode:
        return
    _apply_if_changed(text_comp, "resize_mode", text_comp.resize_mode, mode)
    if bool(text_comp.lock_aspect_ratio):
        _apply_if_changed(text_comp, "lock_aspect_ratio", text_comp.lock_aspect_ratio, False)
    _sync_text_layout_from_ctx(ctx, text_comp)


def _render_common_layout(ctx, comp):
    if not render_compact_section_header(ctx, t("ui_comp.layout"), level="primary"):
        return

    labels = [t("ui_comp.dimensions"), t("ui_comp.size"), t("ui_comp.modify")]
    if isinstance(comp, UIText):
        labels.append(t("ui_comp.resizing"))
    section_lw = max_label_w(ctx, labels)

    if isinstance(comp, UIText):
        _sync_text_layout_from_ctx(ctx, comp)

    if render_compact_section_header(ctx, t("ui_comp.dimensions"), level="secondary"):
        field_label(ctx, t("ui_comp.modify"), section_lw)

        # Build button list: Lock is always present;
        # Set Native Size appears for UIImage / UIButton with texture_path.
        modify_buttons = [("lock", t("ui_comp.lock"))]
        has_texture = bool(getattr(comp, "texture_path", "") or "")
        if has_texture:
            modify_buttons.append(("native_size", t("ui_comp.set_native_size")))

        clicked = Theme.render_inline_button_row(
            ctx,
            "ui_layout_lock_row",
            modify_buttons,
            active_items=["lock"] if bool(getattr(comp, "lock_aspect_ratio", False)) else [],
        )
        if clicked == "lock":
            new_lock = not bool(getattr(comp, "lock_aspect_ratio", False))
            if new_lock and isinstance(comp, UIText) and comp.resize_mode != TextResizeMode.FixedSize:
                _apply_if_changed(comp, "resize_mode", comp.resize_mode, TextResizeMode.FixedSize)
            _apply_if_changed(comp, "lock_aspect_ratio", comp.lock_aspect_ratio, new_lock)
        elif clicked == "native_size" and has_texture:
            _set_native_size(comp)

        field_label(ctx, t("ui_comp.size"), section_lw)
        size_x, size_y = ctx.vector2("Size", float(comp.width), float(comp.height), 1.0, section_lw)
        width_changed = float(size_x) != float(comp.width)
        height_changed = float(size_y) != float(comp.height)
        width_editable = not (isinstance(comp, UIText) and not comp.is_width_editable())
        height_editable = not (isinstance(comp, UIText) and not comp.is_height_editable())
        canvas, _, _ = _canvas_dims(comp)

        if width_changed and width_editable:
            old_w = max(float(comp.width), 1e-6)
            target_w = max(1.0, float(size_x))
            if bool(getattr(comp, "lock_aspect_ratio", False)):
                aspect = old_w / max(float(comp.height), 1e-6)
                target_h = max(1.0, float(size_x) / max(aspect, 1e-6))
                if canvas is None:
                    _apply_if_changed(comp, "width", comp.width, target_w)
                    _apply_if_changed(comp, "height", comp.height, target_h)
                else:
                    _apply_size_preserve_top_left(comp, target_w, target_h, canvas)
            elif canvas is None:
                _apply_if_changed(comp, "width", comp.width, target_w)
            elif isinstance(comp, UIText):
                _apply_size_preserve_top_left(comp, target_w, float(comp.height), canvas)
                _sync_text_layout_from_ctx(ctx, comp)
            else:
                _apply_size_preserve_top_left(comp, target_w, float(comp.height), canvas)

        if height_changed and height_editable:
            old_h = max(float(comp.height), 1e-6)
            target_h = max(1.0, float(size_y))
            if bool(getattr(comp, "lock_aspect_ratio", False)):
                aspect = float(comp.width) / max(old_h, 1e-6)
                target_w = max(1.0, float(size_y) * aspect)
                if canvas is None:
                    _apply_if_changed(comp, "width", comp.width, target_w)
                    _apply_if_changed(comp, "height", comp.height, target_h)
                else:
                    _apply_size_preserve_top_left(comp, target_w, target_h, canvas)
            elif canvas is None:
                _apply_if_changed(comp, "height", comp.height, target_h)
            elif isinstance(comp, UIText):
                _apply_size_preserve_top_left(comp, float(comp.width), target_h, canvas)
                _sync_text_layout_from_ctx(ctx, comp)
            else:
                _apply_size_preserve_top_left(comp, float(comp.width), target_h, canvas)

    if isinstance(comp, UIText) and render_compact_section_header(ctx, t("ui_comp.resizing"), level="secondary"):
        field_label(ctx, t("ui_comp.resizing"), section_lw)
        active = {
            TextResizeMode.AutoWidth: "auto_width",
            TextResizeMode.AutoHeight: "auto_height",
            TextResizeMode.FixedSize: "fixed_size",
        }.get(comp.resize_mode, "fixed_size")
        clicked = Theme.render_inline_button_row(
            ctx,
            "ui_text_resizing_row",
            [
                ("auto_width", t("ui_comp.auto_width")),
                ("auto_height", t("ui_comp.auto_height")),
                ("fixed_size", t("ui_comp.fixed_size")),
            ],
            active_items=[active],
        )
        if clicked == "auto_width":
            _set_text_resize_mode(ctx, comp, TextResizeMode.AutoWidth)
        elif clicked == "auto_height":
            _set_text_resize_mode(ctx, comp, TextResizeMode.AutoHeight)
        elif clicked == "fixed_size":
            _set_text_resize_mode(ctx, comp, TextResizeMode.FixedSize)


def _render_common_appearance(ctx, comp):
    if not render_compact_section_header(ctx, t("ui_comp.appearance"), level="primary"):
        return

    section_lw = max_label_w(ctx, [t("ui_comp.opacity"), t("ui_comp.corner_radius")])

    field_label(ctx, t("ui_comp.opacity"), section_lw)
    opacity_pct = max(0.0, min(100.0, float(getattr(comp, "opacity", 1.0)) * 100.0))
    new_opacity_pct = ctx.drag_float("##ui_opacity_pct", opacity_pct, 1.0, 0.0, 100.0)
    new_opacity = max(0.0, min(1.0, float(new_opacity_pct) / 100.0))
    if not math.isclose(new_opacity, float(getattr(comp, "opacity", 1.0)), rel_tol=1e-5, abs_tol=1e-6):
        _apply_if_changed(comp, "opacity", comp.opacity, new_opacity)

    field_label(ctx, t("ui_comp.corner_radius"), section_lw)
    is_text = isinstance(comp, UIText)
    if is_text and abs(float(getattr(comp, "corner_radius", 0.0))) > 1e-6:
        _apply_if_changed(comp, "corner_radius", comp.corner_radius, 0.0)

    if is_text:
        ctx.begin_disabled(True)
        ctx.drag_float("##ui_corner_radius_disabled", 0.0, 1.0, 0.0, 1000.0)
        ctx.end_disabled()
    else:
        new_radius = ctx.drag_float(
            "##ui_corner_radius",
            max(0.0, float(getattr(comp, "corner_radius", 0.0))),
            1.0,
            0.0,
            1000.0,
        )
        target_radius = max(0.0, float(new_radius))
        if not math.isclose(target_radius, float(getattr(comp, "corner_radius", 0.0)), rel_tol=1e-5, abs_tol=1e-6):
            _apply_if_changed(comp, "corner_radius", comp.corner_radius, target_radius)


def _render_text_typography(ctx, text_comp: UIText):
    IGUI = _igui()
    if not render_compact_section_header(ctx, t("ui_comp.typography"), level="primary"):
        return

    section_lw = max_label_w(ctx, [t("ui_comp.content"), t("ui_comp.font"), t("ui_comp.font_size"), t("ui_comp.line_height"), t("ui_comp.letter_spacing"), t("ui_comp.text_alignment")])

    field_label(ctx, t("ui_comp.content"), section_lw)
    current_text = str(getattr(text_comp, "text", "") or "")
    new_text = ctx.input_text_multiline("##ui_text_content", current_text, 16384, -1.0, 96.0, 0)
    if new_text != current_text:
        _apply_if_changed(text_comp, "text", current_text, new_text)
        _sync_text_layout_from_ctx(ctx, text_comp)

    field_label(ctx, t("ui_comp.font"), section_lw)
    font_values = [value for _, value in font_options]
    font_labels = [label for label, _ in font_options]
    if font_path not in font_values and font_path:
        font_labels.append(font_path)
        font_values.append(font_path)
    try:
        current_font_index = font_values.index(font_path)
    except ValueError:
        current_font_index = 0
    new_font_index = IGUI.searchable_combo(ctx, "ui_text_font_path", current_font_index, font_labels)
    new_font_path = font_values[new_font_index] if 0 <= new_font_index < len(font_values) else font_path
    if new_font_path != font_path:
        _apply_if_changed(text_comp, "font_path", font_path, new_font_path)
        _sync_text_layout_from_ctx(ctx, text_comp)

    field_label(ctx, t("ui_comp.font_size"), section_lw)
    target_font_size = max(4.0, min(1000000.0, float(new_font_size)))
    if not math.isclose(target_font_size, float(text_comp.font_size), rel_tol=1e-5, abs_tol=1e-6):
        _apply_if_changed(text_comp, "font_size", text_comp.font_size, target_font_size)
        _sync_text_layout_from_ctx(ctx, text_comp)

    field_label(ctx, t("ui_comp.line_height"), section_lw)
    target_line_height = max(0.5, min(5.0, float(new_line_height)))
    if not math.isclose(target_line_height, float(text_comp.line_height), rel_tol=1e-5, abs_tol=1e-6):
        _apply_if_changed(text_comp, "line_height", text_comp.line_height, target_line_height)
        _sync_text_layout_from_ctx(ctx, text_comp)

    field_label(ctx, t("ui_comp.letter_spacing"), section_lw)
    target_letter_spacing = max(-20.0, min(100.0, float(new_letter_spacing)))
    if not math.isclose(target_letter_spacing, float(text_comp.letter_spacing), rel_tol=1e-5, abs_tol=1e-6):
        _apply_if_changed(text_comp, "letter_spacing", text_comp.letter_spacing, target_letter_spacing)
        _sync_text_layout_from_ctx(ctx, text_comp)

    if render_compact_section_header(ctx, t("ui_comp.text_alignment"), level="secondary"):
        field_label(ctx, t("ui_comp.text_alignment"), section_lw)
        active_items = []
        horiz = getattr(text_comp, "text_align_h", TextAlignH.Left)
        vert = getattr(text_comp, "text_align_v", TextAlignV.Top)
        active_items.append({
            TextAlignH.Left: "text_left",
            TextAlignH.Center: "text_center_h",
            TextAlignH.Right: "text_right",
        }.get(horiz, "text_left"))
        active_items.append({
            TextAlignV.Top: "text_top",
            TextAlignV.Center: "text_center_v",
            TextAlignV.Bottom: "text_bottom",
        }.get(vert, "text_top"))

        clicked = Theme.render_inline_button_row(
            ctx,
            "ui_text_alignment_row",
            [
                ("text_left", t("ui_comp.text_left")),
                ("text_center_h", t("ui_comp.text_center")),
                ("text_right", t("ui_comp.text_right")),
                ("text_top", t("ui_comp.text_top")),
                ("text_center_v", t("ui_comp.text_middle")),
                ("text_bottom", t("ui_comp.text_bottom")),
            ],
            active_items=active_items,
        )
        if clicked == "text_left":
            _apply_if_changed(text_comp, "text_align_h", text_comp.text_align_h, TextAlignH.Left)
        elif clicked == "text_center_h":
            _apply_if_changed(text_comp, "text_align_h", text_comp.text_align_h, TextAlignH.Center)
        elif clicked == "text_right":
            _apply_if_changed(text_comp, "text_align_h", text_comp.text_align_h, TextAlignH.Right)
        elif clicked == "text_top":
            _apply_if_changed(text_comp, "text_align_v", text_comp.text_align_v, TextAlignV.Top)
        elif clicked == "text_center_v":
            _apply_if_changed(text_comp, "text_align_v", text_comp.text_align_v, TextAlignV.Center)
        elif clicked == "text_bottom":
            _apply_if_changed(text_comp, "text_align_v", text_comp.text_align_v, TextAlignV.Bottom)


def _render_text_fill(ctx, text_comp: UIText):
    if not render_compact_section_header(ctx, t("ui_comp.fill"), level="primary"):
        return

    section_lw = max_label_w(ctx, [t("ui_comp.color")])
    current = getattr(text_comp, "color", [1.0, 1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0, 1.0]
    while len(current) < 4:
        current = list(current) + [1.0]

    field_label(ctx, t("ui_comp.color"), section_lw)
    nr, ng, nb, na = _render_color_bar(ctx, "##ui_text_fill_color", current[0], current[1], current[2], current[3])
    new_color = [nr, ng, nb, na]
    if tuple(new_color) != tuple(current[:4]):
        _apply_if_changed(text_comp, "color", current[:4], new_color)


def _render_canvas_inspector(ctx, canvas: UICanvas):
    lw = max_label_w(ctx, [t("ui_comp.render_mode"), t("ui_comp.sort_order"), t("ui_comp.target_camera"), t("ui_comp.reference_size")])
    if render_compact_section_header(ctx, t("ui_comp.canvas"), level="secondary"):
        members = list(RenderMode)
        labels = [member.name for member in members]
        try:
            current_idx = members.index(canvas.render_mode)
        except ValueError:
            current_idx = 0
        field_label(ctx, t("ui_comp.render_mode"), lw)
        new_idx = ctx.combo("##canvas_render_mode", current_idx, labels, -1)
        new_render_mode = members[new_idx]
        _apply_if_changed(canvas, "render_mode", canvas.render_mode, new_render_mode)

        field_label(ctx, t("ui_comp.sort_order"), lw)
        new_sort_order = int(ctx.drag_int("##canvas_sort_order", int(canvas.sort_order), 1.0, -1000, 1000))
        _apply_if_changed(canvas, "sort_order", canvas.sort_order, new_sort_order)

        if canvas.render_mode == RenderMode.CameraOverlay:
            field_label(ctx, t("ui_comp.target_camera"), lw)
            new_target_camera = int(ctx.drag_int("##canvas_target_camera", int(canvas.target_camera_id), 1.0, -1, 1000000))
            _apply_if_changed(canvas, "target_camera_id", canvas.target_camera_id, new_target_camera)

        new_ref_w, new_ref_h = ctx.vector2("Reference Size", float(canvas.reference_width), float(canvas.reference_height), 1.0, lw)
        _apply_if_changed(canvas, "reference_width", canvas.reference_width, max(1, int(round(new_ref_w))))
        _apply_if_changed(canvas, "reference_height", canvas.reference_height, max(1, int(round(new_ref_h))))


def _render_text_inspector(ctx, text_comp: UIText):
    _render_common_position(ctx, text_comp)
    _render_common_layout(ctx, text_comp)
    _render_common_appearance(ctx, text_comp)
    _render_text_typography(ctx, text_comp)
    _render_text_fill(ctx, text_comp)


# ========================================================================
#  UIImage inspector
# ========================================================================

def _render_image_fill(ctx, img_comp: UIImage):
    if not render_compact_section_header(ctx, t("ui_comp.fill"), level="primary"):
        return

    section_lw = max_label_w(ctx, [t("ui_comp.texture"), t("ui_comp.color")])

    # ── Texture field with IGUI object field + picker ──
    field_label(ctx, t("ui_comp.texture"), section_lw)
    tex_path = str(getattr(img_comp, "texture_path", "") or "")
    display = os.path.basename(tex_path) if tex_path else t("igui.none")
    IGUI = _igui()
    from .inspector_components import _picker_assets

    def _on_tex_drop(payload):
        new_path = str(payload).replace("\\", "/")
        if new_path != tex_path:
            _apply_if_changed(img_comp, "texture_path", tex_path, new_path)

    def _on_tex_pick(picked_path):
        new_path = str(picked_path).replace("\\", "/")
        if new_path != tex_path:
            _apply_if_changed(img_comp, "texture_path", tex_path, new_path)

    def _on_tex_clear():
        if tex_path:
            _apply_if_changed(img_comp, "texture_path", tex_path, "")

    def _tex_asset_items(filt):
        return _picker_assets(filt, "*.png") + _picker_assets(filt, "*.jpg")

    IGUI.object_field(
        ctx,
        "ui_image_texture",
        display, "Texture",
        clickable=False,
        accept="TEXTURE_FILE",
        on_drop=_on_tex_drop,
        picker_asset_items=_tex_asset_items,
        on_pick=_on_tex_pick,
        on_clear=_on_tex_clear,
    )

    # ── Tint color ──
    current = getattr(img_comp, "color", [1.0, 1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0, 1.0]
    while len(current) < 4:
        current = list(current) + [1.0]
    field_label(ctx, t("ui_comp.color"), section_lw)
    nr, ng, nb, na = _render_color_bar(ctx, "##ui_image_fill_color", current[0], current[1], current[2], current[3])
    new_color = [nr, ng, nb, na]
    if tuple(new_color) != tuple(current[:4]):
        _apply_if_changed(img_comp, "color", current[:4], new_color)


def _render_image_inspector(ctx, img_comp: UIImage):
    _render_common_position(ctx, img_comp)
    _render_common_layout(ctx, img_comp)
    _render_common_appearance(ctx, img_comp)
    _render_image_fill(ctx, img_comp)


register_py_component_renderer("UICanvas", _render_canvas_inspector)
register_py_component_renderer("UIText", _render_text_inspector)
register_py_component_renderer("UIImage", _render_image_inspector)


# ── UIButton inspector ──

def _render_button_inspector(ctx, btn_comp: UIButton):
    IGUI = _igui()
    _render_common_position(ctx, btn_comp)
    _render_common_layout(ctx, btn_comp)
    _render_common_appearance(ctx, btn_comp)

    # ── 交互 Interaction ──
    if render_compact_section_header(ctx, t("ui_comp.interaction"), level="primary"):
        new_inter = render_inspector_checkbox(ctx, t("ui_comp.interactable"), btn_comp.interactable)
        _apply_if_changed(btn_comp, "interactable", btn_comp.interactable, new_inter)

        new_rc = render_inspector_checkbox(ctx, t("ui_comp.raycast_target"), btn_comp.raycast_target)
        _apply_if_changed(btn_comp, "raycast_target", btn_comp.raycast_target, new_rc)

    # ── 内容 Content ──
    if render_compact_section_header(ctx, t("ui_comp.content"), level="primary"):
        lw = max_label_w(ctx, [t("ui_comp.label"), t("ui_comp.font"), t("ui_comp.font_size"), t("ui_comp.label_color"),
                                t("ui_comp.line_height"), t("ui_comp.letter_spacing")])
        field_label(ctx, t("ui_comp.label"), lw)
        new_label = ctx.text_input("##btn_label", str(btn_comp.label or ""), 256)
        _apply_if_changed(btn_comp, "label", btn_comp.label, new_label)

        field_label(ctx, t("ui_comp.font"), lw)
        font_path = str(getattr(btn_comp, "font_path", "") or "")
        font_options = _get_project_font_options()
        font_values = [value for _, value in font_options]
        font_labels = [label_str for label_str, _ in font_options]
        if font_path not in font_values and font_path:
            font_labels.append(font_path)
            font_values.append(font_path)
        try:
            current_font_index = font_values.index(font_path)
        except ValueError:
            current_font_index = 0
        new_font_index = IGUI.searchable_combo(ctx, "btn_font_path", current_font_index, font_labels)
        new_font_path = font_values[new_font_index] if 0 <= new_font_index < len(font_values) else font_path
        if new_font_path != font_path:
            _apply_if_changed(btn_comp, "font_path", font_path, new_font_path)

        field_label(ctx, t("ui_comp.font_size"), lw)
        new_fs = ctx.drag_float("##btn_font_size", btn_comp.font_size, 0.5, 4.0, 256.0)
        _apply_if_changed(btn_comp, "font_size", btn_comp.font_size, new_fs)

        field_label(ctx, t("ui_comp.label_color"), lw)
        cur_lc = list(btn_comp.label_color or [1, 1, 1, 1])
        while len(cur_lc) < 4:
            cur_lc.append(1.0)
        nr, ng, nb, na = _render_color_bar(ctx, "##btn_label_color", cur_lc[0], cur_lc[1], cur_lc[2], cur_lc[3])
        new_lc = [nr, ng, nb, na]
        if tuple(new_lc) != tuple(cur_lc[:4]):
            _apply_if_changed(btn_comp, "label_color", cur_lc[:4], new_lc)

        field_label(ctx, t("ui_comp.line_height"), lw)
        target_lh = max(0.5, min(5.0, float(new_lh)))
        if not math.isclose(target_lh, float(btn_comp.line_height), rel_tol=1e-5, abs_tol=1e-6):
            _apply_if_changed(btn_comp, "line_height", btn_comp.line_height, target_lh)

        field_label(ctx, t("ui_comp.letter_spacing"), lw)
        target_ls = max(-20.0, min(100.0, float(new_ls)))
        if not math.isclose(target_ls, float(btn_comp.letter_spacing), rel_tol=1e-5, abs_tol=1e-6):
            _apply_if_changed(btn_comp, "letter_spacing", btn_comp.letter_spacing, target_ls)

        # Text alignment buttons
        if render_compact_section_header(ctx, t("ui_comp.text_alignment"), level="secondary"):
            field_label(ctx, t("ui_comp.text_alignment"), lw)
            active_items = []
            horiz = getattr(btn_comp, "text_align_h", TextAlignH.Center)
            vert = getattr(btn_comp, "text_align_v", TextAlignV.Center)
            active_items.append({
                TextAlignH.Left: "text_left",
                TextAlignH.Center: "text_center_h",
                TextAlignH.Right: "text_right",
            }.get(int(horiz), "text_center_h"))
            active_items.append({
                TextAlignV.Top: "text_top",
                TextAlignV.Center: "text_center_v",
                TextAlignV.Bottom: "text_bottom",
            }.get(int(vert), "text_center_v"))

            clicked = Theme.render_inline_button_row(
                ctx,
                "btn_text_alignment_row",
                [
                    ("text_left", t("ui_comp.text_left")),
                    ("text_center_h", t("ui_comp.text_center")),
                    ("text_right", t("ui_comp.text_right")),
                    ("text_top", t("ui_comp.text_top")),
                    ("text_center_v", t("ui_comp.text_middle")),
                    ("text_bottom", t("ui_comp.text_bottom")),
                ],
                active_items=active_items,
            )
            if clicked == "text_left":
                _apply_if_changed(btn_comp, "text_align_h", btn_comp.text_align_h, TextAlignH.Left)
            elif clicked == "text_center_h":
                _apply_if_changed(btn_comp, "text_align_h", btn_comp.text_align_h, TextAlignH.Center)
            elif clicked == "text_right":
                _apply_if_changed(btn_comp, "text_align_h", btn_comp.text_align_h, TextAlignH.Right)
            elif clicked == "text_top":
                _apply_if_changed(btn_comp, "text_align_v", btn_comp.text_align_v, TextAlignV.Top)
            elif clicked == "text_center_v":
                _apply_if_changed(btn_comp, "text_align_v", btn_comp.text_align_v, TextAlignV.Center)
            elif clicked == "text_bottom":
                _apply_if_changed(btn_comp, "text_align_v", btn_comp.text_align_v, TextAlignV.Bottom)

    # ── 填充 Fill ──
    if render_compact_section_header(ctx, t("ui_comp.fill"), level="primary"):
        lw = max_label_w(ctx, [t("ui_comp.texture"), t("ui_comp.background")])

        # Texture field with IGUI object field + picker
        field_label(ctx, t("ui_comp.texture"), lw)
        tex_path = str(getattr(btn_comp, "texture_path", "") or "")
        display = os.path.basename(tex_path) if tex_path else t("igui.none")
        IGUI = _igui()
        from .inspector_components import _picker_assets

        def _on_tex_drop(payload):
            new_path = str(payload).replace("\\", "/")
            if new_path != tex_path:
                _apply_if_changed(btn_comp, "texture_path", tex_path, new_path)

        def _on_tex_pick(picked_path):
            new_path = str(picked_path).replace("\\", "/")
            if new_path != tex_path:
                _apply_if_changed(btn_comp, "texture_path", tex_path, new_path)

        def _on_tex_clear():
            if tex_path:
                _apply_if_changed(btn_comp, "texture_path", tex_path, "")

        def _tex_asset_items(filt):
            return _picker_assets(filt, "*.png") + _picker_assets(filt, "*.jpg")

        IGUI.object_field(
            ctx,
            "btn_texture",
            display, "Texture",
            clickable=False,
            accept="TEXTURE_FILE",
            on_drop=_on_tex_drop,
            picker_asset_items=_tex_asset_items,
            on_pick=_on_tex_pick,
            on_clear=_on_tex_clear,
        )

        field_label(ctx, t("ui_comp.background"), lw)
        cur_bg = list(btn_comp.background_color or list(Theme.UI_DEFAULT_BUTTON_BG))
        while len(cur_bg) < 4:
            cur_bg.append(1.0)
        nr, ng, nb, na = _render_color_bar(ctx, "##btn_bg_color", cur_bg[0], cur_bg[1], cur_bg[2], cur_bg[3])
        new_bg = [nr, ng, nb, na]
        if tuple(new_bg) != tuple(cur_bg[:4]):
            _apply_if_changed(btn_comp, "background_color", cur_bg[:4], new_bg)

    # ── 色调 ColorTint ──
    if render_compact_section_header(ctx, t("ui_comp.color_tint"), level="secondary"):
        lw = max_label_w(ctx, [t("ui_comp.tint_normal"), t("ui_comp.tint_highlighted"), t("ui_comp.tint_pressed"), t("ui_comp.tint_disabled")])
        for label_str, field_name in [
            (t("ui_comp.tint_normal"), "normal_color"),
            (t("ui_comp.tint_highlighted"), "highlighted_color"),
            (t("ui_comp.tint_pressed"), "pressed_color"),
            (t("ui_comp.tint_disabled"), "disabled_color"),
        ]:
            field_label(ctx, label_str, lw)
            cur = list(getattr(btn_comp, field_name) or [1, 1, 1, 1])
            while len(cur) < 4:
                cur.append(1.0)
            nr, ng, nb, na = _render_color_bar(ctx, f"##btn_{field_name}", cur[0], cur[1], cur[2], cur[3])
            new_val = [nr, ng, nb, na]
            if tuple(new_val) != tuple(cur[:4]):
                _apply_if_changed(btn_comp, field_name, cur[:4], new_val)

    # ── 事件 Events — On Click () ──
    _render_on_click_events(ctx, btn_comp)


def _render_on_click_events(ctx, btn_comp):
    """Render the Unity-style On Click () persistent event list."""
    IGUI = _igui()
    from .inspector_components import render_object_field, _picker_scene_gameobjects, _create_component_ref_from_go
    from InfEngine.components.ref_wrappers import GameObjectRef, ComponentRef
    from InfEngine.ui.ui_event_entry import (
        UIEventEntry,
        UIEventArgument,
        get_callable_methods,
        get_method_parameter_specs,
        normalize_event_arguments,
    )

    def _clone_entry(e):
        """Clone a UIEventEntry without deepcopy (avoids pickling C++ objects)."""
        t = getattr(e, "target", None)
        pid = t.persistent_id if t and hasattr(t, "persistent_id") else 0
        return UIEventEntry(
            target=GameObjectRef(persistent_id=pid),
            component_name=getattr(e, "component_name", "") or "",
            method_name=getattr(e, "method_name", "") or "",
            arguments=[copy.deepcopy(arg) for arg in (getattr(e, "arguments", None) or [])],
        )

    def _clone_entries(lst):
        return [_clone_entry(e) for e in lst]

    entries = list(btn_comp.on_click_entries or [])

    def _on_add():
        new_entries = _clone_entries(entries)
        new_entries.append(UIEventEntry())
        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)

    header_open = IGUI.list_header(
        ctx, t("ui_comp.on_click"), len(entries),
        on_add=_on_add,
        level="primary",
    )
    if not header_open:
        return

    _BTN_W = 24.0
    remove_index = None
    changed = False
    lw = max_label_w(ctx, [t("ui_comp.target"), t("ui_comp.component"), t("ui_comp.method"), t("ui_comp.arguments")])

    for i, entry in enumerate(entries):
        ctx.push_id(i)

        # Collapsible per-entry header with right-aligned [-]
        entry_open = render_compact_section_header(
            ctx, t("ui_comp.click_entry").format(n=i + 1), level="tertiary", allow_overlap=True,
        )

        # Right-aligned remove button on the header row
        ctx.same_line(0, 0)
        avail_w = ctx.get_content_region_avail_width()
        if avail_w >= _BTN_W:
            ctx.set_cursor_pos_x(ctx.get_cursor_pos_x() + avail_w - _BTN_W)
        color_count = Theme.push_inline_button_style(ctx)
        if ctx.button(f"{Theme.ICON_MINUS}##click_rm_{i}", None,
                      width=_BTN_W, height=Theme.INSPECTOR_INLINE_BTN_H):
            remove_index = i
        ctx.pop_style_color(color_count)

        if entry_open:
            # ── Target GameObject ──
            target_ref = getattr(entry, "target", None)
            resolved_go = target_ref.resolve() if target_ref and hasattr(target_ref, "resolve") else None
            display = resolved_go.name if resolved_go else t("igui.none")

            def _make_drop_cb(_idx=i):
                def _on_drop(payload):
                    from InfEngine.lib import SceneManager
                    scene = SceneManager.instance().get_active_scene()
                    if not scene:
                        return
                    obj_id = int(payload) if isinstance(payload, (int, float)) else None
                    if obj_id is None:
                        return
                    go = scene.find_by_id(obj_id)
                    if go is None:
                        return
                    new_entries = _clone_entries(entries)
                    new_entries[_idx].target = GameObjectRef(go)
                    new_entries[_idx].component_name = ""
                    new_entries[_idx].method_name = ""
                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                return _on_drop

            def _make_pick_cb(_idx=i):
                def _on_pick(go):
                    new_entries = _clone_entries(entries)
                    new_entries[_idx].target = GameObjectRef(go)
                    new_entries[_idx].component_name = ""
                    new_entries[_idx].method_name = ""
                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                return _on_pick

            def _make_clear_cb(_idx=i):
                def _on_clear():
                    new_entries = _clone_entries(entries)
                    new_entries[_idx].target = GameObjectRef(persistent_id=0)
                    new_entries[_idx].component_name = ""
                    new_entries[_idx].method_name = ""
                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                return _on_clear

            field_label(ctx, t("ui_comp.target"), lw)
            render_object_field(
                ctx, f"onclick_target_{i}", display, "GameObject",
                clickable=False,
                accept_drag_type="HIERARCHY_GAMEOBJECT",
                on_drop_callback=_make_drop_cb(),
                picker_scene_items=lambda filt: _picker_scene_gameobjects(filt),
                on_pick=_make_pick_cb(),
                on_clear=_make_clear_cb(),
            )

            # ── Component combo ──
            comp_names = []
            if resolved_go:
                for py_comp in resolved_go.get_py_components():
                    cname = type(py_comp).__name__
                    if cname not in comp_names:
                        comp_names.append(cname)

            cur_comp_name = getattr(entry, "component_name", "") or ""
            comp_labels = [t("ui_comp.none")] + comp_names
            comp_values = [""] + comp_names
            try:
                comp_idx = comp_values.index(cur_comp_name)
            except ValueError:
                comp_idx = 0

            field_label(ctx, t("ui_comp.component"), lw)
            new_comp_name = comp_values[new_comp_idx] if 0 <= new_comp_idx < len(comp_values) else cur_comp_name
            if new_comp_name != cur_comp_name:
                new_entries = _clone_entries(entries)
                new_entries[i].component_name = new_comp_name
                new_entries[i].method_name = ""
                new_entries[i].arguments = []
                _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                entries = new_entries
                changed = True

            # ── Method combo ──
            method_names = []
            selected_component = None
            if resolved_go and cur_comp_name:
                for py_comp in resolved_go.get_py_components():
                    if type(py_comp).__name__ == cur_comp_name:
                        selected_component = py_comp
                        method_names = get_callable_methods(py_comp)
                        break

            cur_method = getattr(entry, "method_name", "") or ""
            method_labels = [t("ui_comp.none")] + method_names
            method_values = [""] + method_names
            try:
                method_idx = method_values.index(cur_method)
            except ValueError:
                method_idx = 0

            field_label(ctx, t("ui_comp.method"), lw)
            new_method = method_values[new_method_idx] if 0 <= new_method_idx < len(method_values) else cur_method
            if new_method != cur_method:
                new_entries = _clone_entries(entries)
                new_entries[i].method_name = new_method
                if selected_component is not None and new_method:
                    specs = get_method_parameter_specs(selected_component, new_method)
                    new_entries[i].arguments = normalize_event_arguments([], specs)
                else:
                    new_entries[i].arguments = []
                _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                entries = new_entries
                changed = True

            if selected_component is not None and (getattr(entry, "method_name", "") or ""):
                specs = get_method_parameter_specs(selected_component, getattr(entry, "method_name", "") or "")
                current_args = list(getattr(entry, "arguments", None) or [])
                normalized_args = normalize_event_arguments(current_args, specs)
                if normalized_args != current_args:
                    new_entries = _clone_entries(entries)
                    new_entries[i].arguments = normalized_args
                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                    entries = new_entries
                    entry = new_entries[i]
                    current_args = list(entry.arguments or [])
                    changed = True
                else:
                    current_args = normalized_args

                if specs:
                    if render_compact_section_header(ctx, t("ui_comp.arguments"), level="secondary"):
                        field_label(ctx, t("ui_comp.arguments"), lw)
                        ctx.label(t("ui_comp.params_count").format(n=len(specs)))
                        for arg_index, spec in enumerate(specs):
                            arg = current_args[arg_index]
                            ctx.push_id(arg_index)
                            label = f"{spec.display_name}"
                            kind = spec.kind

                            if kind == "bool":
                                new_value = render_inspector_checkbox(ctx, label, bool(arg.bool_value))
                                if bool(new_value) != bool(arg.bool_value):
                                    new_entries = _clone_entries(entries)
                                    new_entries[i].arguments[arg_index].bool_value = bool(new_value)
                                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                                    entries = new_entries
                                    current_args = list(new_entries[i].arguments or [])
                                    changed = True
                            elif kind == "int":
                                field_label(ctx, label, lw)
                                new_value = int(ctx.drag_int(f"##onclick_arg_int_{i}_{arg_index}", int(arg.int_value), 1.0, -2147483647, 2147483647))
                                if new_value != int(arg.int_value):
                                    new_entries = _clone_entries(entries)
                                    new_entries[i].arguments[arg_index].int_value = int(new_value)
                                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                                    entries = new_entries
                                    current_args = list(new_entries[i].arguments or [])
                                    changed = True
                            elif kind == "float":
                                field_label(ctx, label, lw)
                                new_value = float(ctx.drag_float(f"##onclick_arg_float_{i}_{arg_index}", float(arg.float_value), 0.1, -1000000.0, 1000000.0))
                                if not math.isclose(new_value, float(arg.float_value), rel_tol=1e-5, abs_tol=1e-6):
                                    new_entries = _clone_entries(entries)
                                    new_entries[i].arguments[arg_index].float_value = float(new_value)
                                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                                    entries = new_entries
                                    current_args = list(new_entries[i].arguments or [])
                                    changed = True
                            elif kind == "game_object":
                                target_ref = getattr(arg, "game_object", None)
                                resolved_arg_go = target_ref.resolve() if target_ref and hasattr(target_ref, "resolve") else None
                                display = resolved_arg_go.name if resolved_arg_go else t("igui.none")

                                def _make_arg_go_drop_cb(_entry_idx=i, _arg_idx=arg_index):
                                    def _on_drop(payload):
                                        from InfEngine.lib import SceneManager
                                        scene = SceneManager.instance().get_active_scene()
                                        if not scene:
                                            return
                                        obj_id = int(payload) if isinstance(payload, (int, float)) else None
                                        if obj_id is None:
                                            return
                                        go = scene.find_by_id(obj_id)
                                        if go is None:
                                            return
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].game_object = GameObjectRef(go)
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_drop

                                def _make_arg_go_pick_cb(_entry_idx=i, _arg_idx=arg_index):
                                    def _on_pick(go):
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].game_object = GameObjectRef(go)
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_pick

                                def _make_arg_go_clear_cb(_entry_idx=i, _arg_idx=arg_index):
                                    def _on_clear():
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].game_object = GameObjectRef(persistent_id=0)
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_clear

                                field_label(ctx, label, lw)
                                render_object_field(
                                    ctx, f"onclick_arg_go_{i}_{arg_index}", display, "GameObject",
                                    clickable=False,
                                    accept_drag_type="HIERARCHY_GAMEOBJECT",
                                    on_drop_callback=_make_arg_go_drop_cb(),
                                    picker_scene_items=lambda filt: _picker_scene_gameobjects(filt),
                                    on_pick=_make_arg_go_pick_cb(),
                                    on_clear=_make_arg_go_clear_cb(),
                                )
                            elif kind == "component":
                                comp_ref = getattr(arg, "component", None)
                                display = comp_ref.display_name if isinstance(comp_ref, ComponentRef) else t("igui.none")
                                type_hint = spec.component_type or "Component"

                                def _make_arg_comp_drop_cb(_entry_idx=i, _arg_idx=arg_index, _comp_type=spec.component_type):
                                    def _on_drop(payload):
                                        from InfEngine.lib import SceneManager
                                        scene = SceneManager.instance().get_active_scene()
                                        if not scene:
                                            return
                                        obj_id = int(payload) if isinstance(payload, (int, float)) else None
                                        if obj_id is None:
                                            return
                                        go = scene.find_by_id(obj_id)
                                        if go is None:
                                            return
                                        ref = _create_component_ref_from_go(go, _comp_type)
                                        if ref is None:
                                            return
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].component = ref
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_drop

                                def _make_arg_comp_pick_cb(_entry_idx=i, _arg_idx=arg_index, _comp_type=spec.component_type):
                                    def _on_pick(go):
                                        ref = _create_component_ref_from_go(go, _comp_type)
                                        if ref is None:
                                            return
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].component = ref
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_pick

                                def _make_arg_comp_clear_cb(_entry_idx=i, _arg_idx=arg_index, _comp_type=spec.component_type):
                                    def _on_clear():
                                        new_entries_local = _clone_entries(entries)
                                        new_entries_local[_entry_idx].arguments[_arg_idx].component = ComponentRef(component_type=_comp_type or "")
                                        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries_local)
                                    return _on_clear

                                field_label(ctx, label, lw)
                                render_object_field(
                                    ctx, f"onclick_arg_comp_{i}_{arg_index}", display, type_hint,
                                    clickable=False,
                                    accept_drag_type="HIERARCHY_GAMEOBJECT",
                                    on_drop_callback=_make_arg_comp_drop_cb(),
                                    picker_scene_items=lambda filt, _ct=spec.component_type: _picker_scene_gameobjects(filt, required_component=_ct),
                                    on_pick=_make_arg_comp_pick_cb(),
                                    on_clear=_make_arg_comp_clear_cb(),
                                )
                            else:
                                field_label(ctx, label, lw)
                                new_value = ctx.text_input(f"##onclick_arg_str_{i}_{arg_index}", str(arg.string_value or ""), 1024)
                                if new_value != str(arg.string_value or ""):
                                    new_entries = _clone_entries(entries)
                                    new_entries[i].arguments[arg_index].string_value = new_value
                                    _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)
                                    entries = new_entries
                                    current_args = list(new_entries[i].arguments or [])
                                    changed = True

                            ctx.pop_id()
                elif cur_method:
                    field_label(ctx, t("ui_comp.arguments"), lw)
                    ctx.label(t("ui_comp.no_parameters"))

        ctx.pop_id()

    if remove_index is not None:
        new_entries = [_clone_entry(e) for j, e in enumerate(entries) if j != remove_index]
        _apply_if_changed(btn_comp, "on_click_entries", entries, new_entries)


register_py_component_renderer("UIButton", _render_button_inspector)