"""UIText — a text label UI element (Figma-style properties).

Hierarchy:
    InxComponent → InxUIComponent → InxUIScreenComponent → UIText
"""

from Infernux.components import serialized_field, list_field, add_component_menu
from Infernux.components.fields import FieldType
from .inx_ui_screen_component import InxUIScreenComponent
from .enums import TextAlignH, TextAlignV, TextOverflow, TextResizeMode


_TEXT_MEASURE_FIELDS = frozenset({
    "text", "font_path", "fallback_font_paths", "font_size", "line_height", "letter_spacing",
    "resize_mode", "width",
})


@add_component_menu("UI/Text")
class UIText(InxUIScreenComponent):
    """Figma-style text label rendered with ImGui draw primitives.

    Inherits x, y, width, height from InxUIScreenComponent.
    All fields carry ``group`` metadata so the generic inspector renderer
    displays them in collapsible sections automatically.
    """

    # ── Content ──
    text: str = serialized_field(
        default="New Text", tooltip="Display text",
        group="Content", multiline=True,
    )

    # ── Typography ──
    font_path: str = serialized_field(
        default="", tooltip="Optional font asset path (.ttf/.otf)",
        group="Typography",
    )
    fallback_font_paths: list = list_field(
        element_type=FieldType.STRING,
        tooltip="Ordered fallback font asset paths (.ttf/.otf)",
        group="Typography",
    )
    font_size: float = serialized_field(
        default=18.0, tooltip="Font size in canvas pixels",
        group="Typography", range=(4.0, 256.0), slider=False, drag_speed=0.5,
    )
    line_height: float = serialized_field(
        default=1.2, tooltip="Line height multiplier",
        group="Typography", range=(0.5, 5.0), slider=False, drag_speed=0.01,
    )
    letter_spacing: float = serialized_field(
        default=0.0, tooltip="Extra letter spacing in px",
        group="Typography", range=(-20.0, 100.0), slider=False, drag_speed=0.1,
    )

    # ── Alignment ──
    text_align_h: TextAlignH = serialized_field(
        default=TextAlignH.Left, tooltip="Horizontal alignment",
        group="Alignment",
    )
    text_align_v: TextAlignV = serialized_field(
        default=TextAlignV.Top, tooltip="Vertical alignment",
        group="Alignment",
    )

    # ── Overflow ──
    overflow: TextOverflow = serialized_field(
        default=TextOverflow.Visible, tooltip="Text overflow mode",
        group="Overflow",
    )

    resize_mode: TextResizeMode = serialized_field(
        default=TextResizeMode.FixedSize,
        tooltip="How the text clipping box resizes with content",
        group="Layout",
    )

    # ── Fill ──
    color: list = serialized_field(
        default=[1.0, 1.0, 1.0, 1.0],
        field_type=FieldType.COLOR,
        hdr=True,
        tooltip="Text color (RGBA)",
        group="Fill",
    )

    def __setattr__(self, name, value):
        if name not in _TEXT_MEASURE_FIELDS:
            super().__setattr__(name, value)
            return
        previous = getattr(self, name, object())
        super().__setattr__(name, value)
        if previous != value:
            object.__setattr__(self, "_text_layout_key", None)
            if name == 'resize_mode':
                from .inx_ui_screen_component import _invalidate_rect_cache

                _invalidate_rect_cache()

    def is_auto_width(self) -> bool:
        return self.resize_mode == TextResizeMode.AutoWidth

    def is_auto_height(self) -> bool:
        return self.resize_mode == TextResizeMode.AutoHeight

    def is_fixed_size(self) -> bool:
        return self.resize_mode == TextResizeMode.FixedSize

    def get_wrap_width(self) -> float:
        return 0.0 if self.is_auto_width() else max(1.0, float(self.width))

    def resolve_text_layout(self, measure_text, scale: float = 1.0) -> bool:
        """Resolve intrinsic text size without rewriting authored geometry.

        ``measure_text`` receives the scaled font size and wrap width and must
        return scaled pixel dimensions.  The stored result is normalized back
        into logical canvas pixels, so editor preview zoom and runtime Canvas
        scaling cannot leak into serialized ``width`` / ``height``.

        Returns ``True`` when the effective layout size changed.
        """
        scale = max(1e-6, float(scale))
        wrap_width = self.get_wrap_width()
        key = (
            str(self.text), str(self.font_path or ""), tuple(self.fallback_font_paths or ()), float(self.font_size),
            float(self.line_height), float(self.letter_spacing),
            float(wrap_width), scale,
        )
        if getattr(self, "_text_layout_key", None) == key:
            return False

        from .ui_render_dispatch import _resolve_font_asset_path

        arguments = (
            str(self.text),
            max(1.0, float(self.font_size) * scale),
            0.0 if wrap_width <= 0.0 else wrap_width * scale,
            _resolve_font_asset_path(self.font_path),
            float(self.line_height),
            float(self.letter_spacing) * scale,
        )
        fallback_paths = [
            _resolve_font_asset_path(path)
            for path in (self.fallback_font_paths or ())
        ]
        measured_width, measured_height = (
            measure_text(*arguments, fallback_paths)
            if fallback_paths
            else measure_text(*arguments)
        )
        resolved = (
            max(1.0, float(measured_width) / scale),
            max(1.0, float(measured_height) / scale),
        )
        previous_size = self.get_resolved_size()
        object.__setattr__(self, "_text_intrinsic_size", resolved)
        object.__setattr__(self, "_text_layout_key", key)
        changed = any(
            abs(float(a) - float(b)) > 0.01
            for a, b in zip(previous_size, self.get_resolved_size())
        )
        if changed:
            from .inx_ui_screen_component import _invalidate_rect_cache

            _invalidate_rect_cache()
        return changed

    def get_resolved_size(self) -> tuple[float, float]:
        """Return the effective logical box used by layout, draw, and input."""
        width = float(self.width)
        height = float(self.height)
        intrinsic = getattr(self, "_text_intrinsic_size", None)
        if intrinsic is not None:
            if self.is_auto_width():
                width = float(intrinsic[0])
            elif self.is_auto_height():
                height = float(intrinsic[1])
        return width, height

    def _layout_desired_size(self) -> tuple[float, float]:
        width, height = self.get_resolved_size()
        return (
            self._clamp_layout_extent(width, self.min_width, self.max_width),
            self._clamp_layout_extent(height, self.min_height, self.max_height),
        )

    def is_width_editable(self) -> bool:
        return not self.is_auto_width()

    def is_height_editable(self) -> bool:
        return not self.is_auto_height()
