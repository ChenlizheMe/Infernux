"""UIImage — a rectangular image UI element.

Hierarchy:
    InxComponent → InxUIComponent → InxUIScreenComponent → UIImage
"""

from enum import IntEnum
import math

from infernux.components import serialized_field, add_component_menu
from infernux.components.fields import FieldType
from .inx_ui_screen_component import InxUIScreenComponent
from .ui_sampled_texture import UISampledTextureField, ui_sampled_texture_source


class ImageType(IntEnum):
    SIMPLE = 0
    FILLED = 3


class ImageFillMethod(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    RADIAL90 = 2
    RADIAL180 = 3
    RADIAL360 = 4


@add_component_menu("UI/Image")
class UIImage(InxUIScreenComponent):
    """Screen or world image using a texture asset or live RenderTexture.

    Position and rotation come from the GameObject Transform. Width, height,
    opacity, corner radius and mirroring come from InxUIScreenComponent.
    """

    Type = ImageType
    FillMethod = ImageFillMethod
    image_type: ImageType = serialized_field(
        default=ImageType.SIMPLE,
        display_name_key="ui_comp.image_type",
        enum_labels=["ui_comp.image_type." + e.name for e in ImageType],
    )
    fill_method: ImageFillMethod = serialized_field(
        default=ImageFillMethod.RADIAL360,
        display_name_key="ui_comp.fill_method",
        enum_labels=["ui_comp.fill_method." + e.name for e in ImageFillMethod],
    )
    fill_amount: float = serialized_field(
        default=1.0,
        range=(0.0, 1.0),
        slider=True,
        display_name_key="ui_comp.fill_amount",
    )
    fill_origin: int = serialized_field(
        default=0, range=(0, 3), display_name_key="ui_comp.fill_origin"
    )
    fill_clockwise: bool = serialized_field(
        default=True, display_name_key="ui_comp.fill_clockwise"
    )

    def __setattr__(self, name, value):
        if name == "fill_amount":
            value = float(value)
            if math.isnan(value):
                raise ValueError("Image fill amount must not be NaN")
            value = max(0.0, min(1.0, value))
        if name == "fill_method":
            value = ImageFillMethod(value)
            changed = value != self.fill_method
            super().__setattr__(name, value)
            if changed:
                self.fill_origin = 0
            return
        super().__setattr__(name, value)

    # ── Fill ──
    texture = UISampledTextureField(
        name="texture",
        tooltip="Texture or RenderTexture asset (drag from Project panel)",
    )
    uv_rect = serialized_field(
        default=[0.0, 0.0, 1.0, 1.0],
        field_type=FieldType.VEC4,
        tooltip="Texture region: normalized X, Y, Width, Height",
        group="Fill",
    )
    color: list = serialized_field(
        default=[1.0, 1.0, 1.0, 1.0],
        field_type=FieldType.COLOR,
        hdr=True,
        tooltip="Tint color (RGBA)",
        group="Fill",
    )

    def _image_texture_source(self):
        """Keep static image draws on the GPU asset cache, not CPU pixel loads."""
        return ui_sampled_texture_source(self, type(self).texture)

    def on_validate(self):
        super().on_validate()
        if not 0 <= self.fill_origin <= (1 if int(self.fill_method) < 2 else 3):
            self.fill_origin = 0

    def _deserialize_fields_document(self, data, **kwargs):
        if isinstance(data, dict):
            data = dict(data)
            data.pop("texture_path", None)
        super()._deserialize_fields_document(data, **kwargs)
