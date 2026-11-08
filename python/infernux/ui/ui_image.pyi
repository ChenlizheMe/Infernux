"""Type stubs for infernux.ui.ui_image — rectangular image UI element."""

from __future__ import annotations
from enum import IntEnum

from infernux.ui.inx_ui_screen_component import InxUIScreenComponent
from infernux.core.render_texture import RenderTexture
from infernux.core.texture import Texture


class ImageType(IntEnum):
    SIMPLE = 0
    FILLED = 3


class ImageFillMethod(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    RADIAL90 = 2
    RADIAL180 = 3
    RADIAL360 = 4


class UIImage(InxUIScreenComponent):
    """Screen or world image using a texture asset or live camera output.

    Position and rotation come from the GameObject Transform. The component
    inherits size, opacity, corner radius and mirroring fields.

    Attributes:
        texture: Imported Texture/RenderTexture, or a runtime RenderTexture override.
        color: Tint color as ``[R, G, B, A]`` (0–1 each).

    Example::

        img = game_object.add_component(UIImage)
        img.texture = texture_asset
        img.color = [1.0, 1.0, 1.0, 0.8]
    """

    texture: Texture | RenderTexture | None
    color: list
    Type = ImageType
    FillMethod = ImageFillMethod
    image_type: ImageType
    fill_method: ImageFillMethod
    fill_amount: float
    fill_origin: int
    fill_clockwise: bool

# Unity-compatible public name. It intentionally aliases UIImage so both
# names share the same serialized type, renderer, and managed texture path.
UIRawImage = UIImage
