"""Type stubs for Infernux.ui.ui_image — rectangular image UI element."""

from __future__ import annotations

from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
from Infernux.core.render_texture import RenderTexture
from Infernux.core.texture import Texture


class UIImage(InxUIScreenComponent):
    """Screen or world image using a texture asset or live camera output.

    Inherits ``x``, ``y``, ``width``, ``height``, ``opacity``,
    ``corner_radius``, ``rotation``, ``mirror_x``, ``mirror_y``
    from ``InxUIScreenComponent``.

    Attributes:
        texture: Imported Texture/RenderTexture, or a runtime RenderTexture override.
        texture_path: Legacy authoring shorthand; saves the texture GUID.
        color: Tint color as ``[R, G, B, A]`` (0–1 each).

    Example::

        img = game_object.add_component(UIImage)
        img.texture_path = "Assets/Textures/logo.png"
        img.color = [1.0, 1.0, 1.0, 0.8]
    """

    texture_path: str
    texture: Texture | RenderTexture | None
    color: list

# Unity-compatible public name. It intentionally aliases UIImage so both
# names share the same serialized type, renderer, and managed texture path.
UIRawImage = UIImage
