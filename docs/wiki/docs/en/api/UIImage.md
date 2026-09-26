# UIImage

<div class="class-info">
class in <b>Infernux.ui</b>
</div>

**Inherits from:** `InxUIScreenComponent`

## Description

Screen or world image using a texture asset or live camera output.

Position and rotation come from the GameObject Transform. The component
inherits size, opacity, corner radius and mirroring fields.

Attributes:
    texture: Imported Texture/RenderTexture, or a runtime RenderTexture override.
    color: Tint color as ``[R, G, B, A]`` (0–1 each).

Example::

    img = game_object.add_component(UIImage)
    img.texture = texture_asset
    img.color = [1.0, 1.0, 1.0, 0.8]

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| texture | `Texture | RenderTexture | None` |  |
| color | `list` |  |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Example

<!-- USER CONTENT START --> example
> **Example status:** No curated example has been verified for this symbol in 0.4.0. Use the signatures above; do not infer behavior from similarly named APIs in other engines.
<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
