# UIText

<div class="class-info">
class in <b>InfEngine.ui</b>
</div>

**Inherits from:** [InfUIScreenComponent](InfUIScreenComponent.md)

## Description

Figma-style text label rendered with ImGui draw primitives.

Inherits x, y, width, height from InfUIScreenComponent.
All fields carry ``group`` metadata so the generic inspector renderer
displays them in collapsible sections automatically.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| text | `str` |  *(read-only)* |
| font_path | `str` |  *(read-only)* |
| font_size | `float` |  *(read-only)* |
| line_height | `float` |  *(read-only)* |
| letter_spacing | `float` |  *(read-only)* |
| text_align_h | `TextAlignH` |  *(read-only)* |
| text_align_v | `TextAlignV` |  *(read-only)* |
| overflow | `TextOverflow` |  *(read-only)* |
| resize_mode | `TextResizeMode` |  *(read-only)* |
| color | `list` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `is_auto_width() → bool` |  |
| `is_auto_height() → bool` |  |
| `is_fixed_size() → bool` |  |
| `get_wrap_width() → float` |  |
| `get_layout_tolerance() → float` |  |
| `get_editor_wrap_width() → float` |  |
| `get_auto_size_padding() → tuple[float, float]` |  |
| `is_width_editable() → bool` |  |
| `is_height_editable() → bool` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for UIText
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
