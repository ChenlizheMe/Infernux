# UIButton

<div class="class-info">
class in <b>InfEngine.ui</b>
</div>

**Inherits from:** [UISelectable](UISelectable.md)

## Description

A clickable button with visual state feedback.

Combines **Image** (background) and **Text** (label) capabilities:

* Background can be a solid ``background_color`` or a ``texture_path`` image.
* Label text supports full typography: alignment, line-height, letter-spacing.
* Fires ``on_click`` when the user performs a full click (down + up).

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| label | `str` |  *(read-only)* |
| font_size | `float` |  *(read-only)* |
| font_path | `str` |  *(read-only)* |
| label_color | `list` |  *(read-only)* |
| text_align_h | `TextAlignH` |  *(read-only)* |
| text_align_v | `TextAlignV` |  *(read-only)* |
| line_height | `float` |  *(read-only)* |
| letter_spacing | `float` |  *(read-only)* |
| texture_path | `str` |  *(read-only)* |
| background_color | `list` |  *(read-only)* |
| on_click_entries | `list` |  *(read-only)* |
| on_click | `UIEvent` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `on_pointer_click(event_data)` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Lifecycle Methods

| Method | Description |
|------|------|
| `awake()` |  |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for UIButton
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
