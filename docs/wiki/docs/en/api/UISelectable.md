# UISelectable

<div class="class-info">
class in <b>InfEngine.ui</b>
</div>

**Inherits from:** [InfUIScreenComponent](InfUIScreenComponent.md)

## Description

Base class for interactive UI elements with visual feedback.

Subclass and override pointer hooks to build concrete widgets
(see ``UIButton``).

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| interactable | `bool` |  *(read-only)* |
| transition | `UITransitionType` |  *(read-only)* |
| normal_color | `list` |  *(read-only)* |
| highlighted_color | `list` |  *(read-only)* |
| pressed_color | `list` |  *(read-only)* |
| disabled_color | `list` |  *(read-only)* |
| current_selection_state | `int` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `get_current_tint() → list` | Return the RGBA tint for the current visual state. |
| `on_pointer_enter(event_data)` |  |
| `on_pointer_exit(event_data)` |  |
| `on_pointer_down(event_data)` |  |
| `on_pointer_up(event_data)` |  |

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
# TODO: Add example for UISelectable
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
