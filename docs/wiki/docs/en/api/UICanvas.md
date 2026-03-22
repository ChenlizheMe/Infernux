# UICanvas

<div class="class-info">
class in <b>InfEngine.ui</b>
</div>

**Inherits from:** [InfUIComponent](InfUIComponent.md)

## Description

Screen-space UI canvas.

reference_width / reference_height are the *design* reference resolution.
They are user-editable and default to 1920×1080.  At runtime the Game
View overlay scales all element positions, sizes and font sizes
proportionally from this reference to the actual viewport.

Attributes:
    render_mode: ScreenOverlay or CameraOverlay.
    sort_order: Rendering order (lower draws first).
    target_camera_id: Camera GameObject ID (CameraOverlay mode only).

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| render_mode | `RenderMode` |  *(read-only)* |
| sort_order | `int` |  *(read-only)* |
| target_camera_id | `int` |  *(read-only)* |
| reference_width | `int` |  *(read-only)* |
| reference_height | `int` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `invalidate_element_cache()` | Mark the cached element list as stale. |
| `iter_ui_elements()` | Yield all screen-space UI components on child GameObjects (depth-first). |
| `raycast(canvas_x: float, canvas_y: float)` | Return the front-most element hit at (canvas_x, canvas_y), or None. |
| `raycast_all(canvas_x: float, canvas_y: float)` | Return all elements hit at (canvas_x, canvas_y), front-to-back order. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for UICanvas
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
