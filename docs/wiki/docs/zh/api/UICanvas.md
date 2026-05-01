# UICanvas

<div class="class-info">
类位于 <b>Infernux.ui</b>
</div>

**继承自:** [InxUIComponent](InxUIComponent.md)

## 描述

UI 画布组件。所有 UI 元素的根容器——UI 的舞台。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| render_mode | `RenderMode` | 渲染模式。 |
| sort_order | `int` | 排序顺序。 |
| target_camera_id | `int` |  |
| reference_width | `int` |  |
| reference_height | `int` |  |
| ui_scale_mode | `UIScaleMode` |  |
| screen_match_mode | `ScreenMatchMode` |  |
| match_width_or_height | `float` |  |
| pixel_perfect | `bool` |  |
| reference_pixels_per_unit | `float` |  |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `compute_scale(screen_w: float, screen_h: float) → Tuple[float, float, float]` | Compute ``(scale_x, scale_y, text_scale)`` for a viewport size. |
| `invalidate_element_cache() → None` | Mark the cached element list as stale. |
| `iter_ui_elements() → Iterator[InxUIScreenComponent]` | Yield all screen-space UI components on child GameObjects (depth-first). |
| `raycast(canvas_x: float, canvas_y: float, tolerance: float = ...) → Optional[InxUIScreenComponent]` | Return the front-most element hit at ``(canvas_x, canvas_y)``, or ``None``. |
| `raycast_all(canvas_x: float, canvas_y: float, tolerance: float = ...) → List[InxUIScreenComponent]` | Return all elements hit at the given point, front-to-back order. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
```python
# TODO: Add example for UICanvas
```
<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
