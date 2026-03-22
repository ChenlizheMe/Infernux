# UICanvas

<div class="class-info">
类位于 <b>InfEngine.ui</b>
</div>

**继承自:** [InfUIComponent](InfUIComponent.md)

## 描述

UI 画布组件。所有 UI 元素的根容器——UI 的舞台。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| render_mode | `RenderMode` | 渲染模式。 *(只读)* |
| sort_order | `int` | 排序顺序。 *(只读)* |
| target_camera_id | `int` |  *(只读)* |
| reference_width | `int` |  *(只读)* |
| reference_height | `int` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `invalidate_element_cache()` | Mark the cached element list as stale. |
| `iter_ui_elements()` | Yield all screen-space UI components on child GameObjects (depth-first). |
| `raycast(canvas_x: float, canvas_y: float)` | Return the front-most element hit at (canvas_x, canvas_y), or None. |
| `raycast_all(canvas_x: float, canvas_y: float)` | Return all elements hit at (canvas_x, canvas_y), front-to-back order. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for UICanvas
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
