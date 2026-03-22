# UISelectable

<div class="class-info">
类位于 <b>InfEngine.ui</b>
</div>

**继承自:** [InfUIScreenComponent](InfUIScreenComponent.md)

## 描述

可选择的 UI 元素基类。UIButton 的老爸。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| interactable | `bool` | 是否可交互。 *(只读)* |
| transition | `UITransitionType` | 过渡类型。 *(只读)* |
| normal_color | `list` | 常态颜色。 *(只读)* |
| highlighted_color | `list` | 高亮颜色。 *(只读)* |
| pressed_color | `list` | 按下颜色。 *(只读)* |
| disabled_color | `list` | 禁用颜色。 *(只读)* |
| current_selection_state | `int` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `get_current_tint() → list` | Return the RGBA tint for the current visual state. |
| `on_pointer_enter(event_data)` |  |
| `on_pointer_exit(event_data)` |  |
| `on_pointer_down(event_data)` |  |
| `on_pointer_up(event_data)` |  |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 生命周期方法

| 方法 | 描述 |
|------|------|
| `awake()` |  |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for UISelectable
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
