# UIButton

<div class="class-info">
类位于 <b>Infernux.ui</b>
</div>

**继承自:** [UISelectable](UISelectable.md)

## 描述

UI 按钮组件。用户点击的地方——程序员 Debug 的地方。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| label | `str` |  |
| font_size | `float` |  |
| font_path | `str` |  |
| label_color | `list` |  |
| text_align_h | `TextAlignH` |  |
| text_align_v | `TextAlignV` |  |
| line_height | `float` |  |
| letter_spacing | `float` |  |
| texture_path | `str` |  |
| background_color | `list` |  |
| on_click_entries | `List[UIEventEntry]` |  |
| on_click | `UIEvent` | 点击事件。 *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `on_pointer_click(event_data: PointerEventData) → None` | Internal — fires ``on_click`` and persistent entries on click. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 生命周期方法

| 方法 | 描述 |
|------|------|
| `awake() → None` |  |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## 示例

<!-- USER CONTENT START --> example
```python
# TODO: Add example for UIButton
```
<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
