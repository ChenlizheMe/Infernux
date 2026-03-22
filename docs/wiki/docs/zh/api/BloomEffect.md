# BloomEffect

<div class="class-info">
类位于 <b>InfEngine.renderstack</b>
</div>

**继承自:** [FullScreenEffect](FullScreenEffect.md)

## 描述

泛光效果。让亮处溢出光晕——梦幻感拉满。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| threshold | `float` | 亮度阈值。 *(只读)* |
| intensity | `float` | 泛光强度。 *(只读)* |
| scatter | `float` | 散射范围。 *(只读)* |
| clamp | `float` |  *(只读)* |
| tint_r | `float` |  *(只读)* |
| tint_g | `float` |  *(只读)* |
| tint_b | `float` |  *(只读)* |
| max_iterations | `int` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `get_shader_list() → List[str]` |  |
| `setup_passes(graph: 'RenderGraph', bus: 'ResourceBus') → None` | Inject all bloom passes into the render graph. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for BloomEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
