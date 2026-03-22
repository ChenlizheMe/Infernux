# ToneMappingEffect

<div class="class-info">
类位于 <b>InfEngine.renderstack</b>
</div>

**继承自:** [FullScreenEffect](FullScreenEffect.md)

## 描述

色调映射效果。把 HDR 颜色压到屏幕可显示范围。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| mode | `ToneMappingMode` | 映射模式（ACES / Reinhard / Neutral 等）。 *(只读)* |
| exposure | `float` |  *(只读)* |
| gamma | `float` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `set_params_dict(params)` |  |
| `get_shader_list() → List[str]` |  |
| `setup_passes(graph: 'RenderGraph', bus: 'ResourceBus') → None` | Inject the tonemapping pass into the render graph. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for ToneMappingEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
