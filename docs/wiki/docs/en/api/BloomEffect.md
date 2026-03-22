# BloomEffect

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

**Inherits from:** [FullScreenEffect](FullScreenEffect.md)

## Description

Unity-aligned Bloom post-processing effect.

Uses a progressive downsample/upsample chain with soft threshold
and scatter-based diffusion, matching Unity URP's Bloom implementation.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| threshold | `float` |  *(read-only)* |
| intensity | `float` |  *(read-only)* |
| scatter | `float` |  *(read-only)* |
| clamp | `float` |  *(read-only)* |
| tint_r | `float` |  *(read-only)* |
| tint_g | `float` |  *(read-only)* |
| tint_b | `float` |  *(read-only)* |
| max_iterations | `int` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `get_shader_list() → List[str]` |  |
| `setup_passes(graph: 'RenderGraph', bus: 'ResourceBus') → None` | Inject all bloom passes into the render graph. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for BloomEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
