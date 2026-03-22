# ToneMappingEffect

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

**Inherits from:** [FullScreenEffect](FullScreenEffect.md)

## Description

HDR-to-LDR tone mapping post-processing effect.

Should be the last effect in the post-process chain so that bloom
and other HDR effects can operate on the full dynamic range.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| mode | `ToneMappingMode` |  *(read-only)* |
| exposure | `float` |  *(read-only)* |
| gamma | `float` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_params_dict(params)` |  |
| `get_shader_list() → List[str]` |  |
| `setup_passes(graph: 'RenderGraph', bus: 'ResourceBus') → None` | Inject the tonemapping pass into the render graph. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for ToneMappingEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
