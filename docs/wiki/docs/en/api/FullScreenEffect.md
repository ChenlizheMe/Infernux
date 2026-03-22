# FullScreenEffect

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

**Inherits from:** [RenderPass](RenderPass.md)

## Description

Base class for fullscreen post-processing effects.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `FullScreenEffect.__init__(enabled: bool = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| requires | `ClassVar[Set[str]]` |  *(read-only)* |
| modifies | `ClassVar[Set[str]]` |  *(read-only)* |
| menu_path | `ClassVar[str]` |  *(read-only)* |
| requires_per_frame_rebuild | `ClassVar[bool]` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `setup_passes(graph: RenderGraph, bus: ResourceBus) → None` | Override to add fullscreen passes to the render graph. |
| `get_shader_list() → List[str]` | Return shader paths required by this effect. |
| `inject(graph: RenderGraph, bus: ResourceBus) → None` | Inject this effect into the render graph. |
| `get_params_dict() → Dict[str, Any]` | Get serializable parameters as a dictionary. |
| `set_params_dict(params: Dict[str, Any]) → None` | Restore parameters from a dictionary. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Operators

| Method | Returns |
|------|------|
| `__repr__() → str` | `str` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for FullScreenEffect
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
