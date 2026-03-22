# RenderPass

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

## Description

Base class for custom render passes injected into the render stack.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `RenderPass.__init__(enabled: bool = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` | The unique name of this render pass. *(read-only)* |
| injection_point | `str` | The injection point where this pass is inserted. *(read-only)* |
| default_order | `int` | Default execution order within the injection point. *(read-only)* |
| requires | `ClassVar[Set[str]]` | Resource names this pass reads from. *(read-only)* |
| modifies | `ClassVar[Set[str]]` | Resource names this pass writes to. *(read-only)* |
| creates | `ClassVar[Set[str]]` | Resource names this pass creates. *(read-only)* |
| enabled | `bool` | Whether this pass is currently enabled. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `inject(graph: RenderGraph, bus: ResourceBus) → None` | Inject render commands into the graph using the resource bus. |
| `validate(available_resources: Set[str]) → List[str]` | Validate that all required resources are available. |

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
# TODO: Add example for RenderPass
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
