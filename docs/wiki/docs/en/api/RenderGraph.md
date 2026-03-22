# RenderGraph

<div class="class-info">
class in <b>InfEngine.rendergraph</b>
</div>

## Description

A declarative render graph that defines texture resources and render passes.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `RenderGraph.__init__(name: str = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` | The name of this render graph. *(read-only)* |
| pass_count | `int` | Number of render passes in the graph. *(read-only)* |
| texture_count | `int` | Number of texture resources in the graph. *(read-only)* |
| topology_sequence | `List[Tuple[str, str]]` | Ordered list of (pass_name, type) entries defining the execution order. *(read-only)* |
| injection_points | `list` | List of injection points for pass extension. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_msaa_samples(samples: int) → None` | Set the MSAA sample count for all render targets. |
| `create_texture(name: str, format: Format = ..., camera_target: bool = ..., size: Optional[Tuple[int, int]] = ..., size_divisor: int = ...) → TextureHandle` | Declare a transient texture resource in the render graph. |
| `get_texture(name: str) → Optional[TextureHandle]` | Get a texture handle by name, or None if not found. |
| `has_pass(name: str) → bool` | Check if a render pass with the given name exists. |
| `has_injection_point(name: str) → bool` | Check if an injection point with the given name exists. |
| `injection_point(name: str, display_name: str = ..., resources: Optional[set] = ...) → None` | Declare an injection point where external passes can be inserted. |
| `screen_ui_section(resources: set | None = ...) → None` | Declare a screen UI section in the graph topology. |
| `add_pass(name: str) → RenderPassBuilder` | Add a new render pass to the graph. |
| `remove_pass(name: str) → RenderPassBuilder | None` | Remove a render pass by name. |
| `append_pass(builder: RenderPassBuilder) → None` | Append an existing RenderPassBuilder to the graph. |
| `set_output(texture: str | TextureHandle) → None` | Set the final output texture of the render graph. |
| `validate_no_ip_before_first_pass() → None` | Validate that no injection point appears before the first pass. |
| `build() → RenderGraphDescription` | Compile the graph into a RenderGraphDescription for the backend. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for RenderGraph
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
