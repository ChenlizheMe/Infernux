# RenderStack

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

**Inherits from:** [InfComponent](InfComponent.md)

## Description

Component that manages a stack of render passes driven by a pipeline.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| pipeline_class_name | `str` |  *(read-only)* |
| mounted_passes_json | `str` |  *(read-only)* |
| pipeline_params_json | `str` |  *(read-only)* |
| pipeline | `RenderPipeline` | The currently active render pipeline. *(read-only)* |
| injection_points | `List[InjectionPoint]` | List of injection points defined by the pipeline. *(read-only)* |
| pass_entries | `List[PassEntry]` | All mounted render pass entries. *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `set_pipeline(pipeline_class_name: str) → None` | Set the active render pipeline by class name. |
| `add_pass(render_pass: RenderPass) → bool` | Add a render pass to the stack. |
| `remove_pass(pass_name: str) → bool` | Remove a render pass by name. |
| `set_pass_enabled(pass_name: str, enabled: bool) → None` | Enable or disable a render pass by name. |
| `reorder_pass(pass_name: str, new_order: int) → None` | Change the execution order of a render pass. |
| `move_pass_before(dragged_name: str, target_name: str) → None` | Move a render pass to execute before another pass. |
| `get_passes_at(injection_point: str) → List[PassEntry]` | Get all pass entries at a specific injection point. |
| `invalidate_graph() → None` | Mark the render graph as dirty, triggering a rebuild. |
| `build_graph() → Any` | Build and return the render graph description. |
| `render(context: Any, camera: Any) → None` | Execute the render stack for a camera. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Static Methods

| Method | Description |
|------|------|
| `static RenderStack.discover_pipelines() → Dict[str, type]` | Discover all available render pipeline classes. |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## Lifecycle Methods

| Method | Description |
|------|------|
| `awake() → None` | Initialize the render stack on component awake. |
| `on_destroy() → None` | Clean up the render stack when the component is destroyed. |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for RenderStack
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
