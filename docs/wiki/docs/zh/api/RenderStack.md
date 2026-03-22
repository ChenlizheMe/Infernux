# RenderStack

<div class="class-info">
类位于 <b>InfEngine.renderstack</b>
</div>

**继承自:** [InfComponent](InfComponent.md)

## 描述

后处理效果栈。管理一系列后处理 Pass 的执行顺序。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| pipeline_class_name | `str` |  *(只读)* |
| mounted_passes_json | `str` |  *(只读)* |
| pipeline_params_json | `str` |  *(只读)* |
| pipeline | `RenderPipeline` | The currently active render pipeline. *(只读)* |
| injection_points | `List[InjectionPoint]` | List of injection points defined by the pipeline. *(只读)* |
| pass_entries | `List[PassEntry]` | All mounted render pass entries. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
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

## 静态方法

| 方法 | 描述 |
|------|------|
| `static RenderStack.discover_pipelines() → Dict[str, type]` | Discover all available render pipeline classes. |

<!-- USER CONTENT START --> static_methods

<!-- USER CONTENT END -->

## 生命周期方法

| 方法 | 描述 |
|------|------|
| `awake() → None` | Initialize the render stack on component awake. |
| `on_destroy() → None` | Clean up the render stack when the component is destroyed. |

<!-- USER CONTENT START --> lifecycle_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for RenderStack
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
