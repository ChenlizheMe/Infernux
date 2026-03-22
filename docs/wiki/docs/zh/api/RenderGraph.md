# RenderGraph

<div class="class-info">
类位于 <b>InfEngine.rendergraph</b>
</div>

## 描述

声明式渲染图。用 Pass 描述你想怎么画，引擎帮你调度。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `RenderGraph.__init__(name: str = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` | The name of this render graph. *(只读)* |
| pass_count | `int` | Number of render passes in the graph. *(只读)* |
| texture_count | `int` | Number of texture resources in the graph. *(只读)* |
| topology_sequence | `List[Tuple[str, str]]` | Ordered list of (pass_name, type) entries defining the execution order. *(只读)* |
| injection_points | `list` | List of injection points for pass extension. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `set_msaa_samples(samples: int) → None` | Set the MSAA sample count for all render targets. |
| `create_texture(name: str, format: Format = ..., camera_target: bool = ..., size: Optional[Tuple[int, int]] = ..., size_divisor: int = ...) → TextureHandle` | 创建临时纹理。 |
| `get_texture(name: str) → Optional[TextureHandle]` | Get a texture handle by name, or None if not found. |
| `has_pass(name: str) → bool` | Check if a render pass with the given name exists. |
| `has_injection_point(name: str) → bool` | Check if an injection point with the given name exists. |
| `injection_point(name: str, display_name: str = ..., resources: Optional[set] = ...) → None` | Declare an injection point where external passes can be inserted. |
| `screen_ui_section(resources: set | None = ...) → None` | Declare a screen UI section in the graph topology. |
| `add_pass(name: str) → RenderPassBuilder` | 添加一个渲染 Pass。 |
| `remove_pass(name: str) → RenderPassBuilder | None` | Remove a render pass by name. |
| `append_pass(builder: RenderPassBuilder) → None` | Append an existing RenderPassBuilder to the graph. |
| `set_output(texture: str | TextureHandle) → None` | Set the final output texture of the render graph. |
| `validate_no_ip_before_first_pass() → None` | Validate that no injection point appears before the first pass. |
| `build() → RenderGraphDescription` | Compile the graph into a RenderGraphDescription for the backend. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for RenderGraph
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
