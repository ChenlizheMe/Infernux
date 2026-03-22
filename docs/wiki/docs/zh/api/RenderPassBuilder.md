# RenderPassBuilder

<div class="class-info">
类位于 <b>InfEngine.rendergraph</b>
</div>

## 描述

渲染 Pass 构建器。链式 API 定义输入输出。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `RenderPassBuilder.__init__(name: str, graph: RenderGraph | None = ...) → None` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` | The name of this render pass. *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `read(texture: str | TextureHandle) → RenderPassBuilder` | 声明此 Pass 读取某纹理。 |
| `write_color(texture: str | TextureHandle, slot: int = ...) → RenderPassBuilder` | Declare a color attachment output for this pass. |
| `write_depth(texture: str | TextureHandle) → RenderPassBuilder` | Declare a depth attachment output for this pass. |
| `set_texture(sampler_name: str, texture: str | TextureHandle) → RenderPassBuilder` | Bind a texture to a sampler input for this pass. |
| `set_clear(color: Optional[Tuple[float, float, float, float]] = ..., depth: Optional[float] = ...) → RenderPassBuilder` | Set clear values for color and/or depth attachments. |
| `draw_renderers(queue_range: Tuple[int, int] = ..., sort_mode: str = ..., pass_tag: str = ..., override_material: str = ...) → RenderPassBuilder` | Draw visible renderers filtered by queue range. |
| `draw_skybox() → RenderPassBuilder` | Draw the skybox in this pass. |
| `draw_shadow_casters(queue_range: Tuple[int, int] = ..., light_index: int = ..., shadow_type: str = ...) → RenderPassBuilder` | Draw shadow-casting geometry for a light. |
| `draw_screen_ui(list: str | int = ...) → RenderPassBuilder` | Draw screen-space UI elements in this pass. |
| `fullscreen_quad(shader: str) → RenderPassBuilder` | Draw a fullscreen quad with the specified shader. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 运算符

| 方法 | 返回值 |
|------|------|
| `__repr__() → str` | `str` |

<!-- USER CONTENT START --> operators

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for RenderPassBuilder
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
