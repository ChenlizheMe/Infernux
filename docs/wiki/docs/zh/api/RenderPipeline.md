# RenderPipeline

<div class="class-info">
类位于 <b>InfEngine.renderstack</b>
</div>

**继承自:** [SerializedFieldCollectorMixin](SerializedFieldCollectorMixin.md), [RenderPipelineCallback](RenderPipelineCallback.md)

## 描述

可编程渲染管线基类。继承它来定制整个渲染流程。

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## 构造函数

| 签名 | 描述 |
|------|------|
| `RenderPipeline.__init__()` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## 属性

| 名称 | 类型 | 描述 |
|------|------|------|
| name | `str` |  *(只读)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## 公共方法

| 方法 | 描述 |
|------|------|
| `render(context, cameras)` | 每帧调用，执行渲染。 |
| `should_render_camera(camera) → bool` | Decide whether *camera* should be rendered this frame. |
| `render_camera(context, camera, culling)` | Per-camera render hook. |
| `dispose()` | Override to release resources when the pipeline is replaced. |
| `define_topology(graph: 'RenderGraph') → None` | Define the rendering topology on *graph*. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## 示例

```python
# TODO: Add example for RenderPipeline
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## 另请参阅

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
