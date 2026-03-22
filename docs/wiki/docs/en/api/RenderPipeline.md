# RenderPipeline

<div class="class-info">
class in <b>InfEngine.renderstack</b>
</div>

**Inherits from:** [SerializedFieldCollectorMixin](SerializedFieldCollectorMixin.md), [RenderPipelineCallback](RenderPipelineCallback.md)

## Description

Base class for scriptable render pipelines.

The minimal subclass only needs ``define_topology()`` and optionally
``render_camera()`` for per-camera custom logic::

    class MyPipeline(RenderPipeline):
        name = "My Pipeline"

        def define_topology(self, graph):
            graph.create_texture("color", camera_target=True)
            graph.create_texture("depth", format=Format.D32_SFLOAT)
            with graph.add_pass("OpaquePass") as p:
                p.write_color("color")
                p.write_depth("depth")
                p.draw_renderers(queue_range=(0, 2500))
            graph.set_output("color")

Exposable parameters:
    Use class-level attributes (plain values or ``serialized_field()``)
    just like ``InfComponent``::

        class MyPipeline(RenderPipeline):
            shadow_resolution: int = serialized_field(default=2048, range=(256, 8192))
            enable_ssao: bool = True

    These are collected into ``_serialized_fields_`` and rendered by
    the RenderStack inspector.

RenderStack integration:
    Subclasses implement ``define_topology(graph)`` to declare passes
    and injection points inline.  The ``RenderGraph`` auto-records
    the topology sequence.

<!-- USER CONTENT START --> description

<!-- USER CONTENT END -->

## Constructors

| Signature | Description |
|------|------|
| `RenderPipeline.__init__()` |  |

<!-- USER CONTENT START --> constructors

<!-- USER CONTENT END -->

## Properties

| Name | Type | Description |
|------|------|------|
| name | `str` |  *(read-only)* |

<!-- USER CONTENT START --> properties

<!-- USER CONTENT END -->

## Public Methods

| Method | Description |
|------|------|
| `render(context, cameras)` | Render all cameras. |
| `should_render_camera(camera) → bool` | Decide whether *camera* should be rendered this frame. |
| `render_camera(context, camera, culling)` | Per-camera render hook. |
| `dispose()` | Override to release resources when the pipeline is replaced. |
| `define_topology(graph: 'RenderGraph') → None` | Define the rendering topology on *graph*. |

<!-- USER CONTENT START --> public_methods

<!-- USER CONTENT END -->

## Example

```python
# TODO: Add example for RenderPipeline
```

<!-- USER CONTENT START --> example

<!-- USER CONTENT END -->

## See Also

<!-- USER CONTENT START --> see_also

<!-- USER CONTENT END -->
