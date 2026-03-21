"""
RenderStackPipeline — Engine-level entry point bridge to RenderStack.

This class inherits from ``RenderPipeline`` (the engine's existing render
pipeline callback) and acts as the sole coupling point between the engine's
render loop and the RenderStack system.

When the engine calls ``RenderPipeline.render()``, this class:
1. Finds the scene's RenderStack component
2. If found → delegates to ``RenderStack.render()``
3. If not found → falls back to plain pipeline rendering (no pass injection)

Usage::

    context.set_render_pipeline(RenderStackPipeline())

The C++ engine side does not need to know about RenderStack — it only
interacts with the standard ``RenderPipeline`` interface.
"""

from __future__ import annotations

from InfEngine.renderstack.render_pipeline import RenderPipeline
from InfEngine.renderstack.resource_bus import ResourceBus


class RenderStackPipeline(RenderPipeline):
    """引擎级渲染入口 → RenderStack 的桥梁。

    一个场景只能有一个活跃的 RenderStack。
    如果没有 RenderStack，使用 fallback 路径（等价于 v0.1 行为）。
    """

    # 以 "_" 开头使 discover_pipelines() 跳过此内部类
    name: str = "_RenderStackBridge"

    def __init__(self) -> None:
        super().__init__()
        # Cached fallback graph (built lazily, invalidated never since
        # the fallback pipeline has no user passes to change).
        self._fallback_desc = None
        self._fallback_pipeline = None

    def render(self, context, cameras) -> None:
        """每帧由引擎调用。"""
        for camera in cameras:
            render_stack = self._find_render_stack(context)

            if render_stack is not None:
                render_stack.render(context, camera)
            else:
                self._render_fallback(context, camera)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _find_render_stack(self, context):
        """在场景中查找 RenderStack 组件。

        查找策略（按优先级）：
        1. 类级单例 ``RenderStack._active_instance``（O(1)）
        2. 遍历 ``context.scene`` 中所有 GameObject 的 Python 组件
        3. ``None``（使用 fallback）
        """
        from InfEngine.renderstack.render_stack import RenderStack

        # Fast path: class-level singleton
        inst = RenderStack.instance()
        if inst is not None:
            return inst

        # Slow path: scan scene (e.g. singleton not yet set because
        # awake() hasn't run yet)
        scene = context.scene

        if scene is None:
            return None

        for obj in scene.get_all_objects():
            for comp in obj.get_py_components():
                if isinstance(comp, RenderStack):
                    return comp

        return None

    def _render_fallback(self, context, camera) -> None:
        """无 RenderStack 时的 fallback 渲染。

        直接使用 DefaultForwardPipeline 的 ``define_topology()``
        构建 graph，不注入任何用户 Pass。等价于 v0.1 的行为。
        """
        context.setup_camera_properties(camera)
        culling = context.cull(camera)

        if self._fallback_desc is None:
            from InfEngine.rendergraph.graph import RenderGraph
            from InfEngine.renderstack.default_forward_pipeline import (
                DefaultForwardPipeline,
            )

            if self._fallback_pipeline is None:
                self._fallback_pipeline = DefaultForwardPipeline()

            graph = RenderGraph("Fallback")
            bus = ResourceBus()
            # Define topology (DefaultForwardPipeline inserts screen_ui_section)
            self._fallback_pipeline.define_topology(graph)
            graph.set_output("color")
            self._fallback_desc = graph.build()

        context.apply_graph(self._fallback_desc)
        context.submit_culling(culling)
