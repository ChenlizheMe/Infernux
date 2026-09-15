"""
RenderStackPipeline — Engine-level entry point bridge to RenderStack.

This class inherits from ``RenderPipeline`` (the engine's existing render
pipeline callback) and acts as the sole coupling point between the engine's
render loop and the RenderStack system.

When the engine calls ``RenderPipeline.render()``, this class:
1. Finds the scene's RenderStack component
2. If found → delegates to ``RenderStack.render()``
3. If not found → executes the engine's default forward pipeline

Usage::

    context.set_render_pipeline(RenderStackPipeline())

The C++ engine side does not need to know about RenderStack — it only
interacts with the standard ``RenderPipeline`` interface.
"""

from __future__ import annotations

from Infernux.renderstack.render_pipeline import RenderPipeline


def _scene_cache_key(scene) -> tuple[int, str]:
    if scene is None:
        return (0, "")
    return (id(scene), str(getattr(scene, "name", "")))


class RenderStackPipeline(RenderPipeline):
    """Bridge between the engine render entry point and RenderStack.

    Each scene can have only one active RenderStack. When no RenderStack is
    present, the default forward path is the authoritative pipeline.
    """

    # Leading '_' keeps discover_pipelines() from listing this internal class.
    name: str = "_RenderStackBridge"

    def __init__(self) -> None:
        super().__init__()
        # The no-RenderStack case has one explicit default graph. It is not a
        # recovery path for a broken authored RenderStack.
        self._default_graphs = {}
        self._default_pipeline = None
        # One cache entry per loaded Scene. Multi-camera additive rendering may
        # alternate between scene owners every frame.
        self._cached_stacks: dict[tuple[int, str], tuple[int, object | None]] = {}

    def render(self, context, camera) -> None:
        """Render one engine-owned camera view."""
        render_stack = self._find_render_stack(context)

        if render_stack is not None:
            render_stack.render(context, camera)
        else:
            self._render_default(context, camera)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _find_render_stack(self, context):
        """Find the active RenderStack in the current scene.

        Lookup order:
        1. Per-Scene active RenderStack directory
        2. Per-Scene cached scan result, invalidated by ``structure_version``
        3. Full scene scan across Python components
        4. ``None`` to select the engine default renderer
        """
        from Infernux.renderstack.render_stack import RenderStack

        scene = context.scene
        if scene is None:
            return None

        # The render context owns scene authority.  A process-global cached
        # component may still be alive while a retained old world is being
        # retired, but it must never drive another scene's graph.
        inst = RenderStack.instance(scene)
        if inst is not None:
            return inst

        # Fast path: use cached scan result if structure hasn't changed
        scene_key = _scene_cache_key(scene)
        ver = scene.structure_version
        cached_entry = self._cached_stacks.get(scene_key)
        if cached_entry is not None and cached_entry[0] == ver:
            cached = cached_entry[1]
            if cached is not None and not RenderStack._is_effectively_active(
                cached,
                scene=scene,
            ):
                self._cached_stacks[scene_key] = (ver, None)
                return None
            if cached is not None:
                RenderStack.activate_instance(cached, scene)
            return cached

        # Slow path: scan scene (only when structure changes)
        found = None
        for obj in scene.get_all_objects():
            if not obj.is_active_in_hierarchy():
                continue
            for comp in obj.get_py_components():
                if isinstance(comp, RenderStack) and RenderStack._is_effectively_active(
                    comp,
                    scene=scene,
                ):
                    found = comp
                    break
            if found is not None:
                break

        self._cached_stacks[scene_key] = (ver, found)
        if found is not None:
            RenderStack.refresh_active_instance(scene)
        return found

    def _render_default(self, context, camera) -> None:
        """Authoritative rendering path used when no RenderStack exists.

        This builds a graph directly from ``DefaultForwardPipeline`` without
        injecting any user passes.
        """
        samples = context.output_samples
        description = self._default_graphs.get(samples)
        if description is None:
            from Infernux.rendergraph.graph import RenderGraph
            from Infernux.renderstack.default_forward_pipeline import (
                DefaultForwardPipeline,
            )

            if self._default_pipeline is None:
                self._default_pipeline = DefaultForwardPipeline()

            graph = RenderGraph("Default Forward", output_samples=samples)
            # Define topology (DefaultForwardPipeline inserts screen_ui_section)
            self._default_pipeline.define_topology(graph)
            graph.set_output("color")
            description = self._default_graphs[samples] = graph.build()
            from Infernux.debug import Debug

            screen_ui_passes = tuple(
                render_pass.name
                for render_pass in description.passes
                if "ScreenUI" in render_pass.name
            )
            Debug.log(
                "INFERNUX_RENDER_GRAPH_READY pipeline='Default Forward' "
                f"passes={len(description.passes)} "
                f"screen_ui={','.join(screen_ui_passes) or 'none'}"
            )

        if not context.render_compiled(camera, description.source_revision):
            context.render_with_graph(camera, description)
