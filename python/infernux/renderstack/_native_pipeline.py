"""Bind a Python pipeline definition to the native camera callback owner."""

from infernux.lib import RenderPipelineCallback


class NativeRenderPipeline(RenderPipelineCallback):
    def __init__(self, pipeline):
        super().__init__()
        self._pipeline = pipeline

    def prepare_world_ui(self, context):
        return self._pipeline.prepare_world_ui(context)

    def prepare_view_ui(self, context, view):
        return self._pipeline.prepare_view_ui(context, view)

    def collect_presentation_pose_objects(self):
        return self._pipeline.collect_presentation_pose_objects()

    def should_render_view(self, view):
        return self._pipeline.should_render_view(view)

    def render(self, context, view):
        self._pipeline.render(context, view)

    def dispose(self):
        self._pipeline.dispose()
