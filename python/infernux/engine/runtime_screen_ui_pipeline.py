"""Native render-pipeline integration for runtime Screen UI submission."""

from __future__ import annotations

from infernux.lib import RenderPipelineCallback


class RuntimeScreenUIRenderPipeline(RenderPipelineCallback):
    """Submit Screen UI before delegating each engine-owned camera render."""

    def __init__(self, submission, delegate) -> None:
        super().__init__()
        self._submission = submission
        self._delegate = delegate

    def should_render_view(self, view) -> bool:
        return self._delegate.should_render_view(view)

    def render(self, context, camera) -> None:
        completed = context.capturing_completed_world
        if not context._has_screen_ui_snapshot:
            self._submission.submit(
                force=completed,
                world_id=context.view.world_id,
                target_size=(context.view.output_width, context.view.output_height),
                camera_size=(context.view.pixel_width, context.view.pixel_height),
            )
            if completed and not self._submission.capture_completed(context):
                return
        self._delegate.render(context, camera)

    def dispose(self) -> None:
        self._delegate.dispose()
