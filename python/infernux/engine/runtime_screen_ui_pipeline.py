"""Native render-pipeline integration for runtime Screen UI submission."""

from __future__ import annotations

from infernux.lib import RenderPipelineCallback
from .runtime_worlds import current_world_runtime


class RuntimeScreenUIRenderPipeline(RenderPipelineCallback):
    """Submit Screen UI before delegating each engine-owned camera render."""

    def __init__(self, submission, delegate) -> None:
        super().__init__()
        self._submission = submission
        self._delegate = delegate

    def collect_presentation_pose_objects(self):
        return self._submission.collect_presentation_pose_objects()

    def _capture_empty(self, context, camera_id=0):
        world = current_world_runtime()
        if world is None or world.scheduler.may_have_ui_components:
            return False
        self._submission.capture_empty(context, world.identity, camera_id)
        return True

    def prepare_world_ui(self, context):
        from infernux.lib._Infernux import _PresentationPosePending
        try:
            if not context._has_screen_ui_snapshot:
                if self._capture_empty(context):
                    return True
                self._submission.submit(
                    force=True, world_id=context._ui_world_id, target_size=(1, 1), camera_size=(1, 1),
                    prepared_pose_reads=bool(context._ui_author_revision),
                    capture_key=(context._ui_world_id, context._ui_world_epoch,
                                 context._ui_author_revision or -context._ui_completed_frame, 0),
                    world_only=True,
                )
            return self._submission.capture_completed(context)
        except _PresentationPosePending:
            return False

    def prepare_view_ui(self, context, view):
        from infernux.lib._Infernux import _PresentationPosePending
        try:
            if not context._has_screen_ui_snapshot:
                if self._capture_empty(context, view.camera_id):
                    return True
                self._submission.submit(
                    force=True, world_id=view.world_id,
                    target_size=(view.output_width, view.output_height),
                    camera_size=(view.pixel_width, view.pixel_height),
                    prepared_pose_reads=True,
                    capture_key=(view.world_id, view.simulation_epoch, view.presentation_revision, view.camera_id),
                    include_world=False,
                )
            return self._submission.capture_completed(context)
        except _PresentationPosePending:
            return False

    def should_render_view(self, view) -> bool:
        return self._delegate.should_render_view(view)

    def render(self, context, camera) -> None:
        completed = context.capturing_completed_world
        published_world_ui = context._uses_published_world_ui
        if (not context._has_screen_ui_snapshot and not context._uses_shared_scene_ui
                and not (completed and self._capture_empty(context, camera.camera_id))):
            self._submission.submit(
                force=completed,
                world_id=camera.world_id,
                target_size=(camera.output_width, camera.output_height),
                camera_size=(camera.pixel_width, camera.pixel_height),
                include_world=not published_world_ui,
            )
            if completed and not self._submission.capture_completed(context):
                return
        self._delegate.render(context, camera)

    def dispose(self) -> None:
        self._delegate.dispose()
