"""Editor scene transaction wrapper.

The commit algorithm lives in :mod:`runtime_scene_transaction` so packaged
Players never need to import editor Gizmo services. The editor wrapper keeps
the historical public API and publishes its additional scene-view invalidation
after Python registries have been rebuilt.
"""

from __future__ import annotations

from .runtime_scene_transaction import (
    SceneDocumentTransaction as _RuntimeSceneDocumentTransaction,
    SceneDocumentTransactionError,
    SceneDocumentTransactionState,
)


class SceneDocumentTransaction(_RuntimeSceneDocumentTransaction):
    """Complete an editor scene transaction and refresh editor Gizmos."""

    def __init__(self, scene, *, prepare_content=None, after_publish=None, **kwargs):
        self.content_reconciliation_count = 0

        def finish_content():
            if after_publish is not None:
                after_publish()
            if prepare_content is not None:
                self.content_reconciliation_count = int(bool(prepare_content()))

        super().__init__(scene, after_publish=finish_content, **kwargs)

    def poll(self, *, admit_assets: bool = False) -> bool:
        if (
            admit_assets
            and self.state is SceneDocumentTransactionState.DOCUMENT_READY
            and self._native_engine is not None
            and self._native_engine.has_renderer
            and self._asset_database is not None
        ):
            try:
                self._asset_database._complete_runtime_artifact_publication()
            except Exception as exc:
                self._fail(exc)
                return True
        if self._path is None or self.state is not SceneDocumentTransactionState.RESOURCES_READY:
            return super().poll(admit_assets=admit_assets)

        self._require_owner_thread()
        from infernux.engine.resources_manager import ResourcesManager

        resources = ResourcesManager.instance()
        if resources is None:
            return super().poll(admit_assets=admit_assets)
        try:
            resources.prepare_disk_scene_scripts(self._scene, self._document)
        except Exception as exc:
            self._fail(exc)
            self._error = f"script preflight: {self._error}"
            return True
        return super().poll(admit_assets=admit_assets)

    def _rebuild_python_registries(self) -> None:
        # Rollback restores the retained native world and its Python owners.
        super()._rebuild_python_registries()
        self._invalidate_editor_projections()

    def _reconcile_resident_python_registries(self) -> None:
        # Successful publication takes this path instead of the rollback-only
        # rebuild hook. A partial registry update still replaces native owners.
        super()._reconcile_resident_python_registries()
        self._invalidate_editor_projections()

    @staticmethod
    def _invalidate_editor_projections() -> None:
        from infernux.gizmos.collector import notify_scene_changed
        from infernux.engine.ui.inspector_snapshot import invalidate_rebuilt_scene

        invalidate_rebuilt_scene()
        notify_scene_changed()


__all__ = [
    "SceneDocumentTransaction",
    "SceneDocumentTransactionError",
    "SceneDocumentTransactionState",
]
