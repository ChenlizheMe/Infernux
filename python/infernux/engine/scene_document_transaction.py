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
