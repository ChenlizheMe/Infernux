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
        super()._rebuild_python_registries()
        if not self._clear_registries:
            return
        from infernux.gizmos.collector import notify_scene_changed
        from infernux.engine.ui.inspector_snapshot import invalidate_rebuilt_scene

        invalidate_rebuilt_scene()
        notify_scene_changed()


__all__ = [
    "SceneDocumentTransaction",
    "SceneDocumentTransactionError",
    "SceneDocumentTransactionState",
]
