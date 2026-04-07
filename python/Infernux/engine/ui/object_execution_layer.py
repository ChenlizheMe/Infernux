"""
Object execution layer for Inspector GameObject workflows.

This layer keeps object-resolution and object-level execution helpers out of the
Inspector UI rendering code, mirroring the asset execution layer architecture.
"""

from __future__ import annotations

from Infernux.debug import Debug


class ObjectExecutionLayer:
    """Execution helpers for GameObject selection and object-level actions."""

    @staticmethod
    def resolve_selected_object(selected_object_id: int):
        """Resolve selected GameObject ID to a live object, null-safe."""
        if not selected_object_id:
            return None
        try:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if scene:
                return scene.find_by_id(selected_object_id)
        except (ImportError, RuntimeError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        return None
