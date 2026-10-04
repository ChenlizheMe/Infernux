"""GizmosCollector — per-frame gizmo walk and upload.

Iterates Python and C++ components each frame, invokes gizmo callbacks
(:meth:`on_draw_gizmos`, :meth:`on_draw_gizmos_selected`), and uploads
accumulated geometry and icons to the native renderer.

Usage::

    collector = GizmosCollector()
    collector.collect_and_upload(engine)   # call each frame
    collector.invalidate_cache()           # on scene change

See Also:
    :class:`infernux.gizmos.Gizmos` for drawing primitives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.engine.engine import Engine

def notify_scene_changed() -> None:
    """Signal that the scene has changed; next frame rebuilds the icon cache."""
    ...

class GizmosCollector:
    """Stateful per-frame gizmo collector.

    Maintains a per-type cache of GameObjects to avoid full scene walks
    every frame.  Call :meth:`invalidate_cache` whenever objects are added
    or removed (scene reload, play-mode transitions).

    Example::

        collector = GizmosCollector()
        collector.invalidate_cache()             # on scene change
        collector.collect_and_upload(engine)      # each frame
    """

    def __init__(self) -> None: ...

    def invalidate_cache(self) -> None:
        """Invalidate the per-type GO cache (call on scene change / play-mode exit)."""
        ...

    def collect_and_upload(self, engine: Engine) -> None:
        """Walk the scene, invoke gizmo callbacks, upload geometry to C++.

        Args:
            engine: The live :class:`Engine` instance providing the native backend.
        """
        ...
