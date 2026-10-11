"""SplashPlayer — animated splash sequence for standalone player builds.

Plays a sequence of images/videos with fade transitions before the
game starts.
"""

from __future__ import annotations

from typing import Dict, List


class SplashPlayer:
    """Plays splash items (images / video frame sequences) with fading."""

    def __init__(self, splash_items: List[Dict], data_root: str) -> None: ...

    def is_finished(self) -> bool:
        """``True`` when all splash items have been displayed."""
        ...

    def update(
        self,
        ctx: object,
        native_engine: object,
        x0: float,
        y0: float,
        w: float,
        h: float,
        dt: float,
    ) -> None:
        """Advance the splash animation and render the current frame.

        Args:
            ctx: ImGui rendering context.
            native_engine: Native C++ engine instance.
            x0: Viewport top-left X.
            y0: Viewport top-left Y.
            w: Viewport width.
            h: Viewport height.
            dt: Frame delta time in seconds.
        """
        ...

    def cleanup(self, native_engine: object) -> None:
        """Release GPU resources used by the splash player."""
        ...
