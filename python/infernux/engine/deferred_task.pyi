"""Deferred multi-frame task runner.

Allows long operations to be split into discrete steps that execute one per
frame, giving the status bar a chance to redraw between steps.

Example::

    runner = DeferredTaskRunner.instance()
    runner.submit("Play Mode", [
        ("Saving scene…",  0.2, save_fn),
        ("Rebuilding…",    0.6, rebuild_fn),
    ], on_done=lambda ok: print("Done", ok))
"""

from __future__ import annotations

from typing import Callable, Optional


class DeferredTaskRunner:
    """Execute a sequence of steps across multiple frames.

    Ticked once per frame by the engine's pre-GUI callback.
    """

    @classmethod
    def instance(cls) -> DeferredTaskRunner:
        """Return the singleton runner instance."""
        ...

    def __init__(self) -> None: ...

    @property
    def is_busy(self) -> bool:
        """``True`` while a task is in progress."""
        ...

    def submit(
        self,
        task_name: str,
        steps: list[tuple[str, float, Optional[Callable[[], object]]]],
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Enqueue a multi-step task.

        Args:
            task_name: Human-readable name shown in the status bar.
            steps: List of ``(label, progress_0_1, callable)`` tuples.
            on_done: Called with ``True`` on success, ``False`` on failure.

        Returns:
            ``False`` if the runner is already busy.
        """
        ...

    def tick(self) -> None:
        """Execute the next pending step (called once per frame)."""
        ...

    def cancel(self) -> None:
        """Cancel all outstanding steps."""
        ...
