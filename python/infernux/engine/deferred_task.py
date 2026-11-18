"""Deferred multi-frame task runner.

Allows long operations to be split into discrete steps that execute one per
frame, giving the status bar a chance to redraw between steps.

Usage::

    from infernux.engine.deferred_task import DeferredTaskRunner

    runner = DeferredTaskRunner.instance()
    runner.submit("Play Mode", [
        ("保存场景...",  0.2, save_fn),
        ("重建场景...",  0.6, rebuild_fn),
    ], on_done=lambda ok: EngineStatus.flash("Done", 1.0))

The runner advances once per frame: pre-GUI for graphical hosts, or the
pre-scene maintenance point for Headless hosts (both run and manual tick).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.engine.runtime_dispatch import ReloadableCallbackRef


class DeferredTaskRunner:
    """Execute a sequence of steps across multiple frames."""

    _instance: Optional[DeferredTaskRunner] = None

    @classmethod
    def instance(cls) -> DeferredTaskRunner:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._steps: list[tuple[str, float, Optional[ReloadableCallbackRef]]] = []
        self._task_name: str = ""
        self._on_done: Optional[ReloadableCallbackRef] = None
        self._index: int = 0
        self._failed: bool = False
        self._task_token = None
        self._task_lifetimes = []

    # ── public API ────────────────────────────────────────────────────

    @property
    def is_busy(self) -> bool:
        """True while a task is in progress."""
        return len(self._steps) > 0

    @property
    def active_task_name(self) -> str:
        """Stable name of the active task, or an empty string while idle."""
        return self._task_name if self.is_busy else ""

    @property
    def active_step_label(self) -> str:
        """Human-readable current step without exposing task internals."""
        if not self.is_busy or self._index >= len(self._steps):
            return ""
        return self._steps[self._index][0]

    def submit(
        self,
        task_name: str,
        steps: list[tuple[str, float, Optional[Callable[[], object]]]],
        on_done: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """Enqueue a multi-step task.  Returns False if already busy."""
        if self.is_busy:
            return False
        from infernux.engine.reload.owners import owner_for
        from infernux.engine.runtime_dispatch import ReloadableCallbackRef

        submitted_owner = owner_for()
        references = [
            (label, progress, ReloadableCallbackRef.normalize(callback) if callback is not None else None)
            for label, progress, callback in steps
        ]
        completion = ReloadableCallbackRef.normalize(on_done) if on_done is not None else None
        token = object()
        owners = {reference.reload_owner for _, _, reference in references if reference is not None}
        owners.add(submitted_owner)
        if completion is not None:
            owners.add(completion.reload_owner)
        owners.discard(None)
        if any(owner.closed for owner in owners):
            raise ReferenceError("Cannot defer work from a retired source")
        self._task_name = task_name
        self._steps = references
        self._on_done = completion
        self._index = 0
        self._failed = False
        self._task_token = token
        self._task_lifetimes = [owner.child() for owner in owners]
        for lifetime in self._task_lifetimes:
            lifetime.add_cleanup(lambda target=token: self._cancel_task(target))
        if self._task_token is not token:
            return False
        # Set the first step's status immediately so it's visible this frame
        if self._steps:
            label, progress, _ = self._steps[0]
            from infernux.engine.ui.engine_status import EngineStatus
            EngineStatus.set(label, progress, source="deferred", priority=10)
        else:
            self._finish()
        return True

    def tick(self) -> None:
        """Execute the next pending step (called once per frame)."""
        if not self._steps:
            return

        if self._index >= len(self._steps):
            # All steps done — finalise
            self._finish()
            return

        label, progress, reference = self._steps[self._index]
        token = self._task_token

        # Update status bar *before* executing (visible this frame)
        from infernux.engine.ui.engine_status import EngineStatus
        EngineStatus.set(label, progress, source="deferred", priority=10)

        # Execute the step
        if reference is not None and not self._failed:
            resolution = reference.resolve()
            scope = reference.execution_scope
            if not resolution.resolved or (scope is not None and scope.state == 'closed'):
                self.cancel()
                return
            if scope is not None and scope.state == 'paused':
                return
            try:
                from infernux.execution import component_execution

                owner = reference.reload_owner
                with (owner.enter() if owner is not None else nullcontext()), component_execution(scope):
                    result = resolution.callback()
                if self._task_token is token and result is False:
                    self._failed = True
            except Exception as exc:
                from infernux.debug import Debug
                Debug.log_exception(exc)
                if self._task_token is token:
                    self._failed = True

        # A running callback may retire its own source and cancel this task.
        if self._task_token is not token:
            return
        self._index += 1
        # The final operation is already complete. Finalizing here avoids an
        # empty extra editor frame before input and automation can observe the
        # transition as idle again.
        if self._index >= len(self._steps):
            self._finish()

    def cancel(self) -> None:
        """Cancel outstanding steps."""
        self._steps.clear()
        self._index = 0
        self._on_done = None
        self._failed = False
        self._task_name = ""
        self._task_token = None
        self._close_task_lifetimes()
        from infernux.engine.ui.engine_status import EngineStatus
        EngineStatus.clear(source="deferred")

    # ── internals ─────────────────────────────────────────────────────

    def _cancel_task(self, token) -> None:
        if self._task_token is token:
            self.cancel()

    def _close_task_lifetimes(self) -> None:
        lifetimes, self._task_lifetimes = self._task_lifetimes, []
        for lifetime in lifetimes:
            lifetime.close()

    def _finish(self) -> None:
        on_done = self._on_done
        ok = not self._failed
        self._steps.clear()
        self._index = 0
        self._on_done = None
        self._failed = False
        self._task_name = ""
        self._task_token = None
        self._close_task_lifetimes()
        from infernux.engine.ui.engine_status import EngineStatus
        EngineStatus.clear(source="deferred")
        if on_done:
            try:
                on_done.invoke(ok, propagate_exceptions=True)
            except Exception as exc:
                from infernux.debug import Debug
                Debug.log_exception(exc)
