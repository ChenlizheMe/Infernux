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
from typing import Callable, Optional, List, Tuple


class DeferredTaskRunner:
    """Execute a sequence of steps across multiple frames."""

    _instance: Optional[DeferredTaskRunner] = None

    @classmethod
    def instance(cls) -> DeferredTaskRunner:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._steps: List[Tuple[str, float, Optional[Callable[[], object]]]] = []
        self._task_name: str = ""
        self._on_done: Optional[Callable[[bool], None]] = None
        self._index: int = 0
        self._failed: bool = False
        self._continuation = None
        self._step_owners: List[object] = []
        self._on_done_owner = None
        self._task_token = None
        self._task_lifetimes: List[object] = []

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

        owners = []
        step_owners = []
        for _label, _progress, callback in steps:
            owner = owner_for(callback) if callback is not None else None
            if owner is not None and owner.closed:
                raise ReferenceError("Cannot defer work from a retired source")
            step_owners.append(owner)
            if owner is not None:
                owners.append(owner)
        completion_owner = owner_for(on_done) if on_done is not None else None
        if completion_owner is not None and completion_owner.closed:
            raise ReferenceError("Cannot defer work from a retired source")
        if completion_owner is not None:
            owners.append(completion_owner)
        token = object()
        self._task_name = task_name
        self._steps = list(steps)
        self._step_owners = step_owners
        self._on_done = on_done
        self._on_done_owner = completion_owner
        self._index = 0
        self._failed = False
        self._task_token = token
        self._task_lifetimes = []
        seen = set()
        for owner in owners:
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            lifetime = owner.child()
            lifetime.add_cleanup(lambda target=token: self._cancel_task(target))
            self._task_lifetimes.append(lifetime)
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

        label, progress, fn = self._steps[self._index]
        owner = self._step_owners[self._index] if self._index < len(self._step_owners) else None
        token = self._task_token
        if owner is not None:
            if owner.closed:
                self.cancel()
                return
            if not owner.active:
                return

        # Update status bar *before* executing (visible this frame)
        from infernux.engine.ui.engine_status import EngineStatus
        EngineStatus.set(label, progress, source="deferred", priority=10)

        # Execute the step
        if fn is not None and not self._failed:
            try:
                with (owner.enter() if owner is not None else nullcontext()):
                    if self._continuation is None:
                        result = fn()
                        if result is False:
                            self._failed = True
                        elif hasattr(result, '__next__'):
                            self._continuation = result
                    if self._continuation is not None:
                        try:
                            next(self._continuation)
                        except StopIteration as completed:
                            self._continuation = None
                            if completed.value is False:
                                self._failed = True
                        else:
                            return
            except Exception as exc:
                from infernux.debug import Debug
                Debug.log_exception(exc)
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
        continuation, self._continuation = self._continuation, None
        if continuation is not None:
            continuation.close()
        self._steps.clear()
        self._step_owners = []
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
        continuation, self._continuation = self._continuation, None
        if continuation is not None:
            continuation.close()
        on_done = self._on_done
        ok = not self._failed
        self._steps.clear()
        self._step_owners = []
        self._index = 0
        self._on_done = None
        self._failed = False
        self._task_name = ""
        self._task_token = None
        self._close_task_lifetimes()
        from infernux.engine.ui.engine_status import EngineStatus
        EngineStatus.clear(source="deferred")
        if on_done:
            owner = self._on_done_owner
            self._on_done_owner = None
            if owner is not None and not owner.active:
                return
            try:
                with (owner.enter() if owner is not None else nullcontext()):
                    on_done(ok)
            except Exception as exc:
                from infernux.debug import Debug
                Debug.log_exception(exc)
