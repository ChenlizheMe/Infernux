"""Runtime capabilities and callbacks owned by the shared frame scheduler."""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Literal, get_args
from infernux.execution import capture_execution_scope, component_execution, execution_phase, simulation_state

_lock = RLock()
_services: dict[str, Any] = {}
UpdatePhase = Literal["update", "late_update", "fixed_update", "editor_update",
                      "physics_pre_step", "physics_post_step", "service_update", "render_submitted",
                      "render_presented"]
_update_callbacks: dict[str, tuple[UpdateRegistration, ...]] = {
    phase: () for phase in get_args(UpdatePhase)
}


class UpdateRegistration:
    """One frame callback; disable it when idle and close it when its owner unloads."""

    __slots__ = ("_callback", "_phase", "_unscaled", "_enabled", "_closed", "_scope")

    def __init__(self, callback: Callable[[float], None], phase: UpdatePhase, unscaled: bool) -> None:
        from infernux.engine.runtime_dispatch import (
            _apply_partial_arguments, _partial_callback_target, _PublishedPythonCallback,
        )

        target, arguments = _partial_callback_target(callback)
        binding = _PublishedPythonCallback.bind(target)
        self._callback = (callback if binding is None else
                          _apply_partial_arguments(binding.invoke_if_available, arguments))
        self._phase = phase
        self._unscaled = unscaled
        self._enabled = False
        self._closed = False
        candidate = getattr(target, '__self__', None)
        from infernux.components import InxComponent
        self._scope = capture_execution_scope(candidate if isinstance(candidate, InxComponent) else None)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._closed and enabled:
            raise RuntimeError("Update registration is closed")
        if enabled == self._enabled:
            return
        self._enabled = enabled
        global _update_callbacks
        previous = _update_callbacks[self._phase]
        current = previous + (self,) if enabled else tuple(item for item in previous if item is not self)
        _update_callbacks = {**_update_callbacks, self._phase: current}
        if bool(previous) != bool(current) and self._phase not in (
                'service_update', 'render_submitted', 'render_presented'):
            from infernux.components._component_lifecycle import RuntimeExecutionScheduler

            RuntimeExecutionScheduler.set_runtime_service_phase_active(
                "public-updates:" + self._phase,
                "update" if self._phase == "editor_update" else self._phase,
                bool(current),
            )

    def close(self) -> None:
        self.enabled = False
        self._closed = True
        self._callback = None
        self._scope = None


def register_update(
    callback: Callable[[float], None], *, phase: UpdatePhase = "update",
    unscaled: bool = False, enabled: bool = True,
) -> UpdateRegistration:
    """Register on the engine thread, after component callbacks of the chosen phase.

    New registrations join the next frame snapshot. Disabling/closing suppresses
    pending calls immediately. Normal/Late/Fixed callbacks only run in Play;
    Editor callbacks only run in Edit mode and receive the editor frame duration.
    Physics pre/post callbacks follow real fixed steps. service_update follows
    owner opportunities in all modes; render_submitted follows actual foreground
    submissions. render_presented follows accepted window presentations.
    Read Clock("presentation").now in that callback for the occurrence receipt.
    These external phases preserve the component's execution permission.
    Independent Fixed callbacks still follow physics steps and pause at timeScale=0.
    Preload code should use ``context.add_update()`` to own cleanup automatically.
    """
    if phase not in _update_callbacks:
        raise ValueError(f"Unknown update phase: {phase}")
    if not callable(callback):
        raise TypeError("Update callback must be callable")
    registration = UpdateRegistration(callback, phase, bool(unscaled))
    registration.enabled = enabled
    return registration


def _update_snapshot() -> dict[str, tuple[UpdateRegistration, ...]]:
    return _update_callbacks


def _dispatch_updates(callbacks: tuple[UpdateRegistration, ...], delta: float, unscaled_delta: float,
                      *, gameplay_phase=True) -> None:
    with execution_phase():
        _invoke_updates(callbacks, delta, unscaled_delta, gameplay_phase)


def _invoke_updates(callbacks, delta, unscaled_delta, gameplay_phase):
    admitted_state = simulation_state()
    for registration in callbacks:
        if gameplay_phase and simulation_state() != admitted_state:
            break
        if not registration._enabled:
            continue
        scope = registration._scope
        if scope is not None and scope.state == 'closed':
            registration.close()
            continue
        if scope is not None and scope.state == 'paused':
            continue
        try:
            with component_execution(scope):
                registration._callback(unscaled_delta if registration._unscaled else delta)
        except Exception as exc:
            from infernux.debug import Debug

            Debug.log_exception(exc)


class Signal:
    """Owner-thread event using the same callback ownership as component events.

    emit releases listeners; it does not claim that GPU or IO producers have
    completed. Those dependencies remain on the submitted work's real inputs.
    """
    def __init__(self):
        from infernux.engine.runtime_dispatch import ReloadableCallbackRegistry
        self._listeners = ReloadableCallbackRegistry()
        self._receivers = []

    def _add_receiver(self, callback):
        # Infrastructure receipts collect occurrences even during Pause.
        self._receivers.append(callback)
        return callback

    def _remove_receiver(self, callback):
        self._receivers.remove(callback)

    def add_listener(self, callback):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        return self._listeners.add_listener(callback)

    def remove_listener(self, reference):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        self._listeners.remove_listener(reference)

    def emit(self, *values):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        with execution_phase():
            for receiver in tuple(self._receivers):
                if receiver in self._receivers:
                    receiver(*values)
            return self._listeners.invoke(*values, propagate_exceptions=True)


def install_runtime_service(name: str, service: Any) -> None:
    identity = str(name).strip().casefold()
    if not identity or service is None:
        raise ValueError("runtime service name and implementation are required")
    with _lock:
        existing = _services.get(identity)
        if existing is not None and existing is not service:
            raise RuntimeError(f"runtime service is already installed: {identity}")
        _services[identity] = service


def get_runtime_service(name: str) -> Any | None:
    identity = str(name).strip().casefold()
    if not identity:
        return None
    with _lock:
        return _services.get(identity)


def remove_runtime_service(name: str, service: Any | None = None) -> bool:
    identity = str(name).strip().casefold()
    if not identity:
        return False
    with _lock:
        existing = _services.get(identity)
        if existing is None or (service is not None and existing is not service):
            return False
        del _services[identity]
        return True


__all__ = [
    "UpdatePhase",
    "UpdateRegistration",
    "register_update",
    "Signal",
    "get_runtime_service",
    "install_runtime_service",
    "remove_runtime_service",
]
