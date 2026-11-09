"""Execution ownership, independent of models, rendering and Play policy.

An explicit scope belongs to its caller. Component callbacks supply their own
scope automatically; editor panels and tools can hold one for their own task.
Leaving activate() restores the caller context, not the task's lifetime.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter_ns
import weakref


_current_owner = ContextVar('infernux_execution_owner', default=None)
_active_phase = ContextVar('infernux_execution_phase', default=None)
_active_step = ContextVar('infernux_execution_step', default=None)
_active_result_boundary = ContextVar('infernux_result_boundary', default=None)


@dataclass(frozen=True)
class ClockStamp:
    kind: str
    epoch: int
    tick: int


@dataclass(frozen=True)
class PresentationStamp(ClockStamp):
    """Host opportunity and native monotonic time of window queue acceptance."""

    host_frame: int | None
    accepted_ns: int | None


class Clock:
    """A logical reference, never a worker, budget or simulation driver.

    physics counts completed fixed integrations in the current Play world;
    render counts submitted foreground frames; presentation counts accepted
    window presentations; host counts main-loop service cycles.
    The presentation stamp has no host frame or time before the first acceptance.
    Explicit work may run while its reference clock is stationary.
    """
    def __init__(self, kind):
        if kind not in ('physics', 'render', 'presentation', 'host'):
            raise ValueError("clock must be 'physics', 'render', 'presentation' or 'host'")
        from infernux.application import Application
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError('an execution clock requires an active Engine')
        self._kind = kind
        self._engine = weakref.ref(engine)

    @property
    def kind(self):
        return self._kind

    @property
    def now(self):
        from infernux.application import Application
        engine = self._engine()
        if engine is None or Application._current_engine() is not engine:
            raise RuntimeError('the execution clock owner has retired')
        if self.kind == 'physics':
            from infernux.lib import SceneManager
            manager = SceneManager.instance()
            return ClockStamp(self.kind, int(manager.simulation_epoch), int(manager.completed_fixed_step_index))
        if self.kind == 'presentation':
            sequence, host_frame, accepted_ns = engine.get_native_engine()._last_presentation
            return PresentationStamp(self.kind, id(engine), sequence, host_frame or None, accepted_ns or None)
        tick = (engine.get_native_engine().submitted_render_frames if self.kind == 'render'
                else engine._frame_work.opportunity)
        return ClockStamp(self.kind, id(engine), int(tick))


class ExecutionScope:
    def __init__(self):
        self._closed = False
        self._paused = False
        self._state_provider = None
        self._requests = weakref.WeakSet()
        self._resources = weakref.WeakSet()

    @property
    def state(self):
        if self._closed:
            return 'closed'
        state = 'active' if self._state_provider is None else self._state_provider()
        return 'paused' if self._paused and state in ('active', 'stepping') else state

    @contextmanager
    def activate(self):
        if self.state == 'closed':
            raise RuntimeError('execution scope has closed')
        token = _current_owner.set(self)
        try:
            yield self
        finally:
            _current_owner.reset(token)

    def pause(self):
        self._paused = True
        self._wake()

    def resume(self):
        if self._closed:
            raise RuntimeError('execution scope has closed')
        self._paused = False
        self._wake()

    def close(self):
        if self._closed:
            return
        self._closed = True
        for request in tuple(self._requests):
            request.cancel()
        for resource in tuple(self._resources):
            resource.cancel()

    def _wake(self):
        for request in self._requests:
            owner = request._owner()
            if owner is not None:
                engine = owner._engine()
                if engine is not None:
                    engine._wake_cpu_request(owner, request)


@contextmanager
def component_execution(component):
    token = _current_owner.set(component)
    try:
        yield
    finally:
        _current_owner.reset(token)


def component_scope(component):
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    playing = bool(manager.is_playing())
    domain = (int(manager.simulation_epoch), playing)
    stored = component.__dict__.get('_runtime_execution_scope')
    if stored is not None and stored[0] == domain:
        return stored[1]
    if stored is not None:
        stored[1].close()
    reference = weakref.ref(component)
    def state():
        owner = reference()
        if (owner is None or owner._is_destroyed or owner.__dict__.get('_is_destroying', False)
                or owner.__dict__.get('_play_domain_destroy_finalized', False)
                or int(manager.simulation_epoch) != domain[0]
                or bool(manager.is_playing()) != playing):
            return 'closed'
        if not owner._runtime_active_in_hierarchy or not owner._enabled:
            return 'paused'
        if playing and manager.is_paused():
            permitted = manager.is_stepping() and _active_step.get() == (
                int(manager.simulation_epoch), int(manager._step_permission_index))
            return 'stepping' if permitted else 'paused'
        return 'active'
    scope = ExecutionScope()
    scope._state_provider = state
    component.__dict__['_runtime_execution_scope'] = domain, scope
    return scope


def capture_execution_scope(default_owner=None):
    owner = _current_owner.get()
    # A callback's receiver supplies its ordinary lifetime; an explicitly
    # activated caller scope can choose a different lifetime for the registration.
    explicit = isinstance(owner, ExecutionScope) and owner._state_provider is None
    if default_owner is not None and not explicit:
        owner = default_owner
    phase = _active_phase.get()
    if owner is not None and not explicit and phase is not None and phase != simulation_state():
        raise RuntimeError('the calling component phase has ended')
    scope = owner if owner is None or isinstance(owner, ExecutionScope) else component_scope(owner)
    if scope is not None and scope.state == 'closed':
        raise RuntimeError('the calling execution owner has retired')
    return scope


def capture_step_permission():
    """Only work actually released by a paused logical step inherits its permit."""
    return _active_step.get()


def step_permission_current(permission):
    if permission is None:
        return False
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    return manager.is_stepping() and permission == (
        int(manager.simulation_epoch), int(manager._step_permission_index))


@contextmanager
def step_execution():
    """Native simulation dispatch and required consumers enter the admitted step."""
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    permission = ((int(manager.simulation_epoch), int(manager._step_permission_index))
                  if manager.is_stepping() else None)
    token = _active_step.set(permission)
    try:
        yield
    finally:
        _active_step.reset(token)


@contextmanager
def lifecycle_step_execution():
    """Start/Awake inside actual native Step inherit its precise permission.

    An open step waiting for a result is not an active callback dispatch.
    Editor work during that wait must not acquire a new permission.
    """
    from infernux.lib import SceneManager
    if SceneManager.instance()._is_executing_step():
        with step_execution():
            yield
    else:
        yield


@dataclass(frozen=True, slots=True)
class ResultApplication:
    """One successful result application at its actual consumer boundary."""

    host_frame: int | None
    monotonic_ns: int
    duration_ns: int
    physics_step: ClockStamp | None


class ResultConsumer:
    """One application to components or a graph input at its own boundary.

    A ready-only consumer leaves its boundary available while a request is
    pending. Required edges keep the step open until success or cancellation.
    ``application`` is set when the callback succeeds. A standalone fixed step
    outside the host loop has no host frame. Render application records the
    input change at extraction, independently of any later submission.
    """
    def __init__(self, request, targets, apply, manager, *, boundary, required):
        self.request = request
        self.boundary = boundary
        self.required = required
        self._input = targets[0] if boundary == 'render_input' else None
        self._targets = []
        if self._input is None:
            for target in targets:
                component = target._require_cpp_component()
                epoch = getattr(component, '_result_target_epoch', 0)
                if epoch is None:
                    raise ValueError('result target has no bound state')
                self._targets.append((component.handle, epoch))
        self._target_keys = frozenset((int(handle.world_id), int(handle.id), int(handle.generation))
                                      for handle, _ in self._targets)
        self._apply = apply
        self._scope = capture_execution_scope()
        self._epoch = int(manager.simulation_epoch)
        self._playing = bool(manager.is_playing())
        index = int(manager.fixed_step_index)
        self._step = index if manager.fixed_step_pending and manager.completed_fixed_step_index < index else index + 1
        self.status = 'waiting'
        self.reason = ''
        self.application: ResultApplication | None = None
        self._attached = True
        request.required_consumers += required
        permission = capture_step_permission()
        if permission is not None and (required or request.done):
            request._owner()._permit_step(request, permission)
        if self._input is not None:
            self._input._requirement = self
            self._hold_render_input()
        self._wake()

    def _wake(self):
        owner = self.request._owner()
        if owner is not None:
            owner._plan_dirty = True
            owner.wake()

    def _detach(self):
        if self._attached:
            self._attached = False
            self.request.required_consumers -= self.required
            self._apply = None
            if self._input is not None:
                self._input._native._set_requirement(False, False, ())
                self._input._requirement = None
                self._input = None
            self._wake()

    def cancel(self):
        if self.status in ('applied', 'cancelled', 'rejected'):
            return False
        self.status = 'cancelled'
        self._detach()
        return True

    def _resolve(self, manager):
        if not self._attached:
            return None
        frame = self.request._owner()._frame
        active = frame.active
        if active:
            frame.begin_scope()
        try:
            if self._epoch != int(manager.simulation_epoch) or self._playing != bool(manager.is_playing()) or (
                    self._scope is not None and self._scope.state == 'closed'):
                self.cancel()
                return None
            if self._input is not None:
                return (self._input,)
            targets = []
            for handle, epoch in self._targets:
                world = manager.get_scene_by_world_id(handle.world_id)
                native = None if world is None else world.resolve_component(handle)
                if native is None or world.is_pending_destroy(native.game_object):
                    self.cancel()
                    return None
                if epoch and native._result_target_epoch != epoch:
                    self.reason = 'result target changed after binding'
                    self.cancel()
                    return None
                targets.append(native)
            return tuple(targets)
        finally:
            if active:
                frame.end_scope()

    def _current(self, manager):
        return self._resolve(manager) is not None

    def _step_allowed(self):
        if not step_permission_current(self.request.step_permission):
            return False
        with step_execution():
            return self._scope is None or self._scope.state in ('active', 'stepping')

    def _suspended(self):
        if self._scope is None or self._scope.state != 'paused':
            return False
        if self.boundary == 'physics':
            if self.required:
                return False
            with step_execution():
                return self._scope.state == 'paused'
        return not self._step_allowed()

    def _hold_render_input(self):
        if self._suspended():
            self._input._native._set_requirement(False, False, ())
            return
        captures = (tuple(self.request._owner()._pending_render_captures(self.request))
                    if self.status == 'waiting' else ())
        self._input._native._set_requirement(True, self._step_allowed(), captures)

    def _consume(self, manager, targets):
        if self._input is not None:
            if self._input._native._result_cycle and self.status == 'waiting':
                self.status = 'failed'
                self.reason = 'result dependency cycle: this graph input prevents the view producing its source'
                from infernux.debug import Debug
                Debug.log_error(self.reason)
            self._hold_render_input()
            if self._suspended():
                return True
        permission = int(manager.simulation_epoch), int(manager.fixed_step_index)
        stepping = manager.is_stepping() and (self.boundary == 'physics'
                    or self._step_allowed())
        if stepping:
            self.request._owner()._permit_step(self.request, permission)
        if self.status != 'waiting':
            return not self.required
        # RenderExtraction blocks its producers. A held physics step still
        # allows views to render the most recently published world.
        if self.required and self.boundary == 'render' and self.request._owner()._waits_for_render_input(self.request):
            self.status = 'rejected'
            self.reason = 'result dependency cycle: this boundary prevents the render submission producing its input'
            self._detach()
            from infernux.debug import Debug
            Debug.log_error(self.reason)
            return True
        if not self.request.done:
            return False
        if self.request.status != 'published':
            self.status, self.reason = 'failed', self.request.reason or self.request.status
            if not self.required:
                self._detach()
            return not self.required
        try:
            from contextlib import nullcontext
            permission_scope = step_execution() if stepping else nullcontext()
            with execution_phase(), permission_scope, component_execution(self._scope):
                if self._scope is not None and self._scope.state == 'paused':
                    return not self.required
                self.status = 'applying'
                frame = self.request._owner()._frame
                host_frame = int(frame.opportunity) if frame.active else None
                physics_step = (ClockStamp('physics', *permission) if self.boundary == 'physics' else None)
                if host_frame is not None:
                    frame.begin_scope()
                started = perf_counter_ns()
                token = _active_result_boundary.set(self.boundary)
                try:
                    if self._input is None:
                        from infernux.lib import _wrap_native_builtin_component

                        targets = tuple(_wrap_native_builtin_component(target) for target in targets)
                    self._apply(targets, self.request.result)
                    applied = perf_counter_ns()
                finally:
                    _active_result_boundary.reset(token)
                    if host_frame is not None:
                        frame.end_scope()
        except Exception as error:
            if self.status == 'applying':
                self.status, self.reason = 'failed', str(error)
                if not self.required:
                    self._detach()
            from infernux.debug import Debug
            Debug.log_exception(error)
            return not self.required or self.status == 'cancelled'
        if self.status == 'applying':
            self.application = ResultApplication(host_frame, applied, applied - started, physics_step)
            self.status = 'applied'
            self._detach()
        return True


def close_component_execution(component):
    stored = component.__dict__.pop('_runtime_execution_scope', None)
    if stored is not None:
        stored[1].close()


def simulation_state():
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    return int(manager.simulation_epoch), bool(manager.is_playing()), bool(manager.is_paused())


@contextmanager
def execution_phase():
    if _active_phase.get() is not None:
        yield
        return
    token = _active_phase.set(simulation_state())
    try:
        yield
    finally:
        _active_phase.reset(token)
