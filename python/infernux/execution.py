"""Execution ownership, independent of models, rendering and Play policy.

An explicit scope belongs to its caller. Component callbacks supply their own
scope automatically; editor panels and tools can hold one for their own task.
Leaving activate() restores the caller context, not the task's lifetime.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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
    world_id: int | None = field(default=None, kw_only=True)


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
        from infernux.lib import SceneManager
        self._world_id = int(SceneManager.instance()._current_runtime_world_id) if kind == 'physics' else None
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
            from infernux.lib import _Infernux
            with _Infernux._RuntimeWorldScope(self._world_id):
                return ClockStamp(self.kind, int(manager.simulation_epoch), int(manager.completed_fixed_step_index), world_id=self._world_id)
        if self.kind == 'presentation':
            sequence, host_frame, accepted_ns = engine.get_native_engine()._last_presentation
            return PresentationStamp(self.kind, id(engine), sequence, host_frame or None, accepted_ns or None)
        tick = (engine.get_native_engine().submitted_render_frames if self.kind == 'render'
                else engine._frame_work.opportunity)
        return ClockStamp(self.kind, id(engine), int(tick))


class ExecutionScope:
    def __init__(self):
        # Explicit caller scopes own work, independently of a simulation World.
        # Component scopes and the World domain scope bind this field below.
        self._world_id = None
        self._closed = False
        self._paused = False
        self._component_state = None
        self._requests = weakref.WeakSet()
        self._resources = weakref.WeakSet()

    @property
    def state(self):
        if self._component_state is not None:
            return self._component_state.state(self, _active_step.get())
        if self._closed:
            return 'closed'
        if self._paused:
            return 'paused'
        if self._world_id is not None:
            from infernux.lib import SceneManager, _Infernux
            with _Infernux._RuntimeWorldScope(self._world_id):
                manager = SceneManager.instance()
                if manager.is_playing() and manager.is_paused():
                    permission = _active_step.get()
                    return ('stepping' if permission is not None and permission[0] == self._world_id
                            and step_permission_current(permission) else 'paused')
        return 'active'

    @contextmanager
    def activate(self):
        if self.state == 'closed':
            raise RuntimeError('execution scope has closed')
        with component_execution(self):
            yield self

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
    from contextlib import nullcontext
    from infernux.components.value_ownership import value_world
    from infernux.engine.runtime_worlds import component_world_id
    identity = (component._world_id if isinstance(component, ExecutionScope)
                else component_world_id(component) if component is not None else None)
    token = _current_owner.set(component)
    try:
        with value_world(identity) if identity else nullcontext():
            yield
    finally:
        _current_owner.reset(token)


def component_scope(component):
    from infernux.lib import SceneManager, _Infernux
    manager = SceneManager.instance()
    from infernux.engine.runtime_worlds import component_world_id
    identity = component_world_id(component) or int(manager._current_runtime_world_id)
    with _Infernux._RuntimeWorldScope(identity):
        playing = bool(manager.is_playing())
        domain = (identity, int(manager.simulation_epoch), playing)
    stored = component.__dict__.get('_runtime_execution_scope')
    if stored is not None and stored[0] == domain:
        return stored[1]
    if stored is not None:
        stored[1].close()
    scope = ExecutionScope()
    scope._world_id = identity
    scope._component_state = _Infernux._ComponentExecutionState(component, manager, *domain)
    component.__dict__['_runtime_execution_scope'] = domain, scope
    return scope


def capture_execution_scope(default_owner=None):
    owner = _current_owner.get()
    # A callback's receiver supplies its ordinary lifetime; an explicitly
    # activated caller scope can choose a different lifetime for the registration.
    explicit = isinstance(owner, ExecutionScope) and owner._component_state is None
    if default_owner is not None and not explicit:
        owner = default_owner
    phase = _active_phase.get()
    if owner is not None and not explicit and phase is not None and phase != simulation_state():
        raise RuntimeError('the calling component phase has ended')
    scope = owner if owner is None or isinstance(owner, ExecutionScope) else component_scope(owner)
    if scope is None:
        from infernux.engine.runtime_worlds import current_world_runtime
        domain = current_world_runtime()
        scope = domain.scope if domain is not None else None
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
    from infernux.lib import _Infernux
    with _Infernux._RuntimeWorldScope(permission[0]):
        return manager.is_stepping() and permission == (
            int(manager._current_runtime_world_id), int(manager.simulation_epoch), int(manager._step_permission_index))


@contextmanager
def step_execution(permission=None):
    """Native simulation dispatch and required consumers enter the admitted step."""
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    from contextlib import nullcontext
    from infernux.lib import _Infernux
    with (_Infernux._RuntimeWorldScope(permission[0]) if permission is not None else nullcontext()):
        permission = ((int(manager._current_runtime_world_id), int(manager.simulation_epoch), int(manager._step_permission_index))
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
        self._targets = None
        if self._input is None:
            from infernux.lib import _Infernux

            self._targets = _Infernux._ResultTargetBatch(tuple(target._require_cpp_component() for target in targets))
        self._target_keys = frozenset(
            (handle.world_id, handle.id, handle.generation)
            for handle in (self._targets.handles if self._targets is not None else ())
        )
        self._apply = apply
        self._scope = capture_execution_scope()
        self._world_id = int(manager._current_runtime_world_id)
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

    def _resolve(self, manager, *, require_body_ready=False, collect=False):
        if not self._attached:
            return None
        if self._scope is not None and self._scope.state == 'closed':
            self.cancel()
            return None
        from infernux.lib import _Infernux
        with _Infernux._RuntimeWorldScope(self._world_id):
            return self._resolve_current_world(manager, require_body_ready=require_body_ready, collect=collect)

    def _resolve_current_world(self, manager, *, require_body_ready=False, collect=False):
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
                return True, (self._input,)
            from infernux.lib import _Infernux

            states = _Infernux._ResultTargetStatus
            status, targets = self._targets.inspect(manager, require_body_ready, collect)
            if status == states.Ready:
                return True, targets
            if status == states.Waiting:
                return False, targets
            if status == states.Changed:
                self.reason = 'result target changed after binding'
            self.cancel()
            return None
        finally:
            if active:
                frame.end_scope()

    def _current(self, manager):
        return self._resolve(manager) is not None

    def _step_allowed(self):
        if not step_permission_current(self.request.step_permission):
            return False
        with step_execution(self.request.step_permission):
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
        permission = int(manager._current_runtime_world_id), int(manager.simulation_epoch), int(manager.fixed_step_index)
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
                physics_step = (ClockStamp('physics', *permission[1:], world_id=permission[0]) if self.boundary == 'physics' else None)
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
    return (int(manager.simulation_epoch), bool(manager.is_playing()), bool(manager.is_paused()),
            int(manager._runtime_frame_generation), int(manager._current_runtime_world_id))


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
