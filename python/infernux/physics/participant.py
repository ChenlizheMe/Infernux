"""Public Work declarations for one ordinary component's native physics owner.

This first slice wires state, author controls and native tick stages. Geometry,
rigidbody response and the common residual adapter are not implemented here.
"""
from __future__ import annotations

from infernux.compute import Buffer, Work, buffer
from infernux.lib import Physics as _NativePhysics

__all__ = ['Participant', 'PhysicsBuildPlan']


class Participant:
    """Managed component/World state; set_control records author values only."""

    def __init__(self, native, controls):
        self._native = native
        self._controls = controls

    @property
    def ready(self):
        return self._native is not None and self._native.ready

    def set_control(self, name, value):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        if self._native is None:
            raise RuntimeError('physics participant is retired')
        layout = self._controls[name]
        self._native.set_control(name, layout._coerce_data(value).tobytes(order='C'))

    def set_count(self, count):
        """Record the next admitted batch prefix; never rebuild stage Work."""
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        if self._native is None:
            raise RuntimeError('physics participant is retired')
        if type(count) is not int or not 0 <= count <= 0xffffffff:
            raise ValueError('participant count must be a uint32 batch domain')
        self._native.set_count(count)

    def capture_output(self, name, *, offset=0, count=None):
        """Observe a named state range at the current or earliest complete boundary.

        This does not integrate, apply pending controls, or wait synchronously.
        Transfer the returned ordinary BufferSource with compute.transfer for CPU use.
        """
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        from infernux.compute_source import BufferSource
        from infernux.application import Application
        assert_runtime_owner_thread()
        if self._native is None:
            raise RuntimeError('physics participant is retired')
        if not isinstance(name, str):
            raise TypeError('participant output name must be a string')
        if type(offset) is not int or offset < 0 or type(count) not in (int, type(None)) or (count is not None and count <= 0):
            raise ValueError('participant output requires a nonnegative offset and positive count')
        engine = Application._current_engine()
        session = engine._runtime_world_session if engine is not None else None
        if session is None or session.closed:
            raise RuntimeError('participant output requires its live RuntimeWorld session')
        capture = self._native.capture_output(name, offset, count)
        domain = engine._world_runtimes.get(int(capture.token[0]))
        if domain is None or domain.scope._closed:
            capture.cancel()
            raise RuntimeError('participant output World has retired')
        lease = _ParticipantOutputLease(capture, engine)
        source = BufferSource(lease)
        session.defer_participant_observation(domain.identity, lease)
        return source

    def close(self):
        if self._native is not None:
            self._native.close()
            self._native = None
        self._controls.clear()


class _ParticipantOutputLease:
    """Concrete producer for the existing BufferSource/InputBindings protocol."""
    def __init__(self, capture, engine):
        import weakref
        from infernux import compute
        self._capture = capture
        self.engine = weakref.ref(engine)
        self.shape, self.dtype, self.device = tuple(capture.shape), capture.dtype, 'gpu'
        self.storage_nbytes = capture.storage_byte_size
        self.host_identity = int(compute._native_compute_host().identity)
        self._transfer_key = ('participant-output', self.host_identity, *capture.token)

    @property
    def storage_claims(self):
        from infernux.compute import _native_storage_blocks
        native = self._capture.value if self.ready else self._capture.storage_buffer
        if native is not None:
            return dict(_native_storage_blocks(self.host_identity, native))
        return {('future', *self._transfer_key): self.storage_nbytes}

    @property
    def transfer_key(self):
        # Stable from request creation: provenance does not exist until publication.
        return self._transfer_key

    @property
    def done(self):
        return self._capture.done

    @property
    def ready(self):
        return self._capture.ready

    @property
    def error(self):
        return self._capture.error

    @property
    def identity(self):
        return self._capture.identity

    def acquire(self):
        from infernux import compute
        if not self.ready:
            raise RuntimeError(self.error or 'participant output has not completed')
        return compute.Buffer._from_native(self._capture.value, shape=self.shape)

    def cancel(self):
        self._capture.cancel()


class _ParticipantObservation:
    """Weak admission observation on the existing session Host continuation queue."""
    requires_dispatch_safe_point = False
    def __init__(self, lease):
        import weakref
        self.lease = weakref.ref(lease)

    def advance(self):
        lease = self.lease()
        if lease is None or lease.done:
            return True
        try:
            _NativePhysics._service_participant_outputs()
        except Exception as error:
            lease._capture.fail(str(error))
            return True
        return True if lease.done else False

    def cancel(self):
        lease = self.lease()
        if lease is not None:
            lease.cancel()


class PhysicsBuildPlan:
    """Cold component declaration; the engine supplies this during build_physics.

    Named initialize outputs are persistent per-World allocations. Later Work
    inputs bind those same names, plus optional step/indices/control ports.
    Each Work recipe is one finite indivisible unit under the existing budget.
    """

    def __init__(self, component):
        import weakref
        from infernux.engine.runtime_worlds import current_world_runtime
        self._component = component
        self._world = weakref.ref(current_world_runtime())
        self._participants = []
        self._step = self._indices = None
        self._sealed = False

    @property
    def step(self):
        """f32[2]: actual substep dt, admitted fixed dt; bound at runtime."""
        if self._step is None:
            self._step = buffer(shape=2, dtype=float, device='gpu')
        return self._step

    @property
    def indices(self):
        """u32[6]: substep, substeps, iteration, iterations, count, stage.

        Stages are predict=0, prepare=1, exchange=2, accept=3, initialize=4.
        Initialize receives zero timing/iteration values and its declared count.
        """
        import numpy as np
        if self._indices is None:
            self._indices = buffer(shape=6, dtype=np.uint32, device='gpu')
        return self._indices

    def participant(self, *, initialize, predict=None, prepare=None, exchange=None, accept=None,
                    controls=None, count):
        """Bind multiple named initialize outputs to later Work input ports.

        Input names match initialize output names, step/indices, or controls.
        Work owns stage scratch; all stages share each initialized state owner.
        Controls map names to (prototype Buffer, complete initial author value).
        """
        if self._sealed:
            raise RuntimeError('build_physics has finished')
        if type(count) is not int or not 0 <= count <= 0xffffffff:
            raise ValueError('participant count must be a uint32 batch domain')
        recipes = dict(initialize=initialize, predict=predict, prepare=prepare, exchange=exchange, accept=accept)
        recipes = {name: work for name, work in recipes.items() if work is not None}
        if not isinstance(initialize, Work) or any(not isinstance(work, Work) for work in recipes.values()):
            raise TypeError('participant stages must be public compute.Work values')
        decoded, layouts = {}, {}
        for name, (prototype, value) in (controls or {}).items():
            if not isinstance(prototype, Buffer) or prototype.device != 'gpu':
                raise TypeError('participant control prototypes must be GPU Buffers')
            prototype._require_open()
            if prototype.description.byte_offset or prototype.nbytes != prototype._native.byte_size:
                raise ValueError('control prototype must describe its complete allocation')
            decoded[name] = prototype._native, prototype._coerce_data(value).tobytes(order='C')
            layouts[name] = prototype
        native = _NativePhysics._register_participant(
            self._component._get_bound_native_component(),
            {name: work._require_open() for name, work in recipes.items()}, decoded, count)
        result = Participant(native, layouts)
        self._participants.append(result)
        return result

    def close(self):
        if self._component is None:
            return
        world = self._world()
        if world is not None:
            world.physics_plans.pop(id(self._component), None)
        for participant in self._participants:
            participant.close()
        self._participants.clear()
        for prototype in (self._step, self._indices):
            if prototype is not None:
                prototype.close()
        self._component = None


def _build_component_physics(component):
    """Cold owner-published method resolution; never a numerical stage callback."""
    from infernux.engine.runtime_worlds import current_world_runtime
    domain = current_world_runtime()
    if domain is None or '_physics_build_plan' in component.__dict__:
        return
    callback = component._resolve_runtime_method('build_physics')
    from infernux.components.component import InxComponent
    function = getattr(callback, '__func__', callback)
    if callback is None or getattr(function, '__code__', None) is InxComponent.build_physics.__code__:
        return
    from infernux.execution import component_execution, execution_phase
    plan = PhysicsBuildPlan(component)
    component.__dict__['_physics_build_plan'] = plan
    domain.physics_plans[id(component)] = plan
    try:
        with component_execution(component), execution_phase():
            callback(plan)
        plan._sealed = True
        component._resolve_lifecycle_exception('build_physics')
    except BaseException:
        _retire_component_physics(component)
        raise


def _retire_component_physics(component, *, cancel=True):
    if cancel:
        _clear_preparation_marker(component)
        component.__dict__.pop('_physics_preparation_ticket', None)
        component.__dict__.pop('_physics_preparation_key', None)
        component.__dict__.pop('_physics_rebuild_requested', None)
    plan = component.__dict__.pop('_physics_build_plan', None)
    if plan is not None:
        plan.close()


def _clear_preparation_marker(component):
    from infernux.application import Application
    key = component.__dict__.get('_physics_preparation_key')
    ticket = component.__dict__.get('_physics_preparation_ticket')
    engine = Application._current_engine()
    domain = engine._world_runtimes.get(key[0]) if engine is not None and key is not None else None
    if domain is not None and domain.physics_preparations.get(id(component)) is ticket:
        domain.physics_preparations.pop(id(component), None)


def _note_preparing(component, value, *, ticket=None):
    component.__dict__['_physics_preparing'] = value
    from infernux.application import Application
    from infernux.engine.runtime_worlds import component_world_id
    engine = Application._current_engine()
    domain = engine._world_runtimes.get(component_world_id(component)) if engine is not None else None
    if domain is not None:
        if value:
            domain.physics_preparations[id(component)] = component.__dict__.get('_physics_preparation_ticket')
        elif domain.physics_preparations.get(id(component)) is ticket:
            domain.physics_preparations.pop(id(component), None)
    from infernux.components._component_lifecycle import notify_runtime_component_changed
    notify_runtime_component_changed(component)


def _activate_components(attached, *, activate=True):
    from infernux.engine.runtime_worlds import component_world_id
    from infernux.lib import SceneManager
    manager = SceneManager.instance()
    for target, instance, native in attached:
        if instance._is_destroyed or instance._get_bound_native_component() is not native:
            continue
        identity = component_world_id(instance)
        if identity != int(manager._authoring_runtime_world_id):
            handle = manager._get_runtime_world_handle(identity)
            if handle is None or handle.state not in {'preparing', 'ready'}:
                return
        ticket = instance.__dict__.get('_physics_preparation_ticket')
        target._activate_prepared_py_component(native, activate)
        # Awake may destroy/rebind the owner or request a new cold generation.
        if instance.__dict__.get('_physics_preparation_ticket') is ticket:
            instance.__dict__.pop('_physics_preparation_ticket', None)
            instance.__dict__.pop('_physics_preparation_key', None)
            _note_preparing(instance, False, ticket=ticket)


def _has_physics_declaration(component):
    from infernux.components.component import InxComponent
    callback = getattr(type(component), 'build_physics', None)
    function = getattr(callback, '__func__', callback)
    return callback is not None and getattr(function, '__code__', None) is not InxComponent.build_physics.__code__


class _ComponentPreparation:
    """Cold lifecycle continuation on the existing session Host source only."""
    requires_dispatch_safe_point = True

    def __init__(self, identity, attached, *, build, rebuild):
        self.identity, self.attached, self.build = identity, tuple(attached), build
        self.cursor = 0
        self.tickets = []
        for _target, component, _native in self.attached:
            key = identity, component._native_generation
            if component.__dict__.get('_physics_preparation_key') != key:
                _clear_preparation_marker(component)
                component.__dict__['_physics_preparation_key'] = key
                component.__dict__['_physics_preparation_ticket'] = object()
            self.tickets.append(component.__dict__['_physics_preparation_ticket'])
            if rebuild:
                component.__dict__['_physics_rebuild_requested'] = True
            _note_preparing(component, True)

    def advance(self):
        from infernux.engine.runtime_worlds import component_world_id
        def valid(row, ticket):
            _target, component, native = row
            return (not component._is_destroyed and component_world_id(component) == self.identity
                    and component.__dict__.get('_physics_preparation_ticket') is ticket
                    and component._get_bound_native_component() is native)
        if self.cursor < len(self.attached):
            row, ticket = self.attached[self.cursor], self.tickets[self.cursor]
            self.cursor += 1
            if not valid(row, ticket):
                return None
            _target, component, _native = row
            rebuild = component.__dict__.pop('_physics_rebuild_requested', False)
            if self.build or rebuild:
                if rebuild:
                    _retire_component_physics(component, cancel=False)
                try:
                    _build_component_physics(component)
                except Exception as error:
                    _retire_component_physics(component)
                    component._report_lifecycle_exception(error, 'build_physics')
                    component._disable_after_awake_exception()
                    # Keep native preparation closed; a later valid reload can recover.
            return None  # One cold component per Host quantum, not one whole Scene.
        self.build = False
        attached = []
        for index, (row, ticket) in enumerate(zip(self.attached, self.tickets)):
            if not valid(row, ticket):
                continue
            if row[1].__dict__.get('_physics_rebuild_requested', False):
                self.cursor = index
                return None
            attached.append(row)
        if not attached:
            return True
        # A replacement/schema change may have removed its physics declaration.
        # Its inherited preparation gate needs the same lifecycle hand-off, but
        # has no GPU initialization to await, including unrelated participants.
        has_physics = any(row[1].__dict__.get('_physics_build_plan') is not None for row in attached)
        if has_physics and not _NativePhysics._participants_ready():
            return False
        from infernux.lib import SceneManager
        _activate_components(attached, activate=bool(SceneManager.instance().is_playing()))
        return True

    def cancel(self):
        for (_target, component, _native), ticket in zip(self.attached, self.tickets):
            if component.__dict__.get('_physics_preparation_ticket') is ticket:
                _retire_component_physics(component)


def _queue_component_physics(component, *, rebuild=False):
    """Post a cold declaration/rebuild; never run project Work inside lifecycle."""
    if component._is_destroyed:
        return False
    plan = component.__dict__.get('_physics_build_plan')
    has_physics = plan is not None or _has_physics_declaration(component)
    if not has_physics and not component._physics_preparing:
        return False
    from infernux.application import Application
    from infernux.engine.runtime_worlds import component_world_id
    engine = Application._current_engine()
    identity = component_world_id(component)
    domain = engine._world_runtimes.get(identity) if engine is not None else None
    if domain is None:
        if rebuild or component._physics_preparing:
            _retire_component_physics(component)
            _restore_component_physics_activation(component)
        return False
    session = engine._runtime_world_session
    if session is None or session.closed:
        raise RuntimeError('physics preparation requires its live RuntimeWorld session')
    native = component._get_bound_native_component()
    if native is None:
        return False
    target = native.game_object
    key = identity, component._native_generation
    if rebuild:
        component.__dict__['_physics_rebuild_requested'] = True
    if component.__dict__.get('_physics_preparation_key') == key:
        return True
    target._defer_python_component_activation(native)
    session.defer_component_activation(identity, ((target, component, native),), build=has_physics, rebuild=rebuild)
    return True


def _restore_component_physics_activation(component):
    """A rejected transactional replacement keeps the original ready state."""
    plan = component.__dict__.get('_physics_build_plan')
    if plan is not None and not all(value.ready for value in plan._participants):
        _queue_component_physics(component)
        return
    native = component._get_bound_native_component()
    if native is not None:
        native.game_object._defer_python_component_activation(native)
        _activate_components(((native.game_object, component, native),), activate=False)


def _request_physics_activation(component):
    if '_play_domain_retirement_owner' in component.__dict__:
        _restore_component_physics_activation(component)
        return not component._physics_preparing
    if component.__dict__.get('_physics_build_plan') is not None and not component._physics_preparing:
        return True
    return not _queue_component_physics(component)


def _prepare_component_activation(attached):
    """Use the same readiness gate for every component batch, including no Work."""
    from infernux.engine.runtime_worlds import current_world_runtime
    domain = current_world_runtime()
    if domain is None:
        _activate_components(attached)
        return
    for _target, component, _native in attached:
        _build_component_physics(component)
    if _NativePhysics._participants_ready():
        _activate_components(attached)
    else:
        from infernux.application import Application
        Application._current_engine()._runtime_world_session.defer_component_activation(domain.identity, attached)
