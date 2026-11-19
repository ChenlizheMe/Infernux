"""World lifecycle shared by desktop and platform-owned Player loops.

Native SceneManager owns worlds and clocks. This companion binds Python
realms, phase plans and result consumers to those same native identities.
"""
import weakref

from .work_runtime import WorkRuntime
from .runtime_change_journal import RuntimeChangeDomain, RuntimeFrameBarrier


_RESULT_BOUNDARIES = {'PHYSICS_SIMULATION': 'physics', 'RENDER_EXTRACTION': 'render'}


class RuntimeWorldHost(WorkRuntime):
    def _initialize_world_runtime(self, role):
        from infernux.components._component_lifecycle import RuntimeExecutionScheduler
        from infernux.lib import SceneManager
        SceneManager.instance()._ensure_authoring_world()
        self._world_runtimes = {}
        self._external_update_snapshot_revision = -1
        self._runtime_world_session = None
        from infernux.runtime_services import _author_update_callbacks
        self._authoring_scheduler = RuntimeExecutionScheduler(
            name=role,
            native_bridge=True,
            runtime_updates=_author_update_callbacks,
        )
        self._runtime_scene_manager = None
        self._last_native_transform_serial = None
        from infernux.runtime_services import _external_update_snapshot
        self._clock_callbacks = _external_update_snapshot(self)
        self._clock_delta = 0.0

    def _install_runtime_world_callbacks(self):
        from infernux.components._component_lifecycle import _native_result_phase
        from infernux.execution import simulation_state
        engine_ref = weakref.ref(self)
        from infernux.lib import SceneManager

        scene_manager = SceneManager.instance()
        self._runtime_scene_manager = scene_manager
        def _begin_retire_world(identity):
            engine = engine_ref()
            if engine is None:
                return
            domain = engine._world_runtimes.get(int(identity))
            if domain is not None:
                domain.begin_retirement()
            for consumer in tuple(engine._result_consumers):
                if consumer._world_id == int(identity):
                    consumer.cancel()

        def _retire_world(identity):
            engine = engine_ref()
            if engine is None:
                return
            engine._retire_world_presentation(identity)
            if int(identity) == int(scene_manager._authoring_runtime_world_id):
                from infernux.core.assets import AssetManager
                AssetManager.release_engine(engine)
                from .script_source import clear_module_codes
                clear_module_codes()
                return
            domain = engine._world_runtimes.get(int(identity))
            if domain is not None:
                domain.close()
                del engine._world_runtimes[int(identity)]
                from infernux.runtime_services import _invalidate_external_update_snapshot
                _invalidate_external_update_snapshot(engine)
        scene_manager._set_runtime_world_retirement_callbacks(_begin_retire_world, _retire_world)
        self._last_native_transform_serial = int(
            scene_manager.get_global_transform_serial()
        )
        def _begin_native_frame():
            engine = engine_ref()
            if engine is None:
                return
            engine._runtime_scheduler.begin_native_frame()
            if engine is None or not scene_manager._is_executing_step():
                return
            from infernux.execution import step_execution, capture_step_permission
            with step_execution():
                for consumer in tuple(engine._result_consumers):
                    if (consumer._world_id == int(scene_manager._current_runtime_world_id)
                            and consumer.boundary != 'physics' and (consumer.required or consumer.request.done)
                            and consumer._current(scene_manager)
                            and consumer._scope is not None and consumer._scope.state == 'stepping'):
                        consumer.request._owner()._permit_step(consumer.request, capture_step_permission())
                        scene_manager._retain_step_frame()

        def _dispatch_runtime_input():
            engine = engine_ref()
            if engine is not None:
                engine._dispatch_runtime_input()

        scene_manager.set_runtime_input_callback(_dispatch_runtime_input)

        def _execute_native_phase(phase, delta):
            engine = engine_ref()
            if engine is not None:
                engine._runtime_scheduler.execute_native_phase(phase, delta)

        def _execute_editor_phase(delta):
            engine = engine_ref()
            if engine is not None:
                engine._runtime_scheduler.execute_native_editor_update(delta)

        def _end_native_frame():
            engine = engine_ref()
            if engine is not None:
                engine._runtime_scheduler.end_native_frame()

        scene_manager.set_runtime_lifecycle_callbacks(
            _begin_native_frame,
            lambda delta: _execute_native_phase("fixed_update", delta),
            lambda delta: _execute_native_phase("update", delta),
            lambda delta: _execute_native_phase("late_update", delta),
            _execute_editor_phase,
            _end_native_frame,
        )
        def _consume_native_barrier(barrier):
            engine = engine_ref()
            if engine is None:
                return None
            return engine._consume_runtime_frame_barrier(RuntimeFrameBarrier.from_native(barrier))

        scene_manager.set_runtime_frame_barrier_callback(_consume_native_barrier)
        def _consume_results(barrier):
            engine = engine_ref()
            if engine is None:
                return True
            identity = int(scene_manager._current_runtime_world_id)
            domain = engine._world_runtimes.get(identity)
            scheduler = domain.scheduler if domain is not None else engine._authoring_scheduler
            if domain is None and identity != scheduler.world_id:
                raise KeyError(identity)
            phase = _native_result_phase(barrier)
            if phase is not None:
                # Keep the admitted cursor while Host work prepares the participant.
                if domain is not None and domain.physics_preparations:
                    return False
                if not scheduler._resume_native_phase(phase):
                    return False
            consumers = engine._result_consumers
            if not consumers:
                return True
            boundary = _RESULT_BOUNDARIES.get(barrier.name)
            if boundary is None:
                return True
            state = simulation_state()
            ready = True
            blocked = set()
            waiting = set()
            suspended = set()
            index = 0
            while index < len(consumers):
                consumer = consumers[index]
                index += 1
                if consumer._world_id != identity:
                    continue
                if consumer.boundary == 'render_input' and boundary == 'render':
                    resolution = consumer._resolve(scene_manager)
                    if resolution is not None:
                        consumer._consume(scene_manager, resolution[1])
                    continue
                if consumer.boundary != boundary:
                    continue
                resolution = consumer._resolve(
                    scene_manager,
                    require_body_ready=boundary == 'physics' and (consumer.required or consumer.request.done),
                    collect=consumer.request.status == 'published',
                )
                if resolution is None:
                    continue
                targets_ready, targets = resolution
                if not consumer.required and not consumer.request.done:
                    waiting.update(consumer._target_keys)
                    continue
                if boundary == 'physics' and consumer._step > scene_manager.fixed_step_index:
                    continue
                # The control phase publishes newly created bodies after this
                # gate. Preserve target order without waiting on that phase.
                if consumer._suspended() or not targets_ready:
                    (suspended if consumer.required else waiting).update(consumer._target_keys)
                    continue
                if suspended.intersection(consumer._target_keys):
                    suspended.update(consumer._target_keys)
                    continue
                if not consumer.required and waiting.intersection(consumer._target_keys):
                    waiting.update(consumer._target_keys)
                    continue
                if blocked.intersection(consumer._target_keys) or not consumer._consume(scene_manager, targets):
                    blocked.update(consumer._target_keys)
                    ready = False
                if state != simulation_state():
                    return False
            return ready
        scene_manager.set_runtime_result_consumer_callback(_consume_results)
        scene_manager._set_result_consumers_present(bool(self._result_consumers))
        self._authoring_scheduler.bind_native_bridge(scene_manager)

    def _prepare_runtime_world_schedulers(self):
        from infernux.lib import SceneManager, _Infernux
        manager = SceneManager.instance()
        identities = manager._runtime_world_ids
        author_only = not identities and self._runtime_world_session is None
        if author_only:
            with _Infernux._RuntimeWorldScope(manager._authoring_runtime_world_id):
                self._authoring_scheduler.prepare_frame()
        for identity in identities:
            with _Infernux._RuntimeWorldScope(identity):
                self._world_runtimes[int(identity)].scheduler.prepare_frame()
        return bool(identities) or author_only

    def _retire_world_presentation(self, identity):
        """Platform presentation resources may outlive the retired scene graph."""

    def _on_render_extraction(self):
        """Desktop authoring hosts append their gizmo publication here."""

    def _capture_graph_buffer(self, name, camera_id, scene_view):
        native = self.get_native_engine()
        graph = native.get_scene_render_graph() if scene_view else native.get_camera_render_graph(camera_id)
        if graph is None:
            raise RuntimeError("the selected view has not defined a render graph yet")
        return graph.capture_buffer(name)

    def _watch_graph_events(self, label):
        from infernux.lib import _Infernux
        return _Infernux._watch_graph_events(self.get_native_engine(), label)

    def _take_graph_events(self):
        from infernux.lib import _Infernux
        return _Infernux._take_graph_events(self.get_native_engine())

    def _is_graph_submission_completed(self, submission):
        from infernux.lib import _Infernux
        return _Infernux._is_graph_submission_completed(self.get_native_engine(), submission)

    def _dispatch_runtime_input(self):
        """The platform consumes its captured input inside the native World scope."""

    def _consume_runtime_frame_barrier(self, barrier):
        if not isinstance(barrier, RuntimeFrameBarrier):
            barrier = RuntimeFrameBarrier(barrier)
        if barrier in (RuntimeFrameBarrier.RENDER_SUBMITTED, RuntimeFrameBarrier.RENDER_PRESENTED):
            from infernux.runtime_services import _dispatch_updates
            changes = None
            if barrier == RuntimeFrameBarrier.RENDER_SUBMITTED:
                changes = self._runtime_scheduler.consume_runtime_changes(barrier)
                if self._graph_event_signals is not None:
                    from infernux.rendergraph.events import _dispatch_graph_events
                    _dispatch_graph_events(self)
            callbacks = self._clock_callbacks[barrier.value]
            if callbacks:
                _dispatch_updates(callbacks, self._clock_delta, self._clock_delta, gameplay_phase=False)
            return changes
        scheduler = self._runtime_scheduler
        if barrier == RuntimeFrameBarrier.SNAPSHOT_PUBLICATION:
            scene_manager = getattr(self, "_runtime_scene_manager", None)
            getter = getattr(scene_manager, "get_global_transform_serial", None)
            if callable(getter):
                current_serial = int(getter())
                previous_serial = getattr(
                    self,
                    "_last_native_transform_serial",
                    None,
                )
                self._last_native_transform_serial = current_serial
                if previous_serial is not None and current_serial != previous_serial:
                    journal = scheduler.change_journal
                    journal.publish(RuntimeChangeDomain.TRANSFORM_LOCAL, broad=True)
                    journal.publish(RuntimeChangeDomain.TRANSFORM_WORLD, broad=True)
        changes = scheduler.consume_native_barrier(barrier)
        if barrier == RuntimeFrameBarrier.TRANSFORM_TO_PHYSICS:
            if changes is not None:
                scheduler.execute_native_phase(
                    "physics_pre_step",
                    float(self._runtime_scene_manager.get_fixed_time_step()),
                )
        elif barrier == RuntimeFrameBarrier.ANIMATION_EVALUATION:
            if changes is not None:
                scheduler.execute_native_phase(
                    "animation_update", self._runtime_scene_manager.get_frame_delta_time(),
                )
        elif barrier == RuntimeFrameBarrier.PHYSICS_TO_TRANSFORM:
            if changes is not None:
                scheduler.execute_native_phase(
                    "physics_post_step",
                    float(self._runtime_scene_manager.get_fixed_time_step()),
                )
            # Simulation-owned pose anchors join the same authoritative
            # physics-to-Transform publication point as native rigidbodies.
            # The Transform frame cache is still active here, so publication
            # is applied atomically before Update/LateUpdate and rendering.
            from infernux.compute import _poll_transform_bindings
            _poll_transform_bindings()
        if barrier == RuntimeFrameBarrier.RENDER_EXTRACTION:
            scene_manager = getattr(self, "_runtime_scene_manager", None)
            if scene_manager is not None and (
                not scene_manager.is_playing() or scene_manager.is_paused()
            ):
                # Edit mode and paused Play have no physics barrier. Keep the
                # same Transform-anchor authority exchange alive immediately
                # before their render extraction so W/E/R remains interactive.
                from infernux.compute import _poll_transform_bindings
                _poll_transform_bindings()
            self._on_render_extraction()
        return changes

    @property
    def _runtime_scheduler(self):
        manager = self._runtime_scene_manager
        if manager is None:
            return self._authoring_scheduler
        identity = int(manager._current_runtime_world_id)
        domain = self._world_runtimes.get(identity)
        if domain is not None:
            return domain.scheduler
        if identity != int(manager._authoring_runtime_world_id):
            raise KeyError(identity)
        return self._authoring_scheduler

    def get_runtime_execution_scheduler(self):
        """Get the shared phase-plan service for diagnostics and profiling."""
        return self._runtime_scheduler

    def get_runtime_change_journal(self):
        """Return the typed invalidation stream shared by Editor and Player."""
        return self._runtime_scheduler.change_journal
