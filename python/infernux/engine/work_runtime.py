"""Shared numerical services for Editor and platform Player hosts.

The host supplies its one native FrameWorkOwner and frame wake operation.
Buffers, requests, budgets and result consumers retain the same implementation
on every platform; this class owns no device, worker pool or event loop.
"""


class WorkRuntime:
    def _initialize_work_runtime(self, frame, *, accepting=False):
        self._frame_work = frame
        self._cpu_range_tasks = set()
        self._cpu_range_accepting = accepting
        self._cpu_work_owner = None
        self._result_consumers = []
        self._graph_event_signals = None

    def _shutdown_work_runtime(self):
        cpu_owner = self._cpu_work_owner
        for consumer in self._result_consumers:
            consumer.cancel()
        self._result_consumers.clear()
        if cpu_owner is not None:
            cpu_owner.shutdown_begin()
        self._cpu_range_accepting = False
        if self._cpu_range_tasks:
            from infernux.cpu_execution import _shutdown_cpu_ranges

            _shutdown_cpu_ranges(self)
        if cpu_owner is not None:
            # The common owner waits only for its already accepted GPU tickets
            # here, while the host is live; normal frame retirement never waits.
            cpu_owner.shutdown_finish()
            self._cpu_work_owner = None

    def _cpu_work(self, budget=None):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()
        from infernux.engine.cpu_continuation import CpuWorkOwner
        owner = self._cpu_work_owner
        if owner is None:
            owner = CpuWorkOwner(self, budget)
            self._cpu_work_owner = owner
        elif budget is not None:
            owner.configure_budget(budget)
        owner.wake()
        return owner

    @property
    def work_budget(self):
        """Shared host-work limits for compute and IO.

        Assign a new Budget to tune running work. Accepted work keeps its
        completion ticket; increasing an allowance takes effect next frame.
        Individual submit calls inherit this policy when budget is omitted.
        """
        return self._cpu_work().budget

    @work_budget.setter
    def work_budget(self, budget):
        self._cpu_work().set_budget(budget)

    @property
    def _service_opportunity(self):
        return self._frame_work.opportunity

    def _wake_cpu_request(self, owner, request):
        try:
            self._request_full_speed_frame()
        except BaseException:
            owner._cancel(request, "failed", "initial owner wake failed")
            owner._retire(request)
            raise
        return request

    def _request_nn_inference(self, model, x, *, window_opportunities, budget=None, result_device=None):
        """Continue a cold-prepared CPU or GPU forward on the shared owner."""
        owner = self._cpu_work(budget)
        request = owner.release_nn(model, x, window_opportunities=window_opportunities, result_device=result_device)
        return self._wake_cpu_request(owner, request)

    def _request_nn_training(self, model, x, target, optimizer, *, window_opportunities, record_loss=False, budget=None,
                             result_device=None, objective="mse", clear_grad=False):
        """Continue one prepared MLP step; GPU local VJP remains a whole operation."""
        owner = self._cpu_work(budget)
        request = owner.release_training(model, x, target, optimizer,
            window_opportunities=window_opportunities, record_loss=record_loss, result_device=result_device,
            objective=objective, clear_grad=clear_grad)
        return self._wake_cpu_request(owner, request)

    def _apply_result(self, request, targets, apply, *, boundary, required):
        from infernux.engine.cpu_continuation import CpuRequest
        from infernux.execution import ResultConsumer, _active_result_boundary, capture_execution_scope
        from infernux.lib import SceneManager
        if not isinstance(request, CpuRequest) or request._owner() is not self._cpu_work_owner:
            raise TypeError('a result consumer requires a request from this Engine')
        if not callable(apply):
            raise TypeError('a result consumer requires an application callback')
        targets = tuple(targets)
        if not targets:
            raise ValueError('a result consumer requires its target components')
        manager = SceneManager.instance()
        if boundary == 'physics' and not manager.is_playing():
            raise RuntimeError('physics result application requires a Play world')
        scope = capture_execution_scope()
        if (required and boundary == 'render' and (scope is None or scope.state != 'paused')
                and _active_result_boundary.get() == boundary and
                self._cpu_work_owner._waits_for_render_input(request)):
            raise ValueError('result dependency cycle: the active boundary prevents its input render submission')
        consumer = ResultConsumer(request, targets, apply, manager, boundary=boundary, required=required)
        self._result_consumers.append(consumer)
        manager._set_result_consumers_present(True)
        if boundary != 'physics' and (required or request.done):
            manager._retain_step_frame()
        return consumer

    def _retire_result_consumers(self):
        from infernux.lib import SceneManager
        manager = SceneManager.instance()
        self._result_consumers[:] = [consumer for consumer in self._result_consumers if consumer._current(manager)]
        manager._set_result_consumers_present(bool(self._result_consumers))
