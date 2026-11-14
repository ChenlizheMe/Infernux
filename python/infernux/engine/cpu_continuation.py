"""One host credit for CPU requests and automatic Compute FIFO admission.

Declared scene sources, static programs and model IO share credit, capacity and retirement.
Only frozen POD reaches a worker. Existing JobSystem/FIFO own execution; idle never imports the compiler.
"""
from dataclasses import dataclass, field
from functools import wraps
from itertools import count
from math import isfinite
from time import perf_counter_ns
import weakref

import numpy as np

from infernux.compute import Buffer, _BufferCapacityPending, _storage_blocks
from infernux.compute_source import BufferSource
from infernux.lib import _Infernux as _native
from .service_windows import (ExecutionCosts, ServiceCalendar, ServiceDemand, CpuSegment, GpuSegment,
                              execution_key, transfer_key, model_spans, remaining_model_service,
                              remaining_transfer_service, _service_availability, choose_transfer_range,
                              gpu_prepare_key, gpu_submit_key, _uses)
from .io_service import IoObservation, io_host_key
from infernux.execution import capture_execution_scope


_model_output_ids = count(1)


@dataclass(frozen=True)
class CpuWorkBudget:
    """Host credit, capacity, and separate soft execution targets.

    CPU/GPU/IO targets use measured execution, separate from host credit.
    A default best-effort Store read can stream into its admitted CPU storage
    between host frames. io_frame_ns opts into a soft per-host IO execution
    target for that work. io_slice_ns sets a fixed quantum and disables borrowing.
    One accepted native entry/dispatch/IO action cannot be interrupted here.
    Planning spends the same host credit. Unprofiled work does not receive a
    completion reservation; a wider release window is not a resource limit.
    """
    owner_ns: int = 1_000_000
    foreground_reserve_ns: int = 0
    capacity_bytes: int = 16 * 1024 * 1024
    max_requests: int = 8
    snapshot_rows: int = 4096
    snapshot_ns_per_row: int = 500
    range_elements: int = 64 * 1024
    range_terms: int = 1024
    snapshot_bytes: int = 1024 * 1024
    transfer_bytes: int = 1024 * 1024
    cpu_slice_ns: int = 250_000
    gpu_slice_ns: int = 250_000
    io_slice_ns: int | None = None
    cpu_frame_ns: int = 1_000_000
    gpu_frame_ns: int = 1_000_000
    io_frame_ns: int | None = None
    transfer_frame_bytes: int = _native._FrameWorkOwner.default_transfer_frame_bytes
    segment_operations: int = 8

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for name, value in self.__dict__.items()
               if name != "foreground_reserve_ns" and (name not in ("io_slice_ns", "io_frame_ns") or value is not None)):
            raise ValueError("CPU work limits must be positive integers")
        if type(self.foreground_reserve_ns) is not int or not 0 <= self.foreground_reserve_ns < self.owner_ns:
            raise ValueError("foreground reserve must be an integer in [0, owner_ns)")

    @property
    def _io_frame_allowance_ns(self):
        return self.io_frame_ns or 1_000_000

    @property
    def _io_quantum_ns(self):
        return min(self.io_slice_ns or 250_000, self._io_frame_allowance_ns)


def _metrics():
    return {name: 0 for name in (
        "owner_ns", "owner_overrun_ns", "snapshot_ns", "snapshot_bytes",
        "pack_traffic_bytes", "payload_peak_bytes",
        "completed_elements", "worker_end_to_end_ns", "worker_execution_ns",
        "accepted_submissions")}


@dataclass(slots=True)
class _RequestStorage:
    retained: object
    private_blocks: dict
    reservation_bytes: int
    cold: object = None
    prepare_bytes: int = 0
    native_blocks: object = ()


@dataclass(frozen=True, slots=True)
class _FifoReceipt:
    completion: object
    key: tuple
    submitted_frame: int
    gpu_grant: int
    transfer_grant: int
    transfer_bytes: int
    storage_blocks: tuple


@dataclass(eq=False, kw_only=True)
class CpuRequest:
    """Managed work with owner-thread release and retirement observations.

    ``released_ns`` and ``metrics['retired_ns']`` use the process monotonic
    clock. ``metrics['retired_host_frame']`` identifies the active host frame,
    or is None when retirement runs outside the host loop.
    """

    release_opportunity: int
    deadline: int | None
    reserved_bytes: int
    _owner: object
    deadline_ns: int | None = None
    released_ns: int | None = None
    execution_scope: object = None
    paused_at: int | None = None
    _forecast_deferred: bool = False
    stepping: bool = False
    step_permission: tuple | None = None
    required_consumers: int = 0
    submission_stamp: object = None
    status: str = "pending"
    reason: str = ""
    result: object = None
    task: object = None
    task_end: int = 0
    submitted_ns: int | None = None
    packet: object = None
    gpu_preparation: object = None
    gpu_argument_bytes: int = 0
    host: object = None
    range_count: int = 0
    execution_costs: tuple = ()
    ranges: tuple = ()
    execution_prediction: float = 0
    gpu_calibration_floor_ns: float = 0
    transfer_bytes: int = 0
    transfer_indivisible: bool = False
    execution_grant: int = 0
    transfer_grant: int = 0
    indivisible: bool = False
    preparation_key: tuple | None = None
    preparation_units: int | None = None
    preparation_ns: int = 0
    residency_deferred: bool = False
    metrics: dict = field(default_factory=_metrics)
    input_bindings: object = None
    _retired: bool = field(default=False, init=False)
    _forecast_key: tuple = field(default=(), init=False, repr=False)
    _forecast_work: object = field(default=None, init=False, repr=False)
    service_terms: int = 0

    @property
    def done(self):
        """All accepted work has retired, including a cancelled request's IO."""
        return self._retired

    def cancel(self):
        owner = self._owner()
        if owner is not None:
            owner._check_owner()
            started = perf_counter_ns()
            owner._frame.begin_scope()
            try:
                owner._frame.wake(owner._source)
                return owner._cancel(self, "cancelled", "withdrawn by caller")
            finally:
                owner._charge_host_interval(started, self)
        return False

    def _model_access(self, root):
        return None

    def _blocked(self, owner):
        return False

    def _input_blocked(self, owner):
        return (self.input_bindings is not None and not self.input_bindings.ready) or self._blocked(owner)

    def _release_input_bindings(self):
        if self.input_bindings is not None:
            self.input_bindings.close()
            self.input_bindings = None

    def _execution_domain(self):
        return 'cpu' if self.input_bindings is not None else self._domain()

    def _demand(self, owner, opportunity, capacity_bytes):
        if self.residency_deferred:
            return ServiceDemand(reason="waiting for Vulkan buffer capacity")
        if self.input_bindings is not None:
            reason = ('input binding needs host preparation' if self.input_bindings.ready else
                      'input producer has not published its value')
            return ServiceDemand(reason=reason, prerequisites=self._dependencies(owner))
        return self._service_demand(owner, opportunity, capacity_bytes=capacity_bytes)

    def _dependencies(self, owner):
        return () if self.input_bindings is None else self.input_bindings.producers

    def _poll_external(self, owner, opportunity):
        pass

    def _io_tasks(self):
        return ()

    def _io_ready(self, owner):
        return False

    def _parallel_io_ready(self, owner):
        return False

    def _cancel_external(self):
        return True

    def _retire_external(self):
        return True

    def _wait_external(self, owner):
        pass

    def _on_failure(self, owner, reason):
        pass

    def _domain(self):
        return "cpu"

    def _gpu_preparation_bytes(self):
        if self.gpu_preparation is not None:
            return self.gpu_preparation.reservation_bytes
        return self._owner().budget.segment_operations * self.gpu_argument_bytes

    def _prepared_transfer(self):
        return None

    def _discard_preparation(self, owner):
        if self.packet is not None and self.packet.completion is not None:
            return
        if self.task is not None and (self.task._native is None or self.task._native.submitted):
            return
        if self.gpu_preparation is None and self.packet is None and self.task is None:
            return
        if self.task is not None:
            self.task.cancel()
            self.task._poll()
            self.task = None
        if self.gpu_preparation is not None:
            self.gpu_preparation.close()
            self.gpu_preparation = None
        if self.packet is not None:
            from infernux.compute import _cancel_work_commands
            owner._gpu_busy_packets.pop(self.packet, None)
            _cancel_work_commands(self.host, self.packet)
            self.packet = None
            cursor = self._prepared_transfer()
            if cursor is not None:
                cursor.discard_prepared()
        # Re-select only unexecuted work. Confirmed rows, carry, transfer
        # storage and accepted tickets keep their original positions.
        self.ranges = self.execution_costs = ()
        self.execution_prediction = self.gpu_calibration_floor_ns = self.transfer_bytes = 0
        self.indivisible = self.transfer_indivisible = False
        self.preparation_key = self.preparation_units = None
        self.preparation_ns = 0
        self.metrics['preparation_replans'] = self.metrics.get('preparation_replans', 0) + 1

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        return ServiceDemand(reason="request stages do not yet have a complete service profile")

    def _service_host_key(self):
        return None

    def _observed_host_key(self, key, progress):
        return key

    def _service_host_units(self, key):
        return None

def _advance_transfer(cursor, owner, request):
    """The same bounded transfer step for a standalone value or a model edge."""
    if not owner._available(background=True) or cursor.done:
        return None, False
    metrics = request.metrics
    progressed = cursor.target is None
    cursor.ensure_target()
    if not owner._available(background=True):
        return None, progressed
    step = cursor.next_step()
    begin, end = step.begin, step.end
    if step.kind == "materialize":
        before = cursor.completed_bytes
        cursor.materialize()
        metrics["materialized_bytes"] = metrics.get("materialized_bytes", 0) + cursor.completed_bytes - before
    elif step.kind == "copy_host":
        cursor.copy_host_range(begin, end)
        metrics["pack_traffic_bytes"] += 2 * (end - begin)
    else:
        return (begin, end), progressed
    return None, True


def _is_transfer_host_key(key):
    return key is not None and key[-9:-8] == ("transfer-host",)


def _slice_work(cursor, term_limit):
    # slice may initialize persistent carry storage or select its first inner
    # range. Rebinding the same ready operation alone is not advancement.
    before = bool(cursor.carry), cursor.inner_end
    operation = cursor.slice(term_limit, managed=True)
    return operation, before != (bool(cursor.carry), cursor.inner_end)


def _seal_gpu_costs(request, costs, samples):
    request.execution_costs = tuple(costs)
    request.execution_prediction = sum(value or 0 for _, _, value in costs) * samples.packet_scale(costs)
    key, rows, first_prediction = costs[0]
    # A wider calibration can be unpriced even after a smaller range has a
    # receipt. Keep that measured floor with the sealed packet.
    request.gpu_calibration_floor_ns = samples.calibration_floor(key, rows) if first_prediction is None else 0


def _prepare_transfer(request, owner, cursor, work):
    key = transfer_key(*cursor.cost_identity)
    unit = cursor.dtype.itemsize
    rows, prediction = choose_transfer_range(owner, cursor.cost_identity, unit, work[1]-work[0],
        byte_allowance=owner._frame.resource_remaining(owner._resource('transfer')),
        gpu_allowance=owner._frame.resource_remaining(owner._resource('gpu')))
    if not rows:
        request.metrics['transfer_budget_deferrals'] = request.metrics.get('transfer_budget_deferrals', 0)+1
        return False
    request.packet = cursor.prepare(work[0], work[0]+rows)
    _seal_gpu_costs(request, ((key, rows, prediction),), owner._costs)
    request.transfer_bytes = rows
    request.indivisible = rows == unit
    request.transfer_indivisible = rows == unit
    return True


def _observed_numerical_host_key(request, key, progress):
    if request.status == "published":
        return key
    # GPU preparation/submission have their own sampler. This is the combined
    # CPU entry: a retained binding alone is not its complete host sample.
    if progress[3] or request.metrics["accepted_submissions"] == progress[2]:
        return None
    return key


def _prepare_cpu_work_ranges(request, cursor, owner):
    from infernux.cpu_execution import _prepare_cpu_ranges
    segment = CpuSegment(owner, host_allowance=owner._remaining_credit(background=True),
                         execution_allowance=owner._frame.resource_remaining(owner._resource('cpu')))

    def select(operation, begin, count):
        end = segment.select(execution_key(operation.program, operation.params), begin, count)
        return () if end is None else (end,)

    request.ranges = cursor.preview(cursor.slice(owner.budget.range_terms),
        lambda operation: len(operation.params[operation.program.domain_parameter]),
        select, segment.limit, owner.budget.range_terms)
    request.indivisible = all(work.operation.reverse or work.end - work.begin == 1 for work in request.ranges)
    request.execution_costs = tuple(segment.costs)
    request.execution_prediction = sum(prediction or 0 for _, _, prediction in segment.costs)
    started = perf_counter_ns()
    request.task = _prepare_cpu_ranges((value.operation.program, value.operation.params, value.begin, value.end)
                                       for value in request.ranges)
    elapsed = perf_counter_ns() - started
    owner._costs.observe_host(('cpu', 'prepare-operation'), elapsed / len(request.ranges))
    request.metrics['cpu_prepare_ns'] = request.metrics.get('cpu_prepare_ns', 0) + elapsed
    request.metrics['prepared_operations'] = request.metrics.get('prepared_operations', 0) + len(request.ranges)
    request.metrics['max_segment_operations'] = max(request.metrics.get('max_segment_operations', 0), len(request.ranges))
    return True


def _confirm_work_ranges(request, cursor):
    request.metrics["completed_elements"] += sum(work.end - work.begin for work in request.ranges)
    whole = sum(work.operation.reverse for work in request.ranges)
    if whole:
        request.metrics["whole_vjp_completions"] = request.metrics.get("whole_vjp_completions", 0) + whole
    cursor.confirm(request.ranges)
    request.ranges = ()


def _prepare_gpu_work_ranges(request, cursor, entries, owner):
    from infernux.compute import _WorkPreparation
    from infernux._work import gpu_arguments

    if request.gpu_preparation is not None:
        return _advance_gpu_preparation(request, owner)
    pending_profiles = set()
    for pending in owner.requests:
        packet = pending.packet
        if packet is None or len(pending.execution_costs) != 1:
            continue
        if packet.completion is None and (pending.status != 'pending' or pending.paused_at is not None):
            continue
        key, rows, _ = pending.execution_costs[0]
        if owner._costs.needs_probe(key, rows):
            pending_profiles.add(key)
    segment = GpuSegment(owner,
                         execution_allowance=owner._frame.resource_remaining(owner._resource('gpu')),
                         transfer_allowance=owner._frame.resource_remaining(owner._resource('transfer')),
                         pending_profiles=pending_profiles)

    def select(operation, begin, count):
        entry = entries[operation.program, operation.reverse]
        return segment.select_rows(execution_key(entry, operation.params), begin, count,
                                   argument_bytes=int(entry.artifact.argument_layout['size']),
                                   max_dispatch_rows=entry.max_dispatch_rows, dispatch_grain=entry.dispatch_grain,
                                   whole=operation.reverse)

    def domain(operation):
        entry = entries[operation.program, operation.reverse]
        return operation.params[entry.artifact.domain_parameter].element_count

    request.ranges = cursor.preview(cursor.slice(owner.budget.range_terms, managed=True), domain, select,
                                    segment.limit, owner.budget.range_terms)
    request.indivisible = segment.indivisible
    launches = []
    for value in request.ranges:
        operation = value.operation
        arguments, gradients = gpu_arguments(operation.params, reverse=operation.reverse)
        entry = entries[operation.program, operation.reverse]
        launches.append((entry, arguments, None if operation.reverse else (value.begin, value.end), gradients))
    _seal_gpu_costs(request, segment.costs, owner._costs)
    request.transfer_bytes = sum(int(entry.artifact.argument_layout['size']) for entry, _, _, _ in launches)
    request.transfer_indivisible = len(launches) == 1
    request.gpu_preparation = _WorkPreparation(launches)
    request.metrics["max_segment_operations"] = max(request.metrics.get("max_segment_operations", 0), len(launches))
    return _advance_gpu_preparation(request, owner)


def _advance_gpu_preparation(request, owner):
    preparation = request.gpu_preparation
    operations = preparation.advance(owner._remaining_credit(background=True), owner._gpu_prepare_unit_ns,
        owner._costs.predict(('host-service', 'gpu', 'seal'), len(preparation.launches)) or 0)
    if operations:
        owner._costs.observe_host(("gpu", "prepare-operation"), preparation.binding_ns / operations)
        request.metrics["prepared_operations"] = request.metrics.get("prepared_operations", 0) + operations
    if preparation.seal_ns is not None:
        owner._costs.observe(('host-service', 'gpu', 'seal'), len(preparation.launches), preparation.seal_ns)
    if preparation.packet is None:
        return bool(operations)
    request.packet = preparation.packet
    request.transfer_bytes = request.packet.transfer_bytes
    request.transfer_indivisible = len(preparation.launches) == 1
    request.metrics["prepared_argument_bytes"] = request.metrics.get("prepared_argument_bytes", 0) + request.packet.payload_bytes
    preparation.close()
    request.gpu_preparation = None
    return True


def _input_transfer_key(value):
    if isinstance(value, BufferSource):
        return value._pin().transfer_key
    storage = value if isinstance(value, Buffer) else value._data
    if storage.device == 'cpu':
        return 'cpu', storage._array.ctypes.data, storage.nbytes, storage.dtype
    description = storage.description
    return ('gpu', int(storage._host.identity), description.resource_index,
            description.resource_generation, storage._byte_offset, storage.nbytes, storage.dtype)


@dataclass(eq=False, slots=True, weakref_slot=True)
class _ModelOutputLease:
    identity: tuple
    request_owner: weakref.ReferenceType
    shape: tuple
    dtype: str
    device: str
    producer: CpuRequest | None
    storage: Buffer | None = None
    failure: str = ''

    @property
    def done(self):
        return self.producer is None or self.producer.status != 'pending'

    @property
    def ready(self):
        return self.storage is not None and not self.storage.closed

    @property
    def error(self):
        if self.producer is not None:
            return self.producer.reason
        return '' if self.ready else self.failure or 'published model output has retired'

    @property
    def transfer_key(self):
        return _input_transfer_key(self.storage) if self.ready else self.identity

    @property
    def storage_claims(self):
        # Before publication the producer's live work and reservation already
        # cover the output. Afterwards charge its shared physical allocation.
        return _storage_blocks((self.storage,)) if self.ready else {}

    def acquire(self):
        return self.storage.view(readonly=True, shape=self.shape or (1,))


def _model_result_tensor(buffer, shape, **facts):
    from infernux.nn._tensor import Tensor

    # Model outputs own their activations or transfer destinations. Seal the
    # CPU allocation before borrowers receive views of the published value.
    if buffer.device == 'cpu':
        buffer._array.flags.writeable = False
    return Tensor._from_buffer(buffer, shape, **facts)


@dataclass(eq=False, kw_only=True)
class NnRequest(CpuRequest):
    plan: object
    state: tuple
    inputs: tuple | None
    parameter_version: int
    run: object = None
    optimizer: object = None
    target: object = None
    update_spec: object = None
    grad_facts: tuple = ()
    record_loss: bool = False
    objective: object = "mse"
    clear_grad: bool = False
    sources: object = None
    phase: str = "forward"
    grad_published: bool = False
    gpu_executables: dict = field(default_factory=dict)
    result_device: str | None = None
    outputs: int | tuple[int, ...] | None = None
    output_index: int | None = None
    io: list = field(default_factory=list)
    io_storage: list = field(default_factory=list)
    result_storage: dict = field(default_factory=dict)
    result_ready: bool = False
    service_spans: tuple = ()
    service_signature: tuple = ()
    _output_id: int = field(default=0, init=False)
    _output_leases: weakref.WeakValueDictionary | None = field(default=None, init=False, repr=False)

    def output_source(self, index: int = 0) -> BufferSource:
        """Bind a result port as a deferred input on the Engine owner thread.

        Ports follow the returned value order. A training program that returns
        its loss exposes that scalar as port zero. Closing the source releases
        the caller's reference, and cancelling the request cancels its outputs.
        """
        from infernux.nn._plan import _result_values

        if (owner := self._owner()) is not None:
            owner._check_owner()
        if type(index) is not int or index < 0:
            raise ValueError('model output index must be a nonnegative integer')
        if self.status == 'published':
            ports = () if self.result is None else _result_values(self.result)
        elif self.status == 'pending':
            ports = (tuple(self.plan.result_specs[port] for port in _result_values(self._output_selection()))
                     if self.optimizer is None else (None,) if self.record_loss else ())
        else:
            raise RuntimeError(self.reason or 'model request has no public output')
        if index >= len(ports):
            raise ValueError('model output index is outside the returned ports')
        value = ports[index]
        # Repeated result ports are the same value and share a transfer lease.
        index = next(i for i, port in enumerate(ports) if port is value)
        if self._output_leases is None:
            self._output_id = next(_model_output_ids)
            self._output_leases = weakref.WeakValueDictionary()
        lease = self._output_leases.get(index)
        if lease is None:
            shape, dtype = ((), 'float32') if value is None else (value.shape, np.dtype(value.dtype).name)
            producer = self if self.status == 'pending' else None
            lease = _ModelOutputLease(('model-output', self._output_id, index), self._owner,
                shape, dtype, self.result_device, producer, None if producer is not None else value._data)
            self._output_leases[index] = lease
        return BufferSource(lease)

    def _settle_output_sources(self, result, reason=''):
        if not self._output_leases:
            return
        values = result if isinstance(result, tuple) else (result,)
        for index, lease in self._output_leases.items():
            lease.storage = None if result is None else values[index]._data
            lease.failure = reason
            lease.producer = None

    def _publish_result(self, result, opportunity):
        self._settle_output_sources(result)
        self.result = result
        self.status = 'published'
        self.metrics['terminal_opportunity'] = opportunity

    def _bind_inputs(self, values, owner):
        from infernux.nn._tensor import Tensor
        for name, buffer in values.items():
            previous = self.target if name == 'target' else self.inputs[name]
            value = Tensor._from_buffer(buffer, previous.shape, requires_grad=previous.requires_grad)
            if name == 'target':
                self.target = value
            else:
                self.inputs = (*self.inputs[:name], value, *self.inputs[name+1:])
        if self.gpu_executables:
            owner._gpu_admission(self.gpu_executables,
                tuple(value._data for value in (*self.inputs, self.target) if value is not None and value.device == 'gpu'))
        self._configure_io(self.result_device, owner, admitted=True)

    def _service_host_key(self):
        if self.sources is not None and not self.sources.done:
            return "nn-source", self.sources.service_key
        if not self.service_signature:
            self.service_signature = (self.plan.service_identity, self.plan.device, self.optimizer is not None,
                                      self.record_loss, tuple(old is None for _, old in self.grad_facts),
                                      None if self.optimizer is None else self.optimizer._kind, self.objective, self.clear_grad,
                                      self.update_spec is not None and self.update_spec.max_norm is not None,
                                      self.output_index if self.optimizer is not None else self._output_selection())
        if self.io:
            names, cursor = self.io[0]
            step = cursor.next_step(prepared=self.packet is not None)
            return (*self._transfer_service_tag(names), *cursor.host_cost_key(step,
                finalizing=step.domain == "host" and step.end == cursor.nbytes))
        run = self.run
        if run is None:
            phase = "forward"
        elif run.cursor.current() is not None:
            phase = run.phase
        elif run.phase == "forward" and self.optimizer is not None and self.record_loss:
            phase = "loss"
        elif run.phase in ("forward", "loss") and self.optimizer is not None:
            phase = "backward"
        elif run.phase == "backward":
            phase = "update"
        else:
            phase = "result-bind" if self._needs_result_transfer() and not self.result_storage else "publish"
        return (*self.service_signature, phase)

    def _transfer_service_tag(self, names):
        return (("nn-input-transfer", len(names)) if isinstance(names, tuple)
                else ("nn-result-transfer", self.service_signature))

    def _observed_host_key(self, key, progress):
        if key[0] == "nn-source":
            return key if self.sources is not None and self.sources.done else None
        if _is_transfer_host_key(key):
            if self.metrics.get("model_io_actions", 0) == progress[4]:
                return None
            return key
        if key[-1] == "result-bind":
            return key if self.io and not isinstance(self.io[0][0], tuple) else None
        return _observed_numerical_host_key(self, key, progress)

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        if self.sources is not None and not self.sources.done:
            return ServiceDemand(reason="scene source snapshot has not completed")
        self._service_host_key()
        if not self.service_spans or self.service_terms != owner.budget.range_terms:
            self.service_spans = model_spans(self.plan, self.optimizer, self.grad_facts,
                self.update_spec, self.record_loss, self.gpu_executables, owner.budget.range_terms, self.objective,
                self.clear_grad, output_index=self.output_index,
                frontier=None if self.run is None else (self.run.phase, self.run.cursor, self.ranges))
            self.service_terms = owner.budget.range_terms
            self._forecast_key = ()
        return remaining_model_service(self, owner, opportunity, capacity_bytes=capacity_bytes)

    def _prepared_transfer(self):
        return self.io[0][1] if self.io else None if self.run is None else self.run.cursor.transfer

    def _model_access(self, root):
        return "train" if self.optimizer is not None and self.plan is not None and self.plan.root is root else None

    def _needs_result_transfer(self):
        return (self.result_device is not None and self.result_device != self.plan.device
                and (self.optimizer is None or self.record_loss))

    def _output_selection(self):
        return self.plan.result if self.outputs is None else self.outputs

    def _result_transfer_layouts(self):
        from infernux.nn._plan import _result_values
        if self.optimizer is not None:
            return (("loss", 4, np.dtype("float32")),)
        return tuple((index, self.plan.result_specs[index].nbytes, np.dtype(self.plan.result_specs[index].dtype))
                     for index in dict.fromkeys(_result_values(self._output_selection())))

    def _result_transfer_bytes(self):
        return sum(nbytes for _, nbytes, _ in self._result_transfer_layouts()) if self._needs_result_transfer() else 0

    def _compute_complete(self):
        return (self.run is not None and self.run.cursor.current() is None
                and (self.optimizer is None or self.run.phase == "ready_to_commit"))

    def _domain(self):
        if self.sources is not None and not self.sources.done:
            return "cpu"
        if self.io:
            return self.io[0][1].execution_domain
        if self.run is not None and self.run.phase == "update" and self.run.cursor.current() is None:
            return "cpu"  # Exact final receipt is observed; only host commit preparation remains.
        return "cpu" if self._compute_complete() else self.plan.device

    def _configure_io(self, result_device, owner, *, admitted=False):
        from infernux.compute import _TransferCursor
        if result_device not in (None, "cpu", "gpu"):
            raise ValueError("NN result_device must be 'cpu' or 'gpu'")
        self.result_device = self.plan.device if result_device is None else result_device
        groups = {}
        for name, value in (*enumerate(self.inputs), ("target", self.target)):
            if value is not None and value.device != self.plan.device:
                storage = value._data
                key = _input_transfer_key(value)
                groups.setdefault(key, (storage, []))[1].append(name)
        for storage, names in groups.values():
            # Exact repeated regions share one transfer. Shape remains a fact
            # of each logical input; sharing never packs unrelated fields.
            self.io.append((tuple(names), _TransferCursor(storage, self.plan.device, owner.transfer_chunk_bytes)))
            self.reserved_bytes += storage.nbytes
        self.reserved_bytes += self._result_transfer_bytes()
        if self.io:
            capacity = owner._capacity(None if admitted else self)
            if self.input_bindings is not None:
                capacity -= self.input_bindings.extra_bytes
            peak = max(cursor.prepare_bytes for _, cursor in self.io)
            available = owner.budget.capacity_bytes - capacity + max(0, peak - self._gpu_preparation_bytes())
            for _, cursor in self.io:
                cursor.limit_chunk(available)

    def _adopt_io_input(self, names, cursor):
        from infernux.nn._tensor import Tensor
        target = cursor.target
        for name in names:
            previous = self.target if name == "target" else self.inputs[name]
            value = Tensor._from_buffer(target, previous.shape, requires_grad=previous.requires_grad)
            if name == "target":
                self.target = value
            else:
                self.inputs = (*self.inputs[:name], value, *self.inputs[name + 1:])
            self.io_storage.append(value._data)
            previous._data._close(flush=False)
        cursor.target = None
        target._close(flush=False)
        cursor.close()
        self.io.pop(0)

    def _adopt_io_result(self, index, cursor):
        target = cursor.target
        self.result_storage[index] = target
        self.io_storage.append(target)
        cursor.target = None
        cursor.close()
        self.io.pop(0)

    def _resources(self):
        values = [*(value._data for value in self.inputs or () if not isinstance(value, BufferSource)), *self.state, *_plan_storage(self.plan)]
        private = []
        if self.sources is not None:
            values.extend(self.sources.buffers)
            private.extend(self.sources.buffers)
        values.extend(self.io_storage)
        private.extend(self.io_storage)
        native_blocks = list(self.plan.native_blocks)
        io_peak = 0
        for _, cursor in self.io:
            values.extend((cursor.source, cursor.target, cursor.snapshot))
            private.extend((cursor.target, cursor.snapshot))
            native_blocks.extend(cursor.native_blocks)
            if not cursor.readback_ready and not cursor.done and self.packet is None:
                io_peak = max(io_peak, cursor.prepare_bytes)
        if self.run is not None:
            live_group, private_group = self.run.storage_groups()
            values.extend((live_group, *self.run.cursor.carry))
            private.extend((private_group, *self.run.cursor.carry))
        if self.optimizer is not None:
            if self.target is not None and not isinstance(self.target, BufferSource):
                values.append(self.target._data)
            values.extend(self.optimizer._state_values(self.update_spec.state))
            values.extend(gradient._data for _, gradient in self.grad_facts if gradient is not None)
        cold = (self.plan, self.plan._first_outputs or ()) if self.run is None else None
        peak = self._gpu_preparation_bytes() if (self.gpu_executables and self.packet is None
            and self.status == "pending" and not self._compute_complete()) else 0
        private_blocks = _storage_blocks(private)
        reservation = self._reservation(private_blocks)
        return _RequestStorage(values, private_blocks, reservation, cold, max(peak, io_peak), native_blocks)

    def _reservation(self, private_blocks):
        run = self.run
        if run is None or (self.optimizer is not None and run.phase in ("forward", "loss")):
            return self.reserved_bytes
        current = sum(private_blocks.values())
        carry = sum(_storage_blocks(run.cursor.carry).values())
        owner = self._owner()
        scratch = run.cursor.scratch_reservation(owner.budget.range_terms)
        bound = current - carry + max(carry, scratch)
        output = self._result_transfer_bytes() - sum(value.nbytes for value in self.result_storage.values())
        output -= sum(cursor.target.nbytes for name, cursor in self.io
                      if not isinstance(name, tuple) and cursor.target is not None)
        if self.optimizer is None:
            scalar = self.plan.scalar_storage_bytes if run.cursor.current() is not None else 0
            return max(bound + output + scalar, 0)
        if run.phase == "backward":
            parameters = self.plan.root._state
            grad_parameters = self.plan._result_spec(self.output_index).grad_parameters
            candidates = sum(value.nbytes for value, used, (_, old) in zip(
                parameters, grad_parameters, self.grad_facts) if used or (old is not None and not self.clear_grad)) * self.optimizer._candidate_multiplier
            if self.update_spec.max_norm is not None:
                shapes = tuple(value.shape for value, used, (_, old) in zip(
                    parameters, grad_parameters, self.grad_facts) if used or (old is not None and not self.clear_grad))
                candidates += self.optimizer._clip.storage_bytes(shapes, owner.budget.range_terms)
            gradients = sum(_storage_blocks(value._data for _, value in run.backward_bindings.updates).values())
            private_keys = _storage_blocks(self.io_storage)
            input_bytes = sum(size for key, size in _storage_blocks(value._data for value in self.inputs).items() if key in private_keys)
            if self.sources is not None:
                input_bytes += self.sources.nbytes
            loss = 0 if run.loss_output is None else run.loss_output.nbytes
            scalar = 4 if self.plan.device == "cpu" else 0
            return max(bound, gradients + candidates + input_bytes + loss + scalar + output)
        scalar = 4 if self.plan.device == "cpu" and run.cursor.current() is not None else 0
        return bound + scalar + output

    def _complete(self):
        return self._compute_complete() and (not self._needs_result_transfer() or self.result_ready)


    def _is_current(self, owner):
        root = self.plan.root
        expected_state = self.state if self.run is None or self.run.published_state is None else self.run.published_state
        valid = (self.plan._is_current()
                 and (self.optimizer is None or root._state is expected_state)
                 and all(not value._data.closed for value in self.inputs or () if not isinstance(value, BufferSource))
                 and all(cursor.is_current() for _, cursor in self.io))
        if self.optimizer is not None:
            from infernux.nn._plan import _grads_current
            facts = self.grad_facts if self.run is None else self.run.grad_facts
            valid = valid and root._optimizer() is self.optimizer and _grads_current(facts)
            if self.run is None or self.run.published_state is None:
                valid = valid and self.optimizer._state is self.update_spec.state
            if valid:
                try:
                    self.optimizer._check_domain()
                except (ValueError, KeyError, TypeError):
                    return False
        return valid

    def _work(self, owner, opportunity):
        if self.sources is not None and not self.sources.done:
            return None, self.sources.advance(owner, self.metrics)
        if self.io:
            name, cursor = self.io[0]
            before = cursor.completed_bytes
            work, progressed = _advance_transfer(cursor, owner, self)
            adopted = cursor.done
            if adopted:
                if isinstance(name, tuple):
                    self._adopt_io_input(name, cursor)
                else:
                    self._adopt_io_result(name, cursor)
                progressed = True
            if adopted or cursor.completed_bytes != before:
                self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
            return work, progressed
        if self.result_storage:
            # Price result adoption and publication as separate host actions.
            self.result_ready = True
            return None, False
        progressed = False
        if self.run is None:
            with self.plan.root._lock:
                if not self._is_current(owner):
                    raise RuntimeError("model state changed before forward binding")
                self.run = self.plan._bind_forward(self.inputs, retain_for_backward=self.optimizer is not None,
                                                   state=self.state, managed=bool(self.gpu_executables))
                if self.optimizer is not None:
                    run = self.run
                    run.optimizer, run.target, run.update_spec = self.optimizer, self.target, self.update_spec
                    run.objective = self.objective
                    run.output_index = self.output_index
                    run.clear_grad = self.clear_grad
                    run.grad_facts, run.record_loss = self.grad_facts, self.record_loss
                progressed = True
        phase = self.run.phase
        try:
            operation = self.run._next_operation()
        finally:
            self.phase, self.grad_published = self.run.phase, self.run.grad_published
            if self.optimizer is not None:
                self.grad_facts = self.run.grad_facts
        progressed |= self.run.phase != phase
        if operation is None:
            if self._compute_complete() and self._needs_result_transfer() and not self.result_storage:
                from infernux.compute import _TransferCursor
                for index, _, _ in self._result_transfer_layouts():
                    source = self.run.values[index] if self.optimizer is None else self.run.loss_output
                    self.io.append((index, _TransferCursor(source, self.result_device, owner.transfer_chunk_bytes)))
                peak = max(cursor.prepare_bytes for _, cursor in self.io)
                available = owner.budget.capacity_bytes - owner._capacity() + peak
                for _, cursor in self.io:
                    cursor.limit_chunk(available)
                progressed = True
            return None, progressed
        operation, sliced = _slice_work(self.run.cursor, owner.budget.range_terms)
        progressed |= sliced
        program, params = operation.program, operation.params
        if operation.copy:
            return (None, params, self.run.completed_end, params[0].nbytes), progressed
        if self.gpu_executables:
            program = self.gpu_executables[operation.program, operation.reverse]
            # The local reverse role retains the original primal parameters
            # first; its extra dx/seed bindings are mapped only when sealing.
            count = params[program.artifact.domain_parameter].element_count
        else:
            count = len(params[program.domain_parameter])
        return (program, params, self.run.completed_end, count), progressed

    def _consume_cpu(self):
        self._confirm_ranges()

    def _prepare_cpu(self, owner, work):
        return _prepare_cpu_work_ranges(self, self.run.cursor, owner)

    def _confirm_ranges(self):
        _confirm_work_ranges(self, self.run.cursor)

    def _consume_gpu(self):
        if self.io:
            cursor = self.io[0][1]
            before, end = cursor.completed_bytes, cursor.pending_end
            cursor.consume_execution()
            self.metrics["device_completed_bytes"] = self.metrics.get("device_completed_bytes", 0) + end - before
            return
        if self.run.cursor.current().copy:
            transfer = self.run.cursor.transfer
            before = transfer.completed_bytes
            transfer.consume_execution()
            self.metrics["copy_bytes"] = self.metrics.get("copy_bytes", 0) + transfer.completed_bytes - before
            self.run.cursor.completed_end = transfer.completed_bytes
            if transfer.completed_bytes == transfer.nbytes:
                self.run.cursor.complete()
            return
        self._confirm_ranges()

    def _prepare_gpu(self, owner, work):
        if self.io:
            if _prepare_transfer(self, owner, self.io[0][1], work):
                self.metrics["io_prepared_bytes"] = self.metrics.get("io_prepared_bytes", 0) + self.packet.payload_bytes
                return True
            return False
        program, params, begin, count = work
        if self.run.cursor.current().copy:
            progressed = False
            if self.run.cursor.transfer is None:
                from infernux.compute import _TransferCursor
                source, target = params
                transfer = _TransferCursor(source, target.device, owner.transfer_chunk_bytes)
                transfer.target = target.view()
                self.run.cursor.transfer = transfer
                progressed = True
            transfer = self.run.cursor.transfer
            if _prepare_transfer(self, owner, transfer, transfer.next_range()):
                self.task_end, self.range_count = transfer.pending_end, transfer.nbytes
                progressed = True
            return progressed
        return _prepare_gpu_work_ranges(self, self.run.cursor, self.gpu_executables, owner)

    def _publish(self, owner, opportunity):
        from infernux.nn._plan import _map_result, _result_values
        with self.plan.root._lock:
            if not self._is_current(owner):
                owner._cancel(self, "cancelled", "model state changed before publication")
                return
            if self.optimizer is not None:
                output = self.result_storage["loss"] if self._needs_result_transfer() else self.run.loss_output
                result = None if output is None else _model_result_tensor(output, ())
                try:
                    self.run.published_state = self.run.step_bindings.commit()
                    self.grad_facts = self.run.grad_facts = self.run.step_bindings.facts
                except BaseException:
                    if result is not None:
                        result._data._close(flush=False)
                    raise
                if result is not None:
                    result._data._publish_storage()
                self.run.phase = self.phase = "complete"
                self._publish_result(result, opportunity)
                self.metrics["learning_rate"] = float(self.update_spec.lr)
                return
            outputs = self._output_selection()
            published = {}
            for index in dict.fromkeys(_result_values(outputs)):
                buffer = self.result_storage[index] if self._needs_result_transfer() else self.run.values[index]
                spec = self.plan.result_specs[index]
                tensor = _model_result_tensor(buffer, spec.shape,
                    value_plan=self.plan if buffer.device == self.plan.device else None, value_index=index)
                tensor._data._publish_storage()
                published[index] = tensor
            result = _map_result(outputs, published.__getitem__)
            owner.current_nn = result
        self._publish_result(result, opportunity)

    def _release_private(self):
        if self.status != 'published':
            self._settle_output_sources(None, self.reason)
        for _, cursor in self.io:
            cursor.close()
        self.io.clear()
        from infernux.nn import _primitives as p
        p.retire(self.io_storage, flush=False)
        self.io_storage.clear()
        self.result_storage.clear()
        if self.run is not None:
            self.phase, self.grad_published = self.run.phase, self.run.grad_published
            self.run.retire(flush=False)
        for value in self.inputs or ():
            if not isinstance(value, BufferSource):
                value._data._close(flush=False)
        if self.target is not None and not isinstance(self.target, BufferSource) and not self.target._data.closed:
            self.target._data._close(flush=False)
        self.run = self.inputs = self.state = self.plan = None
        self.target = self.optimizer = self.update_spec = None
        if self.sources is not None:
            self.sources.close()
            self.sources = None
        self.grad_facts = ()
        self.gpu_executables = {}
        self.gpu_argument_bytes = 0
        self.ranges = self.execution_costs = ()
        self.service_spans = ()
        self.service_signature = ()


@dataclass(eq=False, kw_only=True)
class TransferRequest(CpuRequest):
    cursor: object
    destination: str | None = None
    chunk_bytes: int = 0

    def _prepared_transfer(self):
        return self.cursor

    def _bind_inputs(self, values, owner):
        from infernux.compute import _TransferCursor
        self.cursor = _TransferCursor(values['source'], self.destination, self.chunk_bytes)
        self.host = self.cursor.host

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        return remaining_transfer_service(self, owner, opportunity)

    def _service_host_key(self):
        step = self.cursor.next_step(prepared=self.packet is not None)
        return self.cursor.host_cost_key(step,
            finalizing=step.domain == "host" and step.end == self.cursor.nbytes)

    def _observed_host_key(self, key, progress):
        # A final copy that consumed all host credit did not publish. Keep its
        # real cost, but never use it as a sample for the combined operation.
        kind = key[-4]
        if self.status != "published":
            if kind in ("copy_host", "materialize") and self.metrics["completed_elements"] == progress[0]:
                return None
            if kind == "complete":
                return None
        return (*key[:-1], key[-1] and self.status == "published")

    def _domain(self):
        return self.cursor.execution_domain

    def _resources(self):
        c = self.cursor
        if c is None:
            return _RequestStorage((), {}, self.reserved_bytes)
        peak = c.prepare_bytes if self.packet is None and not c.readback_ready and not self._complete() else 0
        return _RequestStorage((c.source, c.target), _storage_blocks((c.snapshot, c.target)), self.reserved_bytes,
                               prepare_bytes=peak, native_blocks=c.native_blocks)

    def _is_current(self, owner):
        return self.cursor is None or self.cursor.is_current()

    def _complete(self):
        return self.cursor.completed_bytes == self.cursor.nbytes

    def _work(self, owner, opportunity):
        c = self.cursor
        work = _advance_transfer(c, owner, self)
        self.metrics["completed_elements"] = c.completed_bytes // c.dtype.itemsize
        return work

    def _prepare_gpu(self, owner, work):
        return _prepare_transfer(self, owner, self.cursor, work)

    def _consume_gpu(self):
        self.metrics["device_completed_bytes"] = self.cursor.pending_end
        self.cursor.consume_execution()
        self.metrics["completed_elements"] = self.cursor.completed_bytes // self.cursor.dtype.itemsize

    def _publish(self, owner, opportunity):
        self.cursor.target._publish_storage()
        self.result, self.cursor.target = self.cursor.target, None
        self.status = "published"
        self.metrics["terminal_opportunity"] = opportunity

    def _release_private(self):
        if self.cursor is not None:
            self.cursor.close()
        self.cursor = None


def _plan_storage(plan):
    optimizer = None if plan.root._optimizer is None else plan.root._optimizer()
    graphs = tuple(dict.fromkeys((*plan.root._plans.values(), plan)))
    return (*plan.root._state, *(parameter.grad._data for _, parameter in plan.root._named if parameter.grad is not None),
            *(() if optimizer is None else optimizer._state_values(optimizer._state)),
            *(value for graph in graphs for value in (graph._first_outputs or ())))


def _admission_cost(function):
    @wraps(function)
    def measured(self, *args, **kwargs):
        self._check_owner()
        from infernux.compute import _command_recording

        started, result = perf_counter_ns(), None
        self._frame.begin_scope()
        depth = getattr(_command_recording, "cost_depth", 0)
        _command_recording.cost_depth = depth + 1
        try:
            result = function(self, *args, **kwargs)
            return result
        finally:
            _command_recording.cost_depth = depth
            self._charge_admission(started, result)
    return measured


def _service_cost(function):
    @wraps(function)
    def measured(self, request, *args, **kwargs):
        started = perf_counter_ns()
        parallel_io = kwargs.get("parallel_io", False)
        key, units, estimate, reserved = ((None, None, None, False) if parallel_io else self._host_estimate(request))
        submitting = not parallel_io and request.packet is not None
        binding = request.input_bindings is not None
        preparing = not parallel_io and not submitting and request._execution_domain() == 'gpu'
        io_acceptances = {task: observation.accepted_submission for task, observation in self._io_observations.items()
                          if observation.request is request}
        progress = (request.metrics["completed_elements"], request.metrics.get("gpu_submit_attempts", 0),
                    request.metrics["accepted_submissions"], request.task is not None or request.packet is not None,
                    request.metrics.get("model_io_actions", 0))
        if reserved and estimate is not None and estimate > self._remaining_credit(background=True):
            self._service_calendar = None
            request.metrics.update(service_plan_status="unavailable",
                service_plan_reason="actual remaining host credit cannot fit the profiled service step")
            self._record_wait(request, "host_budget", request.metrics["service_plan_reason"])
            return False
        result = function(self, request, *args, **kwargs)
        elapsed = perf_counter_ns() - started
        for task in request._io_tasks():
            observation = self._io_observations.get(task)
            if (observation is not None and observation.work is not None
                    and observation.opportunity == self._last_opportunity
                    and observation.accepted_submission != io_acceptances.get(task, 0)):
                observation.host_ns = max(observation.host_ns, elapsed)
                self._costs.observe_host(io_host_key(observation.work), elapsed)
        if request.status in ("pending", "published"):
            measured_units, measured_elapsed = units, elapsed
            observed = None
            if binding:
                observed = key if result and request.input_bindings is None else None
            elif preparing:
                request.metrics['gpu_prepare_ns'] = request.metrics.get('gpu_prepare_ns', 0) + elapsed
                if result:
                    if request.preparation_key is None:
                        request.preparation_key, request.preparation_units = key, units
                    request.preparation_ns += elapsed
                    if request.packet is not None:
                        observed, measured_units = request.preparation_key, request.preparation_units
                        # A transfer's host key ends in kind, byte count,
                        # allocation extent and finalization. Sample the range
                        # sealed by the current GPU and byte allowances.
                        if _is_transfer_host_key(observed):
                            observed = (*observed[:-3], request.transfer_bytes, *observed[-2:])
                        measured_elapsed = request.preparation_ns
                        request.metrics['gpu_prepared_packets'] = request.metrics.get('gpu_prepared_packets', 0) + 1
                        request.preparation_key = request.preparation_units = None
                        request.preparation_ns = 0
            elif submitting:
                observed = key if request.packet is not None and request.packet.completion is not None else None
            elif key is not None:
                observed = request._observed_host_key(key, progress)
            if observed is not None:
                self._costs.observe_host(observed, measured_elapsed)
                if measured_units is not None:
                    self._costs.observe(("host-service", *observed), max(1, measured_units), measured_elapsed)
        return result
    return measured


class CpuWorkOwner:
    def __init__(self, engine, budget=None):
        if budget is not None and not isinstance(budget, CpuWorkBudget):
            raise TypeError("CPU work requires CpuWorkBudget")
        self._engine = weakref.ref(engine)
        self._frame = engine._frame_work
        self.budget = (CpuWorkBudget(owner_ns=self._frame.owner_ns,
            foreground_reserve_ns=self._frame.foreground_reserve_ns) if budget is None else budget)
        self._budget_locked = budget is not None
        self.current_nn = None
        self.requests = []
        self._accepting = True
        if budget is not None:
            self._frame.configure(budget.owner_ns, budget.foreground_reserve_ns)
        self._configure_resources()
        self.automatic_consumer_ns = 0
        self._started = None
        self._gpu_pending = False
        self._background_completion = None
        self._fifo_receipts = []
        self.command_metrics = dict(accepted_submissions=0, transfer_bytes=0,
                                    device_execution_ns=0, missing_device_samples=0)
        self._costs = ExecutionCosts()
        self._io_observations = {}
        self._service_calendar = None
        self._best_effort_due = False
        self._io_peer_turn = None
        self._plan_dirty = False
        self._planned_progress = self._frame.progress_serial
        self._gpu_service_state = (False, 0, False)
        self._gpu_busy_packets = {}
        self._gpu_busy_frame = None
        self._program_bindings = weakref.WeakSet()
        reference = weakref.ref(self)
        def callback(name, *, description=False):
            def call(*args):
                owner = reference()
                if owner is None:
                    return _native._FrameWorkFacts() if description else False
                return getattr(owner, name)(*args)
            return call
        self._source = self._frame.add_source(callback("_observe"), callback("_describe", description=True),
            callback("_advance"), callback("_refresh"))
        self._commands_source = self._frame.add_source(None, callback("_describe_commands", description=True),
            callback("_advance_commands"))

    @property
    def _last_opportunity(self):
        return self._frame.opportunity

    @property
    def remaining_ns(self):
        return max(0, self._frame.remaining())

    @property
    def owner_ns(self):
        return self._frame.total_ns

    @property
    def owner_overrun_ns(self):
        return self._frame.overrun_ns

    @property
    def opportunities(self):
        return self._frame.opportunity

    def wake(self):
        self._frame.wake(self._source)
        self._frame.wake(self._commands_source)

    def configure_budget(self, budget):
        if isinstance(budget, CpuWorkBudget) and self._budget_locked and budget != self.budget:
            raise ValueError("all CPU requests share one fixed owner budget")
        self.set_budget(budget)

    def set_budget(self, budget):
        self._check_owner()
        if not isinstance(budget, CpuWorkBudget):
            raise TypeError("host work requires CpuWorkBudget")
        smaller_segment = any(getattr(budget, name) < getattr(self.budget, name) for name in
            ('cpu_frame_ns', 'cpu_slice_ns', 'gpu_frame_ns', 'gpu_slice_ns',
             'transfer_frame_bytes', 'segment_operations', 'range_elements'))
        if (smaller_segment or budget.range_terms != self.budget.range_terms or
                budget.owner_ns - budget.foreground_reserve_ns < self._max_background_credit()):
            for request in self.requests:
                request._discard_preparation(self)
        # Automatic maintenance may already have spent this opportunity.
        # Selecting a custom budget never recharges an opportunity in progress.
        self._frame.configure(budget.owner_ns, budget.foreground_reserve_ns)
        self.budget = budget
        self._budget_locked = True
        self._configure_resources()
        self._plan_dirty = True

    def _charge_admission(self, started, result):
        elapsed = self._charge_host_interval(started, result)
        if result is not None:
            result.metrics["admission_owner_ns"] = elapsed

    def _charge_host_interval(self, started, result=None, *, automatic=False):
        elapsed = perf_counter_ns() - started
        if automatic:
            self.automatic_consumer_ns += elapsed
        cost = self._frame.end_scope()
        if result is not None:
            result.metrics["owner_ns"] = result.metrics.get("owner_ns", 0) + elapsed
            result.metrics["owner_overrun_ns"] = result.metrics.get("owner_overrun_ns", 0) + cost.overrun_ns
        return elapsed

    @staticmethod
    def _check_owner():
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        assert_runtime_owner_thread()

    def _configure_resources(self):
        budget = self.budget
        self._frame.configure_resources(budget.cpu_frame_ns, budget.gpu_frame_ns,
                                        budget._io_frame_allowance_ns, budget.transfer_frame_bytes)

    @staticmethod
    def _resource(domain):
        resources = _native._FrameWorkResource
        return {'cpu': resources.CPU, 'gpu': resources.GPU,
                'io': resources.IO, 'transfer': resources.TRANSFER}[domain]

    def _preparation_allowance(self, domain):
        resource = self._resource(domain)
        remaining = self._frame.resource_remaining(resource)
        return remaining if remaining > 0 else self._frame.resource_state(resource)['limit']

    @property
    def transfer_chunk_bytes(self):
        return min(self.budget.transfer_bytes, self.budget.transfer_frame_bytes)

    @staticmethod
    def _execution_prediction(request):
        if not request.execution_costs or any(value is None for _, _, value in request.execution_costs):
            return 0
        return max(1, int(np.ceil(request.execution_prediction)))

    def _can_admit_execution(self, request, domain):
        resource = self._resource(domain)
        prediction = self._execution_prediction(request)
        calibration_blocked = (domain == 'gpu' and not prediction
                               and request.gpu_calibration_floor_ns > self._frame.resource_remaining(resource))
        if calibration_blocked or not self._frame.can_admit_resource(resource, prediction, request.indivisible):
            self._record_wait(request, domain + "_budget", f"{domain.upper()} step cannot fit the execution allowance")
            return False
        if request.transfer_bytes and not self._frame.can_admit_resource(
                self._resource('transfer'), request.transfer_bytes, request.transfer_indivisible):
            self._record_wait(request, "transfer_budget", "transfer cannot fit the byte allowance")
            return False
        return True

    def _admit_execution(self, request, domain):
        prediction = self._execution_prediction(request)
        resource = self._resource(domain)
        request.execution_grant = self._frame.record_accepted_resource(resource, prediction)
        if request.transfer_bytes:
            request.transfer_grant = self._frame.record_accepted_resource(self._resource('transfer'), request.transfer_bytes)
        request.metrics['frame_admitted_'+domain+'_ns'] = (
            request.metrics.get('frame_admitted_'+domain+'_ns', 0)+request.execution_grant)
        request.metrics['frame_calibration_grants'] = request.metrics.get('frame_calibration_grants', 0)+(prediction == 0)

    def _accept(self, window, *, finish_within_ms=None):
        from infernux.application import Application
        self._check_owner()
        engine = self._engine()
        if (not self._accepting or not self._frame.accepting or engine is None or Application._current_engine() is not engine
                or not engine._cpu_range_accepting):
            raise RuntimeError("CPU work requires its active accepting Engine owner")
        if len(self.requests) >= self.budget.max_requests:
            raise RuntimeError("CPU request capacity is busy until an active request retires")
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError("release window must be positive host frames")
        if finish_within_ms is not None and (type(finish_within_ms) not in (int, float)
                or not isfinite(finish_within_ms) or finish_within_ms <= 0):
            raise ValueError("finish_within_ms must be positive finite milliseconds")
        released_ns = perf_counter_ns()
        release = int(engine._service_opportunity)
        from infernux.execution import ClockStamp, capture_step_permission
        return dict(submission_stamp=ClockStamp("host", id(engine), release),
                    release_opportunity=release, deadline=None if window is None else release + window,
                    released_ns=released_ns,
                    deadline_ns=None if finish_within_ms is None else released_ns + max(1, int(finish_within_ms*1_000_000)),
                    _owner=weakref.ref(self), execution_scope=capture_execution_scope(),
                    step_permission=capture_step_permission())

    @property
    def _gpu_prepare_unit_ns(self):
        return self._costs.host(("gpu", "prepare-operation")) or 0

    def _has_work(self):
        engine = self._engine()
        return (self._gpu_pending or self._background_completion is not None and not self._background_completion.done or
                any(request.execution_scope is None or request.execution_scope.state != 'paused'
                    or self._step_permitted(request) or self._accepted_service(request) for request in self.requests)
                or bool(engine is not None and engine._cpu_range_tasks))


    def _capacity(self, extra=None):
        values = []
        native_blocks, gpu_payload = {}, 0
        if self.current_nn is not None:
            from infernux.nn._plan import _result_values
            outputs = _result_values(self.current_nn)
            values.extend(output._data for output in outputs)
            for plan in {output._value_plan for output in outputs if output._value_plan is not None}:
                values.extend(_plan_storage(plan))
                native_blocks.update(plan.native_blocks)
        outstanding = list(self.requests)
        if extra is not None:
            outstanding.append(extra)
        future = 0
        credited_cold = set()
        for request in outstanding:
            facts = request._resources()
            if request.input_bindings is not None:
                future += request.input_bindings.extra_bytes
                for lease in request.input_bindings.leases:
                    native_blocks.update(lease.storage_claims)
            values.extend(facts.retained)
            native_blocks.update(facts.native_blocks)
            private = dict(facts.private_blocks)
            if facts.cold is not None:
                identity, buffers = facts.cold
                if identity not in credited_cold:
                    private.update(_storage_blocks(buffers))
                    credited_cold.add(identity)
            if request.task is not None and request.task._owners is not None:
                values.extend(request.task.storage)
                private.update(_storage_blocks(request.task.scalar_storage))
            future += max(0, facts.reservation_bytes - sum(private.values()))
            future += facts.prepare_bytes
            if request.packet is not None:
                native_blocks.update(request.packet.storage_blocks)
                gpu_payload += request.packet.payload_bytes
                if request.packet.native is not None:
                    future += int(request.packet.native.pending_staging_bytes)
        engine = self._engine()
        upload_bytes = 0
        if engine is not None:
            host = getattr(engine, "_infernux_compute_host", None)
            if host is not None:
                upload_bytes = int(host.retained_upload_bytes)
                from infernux.compute import _command_streams, _native_storage_blocks, _readbacks
                host_identity = int(host.identity)
                stream = _command_streams().get(host_identity)
                if stream is not None:
                    for packet in stream['pending']:
                        if packet.native is not None:
                            native_blocks.update(_native_storage_blocks(host_identity, packet.native))
                            gpu_payload += packet.payload_bytes
                # A completed but unread result still owns its staging buffer.
                for readback in _readbacks:
                    if readback._host is host and readback._task.submission_serial:
                        native_blocks.update(_native_storage_blocks(host_identity, readback._task))
            for receipt in self._fifo_receipts:
                native_blocks.update(receipt.storage_blocks)
            for task in engine._cpu_range_tasks:
                if task._owners is not None:
                    values.extend(task.storage)
                    native_blocks.update(task.native_blocks)
        blocks = _storage_blocks(values)
        blocks.update(native_blocks)
        return sum(blocks.values()) + future + gpu_payload + upload_bytes

    def _admit(self, request):
        capacity = self._capacity(request)
        if capacity > self.budget.capacity_bytes:
            raise ValueError("CPU requests and retained storage exceed numerical capacity")
        self._execution_active(request, self._frame.opportunity)
        request.metrics['submitted_host_frame'] = request.release_opportunity
        if request.deadline_ns is not None:
            request.metrics['completion_target_ms'] = (request.deadline_ns-request.released_ns)/1_000_000
        if request.status != 'pending':
            self._retire(request)
            return request
        self._plan_services(extra=request, capacity_bytes=capacity)
        self.requests.append(request)
        if request.execution_scope is not None:
            request.execution_scope._requests.add(request)
        if request.deadline is None:
            self._best_effort_due = True
        self._frame.wake(self._source)
        self._budget_locked = True
        request.metrics["payload_peak_bytes"] = capacity
        return request

    @staticmethod
    def _accepted_worker_service(request):
        return (request.task is not None and request.task._native is not None and request.task._native.submitted
                or getattr(getattr(request, "file", None), "pending", False)
                or getattr(getattr(request, "program_input", None), "pending", False))

    @staticmethod
    def _accepted_service(request):
        return (request.packet is not None and request.packet.completion is not None
                or CpuWorkOwner._accepted_worker_service(request))

    @staticmethod
    def _step_permitted(request):
        from infernux.execution import step_execution, step_permission_current
        if not step_permission_current(request.step_permission):
            return False
        with step_execution(request.step_permission):
            return request.execution_scope is None or request.execution_scope.state in ('active', 'stepping')

    def _execution_active(self, request, opportunity):
        if request.deadline_ns is not None:
            now = perf_counter_ns()
            request.metrics['remaining_target_ms'] = max(0, request.deadline_ns-now)/1_000_000
            if now >= request.deadline_ns and request.status == 'pending':
                request.metrics['wall_target_missed'] = True
                if self._cancel(request, 'expired', 'millisecond completion target elapsed'):
                    return False
                # An accepted file commit cannot be withdrawn. Its real outcome
                # must still publish and report the missed target, never a fake cancel.
        scope = request.execution_scope
        if scope is None:
            return True
        state = scope.state
        if state == 'paused' and self._step_permitted(request):
            state = 'stepping'
        if request.stepping != (state == 'stepping'):
            self._plan_dirty = True
            self._frame.report_progress()
        request.stepping = state == 'stepping'
        if state == 'closed':
            self._cancel(request, 'cancelled', 'execution owner retired')
            return False
        if state == 'paused':
            if request.paused_at is None:
                request.paused_at = opportunity
                self._plan_dirty = True
                self._frame.report_progress()
            return False
        if request.paused_at is not None:
            if request.deadline is not None:
                request.deadline += max(0, opportunity - request.paused_at)
            request.paused_at = None
            self._plan_dirty = True
            self._frame.report_progress()
        return True

    def _plan_services(self, extra=None, *, capacity_bytes=None):
        """Reserve a conditional numerical calendar beside the original queue.

        Unknown work keeps its release window, but receives no completion
        promise. Accepted unknown work blocks forecasts using its own lane.
        Ready peers retain a turn between finite reservation batches.
        """
        if extra is not None and self._frame.active and self._remaining_credit(background=True) <= 0:
            extra._forecast_deferred = True
            extra.metrics.update(service_plan_status="deferred",
                service_plan_reason="host planning allowance is spent for this opportunity")
            return
        started = perf_counter_ns()
        self._plan_dirty = False
        engine = self._engine()
        host = getattr(engine, '_infernux_compute_host', None)
        now = max(int(engine._service_opportunity), self._last_opportunity)
        active = self._frame.active
        if not active:
            now += 1
        requests = (*self.requests, extra) if extra is not None else tuple(self.requests)
        demands = {request: request._demand(self, now, capacity_bytes) if request.input_bindings is not None or request.residency_deferred
                   or request.deadline is not None and request.paused_at is None and not request.stepping else
                   _service_availability(request, self, now)
                   for request in requests if request.status == "pending"}
        known = {request: demand for request, demand in demands.items()
                 if request.deadline is not None and request.paused_at is None and not request.stepping and not demand.reason}
        gpu_state = (self._gpu_pressure_state(host.queue_pressure) if extra is not None and host is not None
                     else self._gpu_service_state)
        _, native_pending, foreground_occupied = gpu_state
        peers = self._frame.peer_work(self._source)
        completion = self._background_completion
        owned_gpu = completion is not None and not completion.done
        profiled_gpu = owned_gpu and any(request.packet is not None and request.packet.completion is completion
            and not demand.reason for request, demand in demands.items())
        unknown_gpu = (owned_gpu and not profiled_gpu) or native_pending > int(profiled_gpu) or peers.gpu_busy
        managed_tasks = {request.task for request in requests if request.task is not None}
        unknown_worker = (bool(engine._cpu_range_tasks - managed_tasks) or peers.worker_busy
                          or any(self._accepted_worker_service(request) and
                                 (request not in demands or demands[request].reason) for request in requests))
        worker_plans = {request for request, demand in known.items()
                        if any(_uses(demand.groups, domain) for domain in ("cpu", "io"))
                        or any(domain in ("cpu", "io") for domain, _ in demand.occupied)}
        unavailable = {}
        for request, demand in known.items():
            # Keep already accepted receipts in the resource ledger. A queued
            # request with no legal submission left cannot reserve new work.
            if request is not extra and (request.deadline < now
                    or (demand.ready_at is not None and demand.ready_at > request.deadline)
                    or (request.deadline == now and any(_uses(demand.groups, domain) for domain in ("cpu", "gpu", "io")))):
                unavailable[request] = "remaining service has no forecast inside its completion window"
            elif foreground_occupied and _uses(demand.groups, "gpu"):
                unavailable[request] = "foreground GPU work has no observed retirement window"
            elif unknown_gpu and (_uses(demand.groups, "gpu") or any(domain == "gpu" for domain, _ in demand.occupied)):
                unavailable[request] = "accepted GPU work has no known availability time"
            elif unknown_worker and request in worker_plans:
                unavailable[request] = "accepted worker or IO work has no known availability time"
        for request, reason in unavailable.items():
            demand = known.pop(request)
            if self._service_calendar is not None and request in self._service_calendar.demands:
                self._service_calendar.discard(request)
            self._record_service_plan(request, demand, now, "unavailable", reason)
        # A host profile is a conditional upper estimate, not a minimum legal
        # execution time. Preserve the same unreserved calibration opportunity
        # already offered when this profile grows after a request is admitted.
        if extra is not None and demands[extra].failure not in ("", "host_target"):
            raise ValueError(demands[extra].reason)
        protected = () if self._service_calendar is None else self._service_calendar.demands
        peers_observed = active and (self._frame.early_advanced or self._frame.late)
        if self._io_peer_turn != peers.io_service_serial or (peers_observed and not peers.io_ready):
            self._io_peer_turn = None
        if peers.io_ready and protected:
            self._io_peer_turn = peers.io_service_serial
        best_effort_ready = any(request.deadline is None and request.paused_at is None and not request._input_blocked(self)
                                and not self._accepted_service(request) for request in demands)
        calibration_ready = any(request.deadline is not None and request.paused_at is None and demand.reason and not request._input_blocked(self)
                                and not self._accepted_service(request)
                                for request, demand in demands.items())
        calibration_ready |= self._best_effort_due and best_effort_ready
        calibration_ready |= bool(protected) and any(request._forecast_deferred and request not in protected
                                                     for request in known)
        # Queued ordinary commands still compete between hosts, when no new
        # GPU grant can be taken. They must not lose their turn to new batches.
        calibration_ready |= self._queued_commands(host) is not None
        calibration_ready |= peers.other_ready
        # An independent GPU batch may start beside ready IO. Once protected,
        # it cannot admit new reservations until a peer takes a real IO turn.
        io_deferred = {request for request in known if self._io_peer_turn is not None
                       or (peers.io_ready and request in worker_plans)}
        deferred = {request for request in known if request not in protected
                    and (calibration_ready or request in io_deferred)}
        for request in deferred:
            known.pop(request)
        required = any(request in demands and request.paused_at is None for request in self._required_requests())
        required |= any(request.deadline_ns is not None and request.paused_at is None for request in demands)
        candidate_calendar = None
        if required:
            # Required consumers change service order. Withdraw the conditional
            # background forecast instead of pretending unprofiled work reserved it.
            self._service_calendar = None
            calendar = None
            reason = "required results or millisecond targets take precedence over background forecasts"
        else:
            credit = max(0, self._remaining_credit(background=True)) if active else self._max_background_credit()
            planning = credit // 2
            service_limits = dict(
                earliest={'gpu': now + 1} if active and self._frame.gpu_foreground_blocked else None,
                owner_ns=self._max_background_credit() - (self._costs.host(("owner", "service_planning")) or 0),
                owner_now_ns=credit,
                resource_limits={"cpu": self.budget.cpu_frame_ns, "gpu": self.budget.gpu_frame_ns,
                                 "io": self.budget._io_frame_allowance_ns, "transfer": self.budget.transfer_frame_bytes},
                resource_remaining=({name: self._frame.resource_remaining(self._resource(name))
                                     for name in ('cpu','gpu','io','transfer')} if active else None),
                resource_admitted=({name: self._frame.resource_state(self._resource(name))['admitted']
                                    for name in ('cpu','gpu','io','transfer')} if active else None))
            planning_started = perf_counter_ns()
            calendar = ServiceCalendar(known, {request: request.deadline for request in known}, now,
                budget_ns=max(1, planning), occupied=tuple(lane for request, demand in demands.items()
                    if request not in known for lane in demand.occupied), **service_limits)
            if (extra in known and calendar.failed_request is not None and calendar.failed_request is not extra
                    and extra not in calendar.scheduled_before_failure and not known[extra].prerequisites):
                # Check the candidate's own window with the same resource
                # facts. An earlier miss cannot reject it or hide its own miss.
                candidate_calendar = ServiceCalendar({extra: known[extra]}, {extra: extra.deadline}, now,
                    budget_ns=max(1, planning - (perf_counter_ns() - planning_started)),
                    occupied=tuple(lane for request, demand in demands.items()
                                   if request is not extra for lane in demand.occupied), **service_limits)
            self._costs.observe_host(("owner", "service_planning"), perf_counter_ns()-started)
            future_credit = self._max_background_credit() - self._costs.host(("owner", "service_planning"))
            calendar.validate_host_supply(now, max(0, self._remaining_credit(background=True)) if active else future_credit,
                future_credit)
            reason = calendar.reason
        # Preserve existing reservations until actual service invalidates them.
        # The candidate may still enter the normal queue without a forecast.
        if extra in known and reason and calendar is not None:
            admission = candidate_calendar if candidate_calendar is not None else calendar
            if admission.failure not in ("", "planning_budget", "host_supply", "host_target"):
                raise ValueError(admission.reason)
            extra._forecast_deferred = True
        if extra is not None and reason and self._reserved_service():
            # A candidate outside complete planning must not revoke an older
            # reservation. Current foreground observations were applied above.
            if extra not in unavailable:
                extra.metrics.update(service_plan_status="best_effort" if extra.deadline is None else "unprofiled",
                    service_plan_reason=demands[extra].reason or reason,
                    remaining_service_steps=None, estimated_completion_opportunity=None,
                    reserved_service_opportunities=0, scheduled_service_this_opportunity=False)
            return
        self._service_calendar = calendar if calendar is not None and not reason else None
        if self._reserved_service():
            if best_effort_ready:
                self._best_effort_due = True
            if peers.io_ready:
                self._io_peer_turn = peers.io_service_serial
        for request, demand in demands.items():
            if request in unavailable:
                continue
            deferred_reason = ""
            if request in deferred:
                if request._forecast_deferred:
                    deferred_reason = "waiting for earlier reservations before retrying the forecast"
                elif request in io_deferred:
                    deferred_reason = "ready file work is due a host turn before new reservations"
                else:
                    deferred_reason = "ready unprofiled work must receive calibration before new reservations"
            if self._service_calendar is not None and request in self._service_calendar.demands:
                request._forecast_deferred = False
            status = ("paused" if request.paused_at is not None else "stepping" if request.stepping else "best_effort" if request.deadline is None else
                      "unavailable" if demand.failure == "host_target" else "infeasible" if demand.failure
                      else "deferred" if request in deferred else "unprofiled"
                      if demand.reason or deferred_reason else "unavailable" if reason else "estimated")
            self._record_service_plan(request, demand, now, status, demand.reason or deferred_reason or reason)

    def _record_service_plan(self, request, demand, now, status, reason):
        calendar = self._service_calendar
        ready_at = None
        if calendar is not None and demand.prerequisites:
            completions = tuple(calendar.completions.get(before) for before in demand.prerequisites)
            if all(value is not None for value in completions):
                ready_at = max(completions)
        request.metrics.update(service_plan_status=status, service_plan_reason=reason,
            remaining_service_steps=None if request.deadline is None or demand.reason else demand.steps,
            minimum_service_atom_ns=demand.minimum_atom_ns or None,
            remaining_window_opportunities=None if request.deadline is None else max(0, request.deadline-now),
            required_service_per_opportunity=None if request.deadline is None or demand.reason else demand.steps / max(1, request.deadline-now+1),
            estimated_completion_opportunity=None if calendar is None else calendar.completions.get(request),
            estimated_prerequisite_available_opportunity=ready_at,
            reserved_service_opportunities=0 if calendar is None else sum(
                any(slot.request is request for slot in slots) for slots in calendar.slots.values()),
            scheduled_service_this_opportunity=calendar is not None and any(
                slot.request is request for slot in calendar.slots.get(now, ())))

    def _reserved_service(self):
        return self._service_calendar is not None and bool(self._service_calendar.demands)

    def _training_writer(self, root):
        return next((request for request in self.requests if request._model_access(root) == "train"), None)

    def _training_available(self, root):
        return not any(request._model_access(root) is not None for request in self.requests)

    @_admission_cost
    def release_model_save(self, model, optimizer, path, *, window_opportunities, finish_within_ms=None):
        from infernux.engine.model_persistence import ModelSaveRequest

        facts = self._accept(window_opportunities, finish_within_ms=finish_within_ms)
        request = ModelSaveRequest.create(self, facts, model, optimizer, path)
        try:
            return self._admit(request)
        except BaseException:
            request._release_input_bindings()
            request._release_private()
            raise

    @_admission_cost
    def release_model_load(self, path, *, device, kind, programs, window_opportunities, finish_within_ms=None):
        from infernux.engine.model_persistence import ModelLoadRequest

        facts = self._accept(window_opportunities, finish_within_ms=finish_within_ms)
        request = ModelLoadRequest.create(self, facts, path, device, kind, programs)
        try:
            return self._admit(request)
        except BaseException:
            request._release_private()
            raise

    @_admission_cost
    def release_transfer(self, source, *, device, window_opportunities, finish_within_ms=None):
        from infernux import compute

        facts = self._accept(window_opportunities, finish_within_ms=finish_within_ms)
        if not isinstance(source, (compute.Buffer, BufferSource)):
            raise TypeError("transfer requires an inx.buffer source")
        if device not in {"cpu", "gpu"}:
            raise ValueError("transfer device must be 'cpu' or 'gpu'")
        if compute._explicit_recording_depth():
            raise RuntimeError("managed transfers cannot release inside compute.recording")
        if isinstance(source, compute.Buffer):
            source = compute._pending_managed_source(source) or source
        if isinstance(source, BufferSource):
            from .input_binding import InputBindings
            source_storage, target_storage = source.device, device
            chunk = compute._transfer_chunk_bytes(self.transfer_chunk_bytes, source.dtype.itemsize)
            request = TransferRequest(**facts, cursor=None, destination=device,
                reserved_bytes=source.nbytes,
                input_bindings=InputBindings(self, {'source': source}))
            try:
                available = self.budget.capacity_bytes - self._capacity(request)
                if available < 0:
                    raise ValueError("transfer source and destination exceed numerical capacity")
                extent = min(source.nbytes, chunk)
                fitted = compute._fit_transfer_chunk(extent, source.dtype.itemsize, available,
                                                     source_storage, target_storage)
                if extent and not fitted:
                    raise ValueError("transfer preparation cannot fit one element in numerical capacity")
                request.chunk_bytes = fitted or chunk
                request.input_bindings.extra_bytes = compute._transfer_prepare_bytes(fitted, source_storage,
                                                                                   target_storage)
                return self._admit(request)
            except BaseException:
                request._release_input_bindings()
                raise
        source._require_open()
        if source._array is not None and source.nbytes > self.budget.snapshot_bytes:
            raise ValueError("CPU source capture exceeds the atomic snapshot_bytes limit")
        cursor = compute._TransferCursor(source, device, self.transfer_chunk_bytes)
        request = TransferRequest(**facts, cursor=cursor, host=cursor.host, reserved_bytes=cursor.reserved_bytes)
        try:
            peak = self._capacity(request)
            available = self.budget.capacity_bytes - peak + cursor.prepare_bytes
            if available < 0:
                raise ValueError("transfer source, snapshot and destination exceed numerical capacity")
            cursor.limit_chunk(available)
            peak = self._capacity(request)
            started = perf_counter_ns()
            cursor.capture_source(self._frame)
            request.metrics["snapshot_ns"] = perf_counter_ns() - started
            request.metrics["snapshot_bytes"] = 0 if cursor.snapshot is None else source.nbytes
            self._admit(request)
            request.metrics["payload_peak_bytes"] = max(peak, request.metrics["payload_peak_bytes"])
            return request
        except BaseException:
            request._release_private()
            request.host = None
            raise

    def _gpu_admission(self, entries, sources):
        """Validate the frozen role closure and already accepted source values."""
        from infernux import compute
        if compute._explicit_recording_depth():
            raise RuntimeError("managed GPU requests cannot release inside compute.recording")
        for value in sources:
            value._require_open()
        host = getattr(self._engine(), "_infernux_compute_host", None)
        if (not entries or host is None
                or any(value.device != "gpu" or int(value._host.identity) != int(host.identity) for value in sources)
                or any(entry.artifact is None or entry.host is None or not entry.tasks
                       or int(entry.host.identity) != int(host.identity) for entry in entries.values())):
            raise RuntimeError("cold prepare with the active host before releasing GPU work")
        if any(int(value._native.last_write_serial) == 0 or not value._native.has_initialized_range(
                value._byte_offset, value.nbytes) for value in sources):
            raise ValueError("GPU inputs, parameters and gradients require an accepted producer for their complete logical range")
        return max(3 * int(entry.artifact.argument_layout["size"]) for entry in entries.values())

    @_admission_cost
    def release_kernel(self, prepared, params, *, window_opportunities, finish_within_ms=None):
        from infernux import compute
        from .kernel_continuation import KernelRequest
        facts = self._accept(window_opportunities, finish_within_ms=finish_within_ms)
        prepared._validate(params, allow_sources=True)
        inputs = list(params)
        for index in prepared.reads:
            value = inputs[index]
            if index not in prepared.writes and isinstance(value, compute.Buffer):
                inputs[index] = compute._pending_managed_source(value) or value
        request = KernelRequest.create(self, facts, prepared, tuple(inputs))
        try:
            if self._capacity(request) > self.budget.capacity_bytes:
                raise ValueError("kernel inputs, outputs and preparation exceed numerical capacity")
            request.freeze(self)
            return self._admit(request)
        except BaseException:
            request._release_input_bindings()
            request._release_private()
            raise

    @_admission_cost
    def release_nn(self, model, x, *, window_opportunities, result_device=None):
        return self._release_model(self._accept(window_opportunities), model, (x,),
                                   result_device=result_device)

    @_admission_cost
    def release_program(self, prepared, arguments, *, window_opportunities, result_device=None, finish_within_ms=None):
        from infernux.nn._plan import _map_result
        from infernux.nn.sources import _SourceInputs
        from infernux.nn._tensor import Tensor
        facts = self._accept(window_opportunities, finish_within_ms=finish_within_ms)
        sources = _SourceInputs((*prepared.inputs, prepared.target), arguments, owner=self)
        try:
            inputs = tuple(sources.values[name].detach() if detached and isinstance(sources.values[name], Tensor) else sources.values[name]
                           for name, detached in zip(prepared.inputs, prepared.detach_inputs))
            target = None if prepared.target is None else sources.values[prepared.target]
            if isinstance(target, Tensor) and prepared.detach_target:
                target = target.detach()
            sources.values.clear()
            outputs = (_map_result(prepared.predictions, lambda value: value.index)
                       if prepared.predictions is not None else None)
            return self._release_model(facts, prepared.model, inputs, optimizer=prepared.optimizer, target=target,
                record_loss=prepared.record_loss, result_device=result_device, objective=prepared.objective,
                clear_grad=prepared.clear_grad, sources=sources, max_norm=prepared.max_norm, outputs=outputs,
                output_index=prepared.output_index)
        except BaseException:
            sources.close()
            raise

    @_admission_cost
    def release_training(self, model, x, target, optimizer, *, window_opportunities,
                         record_loss=False, result_device=None, objective="mse", clear_grad=False):
        facts = self._accept(window_opportunities)
        from infernux.nn.optim import Optimizer
        if (type(record_loss) is not bool or type(clear_grad) is not bool
                or not isinstance(optimizer, Optimizer) or optimizer._root is not model):
            raise TypeError("training requires one model/optimizer and a Tensor input")
        return self._release_model(facts, model, (x,), optimizer=optimizer, target=target,
                                   record_loss=record_loss, result_device=result_device, objective=objective, clear_grad=clear_grad)

    def _release_model(self, facts, model, inputs, *, optimizer=None, target=None,
                       record_loss=False, result_device=None, objective="mse", clear_grad=False, sources=None,
                       max_norm=None, outputs=None, output_index=None):
        from infernux import compute
        from infernux.nn._tensor import Tensor, Parameter, _check_storage
        from infernux.nn._plan import _grad_facts
        from infernux.nn._plan import _objective, _objective_facts, _check_target_layout
        objective = _objective(objective, target)
        from infernux.nn._primitives import program_family
        from infernux.nn.modules import Module
        if (not isinstance(model, Module) or not inputs
                or any(not isinstance(x, (Tensor, BufferSource)) or isinstance(x, Parameter) for x in inputs)):
            raise TypeError("NN work requires a Module and Tensor values")
        if result_device not in (None, "cpu", "gpu"):
            raise ValueError("NN result_device must be 'cpu' or 'gpu'")
        root = model._bind()
        with root._lock:
            prepared = model._plan
            plan = None if prepared is None else model._prepared_plan(inputs, prepared.device)
            if root._dirty or plan is None:
                raise ValueError("prepare a clean model before release")
            if any(isinstance(x, Tensor) and (x._run is not None or x._loss is not None) for x in inputs):
                raise ValueError("NN work requires a matching leaf/detached input")
            training = optimizer is not None
            grads = () if not training else _grad_facts(optimizer._params)
            if training:
                result = plan._result_spec(output_index)
                output_index = result.index
                _objective_facts(objective, result.shape)
                if not self._training_available(root):
                    raise RuntimeError("one model permits one outstanding training writer or pending snapshot")
                if plan.input_grad or not any(result.grad_parameters):
                    raise ValueError("cold prepare a matching differentiable model with a leaf non-gradient input")
                optimizer._check_domain()
                _check_target_layout(target, objective, result.shape, allow_source=True)
                if target is not None:
                    if target.requires_grad or isinstance(target, Tensor) and (target._run is not None or target._loss is not None):
                        raise ValueError("loss targets must be non-differentiable leaf/detached Tensors")
                    if isinstance(target, Tensor):
                        _check_storage((target._data,))
            for x in inputs:
                if isinstance(x, Tensor):
                    _check_storage((x._data,))

            def pending_source(value):
                if isinstance(value, Tensor):
                    pending = compute._pending_managed_source(value._data, shape=value.shape,
                                                              requires_grad=value.requires_grad)
                    if pending is not None:
                        return pending
                return value

            inputs = tuple(pending_source(value) for value in inputs)
            target = pending_source(target)
            mathematical_sources = (*root._state, *(value._data for _, value in grads if value is not None))
            if mathematical_sources:
                _check_storage(mathematical_sources)
                if mathematical_sources[0].device != plan.device:
                    raise ValueError("model parameters and gradients must use the prepared device")
            entries, gpu_peak = {}, 0
            if plan.device == "cpu":
                if training:
                    if (plan.backward_operations is None or optimizer._update is None
                            or (objective, "gradient", result.shape) not in plan.loss_programs
                            or (record_loss and (objective, "value", result.shape) not in plan.loss_programs)):
                        raise RuntimeError("call optimizer._prepare_training before releasing CPU training")
                    programs = (*plan._training_programs(record_loss, objective, output_index=output_index),
                                *optimizer._update_programs)
                    preparation = "optimizer._prepare_training"
                else:
                    programs = tuple(op[0] for op in plan.operations)
                    preparation = "model.prepare"
                if any(program._c_entry is None for declaration in programs for program in program_family(declaration)):
                    raise RuntimeError(f"call {preparation} before releasing CPU work")
            elif training:
                entries = optimizer._gpu_training_entries(plan, record_loss=record_loss, objective=objective,
                                                          output_index=output_index)
            else:
                entries = plan._gpu_forward_executables
                if entries is None or any(entry.artifact is None for entry in entries.values()):
                    plan._prepare_gpu_forward()
                    entries = plan._gpu_forward_executables
            if entries:
                gpu_sources = (*(x._data for x in inputs if isinstance(x, Tensor)),
                               *((target._data,) if isinstance(target, Tensor) else ()), *mathematical_sources)
                gpu_peak = self._gpu_admission(entries, tuple(value for value in gpu_sources if value.device == "gpu"))
            needs_gpu = (plan.device == "gpu" or any(x.device == "gpu" for x in inputs) or result_device == "gpu"
                         or (target is not None and target.device == "gpu"))
            host = compute._native_compute_host() if needs_gpu else None
            shared = tuple(x for x in (*inputs, *((target,) if target is not None else ()))
                           if isinstance(x, Tensor) and x.device == 'cpu' and x._shared_input)
            def capture_key(value):
                array = value._data._array
                return array.ctypes.data, array.nbytes, array.dtype.str
            snapshot_bytes = sum(value._data.nbytes for value in {capture_key(x): x for x in shared}.values())
            if snapshot_bytes > self.budget.snapshot_bytes:
                raise ValueError("CPU NN source capture exceeds the atomic snapshot_bytes limit")
            if snapshot_bytes and self._capacity() + snapshot_bytes > self.budget.capacity_bytes:
                raise ValueError("CPU NN source capture exceeds common numerical capacity")
            snapshots = {}
            def capture(value):
                if isinstance(value, BufferSource):
                    return value
                if value.device == 'gpu':
                    return compute._freeze_gpu_buffer(value._data)
                if not value._shared_input:
                    return value._data
                key = capture_key(value)
                if key not in snapshots:
                    snapshots[key] = compute._freeze_cpu_buffer(value._data, self._frame)
                return snapshots[key].view(readonly=True)
            frozen_inputs, frozen_target, request = [], None, None
            try:
                for x in inputs:
                    storage = capture(x)
                    try:
                        frozen_inputs.append(x if isinstance(x, BufferSource) else
                                             Tensor._from_buffer(storage, x.shape, requires_grad=x.requires_grad))
                    finally:
                        if isinstance(storage, Buffer) and storage is not x._data:
                            storage._close(flush=False)
                if target is not None:
                    storage = capture(target)
                    try:
                        frozen_target = target if isinstance(target, BufferSource) else Tensor._from_buffer(storage, target.shape)
                    finally:
                        if isinstance(storage, Buffer) and storage is not target._data:
                            storage._close(flush=False)
                reserve = (plan._training_storage_bytes(grads, record_loss, self.budget.range_terms, optimizer,
                                                        objective, clear_grad, max_norm, output_index=output_index) if training else
                           plan.activation_bytes + plan.scalar_storage_bytes
                           + plan._inner_storage_bytes(self.budget.range_terms))
                if sources is not None:
                    reserve += sources.nbytes
                request = NnRequest(**facts, plan=plan, state=root._state, parameter_version=root.parameter_version, inputs=tuple(frozen_inputs),
                    optimizer=optimizer, target=frozen_target, update_spec=None if not training else optimizer._step_spec(max_norm),
                    grad_facts=grads, record_loss=record_loss, objective=objective, clear_grad=clear_grad, reserved_bytes=reserve,
                    gpu_executables=entries, gpu_argument_bytes=gpu_peak, host=host, sources=sources,
                    outputs=outputs, output_index=output_index)
                pending = {name: value for name, value in (*enumerate(request.inputs), ('target', request.target))
                           if isinstance(value, BufferSource)}
                if pending:
                    from .input_binding import InputBindings
                    request.result_device = plan.device if result_device is None else result_device
                    io_groups = {_input_transfer_key(value): value for value in (*request.inputs, request.target)
                                 if value is not None and value.device != plan.device}
                    io_bytes = sum(value.nbytes if isinstance(value, BufferSource) else value._data.nbytes
                                   for value in io_groups.values())
                    io_bytes += request._result_transfer_bytes()
                    request.input_bindings = InputBindings(self, pending, extra_bytes=io_bytes)
                else:
                    request._configure_io(result_device, self)
                request.metrics['snapshot_bytes'] = snapshot_bytes
                return self._admit(request)
            except BaseException:
                if request is not None:
                    request._release_input_bindings()
                    request._release_private()
                else:
                    for value in (*frozen_inputs, frozen_target):
                        if isinstance(value, Tensor):
                            value._data._close(flush=False)
                raise
            finally:
                for storage in snapshots.values():
                    storage._close(flush=False)

    def _cancel(self, request, status, reason):
        self._check_owner()
        if request.status != "pending":
            return False
        for task in request._io_tasks():
            self._watch_io(request, task)
        if not request._cancel_external():
            return False  # File commit has started; report its real outcome.
        request.status, request.reason = status, reason
        self._plan_dirty = True
        self._frame.report_progress()
        if self._service_calendar is not None:
            self._service_calendar.discard(request)
        request.metrics["terminal_opportunity"] = max(request.release_opportunity, self._last_opportunity)
        if status == "failed":
            request._on_failure(self, reason)
        if request.task is not None:
            request.task.cancel()
        if request.packet is not None:
            self._gpu_busy_packets.pop(request.packet, None)
            from infernux.compute import _cancel_work_commands
            _cancel_work_commands(request.host, request.packet)
        return True

    def _watch_io(self, request, task):
        observation = self._io_observations.get(task)
        if observation is None:
            observation = IoObservation(request, int(task.execution_ns), int(task.submissions))
            self._io_observations[task] = observation
        return observation

    @staticmethod
    def _io_key(work):
        return ("io", work.operation, work.parameter, bool(work.bytes))

    def _observe_io(self, task, observation):
        observation.complete(task, self)

    def _ready_io(self):
        return (request for request in self.requests
                if request.status == "pending" and request.paused_at is None and request._io_ready(self))

    def _advance_io(self, request, task, maximum, opportunity, *, publish=False, batch=None, batch_bytes=0):
        """Admit one native file grant. The original JobSystem owns execution."""
        started = perf_counter_ns()
        maximum_cap = maximum
        if not self._can_continue(request, opportunity):
            return False
        observation = self._watch_io(request, task)
        self._observe_io(task, observation)
        resource = self._resource('io')
        remaining = self._frame.resource_remaining(resource)
        if not remaining:
            self._record_wait(request, "io_budget", "file work has no IO allowance remaining")
            return False
        quantum = self.budget._io_quantum_ns
        target = min(quantum, remaining)
        work = task.describe_work(maximum, publish=True) if publish else task.describe_work(maximum)
        if not work.operation:
            return False
        best_effort = request.deadline is None and request.deadline_ns is None and not self._reserved_service()
        # Opening still has unknown directory facts. A write with reserved
        # service uses its priced cursor steps rather than an aggregate grant.
        directory = batch == "directory"
        batch = directory or (batch == "write" and best_effort and self.budget.io_slice_ns is None)
        buffered = work.operation == "read.buffer"
        timed = buffered or batch
        idle_io = (timed and best_effort and self.budget.io_slice_ns is None
                    and not any(peer is not request for peer in self._ready_io())
                    and not self._frame.peer_work(self._source).io_ready)
        borrowed = idle_io and target < remaining
        if borrowed:
            target = remaining
        # BufferedProfile 0 is Store into an already claimed CPU destination.
        # A writer can also continue while all its inputs are pinned.
        resident_write = batch and not directory and bool(batch_bytes)
        stream = (idle_io and (buffered and work.parameter == 0 or resident_write)
                  and self.budget.io_frame_ns is None
                  and not any(receipt.streaming and reader.pending
                              for reader, receipt in self._io_observations.items()))
        if work.bytes and not timed:
            full = work
            maximum, _ = self._costs.choose(self._io_key(work), work.bytes, work.bytes,
                                             target, probe=True,
                                             probe_target=remaining)
            # A smaller allowance can turn a flush/finalize tail into a plain
            # byte step. Describe the selected real action again, not the old cap.
            work = task.describe_work(maximum)
            if (self._io_key(work) != self._io_key(full) and full.remaining_bytes is not None
                    and full.remaining_bytes > 1):
                # The full span carries completion work. Price the actual prefix
                # with its own history; repeatedly using the tail's tiny profile
                # would never learn from the byte steps it actually submitted.
                prefix = task.describe_work(min(full.bytes, full.remaining_bytes - 1))
                if self._io_key(prefix) == self._io_key(work):
                    maximum, _ = self._costs.choose(self._io_key(prefix), prefix.bytes, prefix.bytes,
                                                     target, probe=True,
                                                     probe_target=remaining)
                    work = task.describe_work(maximum)
        free_capacity = self.budget.capacity_bytes - self._capacity()
        if free_capacity < 0:
            raise ValueError("IO workspace exceeds common numerical capacity")
        # These cursors already stop within the grant, between native steps.
        prediction = target if timed else self._costs.predict(self._io_key(work), work.bytes or 1)
        if not self._can_continue(request, opportunity):
            return False
        if not self._frame.can_admit_resource(resource, max(0, int(np.ceil(prediction or 0))), work.bytes <= 1):
            self._record_wait(request, "io_budget", "file step cannot fit the IO allowance")
            return False
        observation.begin(task, maximum_cap, quantum, regular=not borrowed and not stream, collect_span=not batch)
        submissions = int(task.submissions)
        if publish:
            task.publish()
        elif directory:
            task.advance(maximum, target, min(free_capacity, self.budget.snapshot_bytes))
        elif stream and resident_write:
            task.advance(max(maximum, batch_bytes), target, continuous=True)
        elif timed:
            task.advance(work.bytes if buffered else max(maximum, batch_bytes), 0 if stream else target)
        else:
            task.advance(maximum)
        if int(task.submissions) == submissions:
            return False
        observation.grant = self._frame.record_accepted_resource(resource,
            max(0, int(np.ceil(prediction or 0))))
        observation.target_ns = target if timed else quantum
        observation.streaming = stream
        observation.work = task.submitted_work
        key = self._io_key(observation.work)
        observation.execution_key = (*key, "stream") if stream else (*key, "borrowed") if borrowed else key
        observation.accepted_submission = int(task.submissions)
        observation.opportunity = opportunity
        metrics = request.metrics
        metrics["io_last_operation"] = observation.work.operation
        metrics["io_last_allowance_bytes"] = observation.work.bytes
        metrics["io_last_remaining_span_bytes"] = observation.work.remaining_bytes
        metrics["io_last_predicted_execution_ns"] = None if stream else prediction
        if borrowed:
            metrics["io_borrowed_submissions"] = metrics.get("io_borrowed_submissions", 0) + 1
        if stream:
            metrics["io_stream_submissions"] = metrics.get("io_stream_submissions", 0) + 1
        observation.host_ns = perf_counter_ns() - started
        return True

    def _observe_gpu_packet(self, request):
        packet = request.packet
        if packet.completion is None:
            if packet.native is None and request.status == "pending":
                self._cancel(request, "cancelled", "prepared GPU range was discarded")
            return packet.native is None
        serial = int(packet.completion.submission_serial)
        if serial != request.metrics.get("gpu_submission_serial"):
            request.metrics["gpu_submission_serial"] = serial
            request.metrics["accepted_submissions"] += 1
        if not packet.completion.done:
            return False
        elapsed = packet.completion.device_elapsed_ns
        if elapsed is None:
            request.metrics["missing_device_samples"] = request.metrics.get("missing_device_samples", 0) + 1
        else:
            request.metrics["device_execution_ns"] = request.metrics.get("device_execution_ns", 0) + elapsed
            self._frame.observe_resource(self._resource('gpu'), int(np.ceil(elapsed)),
                request.metrics['service_submitted_opportunity'], request.execution_grant)
        if request.transfer_bytes:
            self._frame.observe_resource(self._resource('transfer'), request.transfer_bytes,
                request.metrics['service_submitted_opportunity'], request.transfer_grant)
            request.transfer_bytes = 0
            request.transfer_grant = 0
        request.execution_grant = 0
        self._record_execution(request, elapsed)
        if request.status == "pending":
            request._consume_gpu()
        request.packet = None
        return True

    def _record_completion(self, request):
        if request.submitted_ns is not None:
            request.metrics["worker_end_to_end_ns"] += perf_counter_ns() - request.submitted_ns
            request.metrics["worker_execution_ns"] += request.task.execution_ns or 0
            self._frame.observe_resource(self._resource('cpu'), request.task.execution_ns or 0,
                request.metrics['service_submitted_opportunity'], request.execution_grant)
            request.execution_grant = 0
            if not request.task._cancelled and request.task._result_code == 0:
                self._record_execution(request, request.task.execution_ns, request.task.operation_execution_ns)
            request.submitted_ns = None

    def _record_execution(self, request, elapsed, operation_times=None):
        costs, request.execution_costs = request.execution_costs, ()
        request.gpu_calibration_floor_ns = 0
        lag = max(0, self._last_opportunity - request.metrics.get("service_submitted_opportunity", self._last_opportunity))
        gpu = bool(costs and costs[0][0][0] == "gpu")
        predicted_packet, request.execution_prediction = request.execution_prediction, 0
        if gpu:
            self._costs.observe_packet(costs, elapsed)
        if operation_times is not None:
            for (key, rows, _), duration in zip(costs, operation_times):
                self._costs.observe(key, rows, duration, lag=lag)
        elif len(costs) == 1:
            key, rows, _ = costs[0]
            separate_initial = (gpu and key[2] != 'transfer' and elapsed is not None
                                and elapsed > self.budget.gpu_slice_ns and not request.indivisible)
            self._costs.observe(key, rows, elapsed, lag=lag, separate_initial=separate_initial)
        elif costs and elapsed is not None:
            # One receipt measures one segment. Never invent per-layer samples.
            for key, _, _ in costs:
                self._costs.observe_lag(key, lag)
        if costs:
            request.metrics["execution_samples"] = request.metrics.get("execution_samples", 0) + (elapsed is not None)
            target = self.budget.gpu_slice_ns if costs[0][0][0] == "gpu" else self.budget.cpu_slice_ns
            request.metrics["configured_slice_overruns"] = request.metrics.get("configured_slice_overruns", 0) + (elapsed is not None and elapsed > target)
            request.metrics["predicted_execution_ns"] = request.metrics.get("predicted_execution_ns", 0) + predicted_packet

    def _retire(self, request):
        if not request._retire_external():
            return False
        if request.packet is not None:
            if not self._observe_gpu_packet(request):
                return False
        if request.task is not None:
            request.task._poll()
            if not request.task._done:
                return False
            self._record_completion(request)
        for task, observation in tuple(self._io_observations.items()):
            if observation.request is request:
                self._observe_io(task, observation)
                del self._io_observations[task]
        request._release_input_bindings()
        if request.gpu_preparation is not None:
            request.gpu_preparation.close()
            request.gpu_preparation = None
        request._release_private()
        self._gpu_busy_packets.pop(request, None)
        request.task = request.packet = request.host = None
        request._forecast_key, request._forecast_work = (), None
        request.reserved_bytes = 0
        if request in self.requests:
            self.requests.remove(request)
        if self._service_calendar is not None:
            self._service_calendar.discard(request)
            self._plan_dirty = True
        if request.execution_scope is not None:
            request.execution_scope._requests.discard(request)
        now = perf_counter_ns()
        host_frame = int(self._frame.opportunity) if self._frame.active else None
        request.metrics.update(retired_ns=now, retired_host_frame=host_frame)
        if request.deadline_ns is not None:
            request.metrics.update(wall_elapsed_ms=(now-request.released_ns)/1_000_000,
                wall_lateness_ms=max(0, now-request.deadline_ns)/1_000_000,
                wall_target_missed=now >= request.deadline_ns)
        request._retired = True
        return True

    def _max_background_credit(self):
        return self._frame.owner_ns - self._frame.foreground_reserve_ns

    def _remaining_credit(self, *, background=False):
        return self._frame.remaining(background)

    def _available(self, *, background=False):
        return self._remaining_credit(background=background) > 0

    def _begin_charge(self):
        self._started = perf_counter_ns()
        self._started_remaining = self._frame.remaining()
        self._frame.begin_scope()

    def _finish_charge(self, request):
        spent = perf_counter_ns() - self._started
        self._frame.end_scope()
        if request is not None:
            request.metrics["owner_ns"] += spent
            request.metrics["owner_overrun_ns"] += max(0, spent - self._started_remaining)
        self._started = None


    def _publish(self, request, opportunity):
        if not self._execution_active(request, opportunity):
            return False
        if not request._is_current(self):
            return self._cancel(request, "cancelled", "source or model changed before publication")
        status = request.status
        request._publish(self, opportunity)
        return request.status != status

    def _can_continue(self, selected, opportunity):
        """Admit work needing a later receipt; host-only completion is inclusive."""
        if not self._execution_active(selected, opportunity):
            return False
        if selected.deadline is not None and opportunity >= selected.deadline:
            self._cancel(selected, "expired", "no later completion observation remains")
            self._retire(selected)
            return False
        if self._available(background=True):
            return True
        self._record_wait(selected, "host_budget", "no background host credit remains")
        return False

    def _check_working_capacity(self, request):
        capacity = self._capacity()
        request.metrics["payload_peak_bytes"] = max(request.metrics["payload_peak_bytes"], capacity)
        if capacity > self.budget.capacity_bytes:
            raise ValueError("prepared numerical work exceeds numerical capacity")

    @_service_cost
    def _advance_request(self, selected, opportunity, engine, *, parallel_io=False):
        if not self._execution_active(selected, opportunity):
            return False
        if not selected._is_current(self):
            self._cancel(selected, "cancelled", "source or model changed before work")
            self._retire(selected)
            return True
        if parallel_io:
            progressed = selected._advance_parallel_io(self, opportunity)
            self._check_working_capacity(selected)
            return progressed
        if selected.input_bindings is not None:
            if not selected.input_bindings.ready or not self._can_continue(selected, opportunity):
                return False
            selected.input_bindings.bind(selected, self)
            selected._release_input_bindings()
            selected.metrics['input_bindings_completed'] = selected.metrics.get('input_bindings_completed', 0) + 1
            self._check_working_capacity(selected)
            return True
        if selected.packet is not None:
            # Mutable transfer sources are rechecked adjacent to the actual Try,
            # including Busy retries; an earlier frame's check is insufficient.
            if not self._can_continue(selected, opportunity):
                return selected.status != "pending"
            return self._try_gpu(selected)
        domain = selected._domain()
        try:
            work, progressed = selected._work(self, opportunity)
        except _BufferCapacityPending:
            selected.residency_deferred = True
            self._gpu_busy_packets[selected] = self._gpu_retry_key()
            self._plan_dirty = True
            metrics = selected.metrics
            metrics["gpu_residency_deferrals"] = metrics.get("gpu_residency_deferrals", 0) + 1
            work, progressed = None, False
        else:
            if selected.residency_deferred and (progressed or work is not None):
                selected.residency_deferred = False
                self._gpu_busy_packets.pop(selected, None)
        if selected.status != "pending":
            return True  # Work may cancel at its last admission gate.
        complete = selected._complete()
        advance = not complete and work is not None and self._can_continue(selected, opportunity)
        if selected.status != "pending":
            return True
        if advance:
            if domain == "gpu":
                progressed |= selected._prepare_gpu(self, work)
            elif not engine._cpu_range_tasks and selected.task is None:
                progressed |= selected._prepare_cpu(self, work)
        # Binding and packet preparation are one owner-thread transaction.
        # Account their final working set once, before execution/publication.
        self._check_working_capacity(selected)
        if complete:
            if self._available(background=True):
                progressed |= self._publish(selected, opportunity)
                if selected.status != "pending":
                    progressed |= self._retire(selected)
            return progressed
        if not advance:
            return progressed or selected.status != "pending"
        if domain == "gpu":
            return progressed or selected.status != "pending"
        if engine._cpu_range_tasks or not self._available(background=True):
            return progressed
        if not self._execution_active(selected, opportunity) or not self._can_admit_execution(selected, 'cpu'):
            return progressed
        submitted = perf_counter_ns()
        try:
            selected.task.submit()
        finally:
            native = selected.task._native
            if native is not None and native.submitted:
                self._admit_execution(selected, 'cpu')
                selected.submitted_ns = submitted
                selected.metrics["service_submitted_opportunity"] = self._last_opportunity
                selected.metrics["accepted_submissions"] += 1
                progressed = True
        return progressed

    def _try_gpu(self, request):
        from infernux.compute import _submit_commands
        request.transfer_bytes = request.packet.transfer_bytes
        if not self._can_admit_execution(request, 'gpu'):
            return False
        started = perf_counter_ns()
        try:
            _submit_commands(request.packet, block=False,
                on_accepted=lambda packet: self._accepted_gpu_packet(request, packet))
        finally:
            request.metrics["gpu_submit_attempts"] = request.metrics.get("gpu_submit_attempts", 0) + 1
            request.metrics["gpu_submit_ns"] = request.metrics.get("gpu_submit_ns", 0) + perf_counter_ns() - started
        if request.packet.completion is None:
            self._gpu_busy_packets[request.packet] = self._gpu_retry_key()
            self._record_wait(request, "gpu_submission", "Vulkan did not accept the prepared packet")
        else:
            self._gpu_busy_packets.pop(request.packet, None)
        return request.packet.completion is not None

    def _accepted_gpu_packet(self, request, packet):
        request.transfer_bytes = packet.transfer_bytes
        self._admit_execution(request, 'gpu')
        request.metrics["service_submitted_opportunity"] = self._last_opportunity
        if any(task.pending for task in request._io_tasks()):
            request.metrics["device_submissions_with_io_in_flight"] = (
                request.metrics.get("device_submissions_with_io_in_flight", 0) + 1)
        self._background_completion = packet.completion
        self._gpu_pending = True

    def _gpu_retry_key(self):
        opportunity = self._frame.opportunity
        if self._gpu_busy_frame != opportunity:
            self._gpu_busy_frame = opportunity
            self._gpu_busy_packets.clear()
        return opportunity, self._frame.progress_serial

    def _gpu_packet_blocked(self, packet):
        key = self._gpu_retry_key()
        return self._gpu_busy_packets.get(packet) == key

    def _ready_request(self, *, gpu):
        candidates = self.requests
        if self._reserved_service():
            calendar = self._service_calendar
            slots = calendar.slots.get(self._last_opportunity, ())
            candidates = tuple(dict.fromkeys(slot.request for slot in slots if (slot.group.domain == 'gpu') == gpu))
            if not gpu:
                candidates += tuple(request for request in self.requests if request not in calendar.demands)
        required = self._required_requests()
        targeted = sorted((request for request in self.requests
                           if request.deadline_ns is not None and request.status == 'pending'),
                          key=lambda request: request.deadline_ns)
        if targeted:
            candidates = tuple(dict.fromkeys((*(predecessor for request in targeted
                for predecessor in self._required_requests(request)), *candidates)))
        if required:
            candidates = tuple(dict.fromkeys((*required, *candidates)))
        for request in candidates:
            if request.status != "pending":
                continue
            if not self._execution_active(request, self._frame.opportunity):
                continue
            if not gpu and self._parallel_io_ready(request):
                return request
            if request._input_blocked(self):
                continue
            if request.residency_deferred and self._gpu_packet_blocked(request):
                continue
            if request.task is not None and request.task._native.submitted:
                continue
            if request.packet is not None and request.packet.completion is not None:
                continue
            if request.packet is not None and (self._gpu_packet_blocked(request.packet) or
                    not self._can_admit_execution(request, 'gpu')):
                continue
            # Completed GPU values need only publication, not device execution.
            if (request.packet is not None) == gpu:
                return request
        return None

    def _parallel_io_ready(self, request):
        # A profiled calendar currently prices a serial upper bound. Readahead
        # is opportunistic and consumes the same host and IO grants.
        return not self._reserved_service() and request._parallel_io_ready(self)

    def _required_requests(self, root=None):
        roots = (root,) if root is not None else (request for request in self.requests if request.required_consumers)
        result, seen = [], set()
        for request in roots:
            stack = [(request, False)]
            while stack:
                current, ready = stack.pop()
                if ready:
                    result.append(current)
                elif current not in seen:
                    seen.add(current)
                    stack.append((current, True))
                    stack.extend((dependency, False) for dependency in reversed(current._dependencies(self)))
        return tuple(result)

    def _pending_input_bindings(self, request):
        return (predecessor.input_bindings for predecessor in self._required_requests(request)
                if predecessor.status == 'pending' and predecessor.input_bindings is not None)

    def _pending_render_captures(self, request):
        engine = self._engine()
        for bindings in self._pending_input_bindings(request):
            yield from bindings.pending_render_captures(engine)

    def _waits_for_render_input(self, request):
        engine = self._engine()
        return any(bindings.waits_for_render(engine) for bindings in self._pending_input_bindings(request))

    def _permit_step(self, request, permission):
        changed = False
        for predecessor in self._required_requests(request):
            if predecessor.step_permission != permission:
                predecessor.step_permission = permission
                changed = True
        if changed:
            self._plan_dirty = True
            self._frame.report_progress()
            self.wake()

    @staticmethod
    def _queued_commands(host):
        if host is None:
            return None
        from infernux.compute import _command_streams
        stream = _command_streams().get(int(host.identity))
        return stream if stream is not None and stream["failure"] is None and stream["pending"] else None

    def _gpu_pressure_state(self, pressure):
        blocked = self._frame.gpu_foreground_blocked
        return (blocked, pressure.get("compute_submissions", 0),
                bool(pressure.get("recording_frames") or blocked and pressure.get("older_pending_frames")))

    def _observe_gpu_pressure(self, host):
        pressure = {} if host is None else host.queue_pressure
        state = self._gpu_pressure_state(pressure)
        changed = self._gpu_service_state != state
        self._gpu_service_state = state
        for request in self.requests:
            request.metrics["gpu_queue_pressure"] = pressure
        return changed

    def _record_wait(self, request, kind, reason):
        if request.deadline is None and request.deadline_ns is None:
            return
        opportunity = self._last_opportunity
        waits = request.metrics.setdefault("observed_waits", {})
        sample = waits.get(kind)
        if sample is None:
            waits[kind] = {"opportunities": 1, "last_opportunity": opportunity, "reason": reason}
        elif sample["last_opportunity"] != opportunity:
            sample["opportunities"] += 1
            sample["last_opportunity"] = opportunity
            sample["reason"] = reason

    def _observe_request_waits(self, request, gpu_wait):
        # Count observed gates once per host and category. Several independent
        # stages can wait in one host, and a later round can still make progress.
        io_pending = any(task.pending for task in request._io_tasks())
        if io_pending:
            self._record_wait(request, "io_receipt", "accepted file work has no observed receipt")
        packet = request.packet
        if packet is not None and packet.completion is not None:
            self._record_wait(request, "gpu_receipt", "accepted device packet has no observed receipt")
        elif request.task is not None and request.task._native.submitted:
            self._record_wait(request, "cpu_receipt", "accepted worker entry has no observed receipt")
        elif packet is not None and gpu_wait:
            kind = "foreground_gpu" if self._frame.gpu_foreground_blocked else "other_gpu_receipt"
            self._record_wait(request, kind, gpu_wait)
        elif request.residency_deferred:
            self._record_wait(request, "residency", "waiting for Vulkan buffer capacity")
        elif not io_pending:
            if request._input_blocked(self):
                self._record_wait(request, "input", "request prerequisite has not published")
            elif not self._available(background=True):
                self._record_wait(request, "host_budget", "no background host credit remains")

    def _collect_background_receipt(self):
        remaining, changed = [], False
        for receipt in self._fifo_receipts:
            completion = receipt.completion
            if not completion.done:
                remaining.append(receipt)
                continue
            elapsed = completion.device_elapsed_ns
            if elapsed is not None:
                self._frame.observe_resource(self._resource('gpu'), int(np.ceil(elapsed)),
                    receipt.submitted_frame, receipt.gpu_grant)
                self.command_metrics['device_execution_ns'] += elapsed
            else:
                self.command_metrics['missing_device_samples'] += 1
            if receipt.transfer_bytes:
                self._frame.observe_resource(self._resource('transfer'), receipt.transfer_bytes,
                    receipt.submitted_frame, receipt.transfer_grant)
            self._costs.observe(receipt.key, 1, elapsed,
                lag=max(0, self._frame.opportunity-receipt.submitted_frame))
            changed = True
        self._fifo_receipts = remaining
        completion = self._background_completion
        if completion is not None and completion.done:
            self._background_completion = None
            changed = True
        return changed

    @staticmethod
    def _fifo_cost_key(packet):
        return 'gpu', packet.host_identity, 'fifo', packet.execution_signature, packet.transfer_bytes

    def _fifo_prediction(self, packet):
        return max(0, int(np.ceil(self._costs.predict(self._fifo_cost_key(packet), 1) or 0)))

    def _can_admit_fifo(self, packet):
        staging = int(packet.native.pending_staging_bytes)
        if staging and self._capacity() + staging > self.budget.capacity_bytes:
            self.command_metrics['blocked_reason'] = 'GPU staging exceeds available numerical capacity'
            return False
        self.command_metrics['blocked_reason'] = ''
        return (not self._frame.active or
            self._frame.can_admit_resource(self._resource('gpu'), self._fifo_prediction(packet), True)
            and (not packet.transfer_bytes or self._frame.can_admit_resource(
                self._resource('transfer'), packet.transfer_bytes, True)))

    def _can_prepare_fifo(self, packet, *, block=False):
        reserved = packet.native.preparation_bytes
        needed = reserved + (0 if block else packet.native.pending_staging_bytes)
        allowed = not needed or self._capacity() + needed <= self.budget.capacity_bytes
        self.command_metrics['preparation_reserved_bytes'] = reserved
        self.command_metrics['blocked_reason'] = '' if allowed else 'GPU write candidate exceeds shared compute capacity'
        return allowed

    def _prepare_fifo(self, packet, chunk_bytes, *, block):
        if not self._can_prepare_fifo(packet, block=block):
            if block:
                raise MemoryError("GPU write candidate exceeds shared compute capacity")
            return False
        if packet.prepare_storage(chunk_bytes, managed=not block):
            self._gpu_busy_packets.pop(packet, None)
            return True
        self.command_metrics['blocked_reason'] = 'GPU write candidate is waiting for Vulkan memory'
        self._gpu_busy_packets[packet] = self._gpu_retry_key()
        return False

    def _record_fifo_packet(self, packet):
        gpu_grant = self._frame.record_accepted_resource(self._resource('gpu'), self._fifo_prediction(packet))
        transfer_grant = (self._frame.record_accepted_resource(self._resource('transfer'), packet.transfer_bytes)
                          if packet.transfer_bytes else 0)
        self._fifo_receipts.append(_FifoReceipt(packet.completion, self._fifo_cost_key(packet),
            self._frame.opportunity, gpu_grant, transfer_grant, packet.transfer_bytes, packet.storage_blocks))
        self.command_metrics['accepted_submissions'] += 1
        self.command_metrics['transfer_bytes'] += packet.transfer_bytes
        self._background_completion = packet.completion
        self._gpu_pending = True

    def _poll_receipts(self, engine, host, opportunity, *, validate=False):
        changed = False
        for task, observation in tuple(self._io_observations.items()):
            before = observation.submissions
            self._observe_io(task, observation)
            changed |= before != observation.submissions
        if engine is not None and engine._cpu_range_tasks:
            from infernux.cpu_execution import _poll_cpu_ranges
            _poll_cpu_ranges(engine)
        if host is not None:
            from infernux.compute import _collect_commands
            self._gpu_pending = _collect_commands(host)
        else:
            self._gpu_pending = False
        changed |= self._observe_gpu_pressure(host)
        changed |= self._collect_background_receipt()
        late = self._frame.late
        gpu_wait = None
        for request in tuple(self.requests):
            before = request.status, id(request.task), id(request.packet)
            active = self._execution_active(request, opportunity)
            if request.input_bindings is not None:
                changed |= request.input_bindings.poll()
                if request.status == 'pending' and request.input_bindings.error:
                    self._cancel(request, 'failed', request.input_bindings.error)
            request._poll_external(self, opportunity)
            if validate:
                if request.status == 'pending':
                    if active and request.deadline is not None and opportunity > request.deadline:
                        self._cancel(request, 'expired', 'release-to-publication window elapsed')
                    elif not request._is_current(self):
                        self._cancel(request, 'cancelled', 'source or model retired')
            if request.status != 'pending':
                self._retire(request)
            elif request.packet is not None:
                if self._observe_gpu_packet(request) and request.status != 'pending':
                    self._retire(request)
            elif request.task is not None and request.task._done:
                self._record_completion(request)
                if request.task._cancelled or request.task._result_code != 0:
                    self._cancel(request, 'failed', 'worker did not successfully complete the range')
                    self._retire(request)
                else:
                    request._consume_cpu()
                    request.task = None
            if (late and active and request.status == 'pending'
                    and (request.deadline is not None or request.deadline_ns is not None)):
                packet = request.packet
                if gpu_wait is None and packet is not None and packet.completion is None:
                    gpu_wait = self._frame.gpu_unavailable_reason
                    # The coordinator folds accepted source work into the GPU
                    # gate after refresh. Observe those same outstanding tickets.
                    completion = self._background_completion
                    if not gpu_wait and ((completion is not None and not completion.done)
                                         or self._frame.peer_work(self._source).gpu_busy):
                        gpu_wait = "accepted background GPU work has not retired"
                self._observe_request_waits(request, gpu_wait)
            changed |= before != (request.status, id(request.task), id(request.packet))
        return changed

    def _update_plan(self):
        self._plan_services()
        for request in tuple(self.requests):
            if request.metrics.get('service_plan_status') == 'infeasible':
                self._cancel(request, 'failed', request.metrics['service_plan_reason'])
                self._retire(request)

    def _observe(self):
        """Once-per-host lifecycle validation and initial forecast."""
        self._check_owner()
        self._begin_charge()
        try:
            engine = self._engine()
            host = getattr(engine, '_infernux_compute_host', None) if engine is not None else None
            self._poll_receipts(engine, host, self._frame.opportunity, validate=True)
            self._plan_dirty = True  # Describe plans after all sources received their host observation.
        finally:
            self._finish_charge(None)

    def _refresh(self):
        """Collect original completion tickets; no lifecycle tick or eager replan."""
        self._check_owner()
        self._begin_charge()
        try:
            engine = self._engine()
            host = getattr(engine, '_infernux_compute_host', None) if engine is not None else None
            changed = self._poll_receipts(engine, host, self._frame.opportunity)
            self._plan_dirty |= changed
            if changed:
                self._frame.report_progress()
                self._gpu_busy_packets.clear()
        finally:
            self._finish_charge(None)

    def _host_estimate(self, request):
        if request.input_bindings is not None:
            key, units = ('input-bind', type(request).__name__, len(request.input_bindings.leases)), 1
        elif request.packet is not None:
            key, units = gpu_submit_key(request.execution_costs, request.transfer_bytes)
        else:
            key = request._service_host_key()
            units = None if key is None else request._service_host_units(key)
            if request._execution_domain() == 'gpu':
                key = gpu_prepare_key(key)
        estimate = (None if key is None else self._costs.host(key) if units is None else
                    self._costs.predict(("host-service", *key), max(1, units)))
        if request.gpu_preparation is not None and estimate is not None:
            estimate = min(estimate, self._max_background_credit())
        booked = None
        if self._reserved_service():
            lane = 'gpu' if request.packet is not None else 'host' if request._execution_domain() == 'gpu' else None
            booked = next((slot.group.owner_ns
                for slot in self._service_calendar.slots.get(self._last_opportunity, ())
                if slot.request is request and (slot.group.domain != 'gpu' if lane is None else slot.group.domain == lane)), None)
        if booked is not None:
            # The reservation already prices the selected subrange. The
            # unsplit cursor's historical price can be much higher.
            estimate = booked if lane == 'host' and _is_transfer_host_key(key) else max(estimate or 0, booked)
        return key, units, estimate, booked is not None

    def _candidate(self, request):
        if request is None:
            return _native._FrameWorkCandidate()
        _, _, estimate, reserved = self._host_estimate(request)
        if reserved and (request.deadline is not None or request.deadline_ns is not None):
            if (estimate or 0) > self._remaining_credit(background=True):
                self._record_wait(request, "host_budget", "remaining host credit cannot fit the reserved service step")
        # A historical host sample is conditional evidence, not a hard lower
        # bound. Best-effort work must retain one legal calibration attempt.
        return _native._FrameWorkCandidate(id(request), max(0, int(estimate or 0)) if reserved else 0, True, reserved)

    def _describe(self):
        for request in tuple(self.requests):
            self._execution_active(request, self._frame.opportunity)
        if self._plan_dirty or self.requests and self._frame.progress_serial != self._planned_progress:
            self._begin_charge()
            try:
                self._update_plan()
            finally:
                self._finish_charge(None)
            self._planned_progress = self._frame.progress_serial
        host_request = self._ready_request(gpu=False)
        return _native._FrameWorkFacts(self._candidate(host_request),
            self._candidate(self._ready_request(gpu=True)), pending=self._has_work(),
            gpu_busy=self._background_completion is not None and not self._background_completion.done,
            reservation=self._reserved_service(), observe=bool(self.requests),
            io_ready=host_request is not None and host_request._io_ready(self),
            worker_busy=any((request.task is not None and request.task._native is not None
                             and request.task._native.submitted and not request.task._done)
                            or any(task.pending for task in request._io_tasks())
                            for request in self.requests))

    def _advance(self, gpu, token):
        self._check_owner()
        selected = self._ready_request(gpu=bool(gpu))
        if selected is None or id(selected) != token or not self._accepting:
            return False
        engine = self._engine()
        status = selected.status
        progressed = False
        self._begin_charge()
        try:
            self.requests.remove(selected)
            self.requests.append(selected)
            progressed = self._advance_request(selected, self._last_opportunity, engine,
                parallel_io=not gpu and self._parallel_io_ready(selected))
        except BaseException as error:
            self._cancel(selected, "failed", str(error))
            self._retire(selected)
            progressed = selected.status != status
            if not isinstance(error, Exception):
                raise
        finally:
            self._finish_charge(selected)
        progressed |= selected.status != status
        if progressed and selected.deadline is None:
            self._best_effort_due = False
        self._plan_dirty |= progressed
        return progressed

    def _describe_commands(self):
        engine = self._engine()
        host = getattr(engine, "_infernux_compute_host", None) if engine is not None else None
        stream = self._queued_commands(host)
        packet = None if stream is None else next((p for p in stream["pending"] if p.native.has_work), None)
        if packet is not None and packet.needs_preparation:
            candidate = (_native._FrameWorkCandidate(id(packet), 0, True, False)
                         if not self._gpu_packet_blocked(packet) and self._can_prepare_fifo(packet)
                         else _native._FrameWorkCandidate())
            return _native._FrameWorkFacts(candidate, pending=True)
        work = None if packet is None else packet.submission
        candidate = (_native._FrameWorkCandidate() if work is None or self._gpu_packet_blocked(work) or
                     not self._can_admit_fifo(work) else
                     _native._FrameWorkCandidate(id(work), 0, True, False))
        return _native._FrameWorkFacts(gpu=candidate, pending=packet is not None)

    def _advance_commands(self, gpu, token):
        from infernux.compute import _admit_commands, _fail_command_stream
        engine = self._engine()
        host = getattr(engine, "_infernux_compute_host", None) if engine is not None else None
        stream = self._queued_commands(host)
        packet = None if stream is None else next((p for p in stream["pending"] if p.native.has_work), None)
        if packet is None or not self._accepting:
            return False
        if not gpu:
            if id(packet) != token or not packet.needs_preparation or self._gpu_packet_blocked(packet):
                return False
            self._begin_charge()
            try:
                return self._prepare_fifo(packet,
                    max(4, min(self.transfer_chunk_bytes, self._preparation_allowance('transfer'))), block=False)
            except Exception as exception:
                _fail_command_stream(stream, exception)
                return True
            finally:
                self._finish_charge(None)
        if packet.needs_preparation:
            return False
        work = packet.submission
        if id(work) != token or not self._can_admit_fifo(work):
            return False
        self._begin_charge()
        try:
            _admit_commands(host)
        finally:
            if work.completion is not None:
                self._gpu_busy_packets.pop(work, None)
            elif stream['failure'] is None:
                self._gpu_busy_packets[work] = self._gpu_retry_key()
            self._gpu_pending = True
            self._finish_charge(None)
        return work.completion is not None or stream["failure"] is not None

    def shutdown_begin(self):
        self._check_owner()
        self._accepting = False
        for binding in tuple(self._program_bindings):
            binding.close()
        self.current_nn = None
        for request in self.requests:
            self._cancel(request, "cancelled", "engine exiting")

    def shutdown_finish(self):
        self._check_owner()
        for request in tuple(self.requests):
            if request.packet is not None and request.packet.completion is not None:
                request.packet.completion.wait()
            request._wait_external(self)
            request._poll_external(self, self._last_opportunity)
            if not self._retire(request):
                raise RuntimeError("managed work cannot release storage before its accepted task retires")
        if self._background_completion is not None:
            if not self._background_completion.done:
                self._background_completion.wait()
            self._collect_background_receipt()
        for receipt in self._fifo_receipts:
            receipt.completion.wait()
        self._collect_background_receipt()
        self._frame.remove_source(self._source)
        self._frame.remove_source(self._commands_source)
