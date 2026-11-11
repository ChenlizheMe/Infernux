"""Numerical service forecasts for the existing engine owner.

This module submits no work and owns no model, buffer, worker or device queue.
Its calendar is conditional on measured costs and future foreground service;
it is not a worst-case execution-time or hard-real-time guarantee.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np


_GPU_CALIBRATION_ROWS = 1024


def _upper_curve(samples):
    """Nonnegative affine envelope; x/y units belong to the caller."""
    if not samples:
        return None
    low, high = min(x for x, _ in samples), max(x for x, _ in samples)
    largest = max(y for x, y in samples if x == high)
    if low == high:
        return 0., largest/high
    smallest = max(y for x, y in samples if x == low)
    slope = max(0., (largest-smallest)/(high-low))
    intercept = max(0., max(y-slope*x for x, y in samples))
    return intercept, slope


@dataclass(frozen=True)
class IoSpanSample:
    """One completed native span under the existing adaptive IO policy.

    units is the logical extent (at least one), calls is real callbacks,
    and execution_ns is the largest callback duration. Codec counters accompany
    measured codec samples only; estimates carry no invented progress.
    """
    units: int
    calls: int
    execution_ns: int
    host_ns: int
    lag: int
    codec_input_bytes: int | None = None
    codec_output_bytes: int | None = None


class ExecutionCosts:
    def __init__(self):
        self.samples = OrderedDict()
        self.lags = {}
        self.host_samples = OrderedDict()
        self.io_samples = OrderedDict()
        self.packet_ratios = deque(maxlen=8)
        self.revision = 0
        self._curves = {}

    @staticmethod
    def _packet_key(costs):
        return tuple((key, rows) for key, rows, _ in costs)

    def packet_scale(self, costs):
        if len(costs) == 1:
            return 1.
        signature = self._packet_key(costs)
        return max(1., max((ratio for key, ratio in self.packet_ratios
                            if key == signature and ratio is not None), default=1.))

    def observe_packet(self, costs, elapsed):
        prediction = sum(value for _, _, value in costs if value is not None)
        ratio = (elapsed / prediction if prediction and elapsed is not None
                 and all(value is not None for _, _, value in costs) else None)
        # Every actual receipt ages old composition samples, including single
        # operations. A slow packet can therefore be measured again later.
        self.packet_ratios.append((self._packet_key(costs), ratio))
        self.revision += 1

    def needs_probe(self, key, limit):
        samples = self.samples.get(key)
        return bool(samples and samples[0][0] < limit
                    and all(rows == samples[0][0] for rows, _ in samples))

    def curve(self, key):
        if key not in self.samples:
            return None
        if key not in self._curves:
            self._curves[key] = _upper_curve(self.samples.get(key))
        return self._curves[key]

    def observe_io_span(self, key, sample):
        if sample.calls <= 0 or sample.units <= 0:
            return
        values = self.io_samples.pop(key, deque(maxlen=8))
        values.append(sample)
        self.io_samples[key] = values
        if len(self.io_samples) > 256:
            self.io_samples.popitem(last=False)

    def io_span(self, key, units):
        """Conditional callback estimate, separate from attempt timing/lag.

        The key includes IO allowance cap and target. Individual allowances can
        adapt within a fragment; the sample describes that actual policy run.
        """
        values = self.io_samples.get(key)
        if not values:
            return None
        units = max(1, units)
        # Callback counts are staircases. Preserve observed overhead instead
        # of fitting a one-byte change in callback count as a throughput slope.
        calls = int(np.ceil(2 * max(v.calls * max(1., units/v.units) for v in values)))
        return IoSpanSample(units, max(1, calls),
            2 * max(v.execution_ns for v in values), 2 * max(v.host_ns for v in values),
            max(v.lag for v in values))

    @staticmethod
    def _margin(key):
        # Managed device ranges use device timestamps and an observed upper
        # envelope. Host work and mixed packets retain their wider estimate.
        return 1 if key[0] == "gpu" and key[2] != "fifo" else 2

    def predict(self, key, rows):
        curve = self.curve(key)
        if curve is None:
            return None
        launch, per_row = curve
        return (launch + rows*per_row) * self._margin(key)

    def calibration_floor(self, key, rows):
        return max((elapsed for width, elapsed in self.samples.get(key, ()) if width <= rows), default=0)

    def dispatch_probe_width(self, key, limit, allowance, *, grain=1):
        # A nearby width confounds launch cost with timer variation. Explore a
        # wider legal range once, capped independently of the configured grain.
        limit = min(limit, max(grain, _GPU_CALIBRATION_ROWS // grain * grain))
        if not self.needs_probe(key, limit):
            return 0
        rows = min(limit, self.samples[key][0][0] * 4)
        return rows if self.calibration_floor(key, rows) <= allowance else 0

    def observe(self, key, rows, elapsed, *, lag=None, separate_initial=False):
        if elapsed is None or elapsed <= 0 or rows == 0:
            return
        self.revision += 1
        self._curves.pop(key, None)
        values = self.samples.pop(key, None)
        initial = values is None
        if initial:
            values = deque(maxlen=8)
        # A first dispatch can include device pipeline startup. Its actual
        # receipt is charged separately and does not set the per-row slope.
        if not (initial and separate_initial):
            values.append((rows, elapsed))
        self.samples[key] = values
        if lag is not None:
            self.observe_lag(key, lag)
        if len(self.samples) > 256:
            old, _ = self.samples.popitem(last=False)
            self.lags.pop(old, None)
            self._curves.pop(old, None)

    def observe_lag(self, key, lag):
        self.revision += 1
        self.lags.setdefault(key, deque(maxlen=8)).append(lag)

    def lag(self, key):
        values = self.lags.get(key)
        return None if not values else max(values)

    def observe_host(self, key, elapsed):
        if key != ('owner', 'service_planning'):
            self.revision += 1
        values = self.host_samples.pop(key, deque(maxlen=8))
        values.append(elapsed)
        self.host_samples[key] = values
        if len(self.host_samples) > 256:
            self.host_samples.popitem(last=False)

    def host(self, key):
        values = self.host_samples.get(key)
        return None if not values else 2 * max(values)

    def gpu_target(self, key, slice_ns, allowance):
        launch, _ = self.curve(key) or (0, 0)
        return min(allowance, slice_ns + launch)

    def choose(self, key, remaining, cap, target, *, probe=False, probe_target=None, unit=1):
        limit = min(remaining, cap) // unit * unit
        samples = self.samples.get(key)
        probe_limit = target if probe_target is None else probe_target
        device_range = self._margin(key) == 1
        if device_range and samples and all(rows == limit for rows, _ in samples):
            exact = self.predict(key, limit)
            if exact <= probe_limit:
                return limit, exact
        if probe and self.needs_probe(key, limit):
            size = samples[0][0]
            # One size cannot distinguish launch cost from throughput.
            rows = min(limit, size * 2)
            prediction = self.predict(key, rows)
            if prediction <= probe_limit:
                return rows, prediction
            smaller = max(unit, size // (2 * unit) * unit)
            if device_range and smaller == size and self.predict(key, size) <= probe_limit:
                # One legal calibration grant measures a second size instead
                # of executing the same minimum range indefinitely.
                return rows, None
            prediction = self.predict(key, smaller)
            if prediction <= (probe_limit if device_range else target) or smaller == unit:
                return smaller, prediction
        curve = self.curve(key)
        if curve is None:
            return (limit if probe else min(limit, 64)), None
        launch, per_row = curve
        allowance = target / self._margin(key) - launch
        rows = limit
        if allowance <= 0:
            rows = unit
        elif per_row:
            rows = min(rows, max(unit, int(allowance/per_row) // unit * unit))
        return rows, self.predict(key, rows)


def execution_key(program, params):
    """The exact same argument layout for real work and layout-only analysis."""
    gpu = hasattr(program, "artifact")
    control = program.artifact.continuation.inner_range if gpu else program.continuation_plan.inner_range
    inner = None if control is None else (control.kind, int(params[control.end_parameter])
                                         - int(params[control.begin_parameter]))
    layout = tuple((str(value.dtype), tuple(value.shape), value.nbytes)
                   if hasattr(value, "element_count") or isinstance(value, np.ndarray)
                   else type(value).__name__ for value in params)
    return ("gpu" if gpu else "cpu", int(program.host.identity) if gpu else 0,
            program.cost_identity, layout, inner)


class CpuSegment:
    """Pure measured prefix selection, shared by execution and service forecasts."""
    def __init__(self, owner, *, host_allowance, execution_allowance=None):
        self._costs = owner._costs
        self._budget = owner.budget
        self.costs = []
        self.probe_target = owner.budget.cpu_frame_ns if execution_allowance is None else execution_allowance
        self.remaining = min(owner.budget.cpu_slice_ns,
            self.probe_target)
        self.preparation = owner._costs.host(('cpu', 'prepare-operation'))
        self.limit = owner.budget.segment_operations
        if self.preparation:
            self.limit = min(self.limit, max(1, int(host_allowance / self.preparation)))

    def select(self, key, begin, count, *, require_complete=False):
        if len(self.costs) == self.limit:
            return None
        if self.costs:
            minimum = self._costs.predict(key, 1)
            if minimum is None or minimum > self.remaining:
                return None
        probing = not self.costs and self._costs.needs_probe(key, min(count-begin, self._budget.range_elements))
        rows, prediction = self._costs.choose(key, count-begin, self._budget.range_elements,
                                             self.remaining, probe=probing, probe_target=self.probe_target)
        if require_complete and begin + rows != count:
            return None
        self.costs.append((key, rows, prediction))
        self.remaining = 0 if prediction is None or probing else self.remaining - prediction
        return begin + rows


def _gpu_preparation_window(costs, key, units, required_bytes, budget, execution, transfer):
    minimum = costs.predict(key, units)
    if not execution or (minimum is not None and minimum > execution) or required_bytes > transfer:
        # Neither a dispatch nor its arguments can use this window. Prepare
        # both against the next full opportunity; admission still uses the
        # actual current ledger.
        return budget.gpu_frame_ns, budget.transfer_frame_bytes
    return execution, transfer


def _dispatch_rows(rows, grain):
    return (rows + grain - 1) // grain * grain


class GpuSegment:
    """Select a measured dispatch prefix within the GPU and transfer allowance."""
    def __init__(self, owner, *, execution_allowance=None, transfer_allowance=None,
                 pending_profiles=()):
        self._costs = owner._costs
        self._budget = owner.budget
        self.allowance = owner.budget.gpu_frame_ns if execution_allowance is None else execution_allowance
        self.transfer_remaining = (owner.budget.transfer_frame_bytes if transfer_allowance is None
                                   else transfer_allowance)
        self.preparation = self._costs.host(("gpu", "prepare-operation"))
        self._pending_profiles = pending_profiles
        self.limit = owner.budget.segment_operations
        self.costs = []
        self.target = self.remaining = 0
        self.replay_end = 0
        self.indivisible = False

    def select(self, key, begin, count, *, argument_bytes, max_dispatch_rows=None,
               dispatch_grain=1, whole=False, require_complete=False):
        if len(self.costs) == self.limit:
            return None
        extent = count - begin
        first = not self.costs
        limit = min(extent, self._budget.range_elements)
        if max_dispatch_rows is not None:
            limit = min(limit, max_dispatch_rows)
            if whole and extent > max_dispatch_rows:
                from infernux.compute import ComputeCapabilityError

                raise ComputeCapabilityError(_dispatch_limit_message(extent, max_dispatch_rows))
        # Profile the workgroups actually dispatched. The cursor still
        # advances only by logical rows, including a partial final group.
        priced_extent = _dispatch_rows(extent, dispatch_grain)
        priced_limit = _dispatch_rows(limit, dispatch_grain)
        pending_profile = key in self._pending_profiles
        probe_width = (self._costs.dispatch_probe_width(key, priced_limit, self._budget.gpu_frame_ns,
                                                       grain=dispatch_grain)
                       if not whole and (first or pending_profile) else 0)
        minimum_units = priced_extent if whole else dispatch_grain
        if probe_width:
            measured_width = self._costs.samples[key][0][0]
            if pending_profile:
                priced_limit, probe_width = measured_width, 0
            else:
                minimum_units = measured_width
        if first:
            self.allowance, self.transfer_remaining = _gpu_preparation_window(
                self._costs, key, minimum_units, argument_bytes, self._budget,
                self.allowance, self.transfer_remaining)
            # Paying the fixed launch cost again for a smaller range cannot
            # make the dispatch cheaper. Bound useful work and the entire packet.
            self.target = self.remaining = self._costs.gpu_target(key, self._budget.gpu_slice_ns, self.allowance)
        else:
            minimum = self._costs.predict(key, minimum_units)
            if argument_bytes > self.transfer_remaining or minimum is None or minimum > self.remaining:
                return None
        if whole:
            rows, prediction = priced_extent, self._costs.predict(key, priced_extent)
        elif probe_width:
            estimate = self._costs.predict(key, probe_width)
            rows, prediction = probe_width, estimate if estimate <= self.remaining else None
        elif not self._costs.samples.get(key):
            # Leave space for a distinct larger sample while starting with
            # enough rows to measure actual work on wide GPU workgroups.
            initial = max(dispatch_grain, (_GPU_CALIBRATION_ROWS // 4) // dispatch_grain * dispatch_grain)
            rows, prediction = min(priced_limit, initial), None
        else:
            # A complete exact sample can use the rest of this opportunity.
            # A repeated slice at its size cap still needs to meet the target.
            exact_allowance = self.allowance if first and extent == limit else self.remaining
            rows, prediction = self._costs.choose(
                key, priced_extent, priced_limit, self.remaining,
                probe_target=exact_allowance, unit=dispatch_grain)
        logical_rows = extent if whole else min(rows, limit)
        # For profiled rows this choice stays the same until its extent is
        # clamped. A rejected candidate matters too: a shorter tail can fit.
        self.replay_end = begin + logical_rows
        if require_complete and logical_rows != extent:
            return None
        candidate = (*self.costs, (key, rows, prediction))
        if not first and sum(value for _, _, value in candidate) * self._costs.packet_scale(candidate) > self.target:
            return None
        self.costs.append((key, rows, prediction))
        self.indivisible = first and (whole or rows == dispatch_grain)
        self.transfer_remaining -= argument_bytes
        self.remaining = 0 if prediction is None or probe_width else self.remaining - prediction
        return begin + logical_rows

    def select_rows(self, key, begin, count, *, argument_bytes, max_dispatch_rows=None, dispatch_grain=1, whole=False):
        """Pack disjoint row ranges, or append a complete successor operation."""
        if self.costs:
            end = self.select(key, begin, count, argument_bytes=argument_bytes,
                              max_dispatch_rows=max_dispatch_rows, dispatch_grain=dispatch_grain,
                              whole=whole, require_complete=True)
            return () if end is None else (end,)
        ends = []
        while begin < count:
            end = self.select(key, begin, count, argument_bytes=argument_bytes,
                              max_dispatch_rows=max_dispatch_rows, dispatch_grain=dispatch_grain, whole=whole)
            if end is None:
                break
            ends.append(end)
            begin = end
        return tuple(ends)


def _dispatch_limit_message(rows, limit):
    return f"an indivisible Vulkan dispatch requires {rows} rows, but the device limit for this kernel is {limit}"


def transfer_key(source_device, target_device, dtype, host_identity):
    return ("host" if source_device == target_device == "cpu" else "gpu",
            int(host_identity), "transfer", source_device, target_device, str(dtype))


def gpu_prepare_key(key):
    return None if key is None else ("gpu-prepare", *key)


def gpu_submit_key(costs, transfer_bytes=0):
    execution = costs[0][0]
    if execution[2] == 'transfer':
        return ("gpu-submit", *execution[1:]), transfer_bytes
    return ("gpu-submit", execution[1], "dispatch"), len(costs)


def gpu_service(owner, costs, *, transfer_bytes=0, indivisible=False, transfer_indivisible=False):
    """One sealed submission, shared by prediction and the actual submit sampler."""
    key, units = gpu_submit_key(costs, transfer_bytes)
    host = owner._costs.predict(("host-service", *key), units)
    if host is None:
        return ServiceDemand(reason="GPU submission host timing is unprofiled")
    if host > owner._max_background_credit():
        return ServiceDemand(reason="GPU submission exceeds the host opportunity allowance", failure="host_target")
    lags = tuple(owner._costs.lag(key) for key, _, _ in costs)
    if any(prediction is None for _, _, prediction in costs) or any(lag is None for lag in lags):
        return ServiceDemand(reason="GPU execution timing or completion lag is unprofiled")
    execution = sum(prediction for _, _, prediction in costs) * owner._costs.packet_scale(costs)
    return ServiceDemand((ServiceGroup('gpu', 1, max(lags), host, execution, transfer_bytes,
                                      indivisible, transfer_indivisible),))


def _gpu_dispatch_prefix(owner, span, begin):
    segment = GpuSegment(owner)
    ends = segment.select_rows(span.key, begin, span.rows,
                               argument_bytes=span.argument_bytes,
                               max_dispatch_rows=span.max_dispatch_rows, dispatch_grain=span.dispatch_grain,
                               whole=span.whole)
    return segment, ends


def _gpu_preparation_service(owner, operations, total_operations, estimate=0):
    """Binding and sealing are indivisible host actions, not divisible time."""
    unit = owner._costs.host(('gpu', 'prepare-operation'))
    seal = owner._costs.predict(('host-service', 'gpu', 'seal'), total_operations)
    if unit is None or seal is None:
        return ServiceDemand(reason="GPU binding preparation or sealing is unprofiled")
    allowance = owner._max_background_credit()
    unit = max(unit, (estimate - seal) / operations) if operations else unit
    if operations and unit > allowance or seal > allowance:
        return ServiceDemand(reason="GPU preparation action exceeds the host opportunity allowance", failure="host_target")
    groups = []
    if operations:
        per_step = max(1, int(allowance // unit))
        full, tail = divmod(operations, per_step)
        if full:
            groups.append(ServiceGroup('host', full, 0, per_step * unit))
        if tail:
            groups.append(ServiceGroup('host', 1, 0, tail * unit))
    groups.append(ServiceGroup('host', 1, 0, seal))
    return ServiceDemand(tuple(groups))


def _gpu_dispatch_span(owner, span, begin, segment, end, estimate):
    groups = []
    while True:
        costs = tuple(segment.costs)
        preparation = _gpu_preparation_service(owner, len(costs), len(costs), estimate)
        if preparation.reason:
            return preparation, None
        demand = gpu_service(owner, costs,
            transfer_bytes=owner.budget.transfer_frame_bytes - segment.transfer_remaining,
            indivisible=segment.indivisible,
            transfer_indivisible=len(costs) == 1 and bool(span.argument_bytes))
        if demand.reason:
            return demand, None
        width = end - begin
        repeats = 1 + (span.rows - segment.replay_end) // width
        groups.append(ServiceCycle((*preparation.groups, *demand.groups), repeats))
        begin += repeats * width
        if begin == span.rows:
            return ServiceDemand(tuple(groups)), segment
        segment, ends = _gpu_dispatch_prefix(owner, span, begin)
        end = ends[-1]


@dataclass(frozen=True)
class ServiceSpan:
    phase: str
    node: int
    key: tuple
    rows: int
    whole: bool = False
    inner_begin: int = 0
    terms: int = 0
    repeats: int = 1
    copy: bool = False
    unit_bytes: int = 1
    argument_bytes: int = 0
    max_dispatch_rows: int | None = None
    dispatch_grain: int = 1


def model_spans(plan, optimizer, facts, lr, record_loss, entries, term_limit, objective="mse", clear_grad=False,
                *, output_index=None, frontier=None):
    """Consume the sole operation builders; retain numeric facts only."""
    return operation_spans(plan._workload(optimizer, facts, lr, record_loss=record_loss,
        objective=objective, clear_grad=clear_grad, output_index=output_index), entries, plan.device, term_limit,
        frontier=frontier)


def _inner_pieces(count, limit, begin=0, pending_end=0):
    pieces = []
    if pending_end > begin:
        kind = 'initial' if begin == 0 else 'final' if pending_end == count else 'resume'
        pieces.append((kind, begin, pending_end - begin, 1))
        begin = pending_end
    if begin == 0:
        end = min(count, limit)
        pieces.append(('initial', 0, end, 1))
        begin = end
    if begin < count:
        full, tail = divmod(count - begin, limit)
        middle = full if tail else full - 1
        if middle:
            pieces.append(('resume', begin, limit, middle))
            begin += middle * limit
        pieces.append(('final', begin, count - begin, 1))
    return pieces


def operation_spans(operations, entries, device, term_limit, *, frontier=None):
    from infernux._work import inner_range, _ValueLayout
    result, indices = [], {}
    for phase, operation in operations:
        node = indices.get(phase, 0)
        indices[phase] = node + 1
        if operation.copy:
            # This copy is real GPU service, not a zero-cost graph edge. It
            # needs transfer timing before the entire request can be planned.
            host = next(iter(entries.values())).host
            key = transfer_key(device, device, operation.params[0].dtype, host.identity)
            result.append(ServiceSpan(phase, node, key, operation.params[0].nbytes, copy=True,
                                      unit_bytes=np.dtype(operation.params[0].dtype).itemsize))
            continue
        program, params = operation.program, operation.params
        if entries:
            entry = entries[program, operation.reverse]
            rows = params[entry.artifact.domain_parameter].element_count
        else:
            entry = program
            rows = params[program.domain_parameter].element_count
        control = None if operation.reverse else inner_range(program)
        cursor = frontier[1] if frontier is not None and frontier[0] == phase else None
        at_frontier = cursor is not None and cursor.index == node
        current = at_frontier and cursor.carry
        unsplit = at_frontier and (cursor.completed_end or any(
            work.index == node and work.operation.program is program for work in frontier[2]))
        if control is None or not current and (control.count <= term_limit or unsplit):
            result.append(ServiceSpan(phase, node, execution_key(entry, params), rows, operation.reverse,
                argument_bytes=0 if not entries else int(entry.artifact.argument_layout['size']),
                max_dispatch_rows=entry.max_dispatch_rows if entries else None,
                dispatch_grain=entry.dispatch_grain if entries else 1))
            continue
        carry = tuple(_ValueLayout((rows,), device) for _ in range(2))
        pieces = _inner_pieces(control.count, term_limit,
            cursor.inner_begin if current else 0, cursor.inner_end if current else 0)
        for kind, begin, terms, repeats in pieces:
            variant = program._inner_variants[kind]
            entry = entries[variant, False] if entries else variant
            arguments = (*params, *carry, np.int32(begin), np.int32(begin + terms))
            result.append(ServiceSpan(phase, node, execution_key(entry, arguments), rows,
                inner_begin=begin, terms=terms, repeats=repeats,
                argument_bytes=0 if not entries else int(entry.artifact.argument_layout['size']),
                max_dispatch_rows=entry.max_dispatch_rows if entries else None,
                dispatch_grain=entry.dispatch_grain if entries else 1))
    return tuple(result)


@dataclass(frozen=True)
class ServiceGroup:
    domain: str
    steps: int
    lag: int
    owner_ns: float = 0
    execution_ns: float = 0
    transfer_bytes: int = 0
    indivisible: bool = False
    transfer_indivisible: bool = False


@dataclass(frozen=True)
class ServiceCycle:
    groups: tuple
    repeats: int

    @property
    def steps(self):
        return self.repeats * sum(group.steps for group in self.groups)


def _service_units(groups):
    for group in groups:
        if isinstance(group, ServiceCycle):
            for _ in range(group.repeats):
                yield from _service_units(group.groups)
        else:
            for _ in range(group.steps):
                yield group


def _uses(groups, domain):
    return any(_uses(group.groups, domain) if isinstance(group, ServiceCycle) else group.domain == domain
               for group in groups)


@dataclass(frozen=True)
class ServiceDemand:
    groups: tuple = ()
    ready_at: int | None = None
    reason: str = ""
    prerequisites: tuple = ()
    minimum_atom_ns: float = 0
    occupied: tuple = ()
    failure: str = ""

    @property
    def steps(self):
        return sum(group.steps for group in self.groups)


def host_service(owner, key, units=None):
    cost = (owner._costs.host(key) if units is None else
            owner._costs.predict(("host-service", *key), max(1, units)))
    if cost is None:
        return ServiceDemand(reason="host action is unprofiled: " + str(key[-1]))
    if cost > owner._max_background_credit():
        return ServiceDemand(reason="host action exceeds its opportunity allowance", failure="host_target")
    return ServiceDemand((ServiceGroup("host", 1, 0, cost),))


def combine_services(availability, demands):
    if availability.reason:
        return availability
    groups, atom = [], 0
    for demand in demands:
        if demand.reason:
            return demand
        groups.extend(demand.groups)
        atom = max(atom, demand.minimum_atom_ns)
    return ServiceDemand(tuple(groups), availability.ready_at, prerequisites=availability.prerequisites,
                         minimum_atom_ns=atom, occupied=availability.occupied)


def _service_availability(request, owner, opportunity):
    """Observe no receipts: the estimate never advances a real cursor."""
    occupied = []
    for task in request._io_tasks():
        if not task.pending:
            continue
        observation = owner._io_observations.get(task)
        if observation is None:
            return ServiceDemand(reason="accepted IO has no service observation")
        available = observation.availability(owner._costs, opportunity, owner._last_opportunity)
        if available.reason:
            return available
        occupied.extend(available.occupied)
    packet = request.packet
    gpu = packet is not None and packet.completion is not None
    cpu = request.task is not None and request.task._native is not None and request.task._native.submitted
    if gpu or cpu:
        waits = [owner._costs.lag(key) for key, _, _ in request.execution_costs]
        if not waits or any(lag is None for lag in waits):
            return ServiceDemand(reason="accepted work has no observed completion-lag profile")
        observed = owner._last_opportunity
        predicted = request.metrics["service_submitted_opportunity"] + max(waits)
        if observed > predicted:
            return ServiceDemand(reason="accepted work exceeded its observed completion-lag profile")
        # A ticket still pending at the estimated host boundary occupies its
        # lane until another observation, even when earlier samples retired here.
        occupied.append(("gpu" if gpu else "cpu", max(predicted, observed + 1)))
    return ServiceDemand(ready_at=max((opportunity, *(until for _, until in occupied))), occupied=tuple(occupied))


def choose_transfer_range(owner, identity, unit, extent, *, byte_allowance=None, gpu_allowance=None):
    """Pure range policy shared by live preparation and future transfer recipes."""
    byte_allowance = owner.budget.transfer_frame_bytes if byte_allowance is None else byte_allowance
    gpu_allowance = owner.budget.gpu_frame_ns if gpu_allowance is None else gpu_allowance
    key = transfer_key(*identity)
    gpu_allowance, byte_allowance = _gpu_preparation_window(
        owner._costs, key, unit, unit, owner.budget, gpu_allowance, byte_allowance)
    cap = min(extent, owner.transfer_chunk_bytes, byte_allowance)
    if cap < unit:
        return 0, None
    target = owner._costs.gpu_target(key, owner.budget.gpu_slice_ns, gpu_allowance)
    return owner._costs.choose(key, extent, cap, target, probe=True, probe_target=gpu_allowance, unit=unit)


def transfer_service(blocks, identity, owner, host_key, *, unit_bytes=4, project=True, prepared_costs=()):
    """Price numerical transfer layouts with their actual consumer's host key."""
    groups, atom = [], 0
    for steps, repeats in blocks:
        first = steps[0]
        if project and first.kind in ('upload', 'readback', 'device_copy'):
            from infernux.compute import _transfer_work
            chunk, prediction = choose_transfer_range(owner, identity, unit_bytes, first.end-first.begin)
            if not chunk:
                return ServiceDemand(reason="transfer allowance cannot hold one element", failure="configuration")
            if prediction is None:
                return ServiceDemand(reason="transfer range needs a device timing observation")
            projected = _transfer_work(first.end, chunk, begin=first.begin, allocate=first.allocate,
                source_storage='cpu' if first.kind == 'upload' else 'gpu',
                target_storage='cpu' if first.kind == 'readback' else 'gpu')
            demand = transfer_service(projected, identity, owner, host_key, unit_bytes=unit_bytes, project=False)
            if demand.reason:
                return demand
            atom = max(atom, demand.minimum_atom_ns)
            groups.extend((ServiceCycle(demand.groups, repeats),) if repeats > 1 else demand.groups)
            continue
        units = []
        for step in steps:
            if step.domain == "gpu":
                key = transfer_key(*identity)
                extent = step.end - step.begin
                costs = (prepared_costs if step.kind == 'submit' and prepared_costs else
                         ((key, extent, owner._costs.predict(key, extent)),))
                minimum = costs[0][2]
                if minimum is None:
                    return ServiceDemand(reason="transfer execution timing is unprofiled")
                atom = max(atom, minimum)
                if step.kind != 'submit':
                    host = owner._costs.host(gpu_prepare_key(host_key(step)))
                    if host is None:
                        return ServiceDemand(reason="transfer host preparation is unprofiled")
                    if host > owner._max_background_credit():
                        return ServiceDemand(reason="transfer preparation exceeds the host allowance", failure="host_target")
                    units.append(ServiceGroup('host', 1, 0, host))
                demand = gpu_service(owner, costs, transfer_bytes=extent, indivisible=extent == unit_bytes,
                                     transfer_indivisible=extent == unit_bytes)
                if demand.reason:
                    return demand
                units.extend(demand.groups)
            else:
                key = host_key(step)
                host = owner._costs.host(key)
                if ((host is None or host > owner._max_background_credit())
                        and key[-1] is True and step.kind in ("materialize", "copy_host")):
                    from infernux.compute import _TransferStep

                    materialize = host_service(owner, (*key[:-1], False))
                    complete = host_service(owner, host_key(_TransferStep("complete", step.end, step.end)))
                    if not materialize.reason and not complete.reason:
                        units.extend((*materialize.groups, *complete.groups))
                        continue
                if host is None:
                    return ServiceDemand(reason="transfer host materialization/publication is unprofiled")
                if host > owner._max_background_credit():
                    return ServiceDemand(reason="transfer host service exceeds the opportunity allowance", failure="host_target")
                units.append(ServiceGroup('host', 1, 0, host))
        groups.append(ServiceCycle(tuple(units), repeats) if repeats > 1 else units[0] if len(units) == 1
                      else ServiceCycle(tuple(units), 1))
    return ServiceDemand(tuple(groups), minimum_atom_ns=atom)


def value_transfer_service(owner, blocks, identity, nbytes, tag=(), *, unit_bytes=None, append=False,
                           prepared_costs=(), allocate=False):
    """Movement and its consumer's measured adoption or publication."""
    from infernux.compute import _TransferStep, _transfer_host_key

    if not blocks or not append and blocks[-1][0][-1].domain == "gpu":
        blocks += (((_TransferStep("complete", nbytes, nbytes, allocate),), 1),)
    return transfer_service(blocks, identity, owner, lambda step: (*tag, *_transfer_host_key(
        identity, nbytes, step, finalizing=step.domain == "host" and step.end == nbytes)),
        unit_bytes=np.dtype(identity[2]).itemsize if unit_bytes is None else unit_bytes,
        prepared_costs=prepared_costs)


def cursor_transfer_service(request, owner, cursor, tag=(), *, active=True, append=False):
    packet = request.packet if active else None
    prepared = packet is not None and packet.completion is None
    blocks = cursor.remaining_work(prepared=prepared, accepted=packet is not None and not prepared)
    return value_transfer_service(owner, blocks, cursor.cost_identity, cursor.nbytes, tag,
        unit_bytes=cursor.dtype.itemsize, append=append, prepared_costs=request.execution_costs if prepared else (),
        allocate=cursor.target is None and not blocks)


def remaining_transfer_service(request, owner, opportunity):
    """Price the same bounded recipe used by actual transfer execution."""
    availability = _service_availability(request, owner, opportunity)
    if availability.reason:
        return availability
    return combine_services(availability, (cursor_transfer_service(request, owner, request.cursor),))


def remaining_model_service(request, owner, opportunity, *, capacity_bytes):
    """Project input transfer, computation, result transfer and publication."""
    availability = _service_availability(request, owner, opportunity)
    if availability.reason:
        return availability
    stages = []
    for index, (names, cursor) in enumerate(request.io):
        demand = cursor_transfer_service(request, owner, cursor, request._transfer_service_tag(names), active=index == 0)
        if demand.reason:
            return demand
        stages.append(demand)
    if request.io and not isinstance(request.io[0][0], tuple):
        stages.append(host_service(owner, (*request.service_signature, "publish")))
        return combine_services(availability, stages)

    run = request.run
    transfers_result = request._needs_result_transfer() and not request.result_storage
    work = remaining_work_service(request, None if run is None else run.cursor, owner,
        opportunity, "forward" if run is None else run.phase,
        ("forward", "loss", "backward", "update", "ready_to_commit", "complete"),
        availability=availability, active=not request.io, publish=not transfers_result)
    if work.reason:
        return work
    stages.append(work)
    if transfers_result:
        from infernux import compute

        bind = host_service(owner, (*request.service_signature, "result-bind"))
        if bind.reason:
            return bind
        stages.append(bind)
        source, target = request.plan.device, request.result_device
        web = compute._web_cpu_compute_enabled()
        source_storage, target_storage = ("cpu", "cpu") if web else (source, target)
        capacity = owner.budget.capacity_bytes - (owner._capacity() if capacity_bytes is None else capacity_bytes)
        for index, nbytes, dtype in request._result_transfer_layouts():
            chunk = compute._fit_transfer_chunk(min(nbytes, owner.transfer_chunk_bytes), dtype.itemsize, capacity,
                                                source_storage, target_storage)
            if not chunk:
                return ServiceDemand(reason="result staging does not fit the current capacity profile")
            identity = source, target, dtype.name, 0 if web else int(request.host.identity)
            blocks = compute._transfer_work(nbytes, chunk, source_storage=source_storage, target_storage=target_storage)
            stages.append(value_transfer_service(owner, blocks, identity, nbytes, request._transfer_service_tag(index)))
        stages.append(host_service(owner, (*request.service_signature, "publish")))
    return combine_services(availability, stages)


def remaining_work_service(request, cursor, owner, opportunity, phase="compute", phases=("compute",), *,
                           availability=None, active=True, publish=True):
    """Price the same confirmed cursor used by any numerical request."""
    if availability is None:
        availability = _service_availability(request, owner, opportunity)
    if availability.reason:
        return availability
    preparation = request.gpu_preparation
    key = (owner._costs.revision, owner.budget, phase, phases, active, publish,
           None if cursor is None else (id(cursor), cursor.index, cursor.inner_begin, cursor.inner_end,
                                        cursor.inner_count, cursor.completed_end),
           None if preparation is None else preparation.index,
           request.packet is not None, request.packet is not None and request.packet.completion is not None,
           request.transfer_bytes, request.indivisible, request.transfer_indivisible, bool(availability.occupied),
           tuple((value.index, value.begin, value.end, value.count) for value in request.ranges))
    if key != request._forecast_key:
        request._forecast_work = _remaining_work_service(request, cursor, owner, opportunity, phase, phases,
            availability=ServiceDemand(occupied=availability.occupied), active=active, publish=publish)
        request._forecast_key = key
    # Completion observations and their current host boundary are always fresh;
    # only the unchanged numeric work description is reused.
    return combine_services(availability, (request._forecast_work,))


def _remaining_work_service(request, cursor, owner, opportunity, phase, phases, *,
                            availability, active, publish):
    phases = {name: index for index, name in enumerate(phases)}
    node = 0 if cursor is None else cursor.index
    inner = 0 if cursor is None else cursor.inner_begin
    begin = 0 if cursor is None else cursor.completed_end
    preparing = active and request.gpu_preparation is not None
    prepared = preparing or active and request.packet is not None and request.packet.completion is None
    groups, atom = [], 0
    if preparing:
        preparation = request.gpu_preparation
        remaining = _gpu_preparation_service(owner, len(preparation.launches) - preparation.index,
                                              len(preparation.launches))
        if remaining.reason:
            return remaining
        groups.extend(remaining.groups)
    if prepared:
        demand = gpu_service(owner, request.execution_costs, transfer_bytes=request.transfer_bytes,
                             indivisible=request.indivisible,
                             transfer_indivisible=request.transfer_indivisible)
        if demand.reason:
            return demand
        groups.extend(demand.groups)
    if prepared or active and availability.occupied:
        if cursor.current().copy:
            transfer = cursor.transfer
            begin, count = transfer.pending_end, transfer.nbytes
        else:
            last = request.ranges[-1]
            node, begin, count = last.index, last.end, last.count
        if begin == count:
            if cursor.carry and cursor.inner_end < cursor.inner_count:
                inner, begin = cursor.inner_end, 0
            else:
                node, inner, begin = node + 1, 0, 0
    cpu_segment, cpu_phase = None, None
    gpu_segment, gpu_phase = None, None
    for span in request.service_spans:
        if phases[span.phase] < phases[phase] or span.phase == phase and span.node < node:
            continue
        skipped, first = 0, 0
        if span.phase == phase and span.node == node:
            if span.terms:
                skipped = max(0, (inner - span.inner_begin) // span.terms)
                if skipped >= span.repeats:
                    continue
                first = begin if inner == span.inner_begin + skipped * span.terms else 0
            else:
                first = begin
        if first == span.rows:
            continue
        target = min(owner.budget.gpu_slice_ns, owner.budget.gpu_frame_ns) if span.key[0] == "gpu" else min(
            owner.budget.cpu_slice_ns, owner.budget.cpu_frame_ns)
        cap = owner.transfer_chunk_bytes // span.unit_bytes * span.unit_bytes if span.copy else owner.budget.range_elements
        if span.max_dispatch_rows is not None:
            if span.whole and span.rows > span.max_dispatch_rows:
                return ServiceDemand(reason=_dispatch_limit_message(span.rows, span.max_dispatch_rows),
                                     failure="configuration")
            cap = min(cap, span.max_dispatch_rows)
        if not cap:
            return ServiceDemand(reason="transfer_bytes cannot hold one typed element", failure="configuration")
        repeats = span.repeats - skipped
        max_rows = span.rows if repeats > 1 else span.rows - first
        minimum_rows = (_dispatch_rows(span.rows, span.dispatch_grain) if span.whole else
                        span.unit_bytes if span.copy else span.dispatch_grain)
        minimum = owner._costs.predict(span.key, minimum_rows)
        lag = owner._costs.lag(span.key)
        if minimum is None or lag is None:
            return ServiceDemand(reason="operation timing or completion lag is unprofiled")
        if not span.whole and not span.copy:
            probe_limit = min(cap, max_rows)
            probe = (owner._costs.dispatch_probe_width(span.key, _dispatch_rows(probe_limit, span.dispatch_grain),
                                                      owner.budget.gpu_frame_ns, grain=span.dispatch_grain) if span.key[0] == 'gpu'
                     else owner._costs.needs_probe(span.key, probe_limit))
            if probe:
                return ServiceDemand(reason="launch and row cost need a distinct-size observation")
        atom = max(atom, minimum)
        if span.key[0] != "gpu" and minimum > target:
            return ServiceDemand(reason="minimum legal operation exceeds its measured service target",
                                 minimum_atom_ns=minimum, failure="atom")
        host_key = (*request.service_signature, span.phase)
        estimate = owner._costs.host(gpu_prepare_key(host_key) if span.key[0] == 'gpu' else host_key)
        if estimate is None:
            return ServiceDemand(reason="host preparation or publication timing is unprofiled")
        if estimate > owner._max_background_credit() and (span.key[0] != 'gpu' or span.copy):
            return ServiceDemand(reason="host preparation cannot fit the configured opportunity allowance", failure="host_target")
        if span.key[0] == 'gpu' and not span.copy:
            cpu_segment = None
            packable = not span.terms
            ends = (gpu_segment.select_rows(span.key, first, span.rows, argument_bytes=span.argument_bytes,
                                           max_dispatch_rows=span.max_dispatch_rows, dispatch_grain=span.dispatch_grain,
                                           whole=span.whole)
                    if packable and gpu_segment is not None and gpu_phase == span.phase else ())
            following = bool(ends)
            if following:
                segment = gpu_segment
                previous = groups.pop()
                if previous.repeats > 1:
                    groups.append(ServiceCycle(previous.groups, previous.repeats - 1))
                estimate = max(estimate, sum(group.owner_ns * group.steps for group in previous.groups
                                             if group.domain == 'host'))
            else:
                segment, ends = _gpu_dispatch_prefix(owner, span, first)
            demand, tail = _gpu_dispatch_span(owner, span, first, segment, ends[-1], estimate)
            if demand.reason:
                return demand
            gpu_segment = tail if packable and (following or len(tail.costs) == 1) else None
            gpu_phase = span.phase
            if first and repeats > 1:
                groups.extend(demand.groups)
                segment, ends = _gpu_dispatch_prefix(owner, span, 0)
                demand, _ = _gpu_dispatch_span(owner, span, 0, segment, ends[-1], estimate)
                if demand.reason:
                    return demand
                repeats -= 1
            groups.extend(demand.groups if repeats == 1 else (ServiceCycle(demand.groups, repeats),))
            continue
        gpu_segment = None
        if span.whole:
            rows, execution, steps = span.rows, minimum, repeats
            transfer_bytes = span.argument_bytes
        else:
            if span.copy:
                identity = span.key[3], span.key[4], span.key[5], span.key[1]
                rows = choose_transfer_range(owner, identity, span.unit_bytes, span.rows-first)[0]
            else:
                rows = owner._costs.choose(span.key, span.rows, cap, target)[0]
            steps = (span.rows-first + rows-1) // rows
            steps += (repeats-1) * ((span.rows+rows-1) // rows)
            execution = owner._costs.predict(span.key, rows)
            transfer_bytes = rows if span.copy else span.argument_bytes
        if span.key[0] == 'gpu':
            cpu_segment = None
            costs = ((span.key, rows, execution),)
            demand = gpu_service(owner, costs, transfer_bytes=transfer_bytes,
                indivisible=rows == span.unit_bytes, transfer_indivisible=rows == span.unit_bytes)
            if demand.reason:
                return demand
            groups.append(ServiceCycle((ServiceGroup('host', 1, 0, estimate), *demand.groups), steps))
            continue
        packable = steps == 1 and not (span.whole or span.copy or span.terms)
        if not packable:
            cpu_segment = None
            groups.append(ServiceGroup('cpu', steps, lag, estimate, execution, transfer_bytes,
                                       span.whole or rows == (span.unit_bytes if span.copy else 1)))
            continue
        if cpu_segment is None or cpu_phase != span.phase:
            cpu_segment, cpu_phase = CpuSegment(owner, host_allowance=owner._max_background_credit()), span.phase
        following = bool(cpu_segment.costs)
        if cpu_segment.select(span.key, first, span.rows, require_complete=True) is None:
            cpu_segment, following = CpuSegment(owner, host_allowance=owner._max_background_credit()), False
            cpu_segment.select(span.key, first, span.rows, require_complete=True)
        previous = groups.pop() if following else None
        host = max(estimate, (cpu_segment.preparation or 0) * len(cpu_segment.costs),
                   0 if previous is None else previous.owner_ns)
        if host > owner._max_background_credit():
            return ServiceDemand(reason='CPU segment preparation exceeds the host opportunity allowance', failure='host_target')
        groups.append(ServiceGroup('cpu', 1, max(lag, 0 if previous is None else previous.lag), host,
                                   sum(cost for _, _, cost in cpu_segment.costs), 0,
                                   all(rows == 1 for _, rows, _ in cpu_segment.costs)))
    if _uses(groups, "gpu"):
        if not owner._gpu_prepare_unit_ns:
            return ServiceDemand(reason="GPU preparation timing is unprofiled")
        if owner._gpu_prepare_unit_ns > owner._max_background_credit():
            return ServiceDemand(reason="one GPU preparation unit exceeds the host opportunity allowance", failure="host_target")
    computation = ServiceDemand(tuple(groups), minimum_atom_ns=atom)
    if not publish:
        return combine_services(availability, (computation,))
    publication = host_service(owner, (*request.service_signature, "publish"))
    return combine_services(availability, (computation, publication))


@dataclass(frozen=True)
class ServiceSlot:
    request: object
    group: ServiceGroup


class ServiceCalendar:
    """One finite list-scheduling witness, never another runtime executor."""
    def __init__(self, demands, deadlines, opportunity, *, budget_ns, earliest=None,
                 owner_ns=float("inf"), owner_now_ns=float("inf"), occupied=(),
                 resource_limits=None, resource_remaining=None, resource_admitted=None):
        self.slots, self.completions, self.slot_owner_ns, self.reason, self.failure = {}, {}, {}, "", ""
        self.failed_request = None
        self.scheduled_before_failure = frozenset()
        self.demands = demands
        self.slot_resources = {}
        if not demands:
            return
        if any(demand.reason or demand.ready_at is None for demand in demands.values()):
            self.failure = "unknown"
            self.reason = "a prerequisite or service cost is unknown"
            return
        finish = max(deadlines.values())
        pending = {}
        for request, demand in demands.items():
            units = _service_units(demand.groups)
            pending[request] = [units, next(units, None), demand.ready_at]
        resource = lambda domain: "worker" if domain in ("cpu", "io") else domain
        busy = {"worker": opportunity, "gpu": opportunity}
        for domain, until in occupied:
            name = resource(domain)
            busy[name] = max(busy.get(name, opportunity), until)
        for demand in demands.values():
            for domain, until in demand.occupied:
                name = resource(domain)
                busy[name] = max(busy.get(name, opportunity), until)
        earliest = {} if earliest is None else earliest
        limits = {} if resource_limits is None else resource_limits
        remaining = limits if resource_remaining is None else resource_remaining
        admitted = {} if resource_admitted is None else resource_admitted
        started = perf_counter_ns()
        current = opportunity
        while pending:
            if perf_counter_ns() - started >= budget_ns:
                self.failure = "planning_budget"
                self.reason = "service planning exhausted this opportunity's host allowance"
                break
            candidates = []
            next_ready = []
            resolved = False
            for request, (_, group, ready) in tuple(pending.items()):
                before = demands[request].prerequisites
                if any(dependency not in self.completions for dependency in before):
                    continue
                if before:
                    ready = max(ready, max(self.completions[dependency] for dependency in before))
                if group is None:
                    if ready > deadlines[request]:
                        self.failure = "window"
                        self.failed_request = request
                        self.reason = "accepted service retires beyond its completion window"
                        break
                    if ready <= current:
                        self.completions[request] = ready
                        del pending[request]
                        resolved = True
                    else:
                        next_ready.append(ready)
                    continue
                available = max(ready, busy.get(resource(group.domain), current),
                                earliest.get(group.domain, opportunity))
                if group.owner_ns > owner_ns:
                    self.failure = "host_target"
                    self.reason = "one host service unit exceeds the configured opportunity allowance"
                    break
                costs = {group.domain: group.execution_ns} if group.domain != 'host' else {}
                if group.transfer_bytes:
                    costs['transfer'] = group.transfer_bytes
                if any(cost > limits.get(domain, float('inf')) and
                       not (domain == group.domain and group.indivisible or domain == 'transfer' and
                            group.transfer_indivisible) for domain, cost in costs.items()):
                    self.failure = 'resource_target'
                    self.reason = 'one execution unit exceeds its configured resource allowance'
                    break
                host_available = owner_now_ns if current == opportunity else owner_ns
                frame_limits = remaining if current == opportunity else limits
                used = self.slot_resources.get(current, {})
                def fits(domain, cost):
                    allowance = frame_limits.get(domain, float('inf'))
                    spent = used.get(domain, 0)
                    return (spent + cost <= allowance or
                            (domain == group.domain and group.indivisible or domain == 'transfer' and
                             group.transfer_indivisible) and spent == 0 and allowance > 0 and
                            (current != opportunity or admitted.get(domain, 0) == 0))
                if (self.slot_owner_ns.get(current, 0) + group.owner_ns > host_available or
                        any(not fits(domain, cost) for domain, cost in costs.items())):
                    available = max(available, current + 1)
                if available <= current:
                    candidates.append(request)
                else:
                    next_ready.append(available)
            if self.reason:
                break
            if not candidates:
                if not pending:
                    break
                if resolved:
                    continue
                if not next_ready:
                    self.failure = "dependencies"
                    self.reason = "service prerequisites cannot become ready in this plan"
                    break
                current = min(next_ready)
                if current > finish:
                    self.failure = "window"
                    self.reason = "service dependencies extend beyond the completion window"
                    break
                continue
            request = min(candidates, key=lambda value: deadlines[value])
            units, group, _ = pending[request]
            if current == deadlines[request] and group.domain != "host":
                self.failure = "window"
                self.failed_request = request
                self.reason = "new service cannot start at its completion window's final observation opportunity"
                break
            ready = current + group.lag
            if ready > deadlines[request]:
                self.failure = "window"
                self.failed_request = request
                self.reason = "joint service demand cannot be reserved by this scheduling policy"
                break
            self.slots.setdefault(current, []).append(ServiceSlot(request, group))
            self.slot_owner_ns[current] = self.slot_owner_ns.get(current, 0) + group.owner_ns
            used = self.slot_resources.setdefault(current, {})
            if group.domain != 'host':
                used[group.domain] = used.get(group.domain, 0) + group.execution_ns
            if group.transfer_bytes:
                used['transfer'] = used.get('transfer', 0) + group.transfer_bytes
            if resource(group.domain) in busy:
                busy[resource(group.domain)] = ready
            following = next(units, None)
            if following is not None:
                pending[request][1:] = following, ready
            else:
                self.completions[request] = ready
                del pending[request]
        if self.reason:
            if self.failed_request is not None:
                self.scheduled_before_failure = frozenset(slot.request for slots in self.slots.values() for slot in slots)
            self.slots.clear()
            self.completions.clear()
            self.slot_owner_ns.clear()
            self.slot_resources.clear()

    def validate_host_supply(self, opportunity, remaining_ns, future_ns):
        """Check actual planning expense; never charge an unused allowance."""
        if self.reason:
            return
        if any(cost > (remaining_ns if stamp == opportunity else future_ns)
               for stamp, cost in self.slot_owner_ns.items()):
            self.failure = "host_supply"
            self.reason = "measured planning and host service exceed the available opportunity credit"
            self.slots.clear()
            self.completions.clear()
            self.slot_owner_ns.clear()
            self.slot_resources.clear()

    def discard(self, request):
        """Withdraw a forecast; accepted work stays in the real resource ledger."""
        self.demands.pop(request, None)
        self.completions.pop(request, None)
        for stamp, slots in tuple(self.slots.items()):
            slots = [slot for slot in slots if slot.request is not request]
            if not slots:
                self.slots.pop(stamp)
                self.slot_owner_ns.pop(stamp, None)
                self.slot_resources.pop(stamp, None)
                continue
            self.slots[stamp] = slots
            self.slot_owner_ns[stamp] = sum(slot.group.owner_ns for slot in slots)
            resources = self.slot_resources[stamp] = {}
            for slot in slots:
                group = slot.group
                if group.domain != 'host':
                    resources[group.domain] = resources.get(group.domain, 0) + group.execution_ns
                if group.transfer_bytes:
                    resources['transfer'] = resources.get('transfer', 0) + group.transfer_bytes
