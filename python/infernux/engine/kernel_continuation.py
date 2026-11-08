"""Ordinary kernel requests on the same owner, cursor and CPU/Vulkan receipts."""
from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np

from infernux import compute
from infernux.compute_source import BufferSource
from infernux._work import (_Operation, _ValueLayout, _WorkCursor, domain_parameter, inner_range,
                           program_family, retire)
from .cpu_continuation import (CpuRequest, _RequestStorage, _prepare_cpu_work_ranges,
                              _prepare_gpu_work_ranges, _confirm_work_ranges, _observed_numerical_host_key, _slice_work)
from .service_windows import execution_key, operation_spans, remaining_work_service


@dataclass(eq=False, kw_only=True)
class KernelRequest(CpuRequest):
    prepared: object
    params: tuple
    cursor: object = None
    private: list = field(default_factory=list)
    source_versions: tuple = ()
    ranges: tuple = ()
    service_spans: tuple = ()
    service_signature: tuple = ()
    prepare_peak_bytes: int = 0

    @classmethod
    def create(cls, owner, facts, prepared, params):
        snapshots = {}
        for index in prepared.reads:
            value = params[index]
            if prepared.device == "cpu" and isinstance(value, compute.Buffer):
                snapshots.setdefault((value._array.ctypes.data, value.nbytes, value.dtype), value)
        snapshot_bytes = sum(value.nbytes for value in snapshots.values())
        if snapshot_bytes > owner.budget.snapshot_bytes:
            raise ValueError("kernel inputs exceed the atomic host snapshot budget")
        output_bytes = sum(params[index].nbytes for index in prepared.writes)
        control = inner_range(prepared.program)
        scratch = (params[domain_parameter(prepared.program)].element_count * 8 + 8
                   if control is not None and control.count > owner.budget.range_terms else 0)
        scalars = 4 * sum(not isinstance(value, (compute.Buffer, BufferSource)) for value in params)
        request = cls(**facts, prepared=prepared, params=(),
                      reserved_bytes=snapshot_bytes + output_bytes + scratch + scalars)
        values = []
        try:
            for index, value in enumerate(params):
                if prepared.device == "gpu" and index in prepared.writes:
                    bound = _ValueLayout(value.shape, value.device, value.dtype)
                elif isinstance(value, compute.Buffer):
                    bound = value.view(readonly=index not in prepared.writes)
                elif isinstance(value, BufferSource):
                    bound = value
                else:
                    bound = np.int32(value) if prepared.layout[index] == "int32" else np.float32(value)
                values.append(bound)
            request.params = tuple(values)
            if prepared.device == "gpu":
                sources = tuple(request.params[index] for index in prepared.reads if isinstance(request.params[index], compute.Buffer))
                request.prepare_peak_bytes = owner._gpu_admission(prepared.entries, sources)
                request.host = prepared.entries[prepared.program, False].host
            pending = {index: value for index, value in enumerate(request.params) if isinstance(value, BufferSource)}
            if pending:
                from .input_binding import InputBindings
                request.input_bindings = InputBindings(owner, pending)
            request.metrics["snapshot_bytes"] = snapshot_bytes
            return request
        except BaseException:
            retire((value for value in values if isinstance(value, compute.Buffer)), flush=False)
            raise

    def freeze(self, owner):
        started = perf_counter_ns()
        prepared, params, snapshots = self.prepared, list(self.params), {}
        for index, value in enumerate(params):
            if not isinstance(value, compute.Buffer):
                continue
            if index in prepared.writes:
                replacement = compute.Buffer._empty(shape=value.shape, dtype=value.dtype, device=value.device)
                self.private.append(replacement)
            elif prepared.device == "cpu" and index in prepared.reads:
                key = value._array.ctypes.data, value.nbytes, value.dtype
                replacement = snapshots.get(key)
                if replacement is None:
                    replacement = compute._freeze_cpu_buffer(value, owner._frame)
                    snapshots[key] = replacement
                    self.private.append(replacement)
                replacement = replacement.view(readonly=True)
            elif prepared.device == "gpu" and index in prepared.reads:
                replacement = compute._freeze_gpu_buffer(value)
            else:
                continue
            self.params = (*self.params[:index], replacement, *self.params[index + 1:])
            value._close(flush=False)
        if self.input_bindings is None:
            self._configure_cursor(owner)
        self.metrics["snapshot_ns"] = perf_counter_ns() - started

    def _bind_inputs(self, values, owner):
        for index, value in values.items():
            replacement = (compute._freeze_gpu_buffer(value) if value.device == 'gpu' else
                           value.view(readonly=True))
            self.params = (*self.params[:index], replacement, *self.params[index+1:])
        if self.prepared.device == 'gpu':
            owner._gpu_admission(self.prepared.entries, tuple(self.params[index] for index in self.prepared.reads))
        else:
            self.prepared._validate(self.params)
        self._configure_cursor(owner)

    def _configure_cursor(self, owner):
        prepared = self.prepared
        operation = _Operation(prepared.program, self.params)
        if prepared.device == 'gpu':
            self.source_versions = tuple((self.params[index], compute._accepted_buffer_version(self.params[index]))
                                         for index in prepared.reads)
        else:
            self.cursor = _WorkCursor((operation,))
        entry = prepared.entries[prepared.program, False] if prepared.device == "gpu" else prepared.program
        self.service_signature = ("kernel", execution_key(entry, self.params))
        self.service_spans = operation_spans((("compute", operation),), prepared.entries,
                                             prepared.device, owner.budget.range_terms)

    def _bind_gpu_work(self, term_limit):
        params, outputs = list(self.params), []
        try:
            for index in self.prepared.writes:
                layout = params[index]
                value = compute._managed_empty(shape=layout.shape, dtype=layout.dtype, device=layout.device)
                outputs.append(value)
                params[index] = value
            params = tuple(params)
            self.prepared._validate(params)
            cursor = _WorkCursor((_Operation(self.prepared.program, params),))
            # Admit the initial carry with the outputs so a rejected reduction
            # holds no part of its working set against other requests.
            operation = cursor.slice(term_limit, managed=True)
        except BaseException:
            retire(outputs, flush=False)
            raise
        self.private.extend(outputs)
        self.params, self.cursor = params, cursor
        return operation

    def _resources(self):
        values = tuple(value for value in self.params if isinstance(value, compute.Buffer))
        carry = () if self.cursor is None else self.cursor.carry
        blocks = (() if self.prepared.device == "gpu" else
                  tuple(block for variant in program_family(self.prepared.program) for block in variant.native_blocks))
        return _RequestStorage((*values, *carry), (*self.private, *carry), self.reserved_bytes,
            prepare_bytes=self.prepare_peak_bytes if self.packet is None and not self._complete() else 0,
            native_blocks=blocks)

    def _service_host_key(self):
        return (*self.service_signature, "publish" if self._complete() else "compute")

    def _observed_host_key(self, key, progress):
        return _observed_numerical_host_key(self, key, progress)

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        return remaining_work_service(self, self.cursor, owner, opportunity)

    def _is_current(self, owner):
        return all(not value.closed for value in self.params if isinstance(value, compute.Buffer)) and all(
            compute._buffer_version_current(value, version) for value, version in self.source_versions)

    def _domain(self):
        return "cpu" if self._complete() else self.prepared.device

    def _work(self, owner, opportunity):
        if self._complete():
            return None, False
        if self.cursor is None:
            if not owner._available(background=True):
                return None, False
            operation, progressed = self._bind_gpu_work(owner.budget.range_terms), True
        else:
            operation, progressed = _slice_work(self.cursor, owner.budget.range_terms)
        program = (self.prepared.entries[operation.program, False] if self.prepared.device == "gpu"
                   else operation.program)
        return (program, operation.params, self.cursor.completed_end, operation.params[
            domain_parameter(operation.program)].element_count), progressed

    def _prepare_cpu(self, owner, work):
        return _prepare_cpu_work_ranges(self, self.cursor, owner)

    def _prepare_gpu(self, owner, work):
        return _prepare_gpu_work_ranges(self, self.cursor, self.prepared.entries, owner)

    def _consume_cpu(self):
        _confirm_work_ranges(self, self.cursor)

    def _consume_gpu(self):
        _confirm_work_ranges(self, self.cursor)

    def _complete(self):
        return self.cursor is not None and self.cursor.current() is None

    def _publish(self, owner, opportunity):
        result = {self.prepared.names[index]: self.params[index].view(readonly=True)
                  for index in self.prepared.writes}
        for value in result.values():
            value._publish_storage()
        self.result = result
        self.status = "published"
        self.metrics["terminal_opportunity"] = opportunity

    def _release_private(self):
        if self.cursor is not None:
            self.cursor.retire()
        retire((value for value in self.params if isinstance(value, compute.Buffer)), flush=False)
        retire(self.private, flush=False)
        self.private.clear()
        self.params = self.ranges = self.source_versions = self.service_spans = self.service_signature = ()
        self.cursor = self.prepared = None
        self.prepare_peak_bytes = 0
