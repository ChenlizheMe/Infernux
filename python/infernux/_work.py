"""Shared typed programs and continuation cursors; no model or executor."""
from dataclasses import dataclass
import math
import types

import numpy as np
import infernux as inx
from infernux._compiler.dataflow import Access, Dataflow
from infernux.compute import Buffer, _managed_empty, _storage_blocks
from infernux.compute_source import BufferSource


def _empty(shape, device):
    return Buffer._empty(shape=math.prod(shape), dtype="float32", device=device)


def prepare(function, params, **facts):
    """Cold compile one shared row proof and its ordered reduction family."""
    bound = types.FunctionType(function.__code__, {**function.__globals__, **facts}, function.__name__,
                               function.__defaults__, function.__closure__)
    buffers = tuple(value for value in params if isinstance(value, inx.compute.Buffer))
    device = buffers[0].device
    if any(value.device != device for value in buffers):
        raise ValueError("numerical work buffers must use one device")
    if device == "cpu":
        from infernux.jit import _prepare_cpu_work_item, _compile_cpu_work_item, _compile_reduction_work_item
        prepared = _prepare_cpu_work_item(bound, params)
        item, _, _, proof, _ = prepared
    else:
        from infernux._compiler.work_item import normalize_work_item
        from infernux._compiler.slang import _work_item_proof
        item = normalize_work_item(bound, len(params), target="Numerical work")
        proof = _work_item_proof(item, params)
    proof.require()
    if not proof.outputs or set(proof.outputs) != set(proof.fully_written_outputs):
        raise TypeError("numerical work must fully write every output row before publication")
    names = tuple(argument.arg for argument in (*item.definition.args.posonlyargs, *item.definition.args.args))
    access = (tuple(names.index(name) for name in proof.inputs),
              tuple(names.index(name) for name in proof.outputs))
    def bind_access(program):
        program._work_names = names
        program._work_access = access
        if device == "gpu":
            program._work_layout = item.domain_parameter, proof.inner_range
            for variant in program._inner_variants.values():
                lowered = variant._lowered_work_item
                variant._work_layout = lowered.item.domain_parameter, lowered.continuation.inner_range
        return program
    region = proof.ordered_reduction
    sliced = region is not None and region.eligible and region.count > 1
    if device == "cpu":
        return bind_access(_compile_reduction_work_item(bound, params, _prepared=prepared, native_entries=False)["whole"] if sliced else
                           _compile_cpu_work_item(bound, *prepared))
    if not sliced:
        # The original declaration is required for compiler-generated local VJP.
        program = bind_access(inx.compute.Kernel(bound))
    else:
        programs = inx.compute._reduction_kernels(bound, params, _prepared=(item, proof))
        program = programs["whole"]
        program._inner_variants = {kind: variant for kind, variant in programs.items() if kind != "whole"}
        bind_access(program)
    return program


class Continuation:
    """One prepared signature, never mutable metadata on the author Kernel."""
    def __init__(self, declaration, params):
        if not isinstance(declaration, inx.compute.Kernel) or not isinstance(params, tuple):
            raise TypeError("prepare_continuation requires a Kernel and a params tuple")
        buffers = tuple(value for value in params if isinstance(value, Buffer))
        if not buffers or any(value.dtype not in ("float32", "int32", "uint32", "int64")
                              or len(value.shape) != 1 for value in buffers):
            raise TypeError("continuations require one-dimensional scalar typed Buffers")
        self.device = buffers[0].device
        self.layout = self._layout(params)
        self.program = prepare(declaration.function, params)
        self.names = self.program._work_names
        self.reads, self.writes = accesses(self.program)
        self.entries = {}
        if self.device == "cpu":
            for variant in program_family(self.program):
                variant._native_entry()
        else:
            _prepare_gpu_family(self.program, params)
            executable = self.program._executable(params)
            self.entries[self.program, False] = executable
            control = inner_range(self.program)
            samples = []
            try:
                if control is not None:
                    samples = [_empty((params[domain_parameter(self.program)].element_count,), "gpu") for _ in range(2)]
                for kind, variant in self.program._inner_variants.items():
                    bounds = (0, 1) if kind == "initial" else (1, 2) if kind == "resume" else (control.count - 1, control.count)
                    self.entries[variant, False] = variant._executable((*params, *samples, *bounds))
            finally:
                retire(samples, flush=False)
        self._validate(params)

    @staticmethod
    def _layout(params):
        return tuple((value.shape, value.dtype.name if isinstance(value, BufferSource) else value.dtype, value.device)
                     if isinstance(value, (Buffer, BufferSource)) else
                     "int32" if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) else
                     "float32" if isinstance(value, (float, np.floating)) else None for value in params)

    def _validate(self, params, *, allow_sources=False):
        if not isinstance(params, tuple) or self._layout(params) != self.layout:
            raise ValueError("continuation arguments differ from preparation; prepare their current layout")
        for value in params:
            if isinstance(value, Buffer):
                value._require_open()
        pending = tuple(index for index, value in enumerate(params) if isinstance(value, BufferSource))
        if pending:
            if not allow_sources or any(index not in self.reads or index in self.writes for index in pending):
                raise ValueError('deferred buffers are permitted only as submitted read-only kernel inputs')
            return
        if self.device == "cpu":
            self.program._bind(params)
        else:
            entry = self.entries[self.program, False]
            entry.artifact.range_bounds(params, begin=0,
                end=params[domain_parameter(self.program)].element_count, host=entry.host)

    @property
    def inputs(self):
        return tuple(self.names[index] for index in self.reads)

    @property
    def outputs(self):
        return tuple(self.names[index] for index in self.writes)

    def submit(self, params, *, finish_within_frames=None, finish_within_ms=None, budget=None):
        from infernux.application import Application
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError("continuation submission requires an active Engine")
        owner = engine._cpu_work(budget)
        request = owner.release_kernel(self, params, window_opportunities=finish_within_frames,
                                       finish_within_ms=finish_within_ms)
        return engine._wake_cpu_request(owner, request)


def _deferred_prepares():
    stack = getattr(inx.compute._deferred_prepare_state, "stack", None)
    return stack[-1] if stack else None


def _prepare_gpu(program, params, *, own_deferred_samples=False,
                 autodiff_mode="none", gradient_parameters=()):
    if not own_deferred_samples or _deferred_prepares() is None:
        inx.compute.prepare(program, params, autodiff_mode=autodiff_mode,
                            gradient_parameters=gradient_parameters)
        return
    # Frozen graph preparation owns its exact typed views; general
    # compute.prepare keeps the caller's arguments.
    samples, arguments = [], []
    sample = value = None
    try:
        for value in params:
            if isinstance(value, inx.compute.Buffer):
                sample = value.view(readonly=value.readonly)
                samples.append(sample)
                arguments.append(sample)
            else:
                arguments.append(value)
        inx.compute.prepare(program, tuple(arguments), autodiff_mode=autodiff_mode,
                            gradient_parameters=gradient_parameters)
    except BaseException:
        retire(samples, flush=False)
        raise
    finally:
        samples.clear()
        arguments.clear()
        sample = value = params = None


def _prepare_gpu_family(program, params, *, own_deferred_samples=False):
    """Source and built programs prepare through the same deferred sample owner."""
    region = inner_range(program)
    # A loaded provider can be shared by several models/checkpoints. Failure
    # withdraws this preparation, not pipelines previously prepared by another.
    previous = {variant: set(variant._executables) for variant in program_family(program)}
    pending = _deferred_prepares()
    start = 0 if pending is None else len(pending)
    samples = []
    arguments = None
    try:
        if region is not None:
            samples.extend(inx.compute.Buffer._empty(shape=params[domain_parameter(program)].element_count,
                                                    dtype="float32", device="gpu") for _ in range(2))
        _prepare_gpu(program, params, own_deferred_samples=own_deferred_samples)
        for kind, variant in program._inner_variants.items():
            bounds = (0, 1) if kind == "initial" else (1, 2) if kind == "resume" else (region.count - 1, region.count)
            arguments = (*params, *samples, *bounds)
            _prepare_gpu(variant, arguments, own_deferred_samples=True)
    except BaseException:
        if pending is not None:
            del pending[start:]
        for variant in program_family(program):
            for key in set(variant._executables) - previous[variant]:
                variant._executables.pop(key).close()
        raise
    finally:
        retire(samples, flush=False)
        samples.clear()
        arguments = None


def program_family(program):
    return (program, *program._inner_variants.values())


def inner_range(program):
    if not program._inner_variants:
        return None
    initial = program._inner_variants["initial"]
    return initial._work_layout[1] if isinstance(initial, inx.compute.Kernel) else initial.continuation_plan.inner_range


def domain_parameter(program):
    return program._work_layout[0] if isinstance(program, inx.compute.Kernel) else program.domain_parameter


def accesses(program, *, reverse=False):
    # Local VJP binds (primal x, primal y, dx, seed). The compiler's reverse
    # role accumulates dx and consumes seed; both are mutable dependencies.
    return ((0, 1, 2, 3), (2, 3)) if reverse else program._work_access


def inner_domain(program, params):
    control = inner_range(program)
    return None if control is None else (control.count, params[domain_parameter(program)].element_count)


def gpu_arguments(params, *, reverse=False):
    """One role mapping for synchronous and managed GPU operations."""
    if not reverse:
        return params, None
    x, y, dx, seed = params
    return (x, y), {0: dx, 1: seed}


def retire(values, *, flush=True):
    """Drop private binding-cache pins; accepted batches retain their own owners."""
    for value in values:
        if value is not None and not value.closed:
            if value.device == "gpu":
                stream = inx.compute._command_streams().get(int(value._host.identity))
                if flush and stream is not None and stream["failure"] is not None:
                    inx.compute._discard_buffer_launches(value)
                    continue  # Do not replay failed unaccepted commands from close().
            value._close(flush=flush)


@dataclass(frozen=True)
class _Operation:
    program: object
    params: tuple
    reverse: bool = False
    copy: bool = False


@dataclass(frozen=True, eq=False)
class _ValueLayout:
    """Argument layout for service analysis; owns no numerical storage."""
    shape: tuple
    device: str
    dtype: str = "float32"

    @property
    def element_count(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return self.element_count * np.dtype(self.dtype).itemsize


@dataclass(frozen=True)
class _WorkRange:
    index: int
    operation: _Operation
    begin: int
    end: int
    count: int


def _binding_storage_blocks():
    """Read allocation identity once per value while constructing one binding."""
    blocks = {}
    def storage(value):
        identity = id(value)
        if identity not in blocks:
            blocks[identity] = value, tuple(_storage_blocks((value,)))
        return blocks[identity][1]
    return storage


def _operation_access(operation, storage):
    reads, writes = ((0,), (1,)) if operation.copy else accesses(operation.program, reverse=operation.reverse)
    def keys(indices):
        return frozenset(key for index in indices if isinstance(operation.params[index], Buffer)
                         for key in storage(operation.params[index]))
    return Access(keys(reads), keys(writes))


class _WorkCursor:
    """Bound operations only; synchronous callers or the existing owner drive them."""
    def __init__(self, operations, *, retained=(), release=None, graph=None, storage=None):
        self.operations = tuple(operations)
        if graph is None:
            storage = _binding_storage_blocks() if storage is None else storage
            accesses = tuple(_operation_access(operation, storage) for operation in self.operations)
            writes = frozenset(key for access in accesses for key in access.writes)
            initialized = frozenset(key for access in accesses for key in access.reads) - writes
            graph = Dataflow(accesses, initialized=initialized)
        self.graph = graph
        self._flow = self.graph.cursor(retained=retained)
        self._release = release
        self.completed_end = 0
        self.carry = ()
        self.transfer = None
        self.inner_begin = self.inner_end = self.inner_count = 0

    def current(self):
        return None if self.index == len(self.operations) else self.operations[self.index]

    @property
    def index(self):
        return self._flow.index

    def scratch_reservation(self, term_limit):
        largest = 0
        for index in self.graph.order[self._flow.completed:]:
            operation = self.operations[index]
            if operation.copy:
                continue
            control = inner_range(operation.program)
            if control is not None and control.count > term_limit:
                domain = operation.params[domain_parameter(operation.program)]
                largest = max(largest, domain.element_count * 8 + (8 if domain.device == "cpu" else 0))
        return largest

    def complete(self):
        self.retire()
        released = self._flow.complete()
        if self._release is not None:
            self._release(released)
        self.completed_end = 0

    def complete_range(self, end, count):
        self.completed_end = end
        if end == count:
            if self.carry and self.inner_end < self.inner_count:
                self.inner_begin, self.inner_end = self.inner_end, 0
                self.carry = self.carry[::-1]
                self.completed_end = 0
            else:
                self.complete()

    def preview(self, first, domain, select, limit, term_limit):
        """Preview a fixed topological prefix, without completing or releasing it.

        A successor reads whole views, so its producer must finish its full row
        domain in this segment. Copies and carry changes are cuts. A whole
        VJP stays indivisible but may share an ordered GPU packet with peers.
        ``select`` returns successive row ends within the segment allowance.
        A packet that splits a row domain stays within that operation.
        """
        result = []
        for offset, index in enumerate(self.graph.order[self._flow.completed:]):
            if len(result) == limit:
                break
            operation = first if offset == 0 else self.operations[index]
            if operation.copy or (offset and (inner_range(operation.program) is not None
                                             and inner_range(operation.program).count > term_limit)):
                break
            count, begin = domain(operation), self.completed_end if offset == 0 else 0
            ends = select(operation, begin, count)
            if not ends:
                break
            for end in ends:
                result.append(_WorkRange(index, operation, begin, end, count))
                begin = end
            if len(ends) > 1 or end != count or (offset == 0 and self.carry):
                break
        return tuple(result)

    def confirm(self, ranges):
        """Confirm actions only after their exact worker/queue completion."""
        for work in ranges:
            assert self.index == work.index
            if work.operation.reverse:
                self.complete()
            else:
                self.complete_range(work.end, work.count)

    def slice(self, term_limit, *, managed=False):
        """Bind a legal inner stage; only completed rows may advance it."""
        operation = self.current()
        if operation.copy:
            return operation
        control = None if operation.reverse else inner_range(operation.program)
        if control is None or (not self.carry and (control.count <= term_limit or self.completed_end)):
            return operation
        if not self.carry:
            domain = operation.params[domain_parameter(operation.program)]
            values = []
            try:
                allocate = _managed_empty if managed else Buffer._empty
                values.extend(allocate(shape=domain.element_count, dtype="float32", device=domain.device)
                              for _ in range(2))
                self.carry = tuple(values)
            except BaseException:
                retire(values, flush=False)
                raise
            self.inner_count = control.count
        if self.inner_end == 0:
            self.inner_end = min(self.inner_count, self.inner_begin + term_limit)
        kind = "initial" if self.inner_begin == 0 else "final" if self.inner_end == self.inner_count else "resume"
        return _Operation(operation.program._inner_variants[kind],
                          (*operation.params, *self.carry, np.int32(self.inner_begin), np.int32(self.inner_end)))

    def retire(self):
        if self.transfer is not None:
            self.transfer.close()
            self.transfer = None
        retire(self.carry, flush=False)
        self.carry = ()
        self.inner_begin = self.inner_end = self.inner_count = 0
