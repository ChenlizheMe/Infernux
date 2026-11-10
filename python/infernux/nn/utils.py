"""Gradient transforms on the same WorkItems used by optimizer continuations."""
from infernux._work import _Operation, _ValueLayout
from contextlib import nullcontext
from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from . import _primitives as p
from ._plan import _bind_cursor, _execute_work, _grad_facts, _grads_current
from ._tensor import Tensor, Parameter, _empty, _check_options, _check_storage

_TILE = 256
_FLOAT_MAX = float(np.finfo(np.float32).max)


@dataclass
class _ClipWork:
    operations: list
    gradients: list
    norm: object


def _clip_work(facts, max_norm, empty, bind):
    """The numerical, service and capacity builders consume these same rows."""
    operations, norms = [], []
    active = [(parameter, gradient) for parameter, gradient in facts if gradient is not None]
    for _, gradient in active:
        value = gradient._data
        while True:
            count = value.element_count
            width = min(count, _TILE)
            output = empty(((count + width - 1) // width,))
            maxima = empty(output.shape)
            program = bind(p.maximum_tiles, (value, maxima), _COUNT=count, _WIDTH=width)
            operations.append(_Operation(program, (value, maxima)))
            program = bind(p.norm_tiles, (value, maxima, output), _COUNT=count, _WIDTH=width, _MAXIMUM=_FLOAT_MAX)
            operations.append(_Operation(program, (value, maxima, output)))
            value = output
            if value.element_count == 1:
                break
        norms.append(value)
    while len(norms) > 1:
        next_level = []
        for index in range(0, len(norms)-1, 2):
            output = empty(())
            arguments = norms[index], norms[index+1], output
            operations.append(_Operation(bind(p.norm_merge, arguments, _MAXIMUM=_FLOAT_MAX), arguments))
            next_level.append(output)
        if len(norms) % 2:
            next_level.append(norms[-1])
        norms = next_level
    norm = norms[0] if norms else empty(())
    if not active:
        operations.append(_Operation(bind(p.fill_values, (norm,), _SHIFT=0.), (norm,)))
    gradients = []
    for parameter, gradient in active:
        output = empty(parameter.shape)
        arguments = gradient._data, norm, output, np.float32(max_norm)
        operations.append(_Operation(bind(p.clip_gradient, arguments), arguments))
        gradients.append((parameter, output))
    return _ClipWork(operations, gradients, norm)


class _ClipPrograms:
    def __init__(self, device, source=None):
        self.device, self.source, self._programs = device, source, {}

    @property
    def programs(self):
        return tuple(self._programs.values())

    @staticmethod
    def _key(function, facts):
        return function, tuple(sorted(facts.items()))

    def bind(self, function, arguments, **facts):
        return self._programs[self._key(function, facts)]

    def prepare(self, parameters, *, native_entries=False):
        temporary = []
        def empty(shape):
            value = _empty(shape, self.device)
            temporary.append(value)
            return value
        def bind(function, arguments, **facts):
            key = self._key(function, facts)
            if key not in self._programs:
                self._programs[key] = p.prepare(function, arguments, own_deferred_samples=True,
                                               source=self.source, **facts)
            return self._programs[key]
        try:
            facts = tuple((parameter, SimpleNamespace(_data=parameter._data)) for parameter in parameters)
            _clip_work(facts, 1., empty, bind)
            # A later run can have no gradients, even when parameters exist.
            bind(p.fill_values, (empty(()),), _SHIFT=0.)
            if native_entries:
                for declaration in self.programs:
                    for program in p.program_family(declaration):
                        program._native_entry()
        finally:
            p.retire(temporary, flush=False)

    def storage_bytes(self, shapes, term_limit):
        total = 0
        def empty(shape):
            nonlocal total
            result = _ValueLayout(shape, self.device)
            total += result.nbytes
            return result
        facts = tuple((SimpleNamespace(shape=shape), SimpleNamespace(_data=_ValueLayout(shape, self.device)))
                      for shape in shapes)
        work = _clip_work(facts, 1., empty, self.bind)
        rows = max((domain[1] for operation in work.operations
                    if (domain := p.inner_domain(operation.program, operation.params)) is not None
                    and domain[0] > term_limit), default=0)
        return total + 8 * rows + (8 if rows and self.device == "cpu" else 0)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """Clip one float32 gradient domain by its global L2 norm, returning the old norm.

    The result stays on the gradient device. Explicit item()/cpu() is a readback.
    A compiled program currently uses this as an ordered side effect; it can
    return its loss, but cannot return or branch on the norm proxy.
    """
    if norm_type != 2.0 or error_if_nonfinite is not False or foreach not in (None, False):
        raise NotImplementedError("clip_grad_norm_ supports L2, error_if_nonfinite=False and foreach=None/False")
    if (isinstance(max_norm, (bool, np.bool_)) or not isinstance(max_norm, (int, float, np.floating))
            or not math.isfinite(max_norm) or not 0 <= max_norm <= _FLOAT_MAX):
        raise ValueError("max_norm must be a finite nonnegative float32 scalar")
    parameters = (parameters,) if isinstance(parameters, Tensor) else tuple(parameters)
    if any(not isinstance(parameter, Tensor) for parameter in parameters):
        raise TypeError("clip_grad_norm_ requires Tensors")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("clip_grad_norm_ requires distinct Tensors")
    from .program import _capture_program
    if (capture := _capture_program.get()) is not None:
        return capture.clip_grad_norm(parameters, float(max_norm))
    owners = {parameter._owner for parameter in parameters if isinstance(parameter, Parameter) and parameter._owner is not None}
    if len(owners) > 1:
        raise ValueError("clip_grad_norm_ operates on one model parameter domain")
    root = next(iter(owners), None)
    with nullcontext() if root is None else root._lock:
        facts = _grad_facts(parameters)
        values = tuple(gradient._data for _, gradient in facts if gradient is not None)
        device = values[0].device if values else (parameters[0].device if parameters else "cpu")
        _check_options(None, device)
        if values:
            _check_storage(values)
        programs = _ClipPrograms(device, None if root is None else root._program_source)
        programs.prepare(tuple(parameter for parameter, gradient in facts if gradient is not None))
        pending, gradients, result = [], [], None
        published = False
        def empty(shape):
            value = _empty(shape, device)
            pending.append(value)
            return value
        try:
            work = _clip_work(facts, max_norm, empty, programs.bind)
            gradients = [(parameter, Tensor._from_buffer(value, parameter.shape)) for parameter, value in work.gradients]
            result = Tensor._from_buffer(work.norm, ())
            cursor = _bind_cursor(work.operations, private=pending,
                                  retained=(work.norm, *(gradient._data for _, gradient in gradients)))
            _execute_work(cursor, device, programs.programs)
            if not _grads_current(facts):
                raise RuntimeError("gradient bindings changed before clipping publication")
            for parameter, gradient in gradients:
                parameter._grad = gradient
            published = True
            return result
        finally:
            p.retire(pending)
            if not published:
                p.retire((*(gradient._data for _, gradient in gradients), None if result is None else result._data))
