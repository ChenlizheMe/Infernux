"""One counter stream; numerical draws run through the common compute kernels."""
import math
import operator
import threading
import time

import numpy as np


def _signed(word):
    word &= 0xffffffff
    return word if word < 0x80000000 else word - 0x100000000


class _RandomState:
    def __init__(self):
        self._lock = threading.Lock()
        self._seed, self._counter = time.time_ns() & 0xffffffffffffffff, 0

    def manual_seed(self, seed):
        seed = operator.index(seed)
        if not -(1 << 63) <= seed < (1 << 64):
            raise ValueError("seed must fit a signed or unsigned 64-bit integer")
        with self._lock:
            self._seed, self._counter = seed & 0xffffffffffffffff, 0
        return self

    def reserve(self, count):
        return self.reserve_many((count,))[0]

    def reserve_many(self, counts):
        total = sum(counts)
        with self._lock:
            if any(count < 1 for count in counts) or self._counter + total >= 1 << 64:
                raise ValueError("random counter range exceeds its 64-bit domain")
            keys = []
            for count in counts:
                keys.append(tuple(_signed(word) for word in (
                    self._seed & 0xffffffff, self._seed >> 32,
                    self._counter & 0xffffffff, self._counter >> 32)))
                self._counter += count
        return tuple(keys)

    def get_state(self):
        from ._tensor import Tensor
        with self._lock:
            words = (self._seed & 0xffffffff, self._seed >> 32,
                     self._counter & 0xffffffff, self._counter >> 32)
        return Tensor(words, dtype=np.uint32)

    def set_state(self, state):
        from ._tensor import Tensor
        if not isinstance(state, Tensor) or state.device != "cpu" or state.dtype != np.dtype(np.uint32) or state.shape != (4,):
            raise TypeError("RNG state must be a CPU uint32 Tensor with four words")
        low, high, counter_low, counter_high = (int(word) for word in state.numpy())
        with self._lock:
            self._seed, self._counter = low | high << 32, counter_low | counter_high << 32


_default = _RandomState()
manual_seed = _default.manual_seed
get_rng_state = _default.get_state
set_rng_state = _default.set_state


def _shape(size):
    shape = tuple(operator.index(value) for value in size)
    if any(value <= 0 for value in shape):
        raise ValueError("random tensors require positive fixed dimensions")
    return shape


def _uniform(shape, lower, upper, device, *, parameter=False):
    from ._tensor import Tensor, Parameter, _empty
    from . import _primitives as p
    values = _empty(shape, device)
    try:
        params = (values, *_default.reserve(math.prod(shape)))
        program = p.prepare(p.random_uniform, params, _SCALE=float(upper-lower), _SHIFT=float(lower))
        with p.stage(device, (program,)):
            p.run(program, *params)
        result_type = Parameter if parameter else Tensor
        return result_type._from_buffer(values, shape)
    finally:
        p.retire((values,))


def rand(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False):
    """float32 [0,1) samples, sharing the model's per-invocation counter stream."""
    from ._tensor import _check_float_options
    from ._plan import _capture
    shape = _shape(size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size)
    capture = _capture.get()
    device = capture.device if capture is not None and device is None else (device or "cpu")
    _check_float_options(dtype, device)
    if out is not None or layout is not None or pin_memory:
        raise NotImplementedError("static random tensors use ordinary owned contiguous storage")
    if type(requires_grad) is not bool:
        raise TypeError("requires_grad must be bool")
    if capture is not None:
        if device != capture.device or requires_grad:
            raise ValueError("captured random values use the graph device and do not introduce gradient leaves")
        return capture.random_uniform(shape)
    result = _uniform(shape, 0., 1., device)
    result.requires_grad = requires_grad
    return result
