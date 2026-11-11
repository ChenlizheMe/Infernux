"""Logical values on the engine's existing typed CPU/GPU Buffer storage."""
from __future__ import annotations

import math

import numpy as np

from infernux import compute
from infernux._compiler.kernel_contract import ARRAY_KINDS


def _check_options(dtype, device, *, allow_automatic_recording=False):
    if dtype is not None and np.dtype(dtype).name not in ARRAY_KINDS:
        raise TypeError("infernux.nn supports float32, int32, uint32 and int64")
    if device not in (None, "cpu", "gpu"):
        raise ValueError("infernux.nn device must be 'cpu' or 'gpu'")
    if device == "gpu":
        compute.assert_runtime_owner_thread()
        if compute._explicit_recording_depth() or (not allow_automatic_recording and compute._recording_depth()):
            raise RuntimeError("synchronous offline nn cannot run inside compute.recording")
        if compute._web_cpu_compute_enabled():
            raise compute.ComputeCapabilityError("nn GPU execution requires native Vulkan, without a CPU fallback")


def _empty(shape, device, dtype=np.float32, *, managed=False):
    allocate = compute._managed_empty if managed else compute.Buffer._empty
    return allocate(shape=math.prod(shape), dtype=dtype, device=device)


def _check_float_options(dtype, device):
    _check_options(dtype, device)
    if dtype is not None and np.dtype(dtype) != np.dtype(np.float32):
        raise TypeError("this operation requires float32")


def _check_storage(values):
    """Validate a logical execution domain before allocating or recording work."""
    first = values[0]
    for value in values:
        value._require_open()
        if value.device != first.device:
            raise ValueError("nn storage must use the same device")
        if first.device == "gpu" and int(value._host.identity) != int(first._host.identity):
            raise ValueError("nn GPU storage must belong to the same compute host")


def _copy_storage(source, device, dtype=None):
    """Host transfers and native device copies use the existing engine FIFO."""
    _check_options(None, device)
    _check_options(None, source.device)
    source._require_open()
    if dtype is not None and np.dtype(dtype) != np.dtype(source.dtype):
        from . import _primitives as p
        moved = source if source.device == device else _copy_storage(source, device)
        result = _empty((source.element_count,), device, dtype)
        try:
            program = p.prepare(p.copy_values, (moved, result))
            p.run(program, moved, result)
            return result
        except BaseException:
            p.retire((result,))
            raise
        finally:
            if moved is not source:
                p.retire((moved,))
    return compute._copy_buffers((source,), device)[0]


def _copy_storage_batch(sources, device):
    sources = tuple(sources)
    _check_options(None, device)
    for source in sources:
        _check_options(None, source.device)
    return compute._copy_buffers(sources, device)


def _to_device_batch(values, device):
    """Move program operands together, preserving ordinary Tensor.to semantics."""
    values = tuple(values)
    _check_options(None, device)
    moving = []
    for index, value in enumerate(values):
        if value.device == device:
            continue
        if (value._run is not None or value._loss is not None) and value.dtype == np.dtype(np.float32):
            raise RuntimeError("detach a computed Tensor before moving it between devices")
        moving.append(index)
    storage = _copy_storage_batch((values[index]._data for index in moving), device)
    results = list(values)
    try:
        for index, buffer in zip(moving, storage, strict=True):
            value = values[index]
            results[index] = Tensor._from_buffer(buffer, value.shape, requires_grad=value.requires_grad)
        return tuple(results)
    except BaseException:
        for index in moving:
            if results[index] is not values[index]:
                results[index]._data._close(flush=False)
        results.clear()
        raise
    finally:
        for buffer in storage:
            buffer._close(flush=False)


class Tensor:
    def __init__(self, data, *, dtype=None, device=None, requires_grad=False):
        if isinstance(data, Tensor) and device is None:
            device = data.device
        device = "cpu" if device is None else device
        dtype = data.dtype if dtype is None and isinstance(data, Tensor) else np.dtype(np.float32 if dtype is None else dtype)
        _check_options(dtype, device)
        if type(requires_grad) is not bool:
            raise TypeError("requires_grad must be bool")
        if requires_grad and dtype != np.dtype(np.float32):
            raise TypeError("integer Tensors do not require gradients")
        if isinstance(data, Tensor):
            shape, storage = data.shape, _copy_storage(data._data, device, dtype)
        else:
            array = np.asarray(data, dtype=dtype)
            if not array.size:
                raise ValueError("infernux.nn does not support empty tensors")
            shape = array.shape
            storage = compute.Buffer(shape=array.size, dtype=dtype, device=device, data=array.reshape(-1))
        self._shape = tuple(shape)
        self._values = storage.view(readonly=True)
        storage.close()
        self.requires_grad = requires_grad
        self._grad = self._run = self._loss = None
        self._value_plan = None
        self._value_index = None
        self._shared_input = False

    @classmethod
    def _from_buffer(cls, values, shape, *, requires_grad=False, run=None, loss=None, value_plan=None,
                     value_index=None, shared_input=False):
        result = object.__new__(cls)
        result._values, result._shape = values.view(readonly=True), tuple(shape)
        result.requires_grad, result._run, result._loss = requires_grad, run, loss
        result._grad = None
        result._value_plan = value_plan
        result._value_index = value_index
        result._shared_input = shared_input
        return result

    @property
    def _data(self):
        return self._values

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._data.dtype)

    @property
    def device(self):
        return self._data.device

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if type(value) is not bool:
            raise TypeError("requires_grad must be bool")
        storage = getattr(self, "_values", None)
        if value and storage is not None and storage.dtype != "float32":
            raise TypeError("integer Tensors do not require gradients")
        owner = getattr(self, "_owner", None)
        if owner is not None and owner._plans and value != self._requires_grad:
            owner._dirty = True
        self._requires_grad = value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if value is not None and (self.dtype != np.dtype(np.float32) or not isinstance(value, Tensor)
                                  or value.dtype != self.dtype or value.shape != self.shape or value.device != self.device):
            raise ValueError("gradient must be None or a Tensor with the same shape and device")
        self._grad = value

    def numel(self):
        return math.prod(self.shape)

    def reshape(self, *shape):
        from .functional import reshape
        return reshape(self, shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)

    def __getitem__(self, key):
        from .modules import _TensorOperation
        return _TensorOperation("__getitem__", key)(self)

    def squeeze(self, dim=None):
        from .functional import squeeze
        return squeeze(self, dim)

    def unsqueeze(self, dim):
        from .functional import unsqueeze
        return unsqueeze(self, dim)

    def float(self):
        return self.to(dtype=np.float32)

    def int(self):
        return self.to(dtype=np.int32)

    def long(self):
        return self.to(dtype=np.int64)

    def argmax(self, dim=None, keepdim=False):
        from .functional import argmax
        return argmax(self, dim, keepdim)

    def flatten(self, start_dim=0, end_dim=-1):
        from .functional import flatten
        return flatten(self, start_dim, end_dim)

    def transpose(self, dim0, dim1):
        from .functional import transpose
        return transpose(self, dim0, dim1)

    def permute(self, *dims):
        from .functional import permute
        return permute(self, dims[0] if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else dims)

    def sum(self, dim=None, keepdim=False, *, dtype=None):
        from .functional import sum
        return sum(self, dim, keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None):
        from .functional import mean
        return mean(self, dim, keepdim, dtype=dtype)

    def square(self):
        from .functional import square
        return square(self)

    def tanh(self):
        from .functional import tanh
        return tanh(self)

    def sigmoid(self):
        from .functional import sigmoid
        return sigmoid(self)

    def clamp(self, min=None, max=None):
        from .functional import clamp
        return clamp(self, min, max)

    def minimum(self, other):
        from .functional import minimum
        return minimum(self, other)

    def maximum(self, other):
        from .functional import maximum
        return maximum(self, other)

    def sin(self):
        from .functional import sin
        return sin(self)

    def cos(self):
        from .functional import cos
        return cos(self)

    def exp(self):
        from .functional import exp
        return exp(self)

    def log(self):
        from .functional import log
        return log(self)

    def sqrt(self):
        from .functional import sqrt
        return sqrt(self)

    def as_buffer(self, *, shape: int | tuple[int, ...] | None = None):
        """Share read-only storage, flat by default or with the supplied full shape."""
        return self._data.view(count=self.numel(), readonly=True, shape=shape)

    def numpy(self):
        """Observe a CPU value; detach().cpu().numpy() explicitly reads GPU data."""
        if self.device != "cpu":
            raise RuntimeError("GPU Tensor.numpy requires an explicit detach().cpu() transfer")
        return self._data.numpy().reshape(self.shape)

    def item(self):
        if self.numel() != 1:
            raise ValueError("item requires exactly one element")
        return self.detach().cpu()._data.numpy(copy=False)[0].item()

    def detach(self):
        return Tensor._from_buffer(self._data, self.shape, value_plan=self._value_plan, value_index=self._value_index,
                                   shared_input=self._shared_input)

    def to(self, device=None, *, dtype=None):
        """Copy a leaf/detached value; cross-device autograd edges are not supported."""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        _check_options(dtype, device)
        if device == self.device and dtype == self.dtype:
            return self
        if (self._run is not None or self._loss is not None) and dtype == np.dtype(np.float32):
            raise RuntimeError("detach a computed Tensor before moving it between devices")
        storage = _copy_storage(self._data, device, dtype)
        result = Tensor._from_buffer(storage, self.shape, requires_grad=self.requires_grad and dtype == np.dtype(np.float32))
        storage.close()
        return result

    def cpu(self):
        return self.to("cpu")

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if retain_graph not in (None, False) or create_graph or inputs is not None:
            raise NotImplementedError("higher derivatives, retained graphs and selective inputs are unsupported")
        if not self.requires_grad or (self._loss is None and self._run is None):
            raise RuntimeError("backward requires a differentiable result from a static model")
        if self._loss is not None:
            if gradient is not None:
                raise NotImplementedError("a scalar loss currently uses its default unit seed")
            run, target, objective, output_index = self._loss
            run.backward(target, objective=objective, output_index=output_index)
            return
        if gradient is None:
            if self.numel() != 1:
                raise ValueError("a non-scalar output requires an explicit gradient Tensor")
            if self.shape == ():
                self._run.backward(objective="scalar", output_index=self._value_index)
                return
            gradient = Tensor(np.ones(self.shape, np.float32), device=self.device)
        if (not isinstance(gradient, Tensor) or gradient.shape != self.shape or gradient.device != self.device
                or gradient.dtype != self.dtype
                or gradient.requires_grad or gradient._run is not None or gradient._loss is not None):
            raise ValueError("gradient must be a detached Tensor with the same output shape, dtype and device")
        _check_storage((self._data, gradient._data))
        self._run.backward(seed=gradient._data.view(readonly=True), output_index=self._value_index)


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor) and data.dtype != np.dtype(np.float32):
            raise TypeError("model parameters use float32 storage")
        super().__init__(data, requires_grad=requires_grad)
        self._owner = None
        self._slot = None

    @classmethod
    def _from_buffer(cls, values, shape, *, requires_grad=True):
        result = super()._from_buffer(values, shape, requires_grad=requires_grad)
        result._owner = result._slot = None
        return result

    @property
    def _data(self):
        if self._owner is None:
            return self._values
        return self._owner._state[self._slot]


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def as_tensor(data, *, dtype=None, device=None):
    """Reuse contiguous CPU arrays or engine Buffer storage without copying.

    A device/dtype conversion performs the ordinary synchronous conversion.
    Managed submission captures its own stable input value at release.
    """
    if isinstance(data, Tensor):
        if (dtype is None or np.dtype(dtype) == data.dtype) and (device is None or device == data.device):
            return data
        return data.to(device=data.device if device is None else device,
                       dtype=data.dtype if dtype is None else dtype)
    storage = None
    shape = None
    if isinstance(data, compute.Buffer):
        data._require_open()
        shape = data.shape
        storage = data.view(readonly=True)
        if data._dtype.lanes != 1:
            shape = data._storage_shape
            storage._shape = storage._storage_shape = data._storage_shape
            storage._dtype = compute._buffer_dtype(data._dtype.scalar)
            if storage._array is not None:
                storage._array = storage._array.reshape(storage._shape)
            storage._description = None
    elif isinstance(data, np.ndarray):
        shape = data.shape
        chosen = data.dtype if dtype is None else np.dtype(dtype)
        _check_options(chosen, 'cpu' if device is None else device)
        array = np.asarray(data, dtype=chosen)
        if not array.flags.c_contiguous:
            raise ValueError('as_tensor requires contiguous storage in the static runtime')
        storage = object.__new__(compute.Buffer)
        storage._configure(array.shape or (1,), chosen, 'cpu')
        storage._array = array.reshape(storage._storage_shape).view()
        storage._readonly = not array.flags.writeable
        storage._ownership = 'borrowed'
    if storage is None:
        return Tensor(data, dtype=dtype, device=device)
    try:
        _check_options(storage._dtype.scalar, storage.device, allow_automatic_recording=True)
        value = Tensor._from_buffer(storage, shape, shared_input=storage.device == 'cpu')
        if (device is None or device == value.device) and (dtype is None or np.dtype(dtype) == value.dtype):
            return value
        return value.to(device=value.device if device is None else device,
                        dtype=value.dtype if dtype is None else dtype)
    finally:
        storage._close(flush=False)
