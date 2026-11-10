"""Functional operators in the engine's shared static value graph."""
import math

import numpy as np

from ._tensor import Tensor


def _activation(input, kind, dim=None):
    from ._plan import _Symbol
    from . import modules
    if isinstance(input, _Symbol):
        return input.capture.unary(kind, input, dim)
    return modules._FunctionalUnary(kind, dim)(input)


def sigmoid(input):
    return _activation(input, "sigmoid")


def tanh(input):
    return _activation(input, "tanh")


def sin(input):
    return _activation(input, "sin")


def cos(input):
    return _activation(input, "cos")


def square(input):
    return _activation(input, "square")


def exp(input):
    return _activation(input, "exp")


def log(input):
    return _activation(input, "log")


def sqrt(input):
    return _activation(input, "sqrt")


def elu(input, alpha=1.0, inplace=False):
    from ._plan import _Symbol
    from .modules import ELU
    if inplace:
        raise NotImplementedError("static activations produce a new value; inplace=True is unsupported")
    return input.capture.unary("elu", input, options=(alpha,)) if isinstance(input, _Symbol) else ELU(alpha)(input)


def clamp(input, min=None, max=None, *, out=None):
    from ._plan import _Symbol
    from .modules import _FunctionalUnary
    if out is not None or isinstance(min, (Tensor, _Symbol)) or isinstance(max, (Tensor, _Symbol)):
        raise NotImplementedError("static clamp currently uses scalar bounds and a new output value")
    return (input.capture.unary("clamp", input, options=(min, max)) if isinstance(input, _Symbol)
            else _FunctionalUnary("clamp", options=(min, max))(input))


def _extremum(input, other, kind, out):
    from ._plan import _Symbol
    from .modules import _FunctionalBinary
    if out is not None:
        raise NotImplementedError("static extrema produce a new output value")
    if not isinstance(input, (Tensor, _Symbol)) or not isinstance(other, (Tensor, _Symbol)):
        raise TypeError("minimum/maximum require two Tensor operands")
    if isinstance(input, _Symbol) != isinstance(other, _Symbol):
        raise TypeError("captured extrema require operands from the same graph")
    return input.capture.binary(kind, input, other) if isinstance(input, _Symbol) else _FunctionalBinary(kind)(input, other)


def minimum(input, other, *, out=None):
    return _extremum(input, other, "minimum", out)


def maximum(input, other, *, out=None):
    return _extremum(input, other, "maximum", out)


def relu(input, inplace=False):
    if inplace:
        raise NotImplementedError("static activations produce a new value; inplace=True is unsupported")
    return _activation(input, "relu")


def silu(input, inplace=False):
    if inplace:
        raise NotImplementedError("static activations produce a new value; inplace=True is unsupported")
    return _activation(input, "silu")


def _floating(input, dtype):
    from ._tensor import _check_float_options
    _check_float_options(dtype, input.device)
    if dtype is not None:
        input = input.float()
    if input.dtype != np.dtype(np.float32):
        raise TypeError("this operation requires float32; convert integer features with .float()")
    return input


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    return _activation(_floating(input, dtype), "softmax", dim)


def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return _activation(_floating(input, dtype), "log_softmax", dim)


def reshape(input, shape):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    return input.capture.reshape(input, shape) if isinstance(input, _Symbol) else _TensorOperation("reshape", shape)(input)


def transpose(input, dim0, dim1):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    return input.capture.transpose(input, dim0, dim1) if isinstance(input, _Symbol) else _TensorOperation("transpose", dim0, dim1)(input)


def permute(input, dims):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    rank = len(input.shape)
    dims = tuple(dims)
    if len(dims) != rank or any(type(d) is not int or not -rank <= d < rank for d in dims):
        raise ValueError("permute must specify every input axis")
    dims = tuple(d % rank for d in dims)
    if len(set(dims)) != rank:
        raise ValueError("permute axes must be unique")
    if not isinstance(input, _Symbol):
        return _TensorOperation("permute", dims)(input)
    value, order = input, list(range(rank))
    for axis, source in enumerate(dims):
        position = order.index(source)
        if axis != position:
            value = transpose(value, axis, position)
            order[axis], order[position] = order[position], order[axis]
    return reshape(value, value.shape) if value is input else value


def _reduce(input, dim, keepdim, dtype, average):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    input = _floating(input, dtype)
    return (input.capture.reduce(input, dim, keepdim, average) if isinstance(input, _Symbol)
            else _TensorOperation("mean" if average else "sum", dim, keepdim, dtype=dtype)(input))


def sum(input, dim=None, keepdim=False, *, dtype=None):
    return _reduce(input, dim, keepdim, dtype, False)


def mean(input, dim=None, keepdim=False, *, dtype=None):
    return _reduce(input, dim, keepdim, dtype, True)


def argmax(input, dim=None, keepdim=False, *, out=None):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    if out is not None:
        raise NotImplementedError("static argmax produces a new value; out is unsupported")
    return (input.capture.argmax(input, dim, keepdim) if isinstance(input, _Symbol)
            else _TensorOperation("argmax", dim, keepdim)(input))


def flatten(input, start_dim=0, end_dim=-1):
    rank = max(1, len(input.shape))
    if any(type(dim) is not int or not -rank <= dim < rank for dim in (start_dim, end_dim)):
        raise ValueError("flatten axis is outside the input rank")
    start, end = start_dim % rank, end_dim % rank
    if start > end:
        raise ValueError("flatten start_dim must not follow end_dim")
    shape = input.shape or (1,)
    return reshape(input, shape[:start] + (math.prod(shape[start:end + 1]),) + shape[end + 1:])


def squeeze(input, dim=None):
    rank = len(input.shape)
    axes = tuple(range(rank)) if dim is None else (dim,) if type(dim) is int else tuple(dim)
    if any(type(axis) is not int or not -max(1, rank) <= axis < max(1, rank) for axis in axes):
        raise IndexError("squeeze axis is outside the Tensor rank")
    axes = tuple(axis % max(1, rank) for axis in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("squeeze axes must be unique")
    return reshape(input, tuple(size for axis, size in enumerate(input.shape) if axis not in axes or size != 1))


def unsqueeze(input, dim):
    rank = len(input.shape) + 1
    if type(dim) is not int or not -rank <= dim < rank:
        raise IndexError("unsqueeze axis is outside the output rank")
    axis = dim % rank
    return reshape(input, input.shape[:axis] + (1,) + input.shape[axis:])


def _full_like(input, value, dtype, layout, device, requires_grad, memory_format):
    from ._plan import _Symbol
    from ._tensor import _check_options
    device = input.device if device is None else device
    dtype = input.dtype if dtype is None else np.dtype(dtype)
    _check_options(dtype, device)
    if layout is not None or memory_format is not None:
        raise NotImplementedError("static tensors use contiguous layout")
    if isinstance(input, _Symbol):
        if device != input.device or requires_grad:
            raise ValueError("captured constant values use their graph device and have no independent leaf gradient")
        return input.capture.cast(input.capture.constant(input, value), dtype)
    return Tensor(np.full(input.shape, value, dtype=dtype), dtype=dtype, device=device, requires_grad=requires_grad)


def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return _full_like(input, 1., dtype, layout, device, requires_grad, memory_format)


def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return _full_like(input, 0., dtype, layout, device, requires_grad, memory_format)


def cat(tensors, dim=0, *, out=None):
    from ._plan import _Symbol
    values = tuple(tensors)
    if out is not None:
        raise NotImplementedError("captured cat creates its output; out is unsupported")
    if not values or not isinstance(values[0], _Symbol):
        raise TypeError("cat is currently defined inside a Module capture")
    return values[0].capture.cat(values, dim)


def stack(tensors, dim=0, *, out=None):
    from ._plan import _Symbol
    values = tuple(tensors)
    if not values or any(not isinstance(v, _Symbol) or v.shape != values[0].shape for v in values):
        raise ValueError("stack requires same-shaped values inside a Module capture")
    rank = len(values[0].shape) + 1
    if type(dim) is not int or not -rank <= dim < rank:
        raise ValueError("stack axis is outside the output rank")
    axis = dim % rank
    return cat((reshape(v, v.shape[:axis] + (1,) + v.shape[axis:]) for v in values), axis, out=out)


def pixel_shuffle(input, upscale_factor):
    from ._plan import _Symbol
    from .modules import PixelShuffle
    return input.capture.pixel_shuffle(input, upscale_factor) if isinstance(input, _Symbol) else PixelShuffle(upscale_factor)(input)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
                recompute_scale_factor=None, antialias=False):
    from ._plan import _Symbol
    from .modules import _TensorOperation
    if len(input.shape) != 4 or mode not in ("nearest", "bilinear") or antialias:
        raise NotImplementedError("interpolate supports fixed NCHW nearest/bilinear without antialias")
    if (size is None) == (scale_factor is None):
        raise ValueError("interpolate requires exactly one of size or scale_factor")
    if mode == "nearest" and align_corners is not None:
        raise ValueError("align_corners applies to bilinear interpolation")
    if align_corners is not None and type(align_corners) is not bool:
        raise TypeError("align_corners must be bool")
    if scale_factor is not None:
        scales = (scale_factor, scale_factor) if isinstance(scale_factor, (int, float)) else tuple(scale_factor)
        if len(scales) != 2 or any(not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("scale_factor requires two positive finite scales")
        if recompute_scale_factor is not True and any(s != int(s) for s in scales):
            raise NotImplementedError("noninteger scales currently require recompute_scale_factor=True or explicit size")
        size = tuple(math.floor(s * n) for s, n in zip(scales, input.shape[2:]))
    else:
        if recompute_scale_factor:
            raise ValueError("recompute_scale_factor is invalid with an explicit size")
        size = (size, size) if type(size) is int else tuple(size)
    if len(size) != 2 or any(type(s) is not int or s <= 0 for s in size):
        raise ValueError("interpolate output sizes must be two positive integers")
    if isinstance(input, _Symbol):
        return input.capture.resample(input, size, mode, bool(align_corners))
    return _TensorOperation("interpolate", size=size, mode=mode, align_corners=align_corners)(input)


def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    from ._plan import _Symbol
    if mode not in ("nearest", "bilinear") or padding_mode not in ("zeros", "border"):
        raise NotImplementedError("grid_sample currently supports nearest/bilinear and zeros/border padding")
    if align_corners is not None and type(align_corners) is not bool:
        raise TypeError("align_corners must be bool")
    if not isinstance(input, _Symbol):
        raise TypeError("grid_sample currently composes image and coordinates inside a Module forward")
    return input.capture.grid_sample(input, grid, mode, padding_mode, bool(align_corners))


def _loss_value(input, target, objective):
    """Objective values do not require a backward tape or forward activations."""
    from contextlib import nullcontext
    from . import _primitives as p
    from ._grad import is_grad_enabled
    from ._tensor import _empty, _check_storage
    from infernux._work import _WorkCursor, _Operation
    from ._plan import _objective, _objective_values, _check_target_layout, _execute_work
    objective = _objective(objective, target)
    if not isinstance(input, Tensor) or not isinstance(target, Tensor) or input.device != target.device:
        raise ValueError("loss values require Tensors on the same device")
    _check_target_layout(target, objective, input.shape)
    if input.dtype.name != "float32":
        raise TypeError("loss values require float32 predictions")
    if is_grad_enabled() and (input.requires_grad or target.requires_grad):
        raise ValueError("differentiable loss requires a model result and a non-differentiable target")
    _check_storage((input._data, target._data))
    plan = input._value_plan
    values = []
    def empty(shape):
        value = _empty(shape, input.device)
        values.append(value)
        return value
    with plan.root._lock if plan is not None else nullcontext():
        try:
            output = empty(())
            if plan is not None:
                operations = plan._loss_operations(input._data, target, output, empty, objective=objective,
                                                   output_shape=input.shape)
            else:
                operations = tuple(_Operation(p.prepare(function, params, **facts), params)
                    for function, params, facts in _objective_values(objective, input.shape, input._data, target._data, output, empty))
            _execute_work(_WorkCursor(operations), input.device, tuple(op.program for op in operations))
            return Tensor._from_buffer(output, ())
        finally:
            p.retire(values)


def mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    if size_average is not None or reduce is not None or reduction != "mean":
        raise NotImplementedError("only reduction='mean' MSE is supported")
    from .program import _capture_program
    if (capture := _capture_program.get()) is not None:
        return capture.loss(input, target, "mse")
    from ._grad import is_grad_enabled
    if not isinstance(input, Tensor) or input._run is None or not is_grad_enabled():
        return _loss_value(input, target, "mse")
    run = input._run
    with run.plan.root._lock:
        return run.plan.loss(run, target, output_index=input._value_index)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction="mean", label_smoothing=0.0):
    """Scalar, unweighted CE with probability or int64 index targets.

    Index targets must be in [0, C) or equal ignore_index; device values are
    consumed directly, without a CPU validation/readback pass. Invalid labels
    produce NaN rather than indexing outside a buffer or truncating int64.
    """
    if (weight is not None or size_average is not None or reduce is not None
            or reduction not in ("mean", "sum") or label_smoothing != 0.):
        raise NotImplementedError("cross_entropy supports unweighted scalar mean/sum without label smoothing")
    if type(ignore_index) is not int or not -(1 << 63) <= ignore_index < (1 << 63):
        raise TypeError("ignore_index must be a signed int64 integer")
    from ._plan import _CrossEntropy
    from .program import _capture_program, _Value
    capture = _capture_program.get()
    sample = target.sample if isinstance(target, _Value) else target
    if not isinstance(sample, Tensor) or sample.dtype.name not in ("int64", "float32"):
        raise TypeError("cross_entropy targets require float32 probabilities or int64 class indices")
    objective = _CrossEntropy(sample.dtype.name == "int64", ignore_index, reduction)
    if not objective.indexed and ignore_index != -100:
        raise ValueError("ignore_index applies only to class-index targets")
    if capture is not None:
        return capture.loss(input, target, objective)
    from ._grad import is_grad_enabled
    if not isinstance(input, Tensor) or input._run is None or not is_grad_enabled():
        return _loss_value(input, target, objective)
    run = input._run
    with run.plan.root._lock:
        return run.plan.loss(run, target, objective, output_index=input._value_index)
