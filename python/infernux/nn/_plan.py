"""Fixed-shape dataflow over real WorkItems; no private executor or autograd tree."""
from contextvars import ContextVar
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import hashlib
from types import SimpleNamespace

import numpy as np

from . import _primitives as p
from ._tensor import Tensor, _empty, _check_options, _check_storage
from ._grad import is_grad_enabled
from infernux.compute import Buffer, _storage_blocks, _automatic_cost
from infernux._work import _Operation, _ValueLayout, _WorkRange, _WorkCursor, _binding_storage_blocks
from infernux._compiler.dataflow import Access, Dataflow

_capture = ContextVar("infernux_nn_capture", default=None)


@dataclass(frozen=True)
class _UnaryOperation:
    forward: object
    backward: object
    local_ad: bool = True
    output_gradient: bool = False
    dimension: bool = False


_UNARY = {
    "sigmoid": _UnaryOperation(p.sigmoid, p.sigmoid_vjp),
    "relu": _UnaryOperation(p.relu, p.relu_vjp),
    "elu": _UnaryOperation(p.elu, p.elu_vjp, False),
    "clamp": _UnaryOperation(p.clamp, p.clamp_vjp, False),
    "tanh": _UnaryOperation(p.tanh, p.tanh_vjp),
    "silu": _UnaryOperation(p.silu, p.silu_vjp),
    "sin": _UnaryOperation(p.sine, p.sine_vjp, False),
    "cos": _UnaryOperation(p.cosine, p.cosine_vjp, False),
    "square": _UnaryOperation(p.square, p.square_vjp, False),
    "exp": _UnaryOperation(p.exponential, p.exponential_vjp, False, True),
    "log": _UnaryOperation(p.logarithm, p.logarithm_vjp, False),
    "sqrt": _UnaryOperation(p.square_root, p.square_root_vjp, False, True),
    "softmax": _UnaryOperation(p.softmax, p.softmax_vjp, False, True, True),
    "log_softmax": _UnaryOperation(p.log_softmax, p.log_softmax_vjp, False, True, True),
    "probability_log": _UnaryOperation(p.probability_log, p.probability_log_vjp, False),
}
_BINARY = {"subtract": 0, "multiply": 1, "divide": 2, "minimum": 3, "maximum": 4}
_VALUE_KINDS = {"affine", "reshape", "expand", "cat", "pad2d", "pixel_shuffle", "sum", "transpose",
                "slice", "resample", "detach", "constant", "cast", "argmax", "grid_sample",
                "categorical_select", "categorical_entropy", "categorical_sample", "random_uniform", *_BINARY}
_TYPED_VALUES = {"reshape", "expand", "cat", "transpose", "slice", "detach"}
_STOP_GRADIENT = {"detach", "constant", "cast", "argmax", "random_uniform", "categorical_sample"}
_PARAMETER_OPS = {"linear": (p.linear, p.pull_dx, p.pull_dw, p.pull_db),
                  "conv2d": (p.conv2d, p.conv2d_dx, p.conv2d_dw, p.conv2d_db)}


def _gradient_sides(layer):
    if layer.kind in _STOP_GRADIENT:
        return ()
    return (0,) if layer.kind == "categorical_select" else range(len(layer.inputs))


def _image_facts(layer):
    batch, ci, ih, iw = layer.input_shape
    _, co, oh, ow = layer.output_shape
    return dict(_B=batch, _CI=ci, _CO=co, _IH=ih, _IW=iw, _OH=oh, _OW=ow)


def _parameter_facts(layer, parameters):
    if layer.kind == "linear":
        return dict(_B=layer.input_shape[0], _K=layer.input_shape[1], _M=layer.output_shape[1])
    facts = _image_facts(layer)
    sh, sw, dh, dw, groups = layer.options
    _, cig, kh, kw = parameters[layer.weight].shape
    oh, ow = layer.output_shape[2:]
    return dict(**facts, _CIG=cig, _COG=layer.output_shape[1] // groups, _KH=kh, _KW=kw,
                _SH=sh, _SW=sw, _DH=dh, _DW=dw,
                _H_SHIFT=((kh * dh + oh * sh - 1) // (oh * sh)) * oh * sh,
                _W_SHIFT=((kw * dw + ow * sw - 1) // (ow * sw)) * ow * sw)


def _parameter_forward_args(layer, sample, state, output):
    weight = state[layer.weight]
    # The bias-free specialization skips this read and needs no zero upload.
    bias = weight if layer.bias is None else state[layer.bias]
    return sample, weight, bias, output


@dataclass(frozen=True)
class _CrossEntropy:
    indexed: bool = False
    ignore_index: int = -100
    reduction: str = "mean"


def _objective(kind, target=None):
    if kind == "cross_entropy":
        return _CrossEntropy(target is not None and target.dtype == np.dtype(np.int64))
    return kind


def _target_layout(kind, shape):
    if kind == "scalar":
        return None
    kind = _objective(kind)
    if isinstance(kind, _CrossEntropy) and kind.indexed:
        axis = 0 if len(shape) == 1 else 1
        return (*shape[:axis], *shape[axis + 1:]), "int64"
    return shape, "float32"


def _check_target_layout(target, kind, shape, *, allow_source=False):
    layout = _target_layout(kind, shape)
    if layout is None:
        if target is not None:
            raise ValueError("scalar training takes its loss from the model output, without a target")
        return
    from infernux.compute_source import BufferSource
    if (not isinstance(target, (Tensor, BufferSource) if allow_source else Tensor)
            or (target.shape, target.dtype.name) != layout):
        raise ValueError(f"loss requires a {layout[1]} target with shape {layout[0]}")


def _objective_facts(kind, shape):
    kind = _objective(kind)
    count = math.prod(shape)
    if kind == "scalar":
        if shape != ():
            raise ValueError("scalar training requires a rank-zero model output")
        return {}
    if kind == "mse":
        return dict(_COUNT=count, _INV_N=1. / count)
    if not isinstance(kind, _CrossEntropy) or not shape:
        raise ValueError("objective must be mse or cross_entropy with a class axis")
    axis = 0 if len(shape) == 1 else 1
    facts = dict(_COUNT=count, _WIDTH=shape[axis], _STRIDE=math.prod(shape[axis + 1:]),
                 _INV_N=shape[axis] / count if kind.reduction == "mean" else 1.)
    if kind.indexed:
        facts.update(_SAMPLES=count // shape[axis], _IGNORE_INDEX=kind.ignore_index,
                     _MEAN=int(kind.reduction == "mean"))
    return facts


def _objective_values(kind, shape, prediction, target, loss, empty):
    kind = _objective(kind)
    facts = _objective_facts(kind, shape)
    if kind == "scalar":
        return ((p.copy_values, (prediction, loss), {}),)
    if kind == "mse":
        return ((p.mse_mean, (prediction, target, loss), facts),)
    log_probability = empty(shape)
    log_softmax = (p.log_softmax, (prediction, log_probability), dict(_WIDTH=facts['_WIDTH'], _STRIDE=facts['_STRIDE']))
    if kind.indexed:
        rows = empty(_target_layout(kind, shape)[0])
        scale = empty(())
        return (log_softmax, (p.indexed_log_rows, (log_probability, target, rows), facts),
                _class_scale(kind, target, scale, facts),
                (p.scaled_class_loss, (rows, scale, loss), facts))
    return (log_softmax, (p.weighted_log_mean, (log_probability, target, loss), facts))


def _objective_gradients(kind, shape, prediction, target, gradient, empty):
    kind = _objective(kind)
    facts = _objective_facts(kind, shape)
    if kind == "scalar":
        return ((p.fill_values, (gradient,), dict(_SHIFT=1.)),)
    if isinstance(kind, _CrossEntropy) and kind.indexed:
        scale = empty(())
        return (_class_scale(kind, target, scale, facts),
                (p.indexed_cross_entropy_vjp, (prediction, target, scale, gradient), facts))
    return (((p.mse_vjp if kind == "mse" else p.cross_entropy_vjp),
             (prediction, target, gradient), facts),)


def _class_scale(kind, target, scale, facts):
    return ((p.class_normalization, (target, scale), facts) if kind.reduction == "mean" else
            (p.fill_values, (scale,), dict(_SHIFT=1.)))


def _unary_facts(layer):
    if layer.kind == "elu":
        return dict(_ALPHA=layer.options[0])
    if layer.kind == "clamp":
        lower, upper = layer.options
        return dict(_HAS_MINIMUM=int(lower is not None), _HAS_MAXIMUM=int(upper is not None),
                    _MINIMUM=0. if lower is None else lower, _MAXIMUM=0. if upper is None else upper)
    if not _UNARY[layer.kind].dimension:
        return {}
    axis, = layer.options
    return dict(_WIDTH=layer.input_shape[axis] if layer.input_shape else 1,
                _STRIDE=math.prod(layer.input_shape[axis + 1:]))


def _pointwise_options(kind, options):
    count = {"elu": 1, "clamp": 2}.get(kind, 0)
    if len(options) != count:
        raise ValueError("invalid pointwise operation configuration")
    if kind == "clamp" and all(value is None for value in options):
        raise ValueError("clamp requires at least one scalar bound")
    values = []
    for value in options:
        if value is None and kind == "clamp":
            values.append(None)
        elif isinstance(value, (int, float, np.number)) and math.isfinite(value) and abs(value) <= np.finfo(np.float32).max:
            values.append(float(value))
        else:
            raise ValueError("pointwise constants must be finite representable float32 scalars")
    return tuple(values)


def _value_forward(layer):
    kind = layer.kind
    if kind == "random_uniform":
        return p.random_uniform, dict(_SCALE=1., _SHIFT=0.)
    if kind == "categorical_sample":
        return p.categorical_sample, dict(_WIDTH=layer.input_shape[-1], _BATCH=math.prod(layer.input_shape[:-1]))
    if kind == "categorical_select":
        return p.categorical_select, dict(_WIDTH=layer.input_shape[-1])
    if kind == "categorical_entropy":
        return p.categorical_entropy, dict(_WIDTH=layer.input_shape[-1], _FROM_PROBS=int(layer.options[0]))
    if kind == "cast":
        return p.copy_values, {}
    if kind == "grid_sample":
        mode, padding, align = layer.options
        _, channels, height, width = layer.input_shape
        pad = int(padding == "zeros")
        scale_x, scale_y = (width - 2*pad - int(align)) / 2, (height - 2*pad - int(align)) / 2
        return p.grid_sample, dict(_CI=channels, _IH=height, _IW=width,
            _OH=layer.output_shape[2], _OW=layer.output_shape[3], _PAD=pad,
            _X_SCALE=scale_x, _Y_SCALE=scale_y, _X_SHIFT=scale_x + pad - (0. if align else .5),
            _Y_SHIFT=scale_y + pad - (0. if align else .5), _X_LIMIT=float(width - 1),
            _Y_LIMIT=float(height - 1), _INTERPOLATE=int(mode == "bilinear"))
    if kind in ("slice", "resample"):
        axis, *options = layer.options
        facts = dict(_WIDTH=layer.input_shape[axis], _COUNT=layer.output_shape[axis],
                     _STRIDE=math.prod(layer.input_shape[axis + 1:]))
        if kind == "slice":
            return p.slice_axis, dict(**facts, _START=options[0], _STEP=options[1])
        mode, align = options
        width, count = facts['_WIDTH'], facts['_COUNT']
        return p.resample_axis, dict(**facts, _NUMERATOR=2 * (width - 1 if align else width),
            _DENOMINATOR=2 * (max(1, count - 1) if align else count),
            _BIAS=width - 1 if align else count if mode == "bilinear" else width,
            _INTERPOLATE=int(mode == "bilinear"),
            _INV_DENOMINATOR=(1. / (2 * (max(1, count - 1) if align else count))) if mode == "bilinear" else 0.)
    if kind in ("sum", "argmax"):
        axis, _ = layer.options
        return (p.reduce_axis if kind == "sum" else p.argmax_axis), dict(
            _WIDTH=layer.input_shape[axis], _STRIDE=math.prod(layer.input_shape[axis + 1:]))
    if kind == "transpose":
        a, b = layer.options
        return p.transpose, dict(_AXIS_A=layer.input_shape[a], _AXIS_B=layer.input_shape[b],
                                 _BETWEEN=math.prod(layer.input_shape[a + 1:b]), _TAIL=math.prod(layer.input_shape[b + 1:]))
    if kind in ("pad2d", "pixel_shuffle"):
        facts = _image_facts(layer)
        if kind == "pad2d":
            return p.pad2d, dict(**facts, _LEFT=layer.options[0], _TOP=layer.options[2])
        return p.pixel_shuffle, dict(**facts, _FACTOR=layer.options[0])
    if kind in _BINARY:
        return p.binary, dict(_BINARY_OP=_BINARY[kind])
    if kind == "affine":
        return p.affine, dict(_SCALE=layer.options[0], _SHIFT=layer.options[1])
    if kind in ("reshape", "detach"):
        return p.copy_values, {}
    if kind == "constant":
        return p.fill_values, dict(_SHIFT=layer.options[0])
    axis, = layer.options
    width, stride = layer.output_shape[axis], math.prod(layer.output_shape[axis + 1:])
    if kind == "expand":
        return p.expand_axis, dict(_WIDTH=width, _STRIDE=stride)
    left = layer.input_shape[axis]
    return p.concatenate, dict(_WIDTH=width, _STRIDE=stride, _LEFT=left, _RIGHT=width - left)


def _value_pulls(layer, inputs, gradient, empty):
    """The same VJP bindings prepare programs, execute and describe demand."""
    if layer.kind == "grid_sample":
        _, facts = _value_forward(layer)
        yield p.grid_sample_dx, (inputs[1], gradient, empty(inputs[0].shape)), facts
        yield p.grid_sample_dgrid, (*inputs, gradient, empty(inputs[1].shape)), facts
        return
    for side in _gradient_sides(layer):
        before = inputs[side]
        output = empty(before.shape)
        if layer.kind == "categorical_select":
            yield p.categorical_select_vjp, (inputs[1], gradient, output), dict(_WIDTH=layer.input_shape[-1])
        elif layer.kind == "categorical_entropy":
            _, facts = _value_forward(layer)
            yield p.categorical_entropy_vjp, (*inputs, gradient, output), facts
        elif layer.kind in ("slice", "resample"):
            _, facts = _value_forward(layer)
            function = p.slice_axis_vjp if layer.kind == "slice" else p.resample_axis_vjp
            yield function, (gradient, output), facts
        elif layer.kind == "sum":
            _, facts = _value_forward(layer)
            yield p.expand_axis, (gradient, output), facts
        elif layer.kind == "transpose":
            _, facts = _value_forward(replace(layer, input_shape=layer.output_shape, output_shape=layer.input_shape))
            yield p.transpose, (gradient, output), facts
        elif layer.kind in ("pad2d", "pixel_shuffle"):
            _, facts = _value_forward(layer)
            function = p.pad2d_vjp if layer.kind == "pad2d" else p.pixel_shuffle_vjp
            yield function, (gradient, output), facts
        elif layer.kind in _BINARY:
            yield p.binary_vjp, (*inputs, gradient, output), dict(_BINARY_OP=_BINARY[layer.kind], _SIDE=side)
        elif layer.kind == "affine":
            yield p.affine_vjp, (gradient, output), dict(_SCALE=layer.options[0])
        elif layer.kind == "reshape":
            yield p.copy_values, (gradient, output), {}
        else:
            axis, = layer.options
            width, stride = layer.output_shape[axis], math.prod(layer.output_shape[axis + 1:])
            if layer.kind == "expand":
                yield p.expand_axis_vjp, (gradient, output), dict(_WIDTH=width, _STRIDE=stride)
            else:
                left = layer.input_shape[axis]
                yield p.concatenate_vjp, (gradient, output), dict(_WIDTH=width, _STRIDE=stride,
                    _PART=left if side == 0 else width - left, _OFFSET=0 if side == 0 else left)


class _Symbol:
    def __init__(self, shape, device, capture, index, requires_grad=False, dtype="float32"):
        self.shape, self.device, self.capture, self.index = shape, device, capture, index
        self.requires_grad = requires_grad
        self.dtype = np.dtype(dtype)

    def __add__(self, other):
        return self.capture.binary("add", self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return self.capture.binary("subtract", self, other)

    def __rsub__(self, other):
        return self.capture.affine(self, -1., float(other))

    def __mul__(self, other):
        return self.capture.binary("multiply", self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.capture.binary("divide", self, other)

    def __neg__(self):
        return self.capture.affine(self, -1., 0.)

    def __getitem__(self, key):
        return self.capture.slice(self, key)

    def detach(self):
        return self.capture.detach(self)

    def squeeze(self, dim=None):
        from .functional import squeeze
        return squeeze(self, dim)

    def unsqueeze(self, dim):
        from .functional import unsqueeze
        return unsqueeze(self, dim)

    def float(self):
        return self.capture.cast(self, "float32")

    def int(self):
        return self.capture.cast(self, "int32")

    def long(self):
        return self.capture.cast(self, "int64")

    def argmax(self, dim=None, keepdim=False):
        return self.capture.argmax(self, dim, keepdim)

    def to(self, device=None, *, dtype=None):
        if device not in (None, self.device):
            raise ValueError("one model graph executes on one device")
        return self.capture.cast(self, self.dtype if dtype is None else dtype)

    def interpolate(self, *args, **kwargs):
        from .functional import interpolate
        return interpolate(self, *args, **kwargs)

    def reshape(self, *shape):
        return self.capture.reshape(self, shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)

    def flatten(self, start_dim=0, end_dim=-1):
        from .functional import flatten
        return flatten(self, start_dim, end_dim)

    def transpose(self, dim0, dim1):
        return self.capture.transpose(self, dim0, dim1)

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
        return self.capture.unary("square", self)

    def tanh(self):
        return self.capture.unary("tanh", self)

    def sigmoid(self):
        return self.capture.unary("sigmoid", self)

    def clamp(self, min=None, max=None):
        from .functional import clamp
        return clamp(self, min, max)

    def minimum(self, other):
        return self.capture.binary("minimum", self, other)

    def maximum(self, other):
        return self.capture.binary("maximum", self, other)

    def sin(self):
        return self.capture.unary("sin", self)

    def cos(self):
        return self.capture.unary("cos", self)

    def exp(self):
        return self.capture.unary("exp", self)

    def log(self):
        return self.capture.unary("log", self)

    def sqrt(self):
        return self.capture.unary("sqrt", self)

    def _no_value(self, *args, **kwargs):
        raise TypeError("Tensor data cannot control a static Module capture")

    __bool__ = __float__ = __int__ = item = numpy = _no_value


@dataclass(frozen=True)
class _Layer:
    kind: str
    input_shape: tuple
    output_shape: tuple
    weight: int | None = None
    bias: int | None = None
    inputs: tuple = ()
    options: tuple = ()
    dtype: str = "float32"


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    shape: tuple
    requires_grad: bool


@dataclass(frozen=True)
class _InputSpec:
    shape: tuple
    requires_grad: bool
    dtype: str

    @classmethod
    def from_value(cls, value):
        return cls(value.shape, value.requires_grad, value.dtype.name)


def _result_values(result):
    return result if isinstance(result, tuple) else (result,)


def _map_result(result, convert):
    return tuple(convert(value) for value in result) if isinstance(result, tuple) else convert(result)


@dataclass(frozen=True)
class _Definition:
    """Portable fixed value graph; no code, model, buffer or execution state."""
    inputs: tuple
    training: bool
    parameters: tuple
    layers: tuple
    result: int | tuple[int, ...]

    @classmethod
    def from_module(cls, module, inputs, layers, result):
        root = module._bind()
        return cls(tuple(_InputSpec.from_value(x) for x in inputs), module.training,
                   tuple(_ParameterSpec(name, parameter.shape, parameter.requires_grad)
                         for name, parameter in root._named), tuple(layers), result)

    def __post_init__(self):
        def shape(value):
            if not isinstance(value, tuple) or any(type(size) is not int or size <= 0 for size in value):
                raise ValueError("fixed model shapes require positive integer dimensions")
        if not self.inputs or type(self.training) is not bool or not self.layers:
            raise ValueError("fixed model requires an explicit capture mode and a nonempty graph")
        for spec in self.inputs:
            shape(spec.shape)
            _check_options(spec.dtype, "cpu")
            if type(spec.requires_grad) is not bool or spec.requires_grad and spec.dtype != "float32":
                raise ValueError("model input gradient flags require floating storage")
        names = set()
        for spec in self.parameters:
            if type(spec.name) is not str or not spec.name or spec.name in names or type(spec.requires_grad) is not bool:
                raise ValueError("fixed model parameter names/gradient flags are invalid")
            shape(spec.shape)
            names.add(spec.name)
        values = [spec.shape for spec in self.inputs]
        dtypes = [spec.dtype for spec in self.inputs]
        for layer in self.layers:
            if layer.kind not in _PARAMETER_OPS and layer.kind != "add" and layer.kind not in _UNARY and layer.kind not in _VALUE_KINDS:
                raise ValueError("unknown fixed model operation: " + str(layer.kind))
            arity = 2 if layer.kind in ("add", "cat", "grid_sample", "categorical_select", "categorical_sample", *_BINARY) else 1
            if len(layer.inputs) != arity or any(type(index) is not int or not 0 <= index < len(values)
                                                for index in layer.inputs):
                raise ValueError("fixed model input must refer to an earlier value")
            before = values[layer.inputs[0]]
            _check_options(layer.dtype, "cpu")
            if layer.kind in _TYPED_VALUES:
                if any(dtypes[index] != layer.dtype for index in layer.inputs):
                    raise ValueError("shape and copy operations retain their input dtype")
            elif layer.kind in ("argmax", "categorical_sample"):
                if layer.dtype != "int64":
                    raise ValueError("index operations produce int64 values")
                if layer.kind == "categorical_sample" and any(dtypes[index] != "float32" for index in layer.inputs):
                    raise ValueError("categorical sampling requires float32 probabilities and random values")
            elif layer.kind == "constant":
                pass  # A fill has its own declared dtype; the anchor supplies no numerical input.
            elif layer.kind == "random_uniform":
                if layer.dtype != "float32":
                    raise ValueError("random values require float32 storage")
            elif layer.kind == "categorical_select":
                if (layer.dtype != "float32" or dtypes[layer.inputs[0]] != "float32"
                        or dtypes[layer.inputs[1]] not in ("int32", "uint32", "int64")):
                    raise ValueError("categorical log_prob requires floating logits and integer actions")
            elif layer.kind != "cast" and (layer.dtype != "float32" or
                                           any(dtypes[index] != "float32" for index in layer.inputs)):
                raise ValueError("floating model operations require float32 values")
            shape(layer.input_shape)
            shape(layer.output_shape)
            if layer.input_shape != before:
                raise ValueError("fixed model input shape differs from its producer")
            if layer.kind == "linear":
                if type(layer.weight) is not int or not 0 <= layer.weight < len(self.parameters):
                    raise ValueError("Linear requires a valid canonical weight slot")
                weight = self.parameters[layer.weight].shape
                if len(before) != 2 or len(weight) != 2 or before[1] != weight[1]:
                    raise ValueError("fixed Linear input/weight shapes differ")
                expected = before[0], weight[0]
                if layer.bias is not None and (type(layer.bias) is not int or not 0 <= layer.bias < len(self.parameters)
                                               or self.parameters[layer.bias].shape != (weight[0],)):
                    raise ValueError("fixed Linear bias shape/slot differs")
            elif layer.kind == "conv2d":
                if type(layer.weight) is not int or not 0 <= layer.weight < len(self.parameters):
                    raise ValueError("Conv2d requires a canonical weight slot")
                weight = self.parameters[layer.weight].shape
                if len(before) != 4 or len(weight) != 4 or len(layer.options) != 5:
                    raise ValueError("Conv2d requires NCHW input, OIHW weights and stride/dilation/groups")
                sh, sw, dh, dw, groups = layer.options
                if (any(type(v) is not int or v <= 0 for v in layer.options) or before[1] != weight[1] * groups
                        or weight[0] % groups):
                    raise ValueError("Conv2d channel groups and positive strides/dilations must match")
                oh, ow = (before[2] - (weight[2] - 1) * dh - 1) // sh + 1, (before[3] - (weight[3] - 1) * dw - 1) // sw + 1
                if min(oh, ow) <= 0:
                    raise ValueError("Conv2d kernel exceeds its input")
                expected = before[0], weight[0], oh, ow
                if layer.bias is not None and (type(layer.bias) is not int or not 0 <= layer.bias < len(self.parameters)
                                               or self.parameters[layer.bias].shape != (weight[0],)):
                    raise ValueError("Conv2d bias shape/slot differs")
            else:
                expected = before
                if layer.weight is not None or layer.bias is not None:
                    raise ValueError("only parameter operations use model parameter slots")
                if layer.kind in ("add", *_BINARY) and values[layer.inputs[1]] != before:
                    raise ValueError("fixed binary arithmetic requires expanded input shapes")
                if layer.kind in ("random_uniform", "constant"):
                    expected = layer.output_shape
                elif layer.kind == "categorical_sample":
                    expected = values[layer.inputs[1]]
                    if not before or len(expected) < len(before)-1 or expected[len(expected)-len(before)+1:] != before[:-1]:
                        raise ValueError("categorical samples must retain the parameter batch shape")
                elif layer.kind == "reshape":
                    if math.prod(before) != math.prod(layer.output_shape):
                        raise ValueError("fixed reshape must retain the element count")
                    expected = layer.output_shape
                elif layer.kind == "expand":
                    padded = (1,) * (len(layer.output_shape) - len(before)) + before
                    axis, = layer.options
                    if (len(padded) != len(layer.output_shape) or type(axis) is not int
                            or not 0 <= axis < len(padded) or padded[axis] != 1
                            or any(a != b for i, (a, b) in enumerate(zip(padded, layer.output_shape)) if i != axis)):
                        raise ValueError("fixed expand requires one singleton input axis")
                    expected = layer.output_shape
                elif layer.kind == "cat":
                    axis, = layer.options
                    right = values[layer.inputs[1]]
                    if (type(axis) is not int or not 0 <= axis < len(before) or len(right) != len(before)
                            or any(a != b for i, (a, b) in enumerate(zip(before, right)) if i != axis)):
                        raise ValueError("fixed cat requires matching non-concatenated axes")
                    expected = before[:axis] + (before[axis] + right[axis],) + before[axis + 1:]
                elif layer.kind == "pad2d":
                    if len(before) != 4 or len(layer.options) != 4 or any(type(v) is not int or v < 0 for v in layer.options):
                        raise ValueError("pad2d requires NCHW and four nonnegative padding sizes")
                    left, right, top, bottom = layer.options
                    expected = before[:2] + (before[2] + top + bottom, before[3] + left + right)
                elif layer.kind == "pixel_shuffle":
                    if len(before) != 4 or len(layer.options) != 1 or type(layer.options[0]) is not int or layer.options[0] <= 0:
                        raise ValueError("PixelShuffle requires NCHW and a positive factor")
                    factor, = layer.options
                    if before[1] % (factor * factor):
                        raise ValueError("PixelShuffle channels must divide the squared factor")
                    expected = before[0], before[1] // (factor * factor), before[2] * factor, before[3] * factor
                elif layer.kind == "grid_sample":
                    grid = values[layer.inputs[1]]
                    if (len(before) != 4 or len(grid) != 4 or grid[0] != before[0] or grid[-1] != 2
                            or len(layer.options) != 3 or layer.options[0] not in ("nearest", "bilinear")
                            or layer.options[1] not in ("zeros", "border") or type(layer.options[2]) is not bool):
                        raise ValueError("grid_sample requires NCHW, matching NHW2 grid and fixed sampling options")
                    if layer.options[1] == "zeros" and min(before[2:]) <= 2:
                        raise ValueError("zero sampling requires the declared one-pixel source padding")
                    expected = before[:2] + grid[1:3]
                elif layer.kind == "categorical_select":
                    if not before or values[layer.inputs[1]] != before[:-1]:
                        raise ValueError("categorical inputs must have matching batch and class dimensions")
                    expected = before[:-1]
                elif layer.kind == "categorical_entropy":
                    if not before or len(layer.options) != 1 or type(layer.options[0]) is not bool:
                        raise ValueError("categorical entropy requires a class axis and parameterization")
                    expected = before[:-1]
                elif layer.kind == "transpose":
                    if (len(layer.options) != 2 or any(type(v) is not int for v in layer.options)
                            or not 0 <= layer.options[0] < layer.options[1] < len(before)):
                        raise ValueError("transpose requires two ordered axes")
                    a, b = layer.options
                    result = list(before)
                    result[a], result[b] = result[b], result[a]
                    expected = tuple(result)
                elif layer.kind in ("sum", "argmax"):
                    if (len(layer.options) != 2 or type(layer.options[0]) is not int or type(layer.options[1]) is not bool
                            or not 0 <= layer.options[0] < len(before)):
                        raise ValueError("sum requires one axis and a boolean keepdim")
                    axis, keepdim = layer.options
                    expected = before[:axis] + ((1,) if keepdim else ()) + before[axis + 1:]
                elif layer.kind in ("slice", "resample"):
                    axis, *options = layer.options
                    if type(axis) is not int or not 0 <= axis < len(before) or len(layer.output_shape) != len(before):
                        raise ValueError("axis sampling requires a valid fixed rank and axis")
                    count = layer.output_shape[axis]
                    if layer.kind == "slice":
                        if (len(options) != 2 or any(type(v) is not int for v in options)
                                or options[0] < 0 or options[1] <= 0
                                or options[0] + (count - 1) * options[1] >= before[axis]):
                            raise ValueError("slice addresses must fit the input axis")
                    elif len(options) != 2 or options[0] not in ("nearest", "bilinear") or type(options[1]) is not bool:
                        raise ValueError("resample requires an explicit mode and alignment")
                    expected = before[:axis] + (count,) + before[axis + 1:]
            if layer.kind in _UNARY and _UNARY[layer.kind].dimension:
                if len(layer.options) != 1 or type(layer.options[0]) is not int or not 0 <= layer.options[0] < max(1, len(before)):
                    raise ValueError("fixed normalization requires a valid axis")
            elif layer.kind in _UNARY:
                _pointwise_options(layer.kind, layer.options)
            elif layer.kind in ("affine", "constant"):
                if len(layer.options) != (2 if layer.kind == "affine" else 1) or any(type(v) not in (float, int) or not math.isfinite(v) for v in layer.options):
                    raise ValueError("fixed affine constants must be finite numbers")
            elif layer.kind not in ("expand", "cat", "pad2d", "pixel_shuffle", "conv2d", "transpose", "sum", "argmax", "slice", "resample", "grid_sample", "categorical_entropy") and layer.options:
                raise ValueError("operation has no configuration options")
            if layer.output_shape != expected:
                raise ValueError("fixed model output shape differs from the operation")
            values.append(expected)
            dtypes.append(layer.dtype)
        returns = _result_values(self.result)
        if not returns or any(type(value) is not int or not len(self.inputs) <= value < len(values)
                              for value in returns):
            raise ValueError("fixed model results must name graph values")
        needed = set(returns)
        for index in reversed(range(len(self.layers))):
            if index + len(self.inputs) not in needed:
                raise ValueError("fixed model graph must omit dead operations")
            needed.update(self.layers[index].inputs)

    @property
    def used_parameters(self):
        return {slot for layer in self.layers if layer.kind in _PARAMETER_OPS
                for slot in (layer.weight, layer.bias) if slot is not None}

    def backward_values(self, result):
        needed = {result}
        for index in reversed(range(len(self.layers))):
            layer = self.layers[index]
            if index + len(self.inputs) in needed and layer.kind not in _STOP_GRADIENT:
                needed.update(layer.inputs[side] for side in _gradient_sides(layer))
        return needed


def _bind_cursor(operations, *, private=(), retained=()):
    """Release only private allocations after all their real consumers finish."""
    groups = {}
    storage = _binding_storage_blocks()
    for value in private:
        for key in storage(value):
            groups.setdefault(key, {})[id(value)] = value
    def release(keys):
        values = {}
        for key in keys:
            values.update(groups.pop(key, {}))
        p.retire(values.values(), flush=False)
    return _WorkCursor(operations, retained=(key for value in retained for key in storage(value)),
                       release=release, storage=storage)


def _execute_cpu_work(cursor):
    from infernux.cpu_execution import _CpuRangeBinding

    while cursor.current() is not None:
        if cursor.current().copy:
            source,target=cursor.current().params
            np.copyto(target._array,source._array,casting="no")
            cursor.complete()
            continue
        ranges=[]
        binding=None
        try:
            for index in cursor.graph.order[cursor._flow.completed:]:
                operation=cursor.operations[index]
                if operation.copy:
                    break
                program=operation.program
                ranges.append((program,operation.params,0,len(operation.params[program.domain_parameter])))
            binding=_CpuRangeBinding(ranges)
            status,_,durations=binding.execute()
            completed=len(ranges) if status==0 else len(durations)-1
            for _ in range(completed):
                cursor.complete()
            if status:
                raise RuntimeError(f"native CPU range failed with status {status}; output is invalid")
        finally:
            ranges.clear()
            binding=None


def _execute_work(cursor, device, programs, *, gpu_entries=None):
    """The existing synchronous boundary consumes the same bound operations."""
    if device=="cpu":
        _execute_cpu_work(cursor)
        return
    copies = []
    def execute(operation):
        if operation.copy:
            source, target = operation.params
            from infernux.compute import _copy_gpu_into
            copies.append(_copy_gpu_into(source, target))
        else:
            p.run(operation.program, *operation.params, reverse=operation.reverse,
                  executable=None if gpu_entries is None else gpu_entries[operation.program, operation.reverse])
    with p.stage(device, programs):
        # Offline GPU execution lowers a whole graph into the existing
        # ordered recording path. Recording does not complete consumers.
        for index in cursor.graph.order:
            execute(cursor.operations[index])
    for packet in copies:
        if not packet.completion.done:
            packet.completion.wait()
    # p.stage completed the kernels; copy-only graphs also waited above.
    while cursor.current() is not None:
        cursor.complete()


def _grad_facts(destinations):
    return tuple((destination, destination.grad) for destination in destinations)


def _grads_current(facts):
    return all(destination.grad is gradient for destination, gradient in facts)


class _Capture:
    def __init__(self, root, inputs):
        self.root = root
        self.device = inputs[0].device
        self.inputs = tuple(_Symbol(value.shape, self.device, self, index, value.requires_grad, value.dtype)
                            for index, value in enumerate(inputs))
        self.layers = []
        self.slots = {id(parameter): index for index, (_, parameter) in enumerate(root._named)}

    def _append(self, values, layer):
        if any(not isinstance(value, _Symbol) or value.capture is not self for value in values):
            raise ValueError("a captured operation requires values from its own graph")
        if layer.kind in _TYPED_VALUES:
            if any(value.dtype != values[0].dtype for value in values):
                raise TypeError("shape and copy operations require matching input dtypes")
            layer = replace(layer, dtype=values[0].dtype.name)
        elif layer.kind == "categorical_select":
            if values[0].dtype != np.dtype(np.float32) or values[1].dtype.name not in ("int32", "uint32", "int64"):
                raise TypeError("categorical log_prob requires float32 logits and integer actions")
        elif layer.kind not in ("cast", "argmax", "random_uniform", "constant") and any(value.dtype != np.dtype(np.float32) for value in values):
            raise TypeError("floating model operations require float32; convert integer features with .float()")
        layer = replace(layer, inputs=tuple(value.index for value in values))
        self.layers.append(layer)
        requires_grad = layer.kind not in _STOP_GRADIENT and (any(values[side].requires_grad for side in _gradient_sides(layer))
            or any(self.root._named[slot][1].requires_grad for slot in (layer.weight, layer.bias) if slot is not None))
        output = _Symbol(layer.output_shape, self.device, self, len(self.inputs) + len(self.layers) - 1,
                         requires_grad, layer.dtype)
        return self.detach(output) if requires_grad and not is_grad_enabled() else output

    def cast(self, value, dtype):
        dtype = np.dtype(dtype)
        _check_options(dtype, self.device)
        if value.dtype == dtype:
            return value
        return self._append((value,), _Layer("cast", value.shape, value.shape, dtype=dtype.name))

    def categorical_select(self, logits, second):
        shape = np.broadcast_shapes(logits.shape[:-1], second.shape)
        logits = self._expand(logits, shape + (logits.shape[-1],))
        second = self._expand(second, shape)
        return self._append((logits, second), _Layer("categorical_select", logits.shape, shape))

    def categorical_entropy(self, params, from_probs):
        return self._append((params,), _Layer("categorical_entropy", params.shape, params.shape[:-1], options=(from_probs,)))

    def random_uniform(self, shape):
        # This is a shape-domain anchor, like constant(), not a numerical read.
        anchor = self.inputs[0]
        return self._append((anchor,), _Layer("random_uniform", anchor.shape, shape))

    def categorical_sample(self, probs, sample_shape):
        shape = sample_shape + probs.shape[:-1]
        uniform = self.random_uniform(shape)
        return self._append((probs, uniform), _Layer("categorical_sample", probs.shape, shape, dtype="int64"))

    def argmax(self, value, dim, keepdim):
        if type(keepdim) is not bool:
            raise TypeError("argmax keepdim must be bool")
        if dim is None:
            result = self.argmax(self.reshape(value, (math.prod(value.shape),)), 0, False)
            return self.reshape(result, (1,) * len(value.shape) if keepdim else ())
        rank = max(1, len(value.shape))
        if type(dim) is not int or not -rank <= dim < rank:
            raise IndexError("argmax axis is outside the input rank")
        if not value.shape:
            return self.reshape(self.argmax(self.reshape(value, (1,)), 0, False), ())
        axis = dim % rank
        shape = value.shape[:axis] + ((1,) if keepdim else ()) + value.shape[axis + 1:]
        return self._append((value,), _Layer("argmax", value.shape, shape, options=(axis, keepdim), dtype="int64"))

    def detach(self, value):
        return self._append((value,), _Layer("detach", value.shape, value.shape))

    def constant(self, value, fill, *, shape=None, dtype=None):
        shape = value.shape if shape is None else tuple(shape)
        dtype = value.dtype if dtype is None else np.dtype(dtype)
        return self._append((value,), _Layer("constant", value.shape, shape, options=(float(fill),), dtype=dtype.name))

    def linear(self, module, x):
        if not isinstance(x, _Symbol) or len(x.shape) != 2 or x.shape[1] != module.in_features:
            raise ValueError("Linear input must have the exact fixed (batch, in_features) shape")
        if module.weight.shape != (module.out_features, module.in_features):
            raise ValueError("Linear weight must have shape (out_features, in_features)")
        if module.bias is not None and module.bias.shape != (module.out_features,):
            raise ValueError("Linear bias must have shape (out_features,)")
        slots = []
        for parameter in (module.weight, module.bias):
            if parameter is None:
                slots.append(None)
                continue
            slot = self.slots.get(id(parameter))
            if slot is None:
                raise ValueError("a Parameter must belong to the captured model")
            slots.append(slot)
        return self._append((x,), _Layer("linear", x.shape, (x.shape[0], module.out_features), *slots))

    def conv2d(self, module, x):
        if not isinstance(x, _Symbol) or len(x.shape) != 4 or x.shape[1] != module.in_channels:
            raise ValueError("Conv2d requires NCHW with matching input channels")
        if module.padding == "same":
            if module.stride != (1, 1):
                raise ValueError("padding='same' requires unit stride")
            ph, pw = ((k - 1) * d for k, d in zip(module.kernel_size, module.dilation))
            padding = pw // 2, pw - pw // 2, ph // 2, ph - ph // 2
        elif module.padding == "valid":
            padding = (0, 0, 0, 0)
        else:
            ph, pw = module.padding
            padding = pw, pw, ph, ph
        if any(padding):
            left, right, top, bottom = padding
            shape = x.shape[:2] + (x.shape[2] + top + bottom, x.shape[3] + left + right)
            x = self._append((x,), _Layer("pad2d", x.shape, shape, options=padding))
        kh, kw = module.kernel_size
        sh, sw = module.stride
        dh, dw = module.dilation
        expected_weight = module.out_channels, module.in_channels // module.groups, kh, kw
        if module.weight.shape != expected_weight or (module.bias is not None and module.bias.shape != (module.out_channels,)):
            raise ValueError("Conv2d parameters differ from the declared channels/kernel")
        shape = x.shape[0], module.out_channels, (x.shape[2] - (kh - 1) * dh - 1) // sh + 1, (x.shape[3] - (kw - 1) * dw - 1) // sw + 1
        return self._append((x,), _Layer("conv2d", x.shape, shape, self.slots[id(module.weight)],
            None if module.bias is None else self.slots[id(module.bias)], options=(sh, sw, dh, dw, module.groups)))

    def pixel_shuffle(self, x, factor):
        if not isinstance(x, _Symbol) or len(x.shape) != 4 or type(factor) is not int or factor <= 0 or x.shape[1] % (factor * factor):
            raise ValueError("PixelShuffle requires NCHW and channels divisible by a positive squared factor")
        shape = x.shape[0], x.shape[1] // (factor * factor), x.shape[2] * factor, x.shape[3] * factor
        return self._append((x,), _Layer("pixel_shuffle", x.shape, shape, options=(factor,)))

    def grid_sample(self, image, grid, mode, padding, align):
        if (not isinstance(image, _Symbol) or not isinstance(grid, _Symbol) or image.capture is not self or grid.capture is not self
                or len(image.shape) != 4 or len(grid.shape) != 4 or image.shape[0] != grid.shape[0] or grid.shape[-1] != 2):
            raise ValueError("grid_sample requires NCHW image and NHW2 coordinates from the same graph")
        if padding == "zeros":
            shape = image.shape[:2] + (image.shape[2] + 2, image.shape[3] + 2)
            image = self._append((image,), _Layer("pad2d", image.shape, shape, options=(1, 1, 1, 1)))
        shape = image.shape[:2] + grid.shape[1:3]
        return self._append((image, grid), _Layer("grid_sample", image.shape, shape, options=(mode, padding, align)))

    def unary(self, kind, x, dim=None, *, options=()):
        if not isinstance(x, _Symbol):
            raise TypeError("activation requires the captured Tensor")
        if _UNARY[kind].dimension:
            rank = max(1, len(x.shape))
            dim = (0 if rank in (0, 1, 3) else 1) if dim is None else dim
            if type(dim) is not int or not -rank <= dim < rank:
                raise ValueError("normalization axis is outside the input rank")
            options = (dim % rank,)
        else:
            options = _pointwise_options(kind, options)
        return self._append((x,), _Layer(kind, x.shape, x.shape, options=options))

    def sigmoid(self, x):
        return self.unary("sigmoid", x)

    def affine(self, value, scale, shift):
        return self._append((value,), _Layer("affine", value.shape, value.shape, options=(scale, shift)))

    def reshape(self, value, shape):
        shape = tuple(shape)
        if any(type(v) is not int or v == 0 or v < -1 for v in shape) or shape.count(-1) > 1:
            raise ValueError("reshape requires positive sizes and at most one inferred -1")
        count = math.prod(value.shape)
        if -1 in shape:
            known = -math.prod(shape)
            if count % known:
                raise ValueError("reshape sizes must divide the input element count")
            shape = tuple(count // known if v == -1 else v for v in shape)
        if math.prod(shape) != count:
            raise ValueError("reshape must retain the element count")
        return self._append((value,), _Layer("reshape", value.shape, shape))

    def transpose(self, value, dim0, dim1):
        rank = max(1, len(value.shape))
        if any(type(dim) is not int or not -rank <= dim < rank for dim in (dim0, dim1)):
            raise ValueError("transpose axis is outside the input rank")
        a, b = sorted((dim0 % rank, dim1 % rank))
        if a == b:
            return self.reshape(value, value.shape)
        shape = list(value.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return self._append((value,), _Layer("transpose", value.shape, tuple(shape), options=(a, b)))

    def slice(self, value, key):
        keys = key if isinstance(key, tuple) else (key,)
        consumed = sum(item is not None and item is not Ellipsis for item in keys)
        if sum(item is Ellipsis for item in keys) > 1 or consumed > len(value.shape):
            raise IndexError("too many indices for the Tensor")
        missing = len(value.shape) - consumed
        if any(item is Ellipsis for item in keys):
            keys = tuple(part for item in keys for part in ((slice(None),) * missing if item is Ellipsis else (item,)))
        else:
            keys += (slice(None),) * missing
        axis, shape = 0, []
        for item in keys:
            if item is None:
                shape.append(1)
                continue
            size = value.shape[axis]
            if type(item) is int:
                start = item + size if item < 0 else item
                if not 0 <= start < size:
                    raise IndexError("Tensor index is outside its axis")
                count, step = 1, 1
            elif isinstance(item, slice):
                start, stop, step = item.indices(size)
                if step <= 0:
                    raise ValueError("Tensor slices require a positive step")
                count = len(range(start, stop, step))
                if not count:
                    raise ValueError("empty Tensor slices are unsupported")
                shape.append(count)
            else:
                raise TypeError("static Tensor indices must be integers, slices, None or ellipsis")
            if start != 0 or count != size or step != 1:
                sampled = value.shape[:axis] + (count,) + value.shape[axis + 1:]
                value = self._append((value,), _Layer("slice", value.shape, sampled, options=(axis, start, step)))
            axis += 1
        return self.reshape(value, tuple(shape))

    def resample(self, value, size, mode, align):
        for axis, count in zip((2, 3), size):
            if count != value.shape[axis]:
                shape = value.shape[:axis] + (count,) + value.shape[axis + 1:]
                value = self._append((value,), _Layer("resample", value.shape, shape, options=(axis, mode, align)))
        return self.reshape(value, value.shape)

    def reduce(self, value, dim, keepdim, average):
        rank = len(value.shape)
        axes = tuple(range(rank)) if dim is None else ((dim,) if type(dim) is int else tuple(dim))
        if not axes:
            axes = tuple(range(rank))  # Torch's empty dimension list also reduces all axes.
        if type(keepdim) is not bool or any(type(v) is not int or not -max(1, rank) <= v < max(1, rank) for v in axes):
            raise ValueError("sum/mean requires valid axes and a boolean keepdim")
        axes = tuple(v % max(1, rank) for v in axes)
        if len(set(axes)) != len(axes):
            raise ValueError("reduction axes must be unique")
        if rank == 0 or not axes:
            return self.reshape(value, value.shape)
        count = math.prod(value.shape[axis] for axis in axes)
        for axis in sorted(axes, reverse=True):
            shape = value.shape[:axis] + ((1,) if keepdim else ()) + value.shape[axis + 1:]
            value = self._append((value,), _Layer("sum", value.shape, shape, options=(axis, keepdim)))
        return self.affine(value, 1. / count, 0.) if average else value

    def _expand(self, value, shape):
        padded = (1,) * (len(shape) - len(value.shape)) + value.shape
        for axis, size in enumerate(shape):
            if padded[axis] != size:
                expanded = padded[:axis] + (size,) + padded[axis + 1:]
                value = self._append((value,), _Layer("expand", value.shape, expanded, options=(axis,)))
                padded = expanded
        return self.reshape(value, shape) if value.shape != shape else value

    def binary(self, kind, left, right):
        if isinstance(right, (int, float, np.number)):
            if kind in ("minimum", "maximum"):
                raise TypeError("minimum/maximum require two Tensor operands")
            value = float(right)
            scale, shift = {"add": (1., value), "subtract": (1., -value),
                            "multiply": (value, 0.)}.get(kind, (None, None))
            if kind == "divide":
                scale, shift = 1. / value, 0.
            return self.affine(left, scale, shift)
        if not isinstance(right, _Symbol) or right.capture is not self:
            raise ValueError("captured arithmetic requires values from one graph or a scalar constant")
        shape = np.broadcast_shapes(left.shape, right.shape)
        left, right = self._expand(left, shape), self._expand(right, shape)
        return self._append((left, right), _Layer(kind, shape, shape))

    def add(self, left, right):
        return self.binary("add", left, right)

    def cat(self, values, dim):
        values = tuple(values)
        if not values or any(not isinstance(v, _Symbol) or v.capture is not self for v in values):
            raise ValueError("cat requires nonempty values from one captured graph")
        rank = len(values[0].shape)
        if type(dim) is not int or not -rank <= dim < rank:
            raise ValueError("cat axis is outside the input rank")
        axis, left = dim % rank, values[0]
        if len(values) == 1:
            return self.reshape(left, left.shape)
        for right in values[1:]:
            if len(right.shape) != rank or any(a != b for i, (a, b) in enumerate(zip(left.shape, right.shape)) if i != axis):
                raise ValueError("cat requires matching sizes outside the concatenation axis")
            shape = left.shape[:axis] + (left.shape[axis] + right.shape[axis],) + left.shape[axis + 1:]
            left = self._append((left, right), _Layer("cat", left.shape, shape, options=(axis,)))
        return left

    def finish(self, output):
        count = len(self.inputs)
        input_results = {}

        def graph_value(value):
            if not isinstance(value, _Symbol) or value.capture is not self:
                raise ValueError("forward must return graph values from this model")
            if value.index < count:
                if value.index not in input_results:
                    input_results[value.index] = self.reshape(value, value.shape).index
                return input_results[value.index]
            return value.index

        result = _map_result(output, graph_value)
        if not _result_values(result):
            raise ValueError("forward must return at least one graph value")
        needed = set(_result_values(result))
        for index in reversed(range(len(self.layers))):
            if index + count in needed:
                needed.update(self.layers[index].inputs)
        remap, layers = {index: index for index in range(count)}, []
        for index, layer in enumerate(self.layers):
            if index + count in needed:
                layers.append(replace(layer, inputs=tuple(remap[value] for value in layer.inputs)))
                remap[index + count] = count + len(layers) - 1
        return tuple(layers), _map_result(result, remap.__getitem__)


@dataclass(frozen=True)
class _ResultSpec:
    index: int
    shape: tuple
    dtype: str
    backward_values: frozenset
    grad_parameters: tuple
    requires_grad: bool

    @property
    def nbytes(self):
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


class Plan:
    @classmethod
    def capture(cls, entry, inputs, *, binding_precompiled=False):
        root = entry._bind()
        _check_options(None, inputs[0].device, allow_automatic_recording=binding_precompiled)
        _check_storage((*(x._data for x in inputs), *root._state))
        if any(parameter.device != inputs[0].device for _, parameter in root._named):
            raise ValueError("model parameters and input must use the same device")
        return cls(entry, inputs, entry._capture_definition(inputs))

    def __init__(self, entry, inputs, definition):
        root = entry._bind()
        self.entry, self.key = entry, entry._plan_key(inputs)
        self.root, self.inputs = root, definition.inputs
        self.input_count = len(inputs)
        self.input_grad = any(spec.requires_grad for spec in self.inputs)
        self.device = inputs[0].device
        self.definition = definition
        # Portable structure is cold metadata. Online persistence only encodes
        # current SGD/gradient facts, rather than traversing this graph again.
        self.definition_bytes = json.dumps(asdict(definition), ensure_ascii=False, allow_nan=False,
                                          separators=(",", ":")).encode("utf-8")
        self.service_identity = hashlib.sha256(self.definition_bytes).hexdigest()
        self.layers = layers = definition.layers
        # Forward values are SSA slots: inputs/parameters are read-only and
        # every activation has its own allocation. Compile dependency/lifetime
        # structure once; binding another invocation need not rediscover it
        # from physical addresses.
        self.forward_graph = Dataflow(tuple(Access(frozenset(layer.inputs), frozenset((index + self.input_count,)))
                                            for index, layer in enumerate(layers)),
                                      initialized=range(self.input_count))
        self.result = definition.result
        self.random_layers = tuple((index, math.prod(layer.output_shape)) for index, layer in enumerate(layers)
                                   if layer.kind == "random_uniform")
        self.activation_bytes = sum(math.prod(layer.output_shape) * np.dtype(layer.dtype).itemsize for layer in layers)
        self.result_specs = {}
        for result in dict.fromkeys(_result_values(self.result)):
            layer = layers[result - self.input_count]
            backward = frozenset(definition.backward_values(result))
            parameters = {slot for index, op in enumerate(layers)
                          if index + self.input_count in backward and op.kind in _PARAMETER_OPS
                          for slot in (op.weight, op.bias) if slot is not None}
            grads = tuple(index in parameters and parameter.requires_grad
                          for index, (_, parameter) in enumerate(root._named))
            requires_grad = layer.dtype == "float32" and (
                any(spec.requires_grad and index in backward for index, spec in enumerate(self.inputs)) or any(grads))
            self.result_specs[result] = _ResultSpec(result, layer.output_shape, layer.dtype, backward, grads, requires_grad)
        self.backward_values = frozenset().union(*(spec.backward_values for spec in self.result_specs.values()
                                                   if spec.requires_grad))
        self.programs, self.operations = (), []
        self._inner_domains = {}
        outputs = []
        values = [x._data for x in inputs]
        try:
            for layer in layers:
                sample = values[layer.inputs[0]]
                output = _empty(layer.output_shape, self.device, layer.dtype)
                outputs.append(output)
                if layer.kind in _PARAMETER_OPS:
                    facts = _parameter_facts(layer, definition.parameters)
                    if layer.bias is None:
                        facts["_HAS_BIAS"] = 0
                    args = _parameter_forward_args(layer, sample, root._state, output)
                    self.operations.append((self._prepare(_PARAMETER_OPS[layer.kind][0], args, **facts),))
                elif layer.kind == "add":
                    self.operations.append((self._prepare(p.add, (sample, values[layer.inputs[1]], output)),))
                elif layer.kind in _VALUE_KINDS:
                    function, facts = _value_forward(layer)
                    inputs = () if layer.kind in ("constant", "random_uniform") else tuple(values[i] for i in layer.inputs)
                    params = (*inputs, output, 0, 0, 0, 0) if layer.kind == "random_uniform" else (*inputs, output)
                    self.operations.append((self._prepare(function, params, **facts),))
                else:
                    self.operations.append((self._prepare(_UNARY[layer.kind].forward, (sample, output),
                                                         **_unary_facts(layer)),))
                values.append(output)
            self._first_outputs = tuple(outputs) if self.device == "cpu" else None
            if self.device == "gpu":
                p.retire(outputs, flush=False)
            self.operations = tuple(self.operations)
        except BaseException:
            p.retire(outputs, flush=False)
            self.programs, self.operations = (), ()
            self._inner_domains.clear()
            forward = None
            raise
        self.loss_programs = {}
        self.scalar_storage_bytes = (max((4 * (len(program.parameter_kinds)-len(program.array_parameters))
                                         for program in self.programs), default=0) if self.device == "cpu" else 0)
        self.add = self.zero = None
        self.backward_operations = None
        self._gpu_forward_executables = None
        self._native_forward = None
        self._gpu_reverse_programs = ()
        self._forward_scratch = {}
        self._forward_scratch_version = None

    def release_scratch(self):
        if self._native_forward is not None:
            self._native_forward.close()
            self._native_forward = None
        p.retire(self._forward_scratch.values(), flush=False)
        self._forward_scratch.clear()
        self._forward_scratch_version = None

    @property
    def native_blocks(self):
        source = self.root._program_source
        blocks = () if source is None else source.native_blocks
        if self._native_forward is not None:
            blocks = (*blocks, *self._native_forward.storage.storage_blocks.items())
        return blocks

    def _gpu_entries(self, programs, *, reverse_programs=(), required=True):
        """Read the exact cold role closure; never compile during admission."""
        entries = {}
        roles = tuple((variant, False) for program in programs for variant in p.program_family(program)) + tuple(
            (program, True) for program in reverse_programs)
        for program, reverse in roles:
            mode, mask = ("reverse", (0, 1)) if reverse else ("none", ())
            candidates = [executable for executable in program._executables.values()
                          if executable.artifact is not None
                          and executable.artifact.autodiff_mode == mode
                          and executable.artifact.gradient_parameters == mask]
            if len(candidates) != 1:
                if not candidates and not required:
                    return
                raise RuntimeError("GPU cold preparation is not complete")
            executable = candidates[0]
            contract = executable.artifact.continuation
            if not reverse and (not contract.eligible or not contract.output_parameters
                    or set(contract.output_parameters) != set(contract.fully_written_output_parameters)):
                raise TypeError("managed GPU work requires complete independent output rows")
            # Local Taichi VJP is a whole operation. Its inputs include a fully
            # initialized dx and seed; it has no proven begin/end range ABI.
            size = int(executable.artifact.argument_layout["size"])
            if not 0 < size <= 65536 or size % 4:
                raise TypeError("managed GPU work requires an aligned inline argument upload")
            entries[program, reverse] = executable
        return entries

    def _prepare_gpu_forward(self, *, required=True):
        programs = tuple(operation[0] for operation in self.operations)
        entries = self._gpu_entries(programs, required=required)
        if entries is not None:
            self._gpu_forward_executables = entries

    def _prepare_native_forward(self, inputs, *, run=None):
        from ._gpu_forward import GpuForward

        if self._native_forward is None or self._native_forward.layout != GpuForward.input_layout(inputs):
            if self._native_forward is not None:
                self._native_forward.close()
                self._native_forward = None
            self._native_forward = GpuForward(self, inputs, run=run)
        return self._native_forward

    def _prepare(self, function, params, **facts):
        native = p.prepare(function, params, own_deferred_samples=True, source=self.root._program_source, **facts)
        if (domain := p.inner_domain(native, params)) is not None:
            self._inner_domains[native] = domain
        self.programs += (native,)
        return native

    def _result_spec(self, index=None):
        if index is None:
            if len(self.result_specs) != 1:
                raise ValueError("select a model result for this operation")
            return next(iter(self.result_specs.values()))
        return self.result_specs[index]

    def _training_programs(self, record_loss, objective="mse", *, output_index=None):
        """The fixed training closure, with its optional scalar observation."""
        shape = self._result_spec(output_index).shape
        loss_programs = {program for programs in self.loss_programs.values() for program in programs}
        return (tuple(program for program in self.programs if program not in loss_programs)
                + self.loss_programs[objective, "gradient", shape]
                + (self.loss_programs[objective, "value", shape] if record_loss else ()))

    def _prepare_backward(self, values, state, target=None, *, native_entries=False, objective="mse",
                          output_index=None):
        result = self._result_spec(output_index)
        if self.backward_operations is not None:
            if target is not None or objective == "scalar":
                self._prepare_loss_gradient(values[result.index], target, result.shape,
                                            native_entries=native_entries, objective=objective)
            if native_entries:
                for program in self.programs:
                    for variant in p.program_family(program):
                        variant._native_entry()
            return
        programs, operations, samples, reverse_candidates, reverse_programs = [], [], [], [], []
        inner_domains = {}
        gradient_programs = ()
        add = zero = stage = None
        pending = p._deferred_prepares()
        pending_start = 0 if pending is None else len(pending)

        def empty(shape):
            value = _empty(shape, self.device)
            samples.append(value)
            return value

        def prepare(function, params, **facts):
            native = p.prepare(function, params, own_deferred_samples=True, source=self.root._program_source, **facts)
            try:
                if native_entries:
                    for variant in p.program_family(native):
                        variant._native_entry()
            except BaseException:
                native = variant = None
                raise
            programs.append(native)
            if (domain := p.inner_domain(native, params)) is not None:
                inner_domains[native] = domain
            return native

        try:
            for index, layer in enumerate(self.layers):
                if index + self.input_count not in self.backward_values or layer.kind in _STOP_GRADIENT:
                    operations.append(())
                    continue
                if layer.kind == "add":
                    operations.append(())
                    continue
                if layer.kind in _VALUE_KINDS:
                    inputs = tuple(values[i] for i in layer.inputs)
                    operations.append(tuple(prepare(function, params, **facts) for function, params, facts in
                        _value_pulls(layer, inputs, empty(layer.output_shape), empty)))
                    continue
                before = values[layer.inputs[0]]
                # Compilation needs layout, not a forward value that has
                # already reached its last data consumer (notably CPU y).
                output = empty(layer.output_shape)
                dx = empty(before.shape)
                if layer.kind in _PARAMETER_OPS:
                    m = layer.output_shape[1]
                    weight = state[layer.weight]
                    facts = _parameter_facts(layer, self.definition.parameters)
                    _, pull_dx, pull_dw, pull_db = _PARAMETER_OPS[layer.kind]
                    operations.append((
                        prepare(pull_dx, (weight, output, dx), **facts),
                        prepare(pull_dw, (before, output, empty(weight.shape)), **facts),
                        prepare(pull_db, (output, empty((m,))), **facts) if layer.bias is not None else None))
                elif self.device == "gpu" and _UNARY[layer.kind].local_ad:
                    declaration = self.operations[index][0]
                    previous = set(declaration._executables)
                    try:
                        p.prepare_reverse(declaration, (before, output))
                    finally:
                        reverse_candidates.extend((declaration, key) for key in declaration._executables if key not in previous)
                    operations.append((declaration,))
                    reverse_programs.append(declaration)
                else:
                    spec = _UNARY[layer.kind]
                    operations.append((prepare(spec.backward, (output if spec.output_gradient else before, output, dx),
                                               **_unary_facts(layer)),))
            if target is not None or objective == "scalar":
                gradient_programs = tuple(prepare(function, params, **facts) for function, params, facts in
                    _objective_gradients(objective, result.shape, values[result.index],
                        None if target is None else target._data, empty(result.shape), empty))
            add = prepare(p.add, tuple(empty((1,)) for _ in range(3)))
            if self.device == "gpu" and any(layer.kind in _UNARY and _UNARY[layer.kind].local_ad for layer in self.layers):
                zero = prepare(p.zero, (empty((1,)),))
            stage = self.programs + tuple(programs), tuple(operations), add, zero, tuple(reverse_programs)
        except BaseException:
            if pending is not None:
                del pending[pending_start:]  # Preserve earlier startup work.
            for declaration, key in reverse_candidates:
                declaration._executables.pop(key).close()
            programs.clear()
            operations.clear()
            reverse_candidates.clear()
            reverse_programs.clear()
            inner_domains.clear()
            gradient_programs = ()
            add = zero = stage = None
            raise
        finally:
            p.retire(samples, flush=False)
            samples.clear()
            dx = None
        (self.programs, self.backward_operations, self.add,
         self.zero, self._gpu_reverse_programs) = stage
        self._inner_domains.update(inner_domains)

        if gradient_programs:
            self.loss_programs[objective, "gradient", result.shape] = gradient_programs

    def _target(self, target, objective="mse", *, output_index=None):
        result = self._result_spec(output_index)
        if result.dtype != "float32":
            raise TypeError("training losses require a float32 model output")
        _objective_facts(objective, result.shape)
        _check_target_layout(target, objective, result.shape)
        if target is None:
            return None
        if target.device != self.device:
            raise ValueError("loss requires prediction and target on the same device")
        if target.requires_grad or target._run is not None or target._loss is not None:
            raise ValueError("loss targets must be non-differentiable leaf/detached Tensors")
        _check_storage((target._data,))
        return Tensor._from_buffer(target._data, target.shape)

    def _loss_bind(self, objective, role, shape, recipes, *, native_entries=False):
        key = objective, role, shape
        programs = self.loss_programs.get(key)
        if programs is None:
            programs = tuple(p.prepare(function, params, own_deferred_samples=True,
                source=self.root._program_source, **facts) for function, params, facts in recipes)
            self.loss_programs[key] = programs
            self.programs += programs
            for program, (_, params, _) in zip(programs, recipes):
                if (domain := p.inner_domain(program, params)) is not None:
                    self._inner_domains[program] = domain
        if native_entries:
            for program in programs:
                for variant in p.program_family(program):
                    variant._native_entry()
        return tuple(_Operation(program, params) for program, (_, params, _) in zip(programs, recipes))

    def _prepare_loss_gradient(self, output, target, shape, *, native_entries=False, objective="mse"):
        programs = self.loss_programs.get((objective, "gradient", shape))
        if programs is not None:
            if native_entries:
                for program in programs:
                    for variant in p.program_family(program):
                        variant._native_entry()
            return
        samples = []
        def empty(shape):
            value = _empty(shape, self.device)
            samples.append(value)
            return value
        try:
            recipes = _objective_gradients(objective, shape,
                output, None if target is None else target._data, empty(shape), empty)
            self._loss_bind(objective, "gradient", shape, recipes, native_entries=native_entries)
        finally:
            p.retire(samples, flush=False)

    def _loss_operations(self, output, target, loss, empty, *, native_entries=False, objective="mse",
                         output_shape=None):
        shape = self._result_spec().shape if output_shape is None else output_shape
        recipes = _objective_values(objective, shape, output,
                                   None if target is None else target._data, loss, empty)
        return self._loss_bind(objective, "value", shape, recipes, native_entries=native_entries)

    def _inner_storage_bytes(self, term_limit):
        rows = max((rows for terms, rows in self._inner_domains.values() if terms > term_limit), default=0)
        return rows * 8 + (8 if rows and self.device == "cpu" else 0)

    def _training_storage_bytes(self, facts, record_loss, term_limit, optimizer, objective="mse", clear_grad=False,
                                max_norm=None, *, output_index=None):
        """Conservative simultaneous private storage, excluding borrowed input/state."""
        result = self._result_spec(output_index)
        grad_parameters = result.grad_parameters
        backward_layers = tuple((index, layer) for index, layer in enumerate(self.layers, self.input_count)
                                if index in result.backward_values and layer.kind not in _STOP_GRADIENT)
        if clear_grad:
            facts = tuple((parameter, None) for parameter, _ in facts)
        sizes = tuple(value.element_count * 4 for value in self.root._state)
        activations = self.activation_bytes
        value_sizes = (*(math.prod(spec.shape) * np.dtype(spec.dtype).itemsize for spec in self.inputs),
                       *(math.prod(layer.output_shape) * np.dtype(layer.dtype).itemsize for layer in self.layers))
        consumers = Counter(value for _, layer in backward_layers for value in layer.inputs)
        parameter_uses = Counter(slot for _, layer in backward_layers if layer.kind in _PARAMETER_OPS
                                 for slot in (layer.weight, layer.bias)
                                 if slot is not None and grad_parameters[slot])
        adjoints = result.nbytes + sum(value_sizes[source] for _, layer in backward_layers
                                         if layer.kind != "add" for source in layer.inputs)
        adjoints += sum((uses - 1) * value_sizes[value] for value, uses in consumers.items()
                        if uses > 1 and (value >= self.input_count or self.inputs[value].requires_grad))
        if self.device == "gpu":
            adjoints += sum(value_sizes[index] for index, layer in backward_layers
                            if layer.kind in _UNARY and _UNARY[layer.kind].local_ad)
        gradients = sum(size for size, used in zip(sizes, grad_parameters) if used)
        contributions = sum((2 * uses - 1) * sizes[slot] for slot, uses in parameter_uses.items())
        accumulation = sum(size for size, used, (_, old) in zip(sizes, grad_parameters, facts)
                           if used and old is not None)
        candidates = sum(size for size, used, (_, old) in zip(sizes, grad_parameters, facts)
                          if used or old is not None) * optimizer._candidate_multiplier
        clip = (0 if max_norm is None else optimizer._clip.storage_bytes(tuple(
            parameter.shape for used, (parameter, old) in zip(grad_parameters, facts) if used or old is not None), term_limit))
        loss_scratch = result.nbytes if isinstance(objective, _CrossEntropy) and record_loss else 0
        if isinstance(objective, _CrossEntropy) and objective.indexed:
            adjoints += 4  # One private normalization scalar, including loss-free training.
            if record_loss:
                loss_scratch += 4 * (math.prod(result.shape) // _objective_facts(objective, result.shape)['_WIDTH']) + 4
        return max(activations + adjoints + contributions + accumulation + self._inner_storage_bytes(term_limit) + self.scalar_storage_bytes,
                   activations + loss_scratch + self.scalar_storage_bytes, gradients + accumulation + candidates + clip + 4) + (4 if record_loss else 0)

    def _check_current(self):
        _check_options(None, self.device)
        if not self._is_current():
            raise RuntimeError("model structure/gradient requirements changed; call recompile()")

    def _is_current(self):
        return not self.root._dirty and self.root._plans.get(self.key) is self

    def _new_activations(self, *, managed=False, reuse_scratch=False):
        values = []
        try:
            for index, layer in enumerate(self.layers):
                value = self._forward_scratch.get(index) if reuse_scratch else None
                if value is None:
                    value = _empty(layer.output_shape, self.device, layer.dtype, managed=managed)
                    if reuse_scratch and index + self.input_count not in self.result_specs:
                        self._forward_scratch[index] = value
                values.append(value)
            return tuple(values)
        except BaseException:
            scratch = {id(value) for value in self._forward_scratch.values()}
            p.retire((value for value in values if id(value) not in scratch), flush=False)
            raise

    def _check_forward_inputs(self, inputs, *, state=None):
        self._check_current()
        if (tuple(_InputSpec.from_value(x) for x in inputs) != self.inputs
                or any(x.device != self.device for x in inputs)):
            raise ValueError("input shape/device/gradient requirements differ from capture; call recompile()")
        if any(x._run is not None for x in inputs):
            raise ValueError("chaining separately compiled models is not supported; use one Module tree")
        _check_storage((*(x._data for x in inputs), *(self.root._state if state is None else state)))

    def _bind_forward(self, inputs, *, state, retain_for_backward=True, managed=False, reuse_scratch=False):
        self._check_forward_inputs(inputs, state=state)
        if reuse_scratch and self._forward_scratch_version != self.root.parameter_version:
            # Stable intermediates may cache native bindings, but must not pin
            # a succession of parameter publications behind a live model.
            p.release_bindings(self._forward_scratch.values())
            self._forward_scratch_version = self.root.parameter_version
        outputs, self._first_outputs = self._first_outputs, None
        if outputs is None:
            outputs = self._new_activations(managed=managed, reuse_scratch=reuse_scratch)
        elif reuse_scratch:
            self._forward_scratch.update((index, value) for index, value in enumerate(outputs)
                                         if index + self.input_count not in self.result_specs)
        # Synchronous no-grad inputs already expose read-only storage. Borrow
        # those wrappers until this call completes: another identical alias
        # only rebuilds native launch bindings and must then be retired again.
        # Writable inputs and retained/managed runs still get their own views.
        shared_inputs = tuple(x._data for x in inputs if x._data.readonly) if reuse_scratch else ()
        values = tuple(x._data if reuse_scratch and x._data.readonly else x._data.view(readonly=True)
                       for x in inputs)
        return _Run(self, state, inputs, (*values, *outputs),
                    retain_for_backward=retain_for_backward, managed=managed,
                    borrowed=(*self._forward_scratch.values(), *shared_inputs) if reuse_scratch else ())

    def _forward_operation(self, run, index):
        """The sole forward argument binding for synchronous and bounded drivers."""
        layer, operations = self.layers[index], self.operations[index]
        if layer.kind in _PARAMETER_OPS:
            params = _parameter_forward_args(layer, run.values[layer.inputs[0]], run.state,
                                             run.values[index + self.input_count])
        elif layer.kind == "random_uniform":
            params = (run.values[index + self.input_count], *run.random_keys[index])
        elif len(layer.inputs) > 1:
            params = (*(run.values[value] for value in layer.inputs), run.values[index + self.input_count])
        elif layer.kind == "constant":
            params = (run.values[index + self.input_count],)
        else:
            params = (run.values[layer.inputs[0]], run.values[index + self.input_count])
        return operations[0], params

    def _workload(self, optimizer=None, facts=(), update_spec=None, *, record_loss=False, objective="mse",
                  clear_grad=False, output_index=None):
        """Walk the execution bindings with layouts instead of allocating a Run.

        Forward, gradient merges and SGD use the same builders as execution.
        The owner consumes these temporary descriptions as numerical demand;
        they are neither executable bindings nor another persistent graph.
        """
        def empty(shape, dtype="float32"):
            return _ValueLayout((math.prod(shape),), self.device, dtype=dtype)
        state = tuple(empty(value.shape) for value in self.root._state)
        values = (*(empty(spec.shape, spec.dtype) for spec in self.inputs),
                  *(empty(layer.output_shape, layer.dtype) for layer in self.layers))
        run = SimpleNamespace(plan=self, state=state, values=values, output_index=output_index,
                              random_keys={index:(0, 0, 0, 0) for index, _ in self.random_layers},
                              inputs=tuple(SimpleNamespace(shape=spec.shape, requires_grad=spec.requires_grad, _data=values[index])
                                           for index, spec in enumerate(self.inputs)))
        operations = [("forward", _Operation(*self._forward_operation(run, index)))
                      for index in range(len(self.operations))]
        if optimizer is None:
            return tuple(operations)
        result = self._result_spec(output_index)
        target_layout = _target_layout(objective, result.shape)
        target = None if target_layout is None else SimpleNamespace(_data=empty(*target_layout))
        if record_loss:
            recipes = _objective_values(objective, result.shape, values[result.index],
                                       None if target is None else target._data, empty(()), empty)
            operations.extend(("loss", _Operation(program, params)) for program, (_, params, _) in
                              zip(self.loss_programs[objective, "value", result.shape], recipes))
        gradients = tuple((parameter, None if old is None else SimpleNamespace(_data=empty(old.shape), device=self.device))
                          for parameter, old in facts)
        backward, updates = _backward_work(run, target, gradients, empty, objective=objective, clear_grad=clear_grad)
        operations.extend(("backward", operation) for operation in backward)
        published = {id(parameter): SimpleNamespace(_data=value, device=self.device)
                     for parameter, value in updates}
        step_facts = tuple((parameter, published.get(id(parameter), None if clear_grad else old)) for parameter, old in gradients)
        from .optim import _step_work
        steps = _step_work(optimizer, state, step_facts, update_spec, empty,
                           lambda program, *params: program, lambda value: empty(value.shape))
        operations.extend(("update", operation) for operation in steps.operations)
        return tuple(operations)

    @_automatic_cost
    def forward(self, inputs):
        gradients = is_grad_enabled()
        track = gradients and any(spec.requires_grad for spec in self.result_specs.values())
        if self.device == "gpu" and not track:
            self._check_forward_inputs(inputs)
            if self._gpu_forward_executables is None:
                self._prepare_gpu_forward()
            return self._prepare_native_forward(inputs).forward(inputs)
        run = self._bind_forward(inputs, state=self.root._state, retain_for_backward=track, reuse_scratch=not track)
        try:
            if self.device == "gpu" and self._gpu_forward_executables is None:
                self._prepare_gpu_forward()
            if self.device == "gpu":
                self._prepare_native_forward(inputs, run=run).forward_run(run)
            else:
                _execute_work(run.cursor, self.device, self.programs)
            def tensor(index):
                result = self.result_specs[index]
                requires_grad = gradients and result.requires_grad
                return Tensor._from_buffer(run.values[index], result.shape, requires_grad=requires_grad,
                                           run=run if requires_grad else None, value_plan=self, value_index=index)
            return _map_result(self.result, tensor)
        except BaseException:
            p.retire(run.owned_values)
            raise
        finally:
            p.release_bindings(run.owned_values)
            if not track:
                run.retire()

    def loss(self, run, target, objective="mse", *, output_index=None):
        objective = _objective(objective, target)
        self._check_current()
        if run.consumed:
            raise RuntimeError("the forward execution was already consumed")
        result = self._result_spec(output_index)
        target = self._target(target, objective, output_index=result.index)
        prediction = run.values[result.index]
        _check_storage((prediction, *((target._data,) if target is not None else ())))
        output = _empty((), self.device)
        scratch = []
        def empty(shape):
            value = _empty(shape, self.device)
            scratch.append(value)
            return value
        try:
            operations = self._loss_operations(prediction, target, output, empty, objective=objective,
                                               output_shape=result.shape)
            _execute_work(_WorkCursor(operations), self.device, tuple(op.program for op in operations))
            track = is_grad_enabled() and result.requires_grad
            return Tensor._from_buffer(output, (), requires_grad=track,
                                       loss=(run, target, objective, result.index) if track else None)
        finally:
            p.retire((output, *scratch))


def _backward_work(run, target, facts, empty, *, objective="mse", seed=None, clear_grad=False):
    """One gradient recipe for real bindings and storage-free service analysis."""
    plan, gradients, operations = run.plan, {}, []
    result = plan._result_spec(run.output_index)

    def combine(values):
        total = values[0]
        for value in values[1:]:
            output = empty(total.shape)
            operations.append(_Operation(plan.add, (total, value, output)))
            total = output
        return total

    gradient = seed
    if gradient is None:
        prediction = run.values[result.index]
        gradient = empty(prediction.shape)
        recipes = _objective_gradients(objective, result.shape,
            prediction, None if target is None else target._data, gradient, empty)
        operations.extend(_Operation(program, params) for program, (_, params, _) in
                          zip(plan.loss_programs[objective, "gradient", result.shape], recipes))
    adjoints = {result.index: [gradient]}
    for index in reversed(range(len(plan.layers))):
        layer, programs = plan.layers[index], plan.backward_operations[index]
        if index + plan.input_count not in adjoints or layer.kind in _STOP_GRADIENT:
            continue
        gradient = combine(adjoints.pop(index + plan.input_count))
        if layer.kind == "add":
            for source in layer.inputs:
                adjoints.setdefault(source, []).append(gradient)
            continue
        if layer.kind in _VALUE_KINDS:
            inputs = tuple(run.values[i] for i in layer.inputs)
            for side, program, (_, params, _) in zip(_gradient_sides(layer), programs, _value_pulls(layer, inputs, gradient, empty)):
                operations.append(_Operation(program, params))
                adjoints.setdefault(layer.inputs[side], []).append(params[-1])
            continue
        source = layer.inputs[0]
        before = run.values[source]
        dx = empty(before.shape)
        if layer.kind in _PARAMETER_OPS:
            operations.append(_Operation(programs[0], (run.state[layer.weight], gradient, dx)))
            if result.grad_parameters[layer.weight]:
                dw = empty(run.state[layer.weight].shape)
                operations.append(_Operation(programs[1], (before, gradient, dw)))
                gradients.setdefault(layer.weight, []).append(dw)
            if layer.bias is not None and result.grad_parameters[layer.bias]:
                db = empty(run.state[layer.bias].shape)
                operations.append(_Operation(programs[2], (gradient, db)))
                gradients.setdefault(layer.bias, []).append(db)
        elif plan.device == "gpu" and _UNARY[layer.kind].local_ad:
            seed = empty(gradient.shape)
            operations.append(_Operation(None, (gradient, seed), copy=True))
            operations.append(_Operation(plan.zero, (dx,)))
            operations.append(_Operation(programs[0], (before, run.values[index + plan.input_count], dx, seed), True))
        else:
            primal = run.values[index + plan.input_count] if _UNARY[layer.kind].output_gradient else before
            operations.append(_Operation(programs[0], (primal, gradient, dx)))
        adjoints.setdefault(source, []).append(dx)
    destinations = [(plan.root._named[slot][1], combine(values)) for slot, values in gradients.items()]
    for index, value in enumerate(run.inputs):
        if value.requires_grad and index in adjoints:
            destinations.append((value, combine(adjoints[index])))
    # Calling model(x, x) contributes twice to one leaf, not two independent
    # gradient publications which would overwrite each other.
    grouped = {}
    for destination, value in destinations:
        grouped.setdefault(id(destination), (destination, []))[1].append(value)
    destinations = [(destination, combine(values)) for destination, values in grouped.values()]
    previous = {id(destination): value for destination, value in facts}
    updates = []
    for destination, value in destinations:
        old = None if clear_grad and all(destination is not value for value in run.inputs) else previous[id(destination)]
        if old is not None:
            total = empty(value.shape)
            operations.append(_Operation(plan.add, (old._data, value, total)))
            value = total
        updates.append((destination, value))
    return operations, updates


class _BackwardBindings:
    """One backward's private writes and frozen gradient publication domain."""
    def __init__(self, run, target, facts, *, objective="mse", seed=None, clear_grad=False):
        self.run, self.facts = run, facts
        self.clear_grad = clear_grad
        self.storage, self.updates, operations = [], [], []
        self.published = False
        def empty(shape):
            value = _empty(shape, run.plan.device, managed=run.managed)
            self.storage.append(value)
            return value

        try:
            operations, updates = _backward_work(run, target, facts, empty, objective=objective, seed=seed, clear_grad=clear_grad)
            self.updates = [(destination, Tensor._from_buffer(value, destination.shape))
                            for destination, value in updates]
            self.cursor = _bind_cursor(operations, private=(*self.storage, *run.values[run.plan.input_count:]),
                                       retained=(value._data for _, value in self.updates))
        except BaseException:
            self.retire(flush=False)  # Candidate binding has never submitted work.
            operations.clear()
            raise

    def commit(self):
        run, root = self.run, self.run.plan.root
        with root._lock:
            run.plan._check_current()
            if (root._state is not run.state or not _grads_current(self.facts)
                    or any(value.requires_grad != spec.requires_grad for value, spec in zip(run.inputs, run.plan.inputs))):
                raise RuntimeError("backward parameter/gradient bindings changed before publication")
            if run.managed:
                for _, value in self.updates:
                    value._data._publish_storage()
            if self.clear_grad:
                for _, parameter in root._named:
                    parameter._grad = None
            for destination, value in self.updates:
                destination._grad = value
            self.published = True

    def retire(self, *, flush=True):
        p.retire(self.storage, flush=flush)
        if not self.published:
            p.retire((value._data for _, value in self.updates), flush=flush)
        self.storage.clear()
        self.updates.clear()
        self.cursor = None
        self.run = self.facts = None


class _Run:
    """One execution's storage and operation cursor, never an executor or queue."""
    def __init__(self, plan, state, inputs, values, *, retain_for_backward, managed, borrowed=()):
        self.plan, self.state, self.inputs, self.values = plan, state, inputs, values
        self._borrowed = frozenset(id(value) for value in borrowed)
        self.managed = managed
        self.output_index = None
        from ._random import _default
        keys = _default.reserve_many(tuple(count for _, count in plan.random_layers))
        self.random_keys = {index:key for (index, _), key in zip(plan.random_layers, keys)}
        retained = set(plan.result_specs)
        if retain_for_backward:
            retained.update(source for layer in plan.layers if layer.kind != "add" for source in layer.inputs)
            retained.update(index + plan.input_count for index, layer in enumerate(plan.layers)
                            if layer.kind in _UNARY and (_UNARY[layer.kind].output_gradient
                                or plan.device == "gpu" and _UNARY[layer.kind].local_ad))
        private = {index:value for index,value in enumerate(values) if index >= plan.input_count
                   and id(value) not in self._borrowed}
        def release(indices):
            p.retire((private.pop(index) for index in indices if index in private), flush=False)
        self.cursor = _WorkCursor((_Operation(*plan._forward_operation(self, index))
                                  for index in range(len(plan.operations))),
                                 retained=retained, release=release, graph=plan.forward_graph)
        self.consumed = False
        self.phase = "forward"
        self.optimizer = self.target = self.update_spec = self.loss_output = None
        self.objective, self.loss_scratch = "mse", []
        self.clear_grad = False
        self.backward_bindings = self.step_bindings = None
        self.grad_facts = ()
        self.record_loss = self.grad_published = False
        self.published_state = None
        self.training_gradients = ()
        self._storage_cursor = None
        self._storage_groups = ()

    @property
    def owned_values(self):
        return (value for value in self.values if id(value) not in self._borrowed)

    @property
    def completed_end(self):
        return self.cursor.completed_end

    def _complete_range(self, end, count):
        self.cursor.complete_range(end, count)

    def _current(self):
        root = self.plan.root
        expected = self.state if self.published_state is None else self.published_state
        return (self.plan._is_current() and root._state is expected
                and (self.optimizer is None or (root._optimizer() is self.optimizer and _grads_current(self.grad_facts)
                     and (self.published_state is not None or self.optimizer._state is self.update_spec.state))))

    def _begin_backward(self):
        p.retire(self.loss_scratch, flush=False)
        self.loss_scratch.clear()
        self.backward_bindings = _BackwardBindings(self, self.target, self.grad_facts,
                                                   objective=self.objective, clear_grad=self.clear_grad)
        self.cursor = self.backward_bindings.cursor
        self.phase = "backward"

    def _begin_loss(self):
        values = []

        def empty(shape):
            value = _empty(shape, self.plan.device, managed=self.managed)
            values.append(value)
            return value

        try:
            result = self.plan._result_spec(self.output_index)
            output = empty(())
            operations = self.plan._loss_operations(self.values[result.index], self.target, output, empty,
                                                    objective=self.objective, output_shape=result.shape)
            scratch = values[1:]
            self.cursor = _bind_cursor(operations, private=scratch, retained=(output,))
        except BaseException:
            p.retire(values, flush=False)
            raise
        self.loss_output, self.loss_scratch = output, scratch
        self.phase = "loss"

    def _begin_update(self):
        backward = self.backward_bindings
        if not backward.published:
            backward.commit()
            self.grad_published = True
            self.training_gradients = tuple(value._data for _, value in backward.updates)
            self.grad_facts = _grad_facts(self.optimizer._params)
            p.retire((*self.values, None if self.target is None else self.target._data), flush=False)
            self.values, self.target = (), None
        self.step_bindings = self.optimizer._bind_step(self.state, self.grad_facts, self.update_spec,
                                                       managed=self.managed)
        backward.retire(flush=False)
        self.backward_bindings = None
        self.cursor = self.step_bindings.cursor
        self.phase = "update"

    def _next_operation(self):
        while self.cursor.current() is None:
            if self.optimizer is None or self.phase in ("ready_to_commit", "complete"):
                return None
            with self.plan.root._lock:
                if not self._current():
                    raise RuntimeError("training parameter/gradient bindings changed")
                if self.phase == "forward" and self.record_loss:
                    self._begin_loss()
                elif self.phase in ("forward", "loss"):
                    self._begin_backward()
                elif self.phase == "backward":
                    self._begin_update()
                elif self.phase == "update":
                    # Finishing candidates does not publish parameters. The owner
                    # still needs credit and publication eligibility in _publish.
                    self.phase = "ready_to_commit"
        return self.cursor.current()

    def live_values(self):
        values = [*self.values, *self.state, self.loss_output, *self.loss_scratch, *self.cursor.carry]
        if self.target is not None:
            values.append(self.target._data)
        values.extend(value._data for _, value in self.grad_facts if value is not None)
        if self.backward_bindings is not None:
            values.extend(self.backward_bindings.storage)
        if self.step_bindings is not None:
            values.extend(self.step_bindings.live_values())
        return values

    def storage_groups(self):
        # Each phase binds its complete work set once. Weak native endpoints
        # observe retirement and allocation replacement without a Python scan.
        from infernux.compute import _StorageGroup

        if self._storage_cursor is not self.cursor:
            self._storage_groups = (_StorageGroup(self.live_values()), _StorageGroup(self.private_values()))
            self._storage_cursor = self.cursor
        return self._storage_groups

    def private_values(self):
        values = [*self.values[self.plan.input_count:], self.loss_output, *self.loss_scratch, *self.training_gradients, *self.cursor.carry]
        if self.backward_bindings is not None:
            values.extend(self.backward_bindings.storage)
        if self.step_bindings is not None:
            values.extend(self.step_bindings.pending)
        return values

    def retire(self, *, flush=True):
        """Only after the accepted native ticket has actually retired."""
        self.cursor.retire()
        if self.backward_bindings is not None:
            self.backward_bindings.retire(flush=flush)
        if self.step_bindings is not None:
            self.step_bindings.retire(flush=flush)
        p.retire((*self.owned_values, self.loss_output, *self.loss_scratch, None if self.target is None else self.target._data), flush=flush)
        self.values = ()
        self.state = self.inputs = self.target = self.loss_output = None
        self.cursor = self.backward_bindings = self.step_bindings = None
        self.optimizer = self.update_spec = None
        self.grad_facts = ()
        self.training_gradients = ()
        self._storage_cursor = None
        self._storage_groups = ()
        self.loss_scratch.clear()
        self.consumed = True

    def backward(self, target=None, *, objective="mse", seed=None, output_index=None):
        root, plan = self.plan.root, self.plan
        bindings = None
        with root._lock:
            plan._check_current()
            if (self.consumed or root._state is not self.state
                    or any(value.requires_grad != spec.requires_grad for value, spec in zip(self.inputs, plan.inputs))):
                raise RuntimeError("backward execution is already consumed or its parameter state has changed")
            try:
                self.output_index = output_index
                plan._prepare_backward(self.values, self.state, target, objective=objective, output_index=output_index)
                destinations = tuple(value for _, value in root._named) + tuple(x for x in self.inputs if x.requires_grad)
                bindings = _BackwardBindings(self, target, _grad_facts(destinations), objective=objective, seed=seed)
                _execute_work(bindings.cursor, plan.device, plan.programs)
                bindings.commit()
            finally:
                self.consumed = True
                if bindings is not None:
                    bindings.retire()
                p.retire((*self.values, None if target is None else target._data, seed))
                self.values = ()
                self.state = self.inputs = self.cursor = None
