"""Torch-shaped definitions with shared parameters and cached static graphs."""
from collections import OrderedDict, namedtuple
from functools import wraps
import threading
import weakref

import numpy as np

from ._tensor import Parameter, Tensor, _check_options, _copy_storage_batch, _to_device_batch

_IncompatibleKeys = namedtuple("IncompatibleKeys", "missing_keys unexpected_keys")


class Module:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        definition = cls.__dict__.get("forward")
        if definition is None:
            return

        @wraps(definition)
        def forward(self, *inputs):
            from ._plan import _capture
            capture = _capture.get()
            if capture is None:
                return self(*inputs)
            if self._root_owner is not capture.root:
                raise ValueError("forward called a Module outside the captured tree")
            return definition(self, *inputs)

        cls.forward = forward

    def __init__(self):
        self._parameters, self._modules = {}, {}
        self._root_owner = None
        self._plan = None
        self._plans = {}
        self._parameterless_device = None
        self._dirty = False
        self._state = None
        self._parameter_version = 0
        self._named = ()
        self._optimizer = None
        self._program_source = None
        self._clip_programs = {}
        self._lock = threading.RLock()
        self.training = True

    def __setattr__(self, name, value):
        if not name.startswith("_") and "_parameters" in self.__dict__:
            registered = name in self._parameters or name in self._modules or isinstance(value, (Parameter, Module))
            if name in self._parameters and value is not None and not isinstance(value, Parameter):
                raise TypeError("a registered parameter must be a Parameter or None")
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
            if isinstance(value, Parameter):
                self._parameters[name] = value
            elif isinstance(value, Module):
                self._modules[name] = value
            root = self._root_owner or self
            if root._plans:
                from ._plan import _capture
                # Python forward runs only during capture. A counter written
                # there is not a new graph definition; registered state is.
                capture = _capture.get()
                if registered or capture is None or capture.root is not root:
                    root._dirty = True
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if not name.startswith("_"):
            root = self._root_owner or self
            if root._plans:
                root._dirty = True
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
        object.__delattr__(self, name)

    def _tree(self):
        named, modules, seen_parameters, seen_modules, active = [], [], set(), set(), set()

        def visit(module, prefix):
            if id(module) in active:
                raise ValueError("a Module tree cannot contain a cycle")
            if id(module) in seen_modules:
                return
            seen_modules.add(id(module))
            active.add(id(module))
            modules.append(module)
            for name, parameter in module._parameters.items():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                named.append((prefix + name, parameter))
            for name, child in module._modules.items():
                visit(child, prefix + name + ".")
            active.remove(id(module))

        visit(self, "")
        return tuple(named), modules

    def _bind(self):
        root = self._root_owner or self
        if root is not self:
            return root._bind()
        if self._state is not None and self._plans and not self._dirty:
            return self  # Repeated execution does not traverse the Module tree.
        named, modules = self._tree()
        if self._state is not None:
            if named != self._named:
                raise RuntimeError("parameter topology is fixed; create a new model/optimizer for changed parameters")
            for module in modules:
                if module._root_owner not in (None, self):
                    raise ValueError("a Module cannot belong to two model instances")
                object.__setattr__(module, "_root_owner", self)
            return self
        if any(p._owner not in (None, self) for _, p in named):
            raise ValueError("a Parameter cannot belong to two model instances")
        if any(m._root_owner not in (None, self) for m in modules):
            raise ValueError("a Module cannot belong to two model instances")
        self._named = named
        self._state = tuple(p._data for _, p in named)
        for index, (_, parameter) in enumerate(named):
            parameter._owner, parameter._slot = self, index
            parameter._values = None
        for module in modules:
            object.__setattr__(module, "_root_owner", self)
        return self

    def _plan_key(self, inputs, device=None):
        target = inputs[0].device if device is None else device
        host_identity = None
        if target == "gpu":
            from infernux import compute
            # Deferred BufferSource inputs have layout before they have data.
            # Their eventual lease validates the producer; preparation targets
            # the active host without materializing that source here.
            host = (inputs[0]._data._host if isinstance(inputs[0], Tensor) and inputs[0].device == "gpu"
                    else compute._native_compute_host())
            # Queue addresses may be reused after engine shutdown. A weak host
            # identity distinguishes lifetimes without retaining a device lease.
            host_identity = None if host is None else weakref.ref(host)
        return id(self), host_identity, tuple((value.shape, value.device if device is None else device,
                                              value.requires_grad, value.dtype.name) for value in inputs)

    def _prepared_plan(self, inputs, device=None):
        root = self._root_owner or self
        return root._plans.get(self._plan_key(inputs, device))

    def _clear_plans(self):
        for plan in self._plans.values():
            plan.release_scratch()
            plan.entry._plan = None
        self._plans.clear()

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        if not recurse or not remove_duplicate:
            raise NotImplementedError("only recursive unique parameter enumeration is supported")
        self._bind()
        named, _ = self._tree()
        return iter(((prefix + "." if prefix else "") + name, parameter) for name, parameter in named)

    def parameters(self, recurse=True):
        return (parameter for _, parameter in self.named_parameters(recurse=recurse))

    @property
    def parameter_version(self):
        """Instance-local parameter publication count, shared by bound children."""
        return (self._root_owner or self)._parameter_version

    def __call__(self, *inputs):
        from ._plan import Plan, _capture

        capture = _capture.get()
        if capture is not None:
            if self._root_owner is not capture.root:
                raise ValueError("forward called a Module outside the captured tree")
            return self.forward(*inputs)
        from .program import _capture_program
        program_capture = _capture_program.get()
        if program_capture is not None:
            return program_capture.call_model(self, inputs)
        root = self._bind()
        if not inputs or any(not isinstance(x, Tensor) or isinstance(x, Parameter) for x in inputs):
            raise TypeError("model inputs must be nn.Tensor values")
        with root._lock:
            if root._dirty:
                raise RuntimeError("model structure changed; call recompile() before execution")
            plan = self._prepared_plan(inputs)
            if plan is None:
                plan = Plan.capture(self, inputs)
                root._plans[plan.key] = plan
            self._plan = plan
            return plan.forward(inputs)

    def forward(self, *inputs):
        from ._plan import _capture
        if _capture.get() is None:
            return self(*inputs)
        raise NotImplementedError("define forward in your Module")

    def _capture_definition(self, inputs):
        from ._plan import _Capture, _Definition, _capture
        from ._grad import _enabled
        root = self._bind()
        capture = _Capture(root, inputs)
        token = _capture.set(capture)
        grad_token = _enabled.set(True)  # Calling mode does not change the cached model structure.
        try:
            output = self.forward(*capture.inputs)
        finally:
            _enabled.reset(grad_token)
            _capture.reset(token)
        layers, result = capture.finish(output)
        root._bind()
        return _Definition.from_module(self, inputs, layers, result)

    def prepare(self, *example_inputs):
        """Cold fixed-shape forward preparation; no inference or scene work.

        CPU preparation includes the native worker entry. GPU cold work may use
        the startup preparation queue; managed requests require it to finish.
        A loaded, precompiled GPU plan can bind during an engine callback.
        Repeated calls reuse the plan; offline training preparation remains lazy.
        """
        from ._plan import Plan
        root = self._bind()
        if not example_inputs or any(not isinstance(x, Tensor) or isinstance(x, Parameter) for x in example_inputs):
            raise TypeError("prepare requires nn.Tensor examples")
        with root._lock:
            if root._dirty:
                raise RuntimeError("model structure changed; call recompile() before preparation")
            plan = self._prepared_plan(example_inputs)
            if plan is None:
                plan = Plan.capture(self, example_inputs, binding_precompiled=root._program_source is not None)
            if plan.device == "cpu":
                from . import _primitives as p
                for operations in plan.operations:
                    for program in p.program_family(operations[0]):
                        program._native_entry()
            else:
                plan._prepare_gpu_forward(required=False)
                if plan._gpu_forward_executables is not None:
                    plan._prepare_native_forward(example_inputs)
            root._plans[plan.key] = plan
            self._plan = plan
        return self

    def prepare_work(self, *example_inputs):
        """Prepare a fixed GPU forward recipe with per-instance input/output.

        The example supplies layout, not input contents. The graph must fully
        write its input before execution. Weights are a snapshot: prepare and
        publish another recipe explicitly after changing model parameters.
        """
        from infernux import compute

        if not example_inputs or any(not isinstance(x, Tensor) or x.device != "gpu" for x in example_inputs):
            raise TypeError("prepare_work requires GPU Tensor examples")
        self.prepare(*example_inputs)
        if self._plan.random_layers:
            raise ValueError("stochastic models require nn.program execution; fixed GPU Work recipes have no invocation RNG state")
        with self._root_owner._lock:
            run = self._plan._bind_forward(example_inputs, state=self._plan.root._state, retain_for_backward=False)
        operations = work = None
        try:
            operations = tuple((op.program, op.params) for op in run.cursor.operations)
            used = {id(value) for _, params in operations for value in params if isinstance(value, compute.Buffer)}
            private = tuple(value for value in run.values if id(value) in used)
            private_values = {id(value) for value in private}
            private_resources = {(value.description.resource_index, value.description.resource_generation)
                                 for value in private}
            if any(isinstance(value, compute.Buffer) and id(value) not in private_values
                   and (value.description.resource_index, value.description.resource_generation) in private_resources
                   for _, params in operations for value in params):
                raise ValueError("NN work private storage cannot share an allocation with model parameters")
            result = run.plan.result
            outputs = ({f"output{slot}": run.values[index] for slot, index in enumerate(result)}
                       if isinstance(result, tuple) else {"output": run.values[result]})
            work = compute.prepare_work(operations, private_buffers=private,
                inputs={("input" if len(example_inputs) == 1 else f"input{index}"): run.values[index]
                        for index in range(len(example_inputs)) if id(run.values[index]) in used}, outputs=outputs)
            with self._root_owner._lock:
                if not run._current():
                    raise RuntimeError("model changed during work preparation; prepare the current model again")
            return work
        finally:
            work = operations = None
            try:
                run.retire()
            finally:
                run = None

    def recompile(self):
        """Discard this model's cached graphs; the next call captures its signature.

        Existing forward results cannot subsequently backpropagate. The owned
        parameter set is unchanged; replacing it requires a new model.
        """
        root = self._bind()
        with root._lock:
            root._clear_plans()
            root._dirty = False
        return self

    def train(self, mode=True):
        if type(mode) is not bool:
            raise TypeError("training mode must be bool")
        root = self._root_owner or self
        with root._lock:
            _, modules = self._tree()
            for module in modules:
                if module.training != mode:
                    object.__setattr__(module, "training", mode)
                    # Mode changes recapture automatically; an existing real
                    # structure error remains dirty and still needs recompile.
                    root._clear_plans()
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        root = self._bind()
        with root._lock:
            return OrderedDict((name, Tensor._from_buffer(p._data, p.shape))
                               for name, p in self.named_parameters())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign:
            raise NotImplementedError("only complete strict state loading without parameter replacement is supported")
        root = self._bind()
        if root is not self:
            raise RuntimeError("load the complete model root state")
        with root._lock:
            if set(state_dict) != {name for name, _ in root._named}:
                raise ValueError("state_dict keys must exactly match all named parameters")
            candidates, storage = [None] * len(root._named), []
            for name, parameter in root._named:
                value = state_dict[name]
                if not isinstance(value, Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"state_dict shape/type mismatch for {name}")
            from . import _primitives as p
            try:
                groups = {}
                for index, (name, parameter) in enumerate(root._named):
                    groups.setdefault(parameter.device, []).append((index, state_dict[name]._data))
                for device, values in groups.items():
                    copied = _copy_storage_batch((value for _, value in values), device)
                    storage.extend(copied)
                    for (index, _), value in zip(values, copied, strict=True):
                        candidates[index] = value.view(readonly=True)
                root._state = tuple(candidates)
                root._parameter_version += 1
            except BaseException:
                p.retire(candidates)
                candidates.clear()
                raise
            finally:
                p.retire(storage, flush=False)
        return _IncompatibleKeys([], [])

    def to(self, device):
        """Move a whole parameter set or select a parameterless program's device."""
        _check_options(None, device)
        root = self._bind()
        if root is not self:
            raise RuntimeError("move the complete model root")
        with root._lock:
            if root._program_source is not None:
                root._program_source.require_device(device)
            if not root._state:
                if root._parameterless_device != device and any(plan.device != device for plan in root._plans.values()):
                    root._clear_plans()
                root._parameterless_device = device
                return self
            if all(value.device == device for value in root._state):
                return self
            from . import _primitives as p
            candidates, gradients, storage, moved = [], [], (), None
            try:
                storage = _copy_storage_batch(root._state, device)
                candidates.extend(value.view(readonly=True) for value in storage)
                present = [parameter.grad for _, parameter in root._named if parameter.grad is not None]
                moved = iter(_to_device_batch(present, device))
                gradients = [None if parameter.grad is None else next(moved) for _, parameter in root._named]
                optimizer = None if root._optimizer is None else root._optimizer()
                optimizer_state = None if optimizer is None else optimizer._copy_state(device)
                root._state = tuple(candidates)
                for (_, parameter), gradient in zip(root._named, gradients):
                    parameter._grad = gradient
                root._clear_plans()  # Preserve genuine structural dirtiness.
                if optimizer is not None:
                    optimizer._state = optimizer_state
                    optimizer._clear_programs()
            except BaseException:
                p.retire(candidates)
                candidates.clear()
                gradients.clear()
                raise
            finally:
                p.retire(storage, flush=False)
                moved = None
        return self

    def cpu(self):
        return self.to("cpu")


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        from ._tensor import _check_float_options
        _check_float_options(dtype, device)
        if any(type(v) is not int or v <= 0 for v in (in_features, out_features)) or type(bias) is not bool:
            raise ValueError("Linear requires positive integer widths and a bool bias")
        self.in_features, self.out_features = in_features, out_features
        bound = 1.0 / np.sqrt(in_features)
        from ._random import _uniform
        self.weight = _uniform((out_features, in_features), -bound, bound, device or "cpu", parameter=True)
        self.bias = _uniform((out_features,), -bound, bound, device or "cpu", parameter=True) if bias else None

    def forward(self, x):
        from ._plan import _capture
        return _capture.get().linear(self, x)


def _pair(value, name, minimum=1):
    pair = (value, value) if type(value) is int else tuple(value)
    if len(pair) != 2 or any(type(v) is not int or v < minimum for v in pair):
        raise ValueError(f"{name} requires two integers >= {minimum}")
    return pair


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None):
        super().__init__()
        from ._tensor import _check_float_options
        _check_float_options(dtype, device)
        if (any(type(v) is not int or v <= 0 for v in (in_channels, out_channels, groups))
                or in_channels % groups or out_channels % groups or type(bias) is not bool):
            raise ValueError("Conv2d channels must divide its positive group count")
        if padding_mode != "zeros":
            raise NotImplementedError("Conv2d currently supports zero padding")
        self.in_channels, self.out_channels, self.groups = in_channels, out_channels, groups
        self.kernel_size, self.stride, self.dilation = _pair(kernel_size, "kernel_size"), _pair(stride, "stride"), _pair(dilation, "dilation")
        if isinstance(padding, str):
            if padding not in ("same", "valid"):
                raise ValueError("padding must be a pair, 'same' or 'valid'")
            self.padding = padding
        else:
            self.padding = _pair(padding, "padding", minimum=0)
        shape = out_channels, in_channels // groups, *self.kernel_size
        bound = 1. / np.sqrt(np.prod(shape[1:]))
        from ._random import _uniform
        self.weight = _uniform(shape, -bound, bound, device or "cpu", parameter=True)
        self.bias = _uniform((out_channels,), -bound, bound, device or "cpu", parameter=True) if bias else None

    def forward(self, x):
        from ._plan import _capture
        return _capture.get().conv2d(self, x)


class PixelShuffle(Module):
    def __init__(self, upscale_factor):
        super().__init__()
        if type(upscale_factor) is not int or upscale_factor <= 0:
            raise ValueError("upscale_factor must be a positive integer")
        self.upscale_factor = upscale_factor

    def forward(self, x):
        from ._plan import _capture
        return _capture.get().pixel_shuffle(x, self.upscale_factor)


class _Activation(Module):
    def forward(self, x):
        from ._plan import _capture
        return _capture.get().unary(self._kind, x, getattr(self, "dim", None), options=getattr(self, "options", ()))


class _FunctionalUnary(_Activation):
    def __init__(self, kind, dim=None, *, options=()):
        super().__init__()
        self._kind, self.dim = kind, dim
        self.options = options


class _FunctionalBinary(Module):
    def __init__(self, kind):
        super().__init__()
        self._kind = kind

    def forward(self, left, right):
        return left.capture.binary(self._kind, left, right)


class Sigmoid(_Activation):
    _kind = "sigmoid"


class Tanh(_Activation):
    _kind = "tanh"


class ReLU(_Activation):
    _kind = "relu"

    def __init__(self, inplace=False):
        super().__init__()
        if inplace:
            raise NotImplementedError("static activations produce a new value; inplace=True is unsupported")


class SiLU(ReLU):
    _kind = "silu"


class ELU(ReLU):
    _kind = "elu"

    def __init__(self, alpha=1.0, inplace=False):
        super().__init__(inplace)
        self.alpha = alpha

    def forward(self, x):
        from ._plan import _capture
        return _capture.get().unary(self._kind, x, options=(self.alpha,))


class Softmax(_Activation):
    _kind = "softmax"

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim


class LogSoftmax(Softmax):
    _kind = "log_softmax"


class _TensorOperation(Module):
    def __init__(self, operation, *args, **kwargs):
        super().__init__()
        self._operation, self._args, self._kwargs = operation, args, kwargs

    def forward(self, x):
        return getattr(x, self._operation)(*self._args, **self._kwargs)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        items = args[0].items() if len(args) == 1 and isinstance(args[0], OrderedDict) else enumerate(args)
        for name, module in items:
            name = str(name)
            if not isinstance(module, Module) or not name or "." in name or hasattr(self, name):
                raise ValueError("Sequential requires uniquely named Module children")
            setattr(self, name, module)

    def __getitem__(self, index):
        return tuple(self._modules.values())[index]

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
