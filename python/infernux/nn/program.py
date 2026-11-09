"""Author training/inference recipes over the existing Plan and frame owner."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import update_wrapper
import inspect

from . import _primitives as p
from ._tensor import Parameter, Tensor, _empty
from ._grad import _enabled, is_grad_enabled
from ._plan import _map_result, _result_values
from .sources import (
    _BatchChannel, _MaterialChannel, _SourceInputs, _SourceProxy, _TensorChannel,
    _buffer_source_sample, _materials,
)
from infernux.engine.cpu_continuation import CpuWorkBudget as Budget
from infernux.compute_source import BufferSource

_capture_program = ContextVar("infernux_nn_program", default=None)


class _Value:
    def __init__(self, capture, sample, argument=None, kind="input", detached=False):
        self.capture, self.sample, self.argument = capture, sample, argument
        self.kind, self.detached = kind, detached

    @property
    def shape(self):
        if self.kind == "gradient_norm":
            raise NotImplementedError("compiled gradient norms are ordered side effects, not program values")
        if self.kind == "loss":
            return ()
        return self.sample.shape

    @property
    def device(self):
        return self.capture.plan.device if self.kind != "input" else self.sample.device

    def detach(self):
        return _Value(self.capture, self.sample, self.argument, self.kind, True)

    def backward(self):
        self.capture.backward(self)

    def __bool__(self):
        raise TypeError("Tensor values cannot control a compiled program's Python branches")


@dataclass(frozen=True)
class _Prediction:
    index: int
    detached: bool


@dataclass(frozen=True)
class _Prepared:
    model: object
    plan: object
    inputs: tuple
    target: object
    optimizer: object
    objective: object
    clear_grad: bool
    record_loss: bool
    predictions: _Prediction | tuple[_Prediction, ...] | None
    output_index: int | None
    detach_inputs: tuple
    detach_target: bool
    max_norm: float | None

    def offline(self, arguments):
        sources = _SourceInputs((*self.inputs, self.target), arguments)
        try:
            sources.offline()
            return self._offline(sources.values)
        finally:
            sources.close()

    def _offline(self, arguments):
        model = self.model
        with self.plan.root._lock:
            self.plan._check_current()
            if self.clear_grad:
                self.optimizer.zero_grad()
            inputs = tuple((arguments[name].detach() if detached else arguments[name]).to(self.plan.device)
                           for name, detached in zip(self.inputs, self.detach_inputs))
            values = dict(zip(_result_values(self.plan.result), _result_values(model(*inputs))))
            if self.optimizer is None:
                return _map_result(self.predictions,
                    lambda prediction: values[prediction.index].detach() if prediction.detached else values[prediction.index])
            prediction = values[self.output_index]
            target = None if self.target is None else arguments[self.target]
            if target is not None:
                target = (target.detach() if self.detach_target else target).to(self.plan.device)
            if self.record_loss:
                result = self.plan.loss(prediction._run, target, self.objective, output_index=self.output_index)
                result.backward()
                result = result.detach()
            else:
                prediction._run.backward(target, objective=self.objective, output_index=self.output_index)
                result = None
            if self.max_norm is not None:
                from .utils import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()
            return result


class _Capture:
    def __init__(self):
        self.model = self.plan = self.target = self.optimizer = None
        self.output_index = None
        self.inputs = ()
        self.objective = "mse"
        self.clear_grad = False
        self.max_norm = None
        self.phase = "start"
        self.samples = []

    def _sample(self, value, device):
        sample = value.sample
        if value.detached:
            sample = sample.detach()
            self.samples.append(sample)
        if sample.device == device:
            return sample
        storage = _empty(sample.shape, device, sample.dtype)
        sample = Tensor._from_buffer(storage, sample.shape, requires_grad=sample.requires_grad)
        storage._close(flush=False)
        self.samples.append(sample)
        return sample

    def call_model(self, model, values):
        if (not values or any(not isinstance(value, _Value) or value.capture is not self or value.kind != "input"
                              for value in values) or self.phase not in ("start", "zero")):
            raise TypeError("a program calls one model with Tensor arguments")
        root = model._bind()
        with root._lock:
            device = (root._state[0].device if root._state else
                      root._parameterless_device or values[0].sample.device)
            model.prepare(*(self._sample(value, device) for value in values))
            self.plan = model._plan
        self.model, self.inputs = model, values
        self.phase = "model"
        # Only shape/device facts are observed; capture executes no inference.
        return _map_result(self.plan.result,
            lambda index: _Value(self, self.plan.result_specs[index], index, "prediction", not is_grad_enabled()))

    def loss(self, prediction, target, objective):
        from ._plan import _objective_facts, _target_layout
        if (not isinstance(prediction, _Value) or prediction.capture is not self or prediction.kind != "prediction" or prediction.detached
                or not isinstance(target, _Value) or target.capture is not self or target.kind != "input"
                or self.phase != "model"):
            raise TypeError("a program loss requires its model prediction and a Tensor argument")
        shape = prediction.shape
        _objective_facts(objective, shape)
        if (target.shape, target.sample.dtype.name) != _target_layout(objective, shape):
            raise ValueError("program loss target differs from its objective layout")
        if target.sample.requires_grad and not target.detached:
            raise ValueError("a program target must be non-differentiable")
        self.target, self.objective, self.phase = target, objective, "loss"
        self.output_index = prediction.argument
        return _Value(self, target.sample, kind="loss", detached=not is_grad_enabled())

    def zero_grad(self, optimizer, set_to_none):
        if self.phase != "start" or not set_to_none:
            raise NotImplementedError("program zero_grad uses set_to_none=True before its model call")
        self.optimizer, self.clear_grad, self.phase = optimizer, True, "zero"

    def backward(self, value):
        if value.capture is self and value.kind == "prediction" and not value.detached and self.phase == "model":
            from ._plan import _objective_facts
            _objective_facts("scalar", value.shape)
            self.objective, self.phase = "scalar", "loss"
            self.output_index = value.argument
        elif value.kind != "loss":
            raise ValueError("a training program backpropagates its scalar loss once")
        if value.capture is not self or value.detached or self.phase != "loss":
            raise ValueError("a training program backpropagates its scalar loss once")
        if not self.plan.result_specs[self.output_index].requires_grad:
            raise RuntimeError("backward requires a differentiable model result")
        self.phase = "backward"

    def step(self, optimizer):
        if self.phase != "backward" or (self.optimizer is not None and optimizer is not self.optimizer):
            raise ValueError("a training program steps one optimizer after loss.backward()")
        self.optimizer, self.phase = optimizer, "step"

    def clip_grad_norm(self, parameters, max_norm):
        if (self.phase != "backward" or self.max_norm is not None
                or parameters != tuple(self.model.parameters())):
            raise ValueError("a program clips its complete model gradients once after backward and before step")
        self.max_norm = max_norm
        # Only an ordering declaration; no norm is evaluated during capture.
        return _Value(self, None, kind="gradient_norm")

    def finish(self, output):
        if self.model is None:
            raise ValueError("a program must call a model")
        predictions = None
        if self.phase == "model" and self.optimizer is None:
            def prediction(value):
                if not isinstance(value, _Value) or value.capture is not self or value.kind != "prediction":
                    raise ValueError("an inference program returns its predictions")
                return _Prediction(value.argument, value.detached)

            predictions = _map_result(output, prediction)
            if not _result_values(predictions):
                raise ValueError("an inference program returns at least one prediction")
        elif self.phase != "step" or self.optimizer._root is not self.model:
            raise ValueError("a training program must backpropagate and step its root model's optimizer")
        record_loss = (isinstance(output, _Value) and output.capture is self
                       and (output.kind == "loss" or self.objective == "scalar" and output.kind == "prediction"
                            and output.argument == self.output_index))
        if self.optimizer is not None and output is not None and not (record_loss and output.detached):
            raise ValueError("a training program returns loss.detach() or None")
        if self.optimizer is not None:
            self.optimizer._prepare_training(tuple(self._sample(value, self.plan.device) for value in self.inputs),
                None if self.target is None else self._sample(self.target, self.plan.device),
                record_loss=record_loss, objective=self.objective, max_norm=self.max_norm, output_index=self.output_index)
        return _Prepared(self.model, self.plan, tuple(value.argument for value in self.inputs),
                         None if self.target is None else self.target.argument,
                         self.optimizer, self.objective, self.clear_grad, record_loss,
                         predictions, self.output_index, tuple(value.detached for value in self.inputs),
                         self.target is not None and self.target.detached, self.max_norm)


class Program:
    """Capture Tensor/scene-field inference or one-optimizer training recipes.

    prepare() is cold. Ordinary calls synchronously execute the prepared recipe.
    submit() only admits prepared work to the current Engine's existing owner;
    its window covers source transfers, compute and result publication.
    """
    def __init__(self, function):
        self.function = function
        self.signature = inspect.signature(function)
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in self.signature.parameters.values()):
            raise TypeError("program arguments must be explicitly named values")
        self._prepared = {}
        update_wrapper(self, function)

    def _arguments(self, args, kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = []
        for name, value in bound.arguments.items():
            materials = _materials(value)
            if materials is not None:
                resident = value.resident_signature if isinstance(value, _MaterialChannel) else ()
                key.append((name, "materials", len(materials), resident))
            elif isinstance(value, _TensorChannel):
                key.append((name, "tensor-fields", value.signature))
            elif isinstance(value, _BatchChannel):
                key.append((name, "scene-fields", value.signature))
            elif isinstance(value, BufferSource):
                key.append((name, value.shape, value.device, False, value.dtype.name))
            elif (isinstance(value, Tensor) and not isinstance(value, Parameter)
                    and value._run is None and value._loss is None):
                key.append((name, value.shape, value.device, value.requires_grad, value.dtype.name))
            else:
                raise TypeError("program arguments must be leaf Tensors, buffer sources, Material or declared Channels")
        return bound, tuple(key)

    def prepare(self, *args, **kwargs):
        bound, key = self._arguments(args, kwargs)
        capture = _Capture()
        proxies = {}
        try:
            for name, value in bound.arguments.items():
                if isinstance(value, BufferSource):
                    sample = _buffer_source_sample(value)
                    capture.samples.append(sample)
                    proxies[name] = _Value(capture, sample, name)
                else:
                    proxies[name] = (_Value(capture, value, name) if isinstance(value, Tensor)
                                     else _SourceProxy(capture, name, value))
            token = _capture_program.set(capture)
            grad_token = _enabled.set(True)
            try:
                call = inspect.BoundArguments(self.signature, proxies)
                output = self.function(*call.args, **call.kwargs)
            finally:
                _enabled.reset(grad_token)
                _capture_program.reset(token)
            self._prepared[key] = capture.finish(output)
        finally:
            p.retire((sample._data for sample in capture.samples), flush=False)
        return self

    def is_prepared(self, *args, **kwargs):
        """Query this complete input signature and current captured structure, without compiling."""
        _, key = self._arguments(args, kwargs)
        prepared = self._prepared.get(key)
        return prepared is not None and prepared.plan._is_current()

    def __call__(self, *args, **kwargs):
        bound, key = self._arguments(args, kwargs)
        if (prepared := self._prepared.get(key)) is None or not prepared.plan._is_current():
            self.prepare(*args, **kwargs)
        return self._prepared[key].offline(bound.arguments)

    def submit(self, *args, finish_within_frames=None, finish_within_ms=None, budget=None, result_device=None,
               expected_parameter_version=None, **kwargs):
        """Submit a prepared program against the shared host-frame budget.

        The optional completion window counts host service cycles, independent
        of the device, physics steps, rendering and the event that triggered it.
        A millisecond target uses monotonic real time and does not pause. Both
        targets may be specified; neither creates more execution budget.
        BufferSource inputs bind when their producer publishes a value. Use
        request.output_source(index) to submit a dependent model in advance.
        Model and ordinary input values are fixed now, and waiting preserves targets.
        """
        from infernux.application import Application
        bound, key = self._arguments(args, kwargs)
        prepared = self._prepared.get(key)
        if prepared is None or not prepared.plan._is_current():
            raise RuntimeError("call program.prepare with this Tensor signature before submission")
        engine = Application._current_engine()
        if engine is None:
            raise RuntimeError("program submission requires an active Engine")
        with prepared.plan.root._lock:
            if expected_parameter_version is not None:
                if type(expected_parameter_version) is not int or expected_parameter_version < 0:
                    raise TypeError("expected_parameter_version must be a nonnegative integer")
                if prepared.plan.root.parameter_version != expected_parameter_version:
                    raise ValueError("model parameters changed since this input batch was collected")
            prepared.model._plan = prepared.plan
            owner = engine._cpu_work(budget)
            request = owner.release_program(prepared, bound.arguments,
                window_opportunities=finish_within_frames, result_device=result_device,
                finish_within_ms=finish_within_ms)
            return engine._wake_cpu_request(owner, request)

    def bake(self, path, *args, **kwargs):
        """Bake this prepared signature, including its actual training objective."""
        from .model import _bake
        _, key = self._arguments(args, kwargs)
        prepared = self._prepared[key]
        return _bake(prepared.model, path, optimizer=prepared.optimizer,
                     objective=prepared.objective, plan=prepared.plan, output_index=prepared.output_index)

    def schedule(self, arguments, *, clock, every_ticks, overlap="queue", **submit_options):
        """Release on real clock ticks; arguments(stamp) supplies named inputs.

        Bind after prepare. The periodic subscription belongs to its clock epoch;
        a replaced physics world requires a new binding. Its caller scope still
        controls whether callbacks and outstanding requests may advance.
        """
        from .scheduling import ProgramBinding
        from infernux.execution import Clock
        if not isinstance(clock, Clock):
            raise TypeError('periodic programs require nn.Clock')
        if type(every_ticks) is not int or every_ticks <= 0:
            raise ValueError('every_ticks must be a positive integer')
        return ProgramBinding(self, arguments, clock=clock, period=every_ticks,
                              overlap=overlap, options=submit_options)

    def on(self, signal, arguments, *, overlap="queue", **submit_options):
        """Release from Signal values; arguments(*values) supplies named inputs."""
        from .scheduling import ProgramBinding, Signal
        if not isinstance(signal, Signal):
            raise TypeError('event programs require nn.Signal')
        return ProgramBinding(self, arguments, clock=None, signal=signal,
                              overlap=overlap, options=submit_options)


def program(function):
    return Program(function)
