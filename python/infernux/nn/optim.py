"""Optimizer recipes share one cursor and one parameter/state publication."""
from infernux._work import _Operation
from dataclasses import dataclass
from types import SimpleNamespace
import weakref

import numpy as np

from . import _primitives as p
from ._tensor import Parameter, Tensor, _empty, _check_options, _check_storage, _copy_storage
from ._plan import _bind_cursor, _execute_work, _grad_facts, _grads_current

_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class _StepSpec:
    settings: tuple
    state: tuple
    max_norm: float | None = None

    @property
    def lr(self):
        return self.settings[0]


@dataclass
class _StepWork:
    operations: list
    parameters: list
    changed: list
    state: tuple
    gradients: tuple = ()


def _copy_optimizer_state(state, device):
    result, owned = [], []
    try:
        for entry in state:
            if entry is None:
                result.append(None)
                continue
            values = []
            for value in entry[1:]:
                storage = _copy_storage(value, device)
                view = storage.view(readonly=True)
                owned.append(view)
                storage.close()
                values.append(view)
            result.append((entry[0], *values))
        return tuple(result)
    except BaseException:
        p.retire(owned)
        raise


def _step_work(optimizer, state, facts, spec, empty, prepare, borrow=lambda value: value):
    """One builder for numerical execution and storage-free service analysis."""
    clip = None
    if spec.max_norm is not None:
        from .utils import _clip_work
        clip = _clip_work(facts, spec.max_norm, empty, optimizer._clip.bind)
        gradients = {id(parameter): value for parameter, value in clip.gradients}
        facts = tuple((parameter, None if gradient is None else SimpleNamespace(
            _data=gradients[id(parameter)], device=parameter.device)) for parameter, gradient in facts)
    work = optimizer._step_work(state, facts, spec, empty, prepare, borrow)
    if clip is not None:
        work.operations[:0] = clip.operations
        work.gradients = tuple(clip.gradients)
    return work


def _sgd_work(optimizer, state, facts, spec, empty, prepare, borrow):
    candidates, changed, operations = list(state), [], []
    for index, (parameter, gradient) in enumerate(facts):
        if gradient is None:
            continue
        if gradient.device != parameter.device:
            raise ValueError("optimizer gradients and parameters must use the same device")
        data = empty(parameter.shape)
        changed.append(index)
        candidates[index] = data
        params = state[index], gradient._data, data, spec.lr
        operations.append(_Operation(prepare(optimizer._update, *params), params))
    return _StepWork(operations, candidates, changed, spec.state)


class _StepBindings:
    """Private parameter and optimizer candidates, published together."""
    def __init__(self, optimizer, state, facts, spec, *, managed):
        self.optimizer, self.state, self.facts, self.spec = optimizer, state, facts, spec
        self.managed = managed
        self.candidates, self.pending, self.changed = list(state), [], []
        self.next_state = ()
        self.gradients = []
        self.published = False
        operations = []
        def empty(shape):
            value = _empty(shape, optimizer._params[0].device, managed=managed)
            self.pending.append(value)
            return value
        def prepare(program, *params):
            return program
        try:
            work = _step_work(optimizer, state, facts, spec, empty, prepare)
            operations, self.candidates, self.changed = work.operations, work.parameters, work.changed
            self.gradients = [(parameter, Tensor._from_buffer(value, parameter.shape))
                              for parameter, value in work.gradients]
            self.next_state = tuple(None if entry is None else
                (entry[0], entry[1].view(readonly=True), entry[2].view(readonly=True)) for entry in work.state)
            for index in self.changed:
                self.candidates[index] = self.candidates[index].view(readonly=True)
            self.cursor = _bind_cursor(operations, private=self.pending,
                retained=(*self.candidates, *optimizer._state_values(self.next_state),
                          *(gradient._data for _, gradient in self.gradients)))
        except BaseException:
            self.retire(flush=False)  # No operation from this binding was submitted.
            operations.clear()
            raise

    def commit(self):
        root = self.optimizer._root
        with root._lock:
            self.optimizer._check_domain()
            if (root._state is not self.state or self.optimizer._state is not self.spec.state
                    or not _grads_current(self.facts)):
                raise RuntimeError("optimizer parameter/gradient/state bindings changed before publication")
            if self.managed:
                for value in (*(self.candidates[index] for index in self.changed),
                              *self.optimizer._state_values(self.next_state),
                              *(gradient._data for _, gradient in self.gradients)):
                    value._publish_storage()
            if self.changed:
                state = tuple(self.candidates)
                step = self.optimizer._committed_steps + 1
                root._state = state
                root._parameter_version += 1
                self.optimizer._state = self.next_state
                self.optimizer._committed_steps = step
            for parameter, gradient in self.gradients:
                parameter._grad = gradient
            self.facts = _grad_facts(self.optimizer._params)
            self.published = True
            return root._state

    def live_values(self):
        return (*self.state, *self.optimizer._state_values(self.spec.state),
                *(gradient._data for _, gradient in self.facts if gradient is not None), *self.pending)

    def retire(self, *, flush=True):
        p.retire(self.pending, flush=flush)
        if not self.published:
            p.retire((gradient._data for _, gradient in self.gradients), flush=flush)
            p.retire((self.candidates[index] for index in self.changed
                      if self.candidates[index] is not self.state[index]), flush=flush)
            p.retire(self.optimizer._state_values(self.next_state), flush=flush)
        self.pending.clear()
        self.candidates.clear()
        self.gradients.clear()
        self.cursor = self.state = self.facts = None
        self.next_state = self.spec = None


class Optimizer:
    _candidate_multiplier = 1
    def __init__(self, params, defaults):
        params = tuple(params)
        if not params or any(not isinstance(value, Parameter) for value in params):
            raise TypeError("optimizer requires the complete model.parameters() iterable")
        root = params[0]._owner
        if root is None or params != tuple(value for _, value in root._named):
            raise ValueError("optimizer requires exactly one complete model parameter tree")
        if root._optimizer is not None and root._optimizer() is not None:
            raise ValueError("a model supports one live optimizer")
        self._root, self._params = root, params
        self._committed_steps = 0
        self.param_groups = [{"params": list(params), **defaults}]
        self._group_fields = frozenset(defaults)
        self._state = ()
        self._settings()
        self._update = self._zero = self._clip = None
        root._optimizer = weakref.ref(self)

    def _check_domain(self):
        if len(self.param_groups) != 1 or tuple(self.param_groups[0]["params"]) != self._params:
            raise ValueError("optimizer parameter groups cannot replace the model's publication domain")
        if self._root._optimizer() is not self or tuple(value for _, value in self._root._named) != self._params:
            raise ValueError("optimizer requires its original model parameter domain")

    def _learning_rate(self, group=None):
        # Construction validates groups before registering the weak optimizer owner.
        if len(self.param_groups) != 1 or tuple(self.param_groups[0]["params"]) != self._params:
            raise ValueError("optimizer parameter groups cannot replace the model's publication domain")
        group = self.param_groups[0] if group is None else group
        lr = group["lr"]
        if isinstance(lr, (bool, np.bool_)) or not isinstance(lr, (int, float, np.floating)) or not np.isfinite(lr) or lr < 0:
            raise ValueError("optimizer lr must be finite and nonnegative")
        if lr > _FLOAT32_MAX:
            raise ValueError("optimizer lr exceeds float32 range")
        return np.float32(lr)

    def _settings(self, group=None):
        return (self._learning_rate(group),)

    def _step_spec(self, max_norm=None):
        return _StepSpec(self._settings(), self._state, max_norm)

    @staticmethod
    def _state_values(state):
        return tuple(value for entry in state if entry is not None for value in entry[1:])

    @property
    def _update_programs(self):
        return (() if self._update is None else (self._update,)) + (() if self._clip is None else self._clip.programs)

    def _copy_state(self, device):
        return _copy_optimizer_state(self._state, device)

    def _require_idle(self):
        from infernux.application import Application

        engine = Application._current_engine()
        owner = None if engine is None else engine._cpu_work_owner
        if owner is not None and owner._training_writer(self._root) is not None:
            raise RuntimeError("finish the model's training request before transferring optimizer state")

    def state_dict(self):
        """Snapshot committed optimizer values, indexed by parameter order.

        Tensor values retain their immutable storage. The dictionary is an
        in-memory value; checkpoint files continue to use the existing INM codec.
        """
        with self._root._lock:
            self._check_domain()
            self._require_idle()
            record = self._record()
            group = {key: value for key, value in record.items()
                     if key not in ("kind", "committed_steps", "steps")}
            group["params"] = list(range(len(self._params)))
            state = {index: dict(step=entry[0],
                exp_avg=Tensor._from_buffer(entry[1], parameter.shape),
                exp_avg_sq=Tensor._from_buffer(entry[2], parameter.shape))
                for index, (parameter, entry) in enumerate(zip(self._params, self._state)) if entry is not None}
            return dict(state=state, param_groups=[group], kind=self._kind, committed_steps=self._committed_steps)

    def load_state_dict(self, state_dict):
        """Restore settings and moments into this model's parameter domain.

        This explicit cold boundary copies through the ordinary CPU/Vulkan
        transfer path. It publishes only after every state tensor is ready.
        """
        with self._root._lock:
            self._check_domain()
            self._require_idle()
            if state_dict["kind"] != self._kind or len(state_dict["param_groups"]) != 1:
                raise ValueError("optimizer state requires the same optimizer kind and one parameter group")
            group = dict(state_dict["param_groups"][0])
            ids, state, committed = tuple(group["params"]), state_dict["state"], state_dict["committed_steps"]
            if (len(ids) != len(self._params) or any(type(index) is not int for index in ids)
                    or len(set(ids)) != len(ids) or not set(state).issubset(ids)):
                raise ValueError("optimizer state parameter IDs must match the model's parameter count")
            if type(committed) is not int or committed < 0:
                raise ValueError("optimizer committed_steps must be a nonnegative integer")
            if set(group) != self._group_fields | {"params"}:
                raise ValueError("optimizer state group settings differ from the supported optimizer")
            group["params"] = list(self._params)
            settings = self._settings(group)
            group = dict(params=list(self._params), lr=float(settings[0]))
            if self._kind == "adam":
                group.update(betas=tuple(settings[1:3]), eps=float(settings[3]), weight_decay=float(settings[4]))
            incoming = []
            if self._kind == "sgd" and state:
                raise ValueError("ordinary SGD has no per-parameter state")
            if self._kind == "adam":
                for index, parameter in zip(ids, self._params):
                    if index not in state:
                        incoming.append(None)
                        continue
                    value = state[index]
                    step, first, second = value["step"], value["exp_avg"], value["exp_avg_sq"]
                    if type(step) is not int or not 0 < step <= committed:
                        raise ValueError("Adam parameter step must be positive and no greater than committed_steps")
                    if any(not isinstance(moment, Tensor) or moment.shape != parameter.shape
                           or moment.dtype != parameter.dtype for moment in (first, second)):
                        raise ValueError("Adam state tensors must match their parameter shape and dtype")
                    incoming.append((step, first._data, second._data))
            copied = _copy_optimizer_state(incoming, self._params[0].device)
            self.param_groups = [group]
            self._state = copied
            self._committed_steps = committed

    def _clear_programs(self):
        self._update = self._zero = self._clip = None

    def _record(self):
        return dict(kind=self._kind, lr=float(self._learning_rate()), committed_steps=self._committed_steps)

    def _checkpoint_values(self):
        return tuple((f"optimizer/{index}/{name}.f32", value)
                     for index, entry in enumerate(self._state) if entry is not None
                     for name, value in zip(("first", "second"), entry[1:]))

    def _checkpoint_specs(self):
        return tuple((f"optimizer/{index}/{name}.f32", parameter._data.nbytes, False)
                     for index, parameter in enumerate(self._params) for name in
                     (("first", "second") if self._kind == "adam" else ()))

    def _prepare_update(self, current, gradient, output, lr, *, native_entry=False):
        if self._update is None:
            update = p.prepare(p.sgd, (current, gradient, output, lr), own_deferred_samples=True,
                               source=self._root._program_source)
            try:
                if native_entry:
                    update._native_entry()
            except BaseException:
                update = None
                raise
            self._update = update
        elif native_entry:
            self._update._native_entry()

    def _prepare_training(self, example_input, example_target=None, *, record_loss=False, objective="mse",
                          max_norm=None, output_index=None):
        """Cold fixed-shape training preparation; no numerical update executes."""
        from ._plan import _objective
        objective = _objective(objective, example_target)
        if type(record_loss) is not bool:
            raise TypeError("record_loss must be bool")
        inputs = (example_input,) if isinstance(example_input, Tensor) else tuple(example_input)
        if not inputs or any(not isinstance(x, Tensor) or x.requires_grad for x in inputs):
            raise ValueError("training continuation requires non-differentiable Tensor inputs")
        root = self._root
        with root._lock:
            self._check_domain()
            root.prepare(*inputs)
            plan = root._plan
            result = plan._result_spec(output_index)
            native_entries = plan.device == "cpu"
            if not any(result.grad_parameters):
                raise ValueError("training requires a differentiable parameter path")
            target = plan._target(example_target, objective, output_index=result.index)
            temporary = []
            try:
                outputs = plan._first_outputs
                if outputs is None:
                    outputs = plan._new_activations()
                    temporary.extend(outputs)
                values = (*(x._data for x in inputs), *outputs)
                plan._prepare_backward(values, root._state, target, native_entries=native_entries, objective=objective,
                                       output_index=result.index)
                if record_loss:
                    loss = _empty((), plan.device)
                    temporary.append(loss)
                    def empty(shape):
                        value = _empty(shape, plan.device)
                        temporary.append(value)
                        return value
                    plan._loss_operations(values[result.index], target, loss, empty, native_entries=native_entries,
                                          objective=objective, output_shape=result.shape)
                current = root._state[0]
                updated = _empty(current.shape, plan.device)
                temporary.append(updated)
                self._prepare_update(current, current, updated, self._learning_rate(), native_entry=native_entries)
                if max_norm is not None:
                    from .utils import _ClipPrograms
                    if self._clip is None:
                        self._clip = _ClipPrograms(plan.device, root._program_source)
                    self._clip.prepare(self._params, native_entries=native_entries)
                if plan.device == "gpu":
                    self._gpu_training_entries(plan, record_loss=record_loss, required=False, objective=objective,
                                               output_index=result.index)
            finally:
                p.retire((*temporary, None if target is None else target._data), flush=False)
                temporary.clear()
        return self

    def _gpu_training_entries(self, plan, *, record_loss, required=True, objective="mse", output_index=None):
        shape = None if plan is None else plan._result_spec(output_index).shape
        if (plan is None or plan.backward_operations is None or self._update is None
                or (objective, "gradient", shape) not in plan.loss_programs
                or (record_loss and (objective, "value", shape) not in plan.loss_programs)):
            raise RuntimeError("call optimizer._prepare_training before releasing GPU training")
        return plan._gpu_entries((*plan._training_programs(record_loss, objective, output_index=output_index),
                                  *self._update_programs),
                                 reverse_programs=plan._gpu_reverse_programs, required=required)

    def _bind_step(self, state, facts, spec, *, managed=False):
        self._check_domain()
        _check_storage((*state, *self._state_values(spec.state), *(gradient._data for _, gradient in facts if gradient is not None)))
        return _StepBindings(self, state, facts, spec, managed=managed)

    def _prepare_zero(self, data):
        if self._zero is None:
            self._zero = p.prepare(p.zero, (data,), source=self._root._program_source)

    def zero_grad(self, set_to_none=True):
        if type(set_to_none) is not bool:
            raise TypeError("set_to_none must be bool")
        from .program import _capture_program
        if (capture := _capture_program.get()) is not None:
            return capture.zero_grad(self, set_to_none)
        with self._root._lock:
            device = self._params[0].device
            _check_options(None, device)
            if set_to_none:
                for parameter in self._params:
                    parameter._grad = None
                return
            pending, updates = [], []
            published = False
            try:
                for parameter in self._params:
                    data = None if parameter.grad is None else _empty(parameter.shape, device)
                    pending.append(data)
                    if data is not None and self._zero is None:
                        self._prepare_zero(data)
                if not any(data is not None for data in pending):
                    return
                with p.stage(device, () if self._zero is None else (self._zero,)):
                    for parameter, data in zip(self._params, pending):
                        if data is not None:
                            p.run(self._zero, data)
                        updates.append(None if data is None else Tensor._from_buffer(data, parameter.shape))
                for parameter, value in zip(self._params, updates):
                    parameter._grad = value
                published = True
            finally:
                p.retire(pending)
                if not published:
                    p.retire(value._data for value in updates if value is not None)
                pending.clear()
                updates.clear()

    def step(self, closure=None):
        """Synchronous offline step; it has no frame-budget guarantee."""
        if closure is not None:
            raise NotImplementedError("optimizer closure execution is not supported")
        from .program import _capture_program
        if (capture := _capture_program.get()) is not None:
            return capture.step(self)
        root = self._root
        with root._lock:
            root._bind()
            spec = self._step_spec()
            device = self._params[0].device
            _check_options(None, device)
            _check_storage((*root._state, *(parameter.grad._data for parameter in self._params if parameter.grad is not None)))
            bindings = None
            try:
                facts = _grad_facts(self._params)
                if self._update is None and any(gradient is not None for _, gradient in facts):
                    current = root._state[0]
                    scratch = _empty(current.shape, device)
                    try:
                        self._prepare_update(current, current, scratch, spec.lr)
                    finally:
                        scratch.close()
                bindings = self._bind_step(root._state, facts, spec)
                _execute_work(bindings.cursor, device, self._update_programs)
                bindings.commit()
            finally:
                if bindings is not None:
                    bindings.retire()


class SGD(Optimizer):
    _kind = "sgd"
    _step_work = _sgd_work

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0,
                 nesterov=False, *, maximize=False, foreach=None, differentiable=False, fused=None):
        if momentum or dampening or weight_decay or nesterov or maximize or foreach is not None or differentiable or fused is not None:
            raise NotImplementedError("only ordinary SGD without momentum/weight decay is supported")
        super().__init__(params, dict(lr=lr))


class Adam(Optimizer):
    _kind = "adam"
    _candidate_multiplier = 3

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 amsgrad=False, *, foreach=None, maximize=False, capturable=False,
                 differentiable=False, fused=None, decoupled_weight_decay=False):
        if amsgrad or foreach is not None or maximize or capturable or differentiable or fused is not None or decoupled_weight_decay:
            raise NotImplementedError("Adam supports coupled weight decay without optional execution variants")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self._state = (None,) * len(self._params)
        self._moments = ()

    def _settings(self, group=None):
        group = self.param_groups[0] if group is None else group
        betas = tuple(group["betas"])
        if len(betas) != 2 or any(not np.isfinite(value) or not 0 <= value < 1 for value in betas):
            raise ValueError("Adam betas require two finite values in [0,1)")
        if any(not np.isfinite(group[name]) or not 0 <= group[name] <= _FLOAT32_MAX
               for name in ("eps", "weight_decay")):
            raise ValueError("Adam epsilon and weight_decay must be finite nonnegative float32 values")
        return self._learning_rate(group), float(betas[0]), float(betas[1]), np.float32(group["eps"]), np.float32(group["weight_decay"])

    @property
    def _update_programs(self):
        return (*self._moments, *super()._update_programs)

    def _clear_programs(self):
        super()._clear_programs()
        self._moments = ()

    def _record(self):
        lr, beta1, beta2, epsilon, decay = self._settings()
        return dict(kind=self._kind, lr=float(lr), betas=(beta1, beta2), eps=float(epsilon),
                    weight_decay=float(decay), committed_steps=self._committed_steps,
                    steps=[0 if entry is None else entry[0] for entry in self._state])

    def _prepare_update(self, current, gradient, output, lr, *, native_entry=False):
        if self._update is None:
            settings = self._settings()
            self._moments = tuple(p.prepare(p.adam_moment,
                (current, gradient, current, output, np.float32(beta), np.float32(1-beta), settings[4], np.float32(1)),
                _MOMENT_POWER=power, own_deferred_samples=True, source=self._root._program_source)
                for power, beta in enumerate(settings[1:3], 1))
            self._update = p.prepare(p.adam_update,
                (current, current, current, output, lr, np.float32(1), np.float32(1), settings[3]),
                own_deferred_samples=True, source=self._root._program_source)
        if native_entry:
            for program in self._update_programs:
                for variant in p.program_family(program):
                    variant._native_entry()

    def _step_work(self, state, facts, spec, empty, prepare, borrow):
        lr, beta1, beta2, epsilon, decay = spec.settings
        operations, candidates, changed = [], list(state), []
        moments = list(spec.state)
        for index, (parameter, gradient) in enumerate(facts):
            if gradient is None:
                continue
            old = spec.state[index]
            step = 1 if old is None else old[0] + 1
            first, second, updated = (empty(parameter.shape) for _ in range(3))
            for program, output, beta, previous in zip(self._moments, (first, second), (beta1, beta2),
                    (state[index], state[index]) if old is None else (borrow(old[1]), borrow(old[2]))):
                params = (state[index], gradient._data, previous, output,
                          np.float32(beta), np.float32(1-beta), decay, np.float32(old is None))
                operations.append(_Operation(prepare(program, *params), params))
            params = (state[index], first, second, updated, lr,
                      np.float32(1 / (1 - float(beta1) ** step)),
                      np.float32(1 / (1 - float(beta2) ** step)), epsilon)
            operations.append(_Operation(prepare(self._update, *params), params))
            candidates[index], moments[index] = updated, (step, first, second)
            changed.append(index)
        return _StepWork(operations, candidates, changed, tuple(moments))
