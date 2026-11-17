"""Forward differentiation over ordinary NN invocations and native Work."""

from contextlib import ExitStack

from infernux._work import _Operation

from . import _primitives as p
from ._grad import _enabled
from ._plan import (_BINARY, _STOP_GRADIENT, _UNARY, _binary_pull, _bind_cursor, _capture,
                    _check_gradient_run, _execute_work, _invocation_graph, _jvp_scope,
                    _parameter_facts, _unary_facts)
from ._tensor import Tensor, _check_options, _check_storage, _empty


def _pushforward(run, indices, seeds, empty, operations):
    plan = run.plan
    needed = set().union(*(plan.result_specs[index].backward_values for index in indices))
    directions = dict(enumerate(seeds))

    def bind(program, params):
        operations.append(_Operation(program, params))
        return params[-1]

    def prepare(function, params, **facts):
        key = function, tuple(facts.items())
        program = plan._tangent_programs.get(key)
        if program is None:
            program = p.prepare(function, params, **facts)
            plan._tangent_programs[key] = program
        return bind(program, params)

    for index, layer in enumerate(plan.layers):
        value_index = plan.input_count + index
        if value_index not in needed or layer.kind in _STOP_GRADIENT:
            continue
        incoming = tuple(directions.get(source) for source in layer.inputs)
        active = tuple(value for value in incoming if value is not None)
        if not active:
            continue
        if plan.root._program_source is not None:
            raise NotImplementedError("jvp requires source-backed model programs")
        kind, program = layer.kind, plan.operations[index][0]
        if kind == "reshape" or (kind == "add" and len(active) == 1):
            result = active[0]
        elif kind in ("add", "expand", "transpose", "slice", "sum"):
            result = bind(program, (*active, empty(layer.output_shape)))
        elif kind == "linear":
            weight = run.state[layer.weight]
            params = (active[0], weight, weight, empty(layer.output_shape))
            if layer.bias is None:
                result = bind(program, params)
            else:
                result = prepare(p.linear, params, **_parameter_facts(layer, plan.definition.parameters), _HAS_BIAS=0)
        elif kind == "affine":
            result = prepare(p.affine_vjp, (active[0], empty(layer.output_shape)), _SCALE=layer.options[0])
        elif kind in _BINARY:
            primals = tuple(run.values[source] for source in layer.inputs)
            recipes = (_binary_pull(layer, primals, value, empty(layer.output_shape), side)
                       for side, value in enumerate(incoming) if value is not None)
            contributions = tuple(prepare(function, params, **facts) for function, params, facts in recipes)
            result = contributions[0]
            if len(contributions) == 2:
                result = prepare(p.add, (*contributions, empty(layer.output_shape)))
        elif (spec := _UNARY.get(kind)) is not None and spec.backward is not None and not spec.dimension:
            # Scalar pointwise Jacobians are diagonal, so the JVP and VJP use
            # the same derivative of each element and its saved primal value.
            primal = run.values[value_index if spec.output_gradient else layer.inputs[0]]
            result = prepare(spec.backward, (primal, active[0], empty(layer.output_shape)), **_unary_facts(layer))
        else:
            raise NotImplementedError(f"jvp has no directional derivative for the NN operation {kind!r}")
        directions[value_index] = result
    return tuple(directions.get(index) for index in indices)


def _directions(outputs, primals, tangents, scope):
    def port(value):
        run = value._run
        return (run, value._value_index) if run is not None and run.jvp_scope is scope else value

    ports = tuple(port(value) for value in outputs)
    order, results, _ = _invocation_graph((key for key in ports if isinstance(key, tuple)), scope=scope)
    directions = {primal: tangent._data for primal, tangent in zip(primals, tangents)}
    device = primals[0].device
    operations, storage = [], []
    zero = None

    def empty(shape):
        value = _empty(shape, device)
        storage.append(value)
        return value

    roots = {id(run.plan.root): run.plan.root for run in order}
    with ExitStack() as locks:
        for identity in sorted(roots):
            locks.enter_context(roots[identity]._lock)
        try:
            for run in order:
                _check_gradient_run(run, "jvp")
                indices = results[run]
                seeds = tuple(directions.get(port(value)) for value in run.inputs)
                for index, value in zip(indices, _pushforward(run, indices, seeds, empty, operations)):
                    directions[run, index] = value
            buffers = []
            for output, key in zip(outputs, ports):
                value = directions.get(key)
                if value is None:
                    value = empty(output.shape)
                    if zero is None:
                        zero = p.prepare(p.zero, (value,))
                    operations.append(_Operation(zero, (value,)))
                buffers.append(value)
            if operations:
                cursor = _bind_cursor(operations, private=storage, retained=buffers)
                programs = tuple(dict.fromkeys(operation.program for operation in operations))
                _execute_work(cursor, device, programs)
            return tuple(Tensor._from_buffer(buffer, output.shape) for output, buffer in zip(outputs, buffers))
        finally:
            p.retire(storage)


def jvp(func, primals, tangents):
    """Evaluate ``func`` once and return its value and first directional derivative.

    Primals and tangents are float32 Tensors with matching shapes on one device.
    The callable may compose ordinary NN invocations and return a Tensor or a
    tuple of Tensors. Its captured values and parameters are held fixed, and
    both returned values are detached. Unsupported NN operations raise if
    they lie on a differentiated path.
    """
    from .program import _capture_program

    if _jvp_scope.get() is not None or _capture.get() is not None or _capture_program.get() is not None:
        raise NotImplementedError("jvp requires an ordinary callable outside a nested transform or managed capture")
    if not callable(func):
        raise TypeError("jvp requires a callable")
    primals = (primals,) if isinstance(primals, Tensor) else tuple(primals)
    tangents = (tangents,) if isinstance(tangents, Tensor) else tuple(tangents)
    if not primals or len(primals) != len(tangents):
        raise ValueError("jvp requires one tangent for each primal")
    if any(not isinstance(value, Tensor) or value.dtype.name != "float32" for value in (*primals, *tangents)):
        raise TypeError("jvp primals and tangents must be float32 Tensors")
    if any(primal.shape != tangent.shape for primal, tangent in zip(primals, tangents)):
        raise ValueError("jvp tangent shapes must match their primals")
    _check_options(None, primals[0].device)
    _check_storage(tuple(value._data for value in (*primals, *tangents)))

    inputs = tuple(primal.detach() for primal in primals)
    for value in inputs:
        value.requires_grad = True
    scope = object()
    token = _jvp_scope.set(scope)
    grad = _enabled.set(True)
    try:
        result = func(*inputs)
    finally:
        _enabled.reset(grad)
        _jvp_scope.reset(token)
    outputs = result if isinstance(result, tuple) else (result,)
    if not outputs or any(not isinstance(value, Tensor) or value.dtype.name != "float32" for value in outputs):
        raise TypeError("jvp callable must return a float32 Tensor or a nonempty tuple of float32 Tensors")
    _check_storage((inputs[0]._data, *(value._data for value in outputs)))
    if any(value._loss is not None and value._loss[0].jvp_scope is scope for value in outputs):
        raise NotImplementedError("jvp loss outputs require a scalar Tensor expression in the callable")
    directional = _directions(outputs, inputs, tangents, scope)
    detached = tuple(value.detach() for value in outputs)
    return (detached, directional) if isinstance(result, tuple) else (detached[0], directional[0])
