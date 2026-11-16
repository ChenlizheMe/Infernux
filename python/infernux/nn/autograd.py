"""First-order differentiation of NN values."""

from ._tensor import Tensor
from ._jvp import jvp


def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,
         only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False):
    """Return the summed VJP for the requested leaves, in input order.

    Outputs can come from several composed or independent invocations on the
    same device and compute host. The call consumes the contributing forward
    executions and leaves existing .grad values untouched. Disconnected leaves
    return None with allow_unused=True, or zero Tensors with materialize_grads=True.
    """
    if retain_graph not in (None, False) or create_graph or is_grads_batched or only_inputs is not True:
        raise NotImplementedError("grad supports first-order, unbatched differentiation without retaining the graph")
    if allow_unused is None:
        allow_unused = materialize_grads
    if materialize_grads and not allow_unused:
        raise ValueError("materialize_grads requires allow_unused=True")

    outputs = (outputs,) if isinstance(outputs, Tensor) else tuple(outputs)
    inputs = (inputs,) if isinstance(inputs, Tensor) else tuple(inputs)
    if not outputs or any(not isinstance(value, Tensor) for value in outputs):
        raise TypeError("grad outputs must be a Tensor or a nonempty sequence of Tensors")
    if not inputs or any(not isinstance(value, Tensor) for value in inputs):
        raise TypeError("grad inputs must be a Tensor or a nonempty sequence of Tensors")
    if any(not value.requires_grad for value in inputs):
        raise RuntimeError("grad inputs must require gradients")
    if any(value._run is not None or value._loss is not None for value in inputs):
        raise NotImplementedError("grad inputs must be leaf Tensors or Parameters")

    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    elif isinstance(grad_outputs, Tensor):
        grad_outputs = (grad_outputs,)
    else:
        grad_outputs = tuple(grad_outputs)
    if len(outputs) != len(grad_outputs):
        raise ValueError("grad_outputs must provide one seed per output")

    from ._plan import _reverse_runs
    sources = tuple(value._backward_source(seed) for value, seed in zip(outputs, grad_outputs))
    gradients = _reverse_runs(sources, inputs=inputs, allow_unused=allow_unused)
    if materialize_grads:
        from .functional import zeros_like
        return tuple(zeros_like(value) if gradient is None else gradient for value, gradient in zip(inputs, gradients))
    return gradients
