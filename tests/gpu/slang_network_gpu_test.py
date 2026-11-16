"""A real Slang MLP retains training, model IO and the public NN syntax."""

from itertools import chain
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import infernux as inx
import infernux.nn as nn
from infernux.nn import functional as F


class _SharedOutputs(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layer = nn.Linear(2, 1, device=device)
        self.unused = nn.Parameter(nn.tensor([1.], device=device))

    def forward(self, first, second, zero):
        prediction = self.layer(first + second).sigmoid()
        return prediction, prediction.sum() + (zero * 0.).sum()


def check_grad(device):
    model = _SharedOutputs(device)
    weight = np.array([[2., -1.]], dtype=np.float32)
    model.load_state_dict({
        "layer.weight": nn.tensor(weight),
        "layer.bias": nn.tensor([.25]),
        "unused": nn.tensor([1.]),
    })
    inputs = np.array([[.25, .5], [-.5, .25]], dtype=np.float32)
    x = nn.tensor(inputs, device=device, requires_grad=True)
    zero = nn.tensor([3.], device=device, requires_grad=True)
    existing = nn.ones_like(x)
    x.grad = existing
    model.layer.weight.grad = existing_weight = nn.ones_like(model.layer.weight)
    target_values = np.array([[.1], [.2]], dtype=np.float32)
    seed_values = np.array([[.6], [-.4]], dtype=np.float32)
    target = nn.tensor(target_values, device=device)
    seed = nn.tensor(seed_values, device=device)
    prediction, total = model(x, x, zero)
    loss = F.mse_loss(prediction, target)

    with np.testing.assert_raises_regex(RuntimeError, "unused"):
        nn.autograd.grad(loss, model.unused)
    db, dx, missing, dzero, dw, repeated = nn.autograd.grad(
        (prediction, loss, total), (model.layer.bias, x, model.unused, zero, model.layer.weight, x),
        grad_outputs=(seed, None, None), allow_unused=True,
    )
    expected = 1. / (1. + np.exp(-((2. * inputs) @ weight.T + .25)))
    upstream = seed_values + 2. * (expected - target_values) / expected.size + 1.
    adjoint = upstream * expected * (1. - expected)
    expected_dx = 2. * adjoint @ weight
    np.testing.assert_allclose(db.cpu().numpy(), adjoint.sum(axis=0), rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(dx.cpu().numpy(), expected_dx, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(repeated.cpu().numpy(), expected_dx, rtol=3e-4, atol=3e-6)
    np.testing.assert_allclose(dw.cpu().numpy(), adjoint.T @ (2. * inputs), rtol=3e-4, atol=3e-6)
    np.testing.assert_array_equal(dzero.cpu().numpy(), [0.])
    np.testing.assert_array_equal(seed.cpu().numpy(), seed_values)
    assert missing is None and not any(value.requires_grad for value in (db, dx, dzero, dw, repeated))
    assert x.grad is existing and model.layer.weight.grad is existing_weight
    assert zero.grad is model.layer.bias.grad is model.unused.grad is None
    with np.testing.assert_raises_regex(RuntimeError, "consumed"):
        nn.autograd.grad(total, x)

    prediction, _ = model(x, x, zero)
    missing_zero, = nn.autograd.grad(prediction, model.unused, seed, materialize_grads=True)
    assert missing_zero.device == device
    np.testing.assert_array_equal(missing_zero.cpu().numpy(), [0.])
    print("Shared-output VJP passed on", device, flush=True)


def check_composition(device):
    from infernux.nn import _plan

    weight = np.array([[.4, -.5], [.25, .75]], np.float32)
    bias = np.array([.2, -.1], np.float32)
    projection = np.array([[1.5, -.6]], np.float32)
    shift = np.array([.15], np.float32)
    encoder = nn.Sequential(nn.Linear(2, 2, device=device), nn.Sigmoid())
    head = nn.Linear(2, 1, device=device)
    encoder.load_state_dict({"0.weight": nn.tensor(weight), "0.bias": nn.tensor(bias)})
    head.load_state_dict({"weight": nn.tensor(projection), "bias": nn.tensor(shift)})
    xs = np.array([[.3, -.2], [-.4, .8]], np.float32)
    x = nn.tensor(xs, device=device, requires_grad=True)
    targets = tuple(np.array(values, np.float32) for values in ([[-.2], [.1]], [[.6], [-.4]], [[.3], [.7]]))
    t0, t1, t2 = (nn.tensor(values, device=device) for values in targets)
    leaves = (x, *encoder.parameters(), *head.parameters())

    latent = 1. / (1. + np.exp(-(xs @ weight.T + bias)))
    expected = latent @ projection.T + shift

    def analytic(dy, direct=None):
        dz = (dy @ projection) * latent * (1. - latent)
        dx, dh, db = dz @ weight, dy.T @ latent, dy.sum(axis=0)
        if direct is not None:
            dx += direct @ projection
            dh += direct.T @ xs
            db += direct.sum(axis=0)
        return dx, dz.T @ xs, dz.sum(axis=0), dh, db

    def assert_gradients(actual, wanted):
        for gradient, reference in zip(actual, wanted, strict=True):
            assert gradient.device == device and not gradient.requires_grad
            np.testing.assert_allclose(gradient.cpu().numpy(), reference, rtol=3e-4, atol=3e-6)

    hidden = encoder(x)
    prediction = head(hidden)
    F.mse_loss(prediction, t0).backward()
    old_gradients = tuple(value.grad for value in leaves)
    dy0 = 2. * (expected - targets[0]) / expected.size
    assert_gradients(old_gradients, analytic(dy0))
    assert hidden.grad is prediction.grad is None

    hidden = encoder(x)
    losses = (F.mse_loss(head(hidden), t0), F.mse_loss(head(hidden), t1), F.mse_loss(head(x), t2))
    dy1 = 2. * (expected - targets[1]) / expected.size
    direct = 2. * (xs @ projection.T + shift - targets[2]) / expected.size
    with patch.object(_plan, "_execute_work", wraps=_plan._execute_work) as execute, \
         patch.object(_plan, "_pullback_work", wraps=_plan._pullback_work) as pullback:
        gradients = nn.autograd.grad(losses, leaves)
        assert execute.call_count == 1
        assert pullback.call_count == len({id(call.args[0]) for call in pullback.call_args_list}) == 4
    assert_gradients(gradients, analytic(dy0 + dy1, direct))
    assert all(value.grad is previous for value, previous in zip(leaves, old_gradients, strict=True))
    with np.testing.assert_raises_regex(RuntimeError, "consumed"):
        losses[1].backward()

    hidden = encoder(x)
    detached = hidden.detach()
    detached.requires_grad = True
    loss = F.mse_loss(head(detached), t0)
    dx, dh = nn.autograd.grad(loss, (x, detached), allow_unused=True)
    assert dx is None
    assert_gradients((dh,), (dy0 @ projection,))
    assert nn.autograd.grad(hidden.sum(), x)[0] is not None

    stale = F.mse_loss(head(encoder(x)), t0)
    encoder.load_state_dict(encoder.state_dict())
    with np.testing.assert_raises_regex(RuntimeError, "changed"):
        stale.backward()
    assert all(value.grad is previous for value, previous in zip(leaves, old_gradients, strict=True))
    F.mse_loss(head(encoder(x)), t0).backward()
    assert_gradients((value.grad for value in leaves), (2. * value for value in analytic(dy0)))
    print("Composed-model and shared-invocation VJPs passed on", device, flush=True)


def check_jvp(device):
    from infernux.nn import _jvp

    class Mix(nn.Module):
        def forward(self, left, right, gain):
            mixed = left + right
            return ((mixed * gain * 1.5 + .25).sum(),
                    mixed.sigmoid().transpose(0, 1)[1:, :].reshape(2))

    encoder = nn.Sequential(nn.Linear(2, 2, device=device), nn.Tanh())
    head = nn.Linear(2, 2, device=device)
    mix = Mix()
    weight = np.array([[.5, -.75], [-.25, 1.25]], dtype=np.float32)
    projection = np.array([[1., .5], [-1., .25]], dtype=np.float32)
    bias, offset = np.array([.25, -.4], dtype=np.float32), np.array([.1, -.2], dtype=np.float32)
    encoder.load_state_dict({"0.weight": nn.tensor(weight), "0.bias": nn.tensor(bias)})
    head.load_state_dict({"weight": nn.tensor(projection), "bias": nn.tensor(offset)})
    xs, gs = np.array([[-.6, .25], [.5, -.75]], dtype=np.float32), np.array([[.8]], dtype=np.float32)
    vs = (np.array([[.2, -.3], [-.75, .5]], dtype=np.float32),
          np.array([[-.1, .6], [.5, -.8]], dtype=np.float32), np.array([[-.25]], dtype=np.float32))
    x, gain = (nn.tensor(value, device=device, requires_grad=True) for value in (xs, gs))
    tangents = tuple(nn.tensor(value, device=device) for value in vs)
    x.grad = old_grad = nn.ones_like(x)
    old = head(x)
    calls = []

    def model(first, second, scale):
        calls.append(True)
        left, right = head(encoder(first)), head(encoder(second))
        return (*mix(left, right, scale), old, second, left.detach())

    with nn.no_grad(), patch.object(_jvp, "_execute_work", wraps=_jvp._execute_work) as execute:
        values, directions = nn.autograd.jvp(model, (x, x, gain), tangents)
        assert calls == [True] and execute.call_count == 1
        assert not nn.is_grad_enabled()

    hidden = np.tanh(xs @ weight.T + bias)
    raw = hidden @ projection.T + offset
    d_mixed = ((vs[0] + vs[1]) @ weight.T * (1. - hidden ** 2)) @ projection.T
    sigmoid = 1. / (1. + np.exp(-2. * raw))
    expected = (((2. * raw * gs) * 1.5 + .25).sum(), sigmoid[:, 1], xs @ projection.T + offset, xs, raw)
    derivatives = ((1.5 * (d_mixed * gs + 2. * raw * vs[2])).sum(),
                   (sigmoid * (1. - sigmoid) * d_mixed)[:, 1], np.zeros_like(xs), vs[1], np.zeros_like(xs))
    for actual, wanted in zip((*values, *directions), (*expected, *derivatives), strict=True):
        assert actual.device == device and not actual.requires_grad
        np.testing.assert_allclose(actual.cpu().numpy(), wanted, rtol=3e-5, atol=3e-6)

    inputs = tuple(nn.tensor(value, device=device, requires_grad=True) for value in (xs, xs, gs))
    seeds = (np.array(.6, dtype=np.float32), np.array([-.75, .25], dtype=np.float32))
    vjp = nn.autograd.grad(model(*inputs)[:2], inputs, tuple(nn.tensor(seed, device=device) for seed in seeds))
    left_dot = sum((value.cpu().numpy() * seed).sum() for value, seed in zip(directions[:2], seeds))
    right_dot = sum((value.cpu().numpy() * seed).sum() for value, seed in zip(vjp, vs))
    np.testing.assert_allclose(left_dot, right_dot, rtol=3e-5, atol=3e-6)
    assert x.grad is old_grad and head.weight.grad is None
    assert nn.autograd.grad(old.sum(), x)[0] is not None

    detached = x.detach()
    squared, squared_direction = nn.autograd.jvp(F.square, detached, tangents[0])
    assert not detached.requires_grad
    np.testing.assert_allclose(squared.cpu().numpy(), xs * xs, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(squared_direction.cpu().numpy(), 2. * xs * vs[0], rtol=3e-5, atol=3e-6)

    # Consuming the earlier reverse tape leaves its primal usable as a constant.
    _, captured_direction = nn.autograd.jvp(lambda value: mix(value, old, gain)[0], detached, tangents[0])
    np.testing.assert_allclose(captured_direction.item(), (1.5 * vs[0] * gs).sum(), rtol=3e-5, atol=3e-6)
    with np.testing.assert_raises_regex(NotImplementedError, "log_softmax"):
        nn.autograd.jvp(lambda value: F.log_softmax(value, dim=1), x, tangents[0])
    print("Composed-model directional derivatives and JVP/VJP duality passed on", device, flush=True)


def check_composed_optimizers(device):
    from infernux.nn import optim

    for kind in (optim.SGD, optim.Adam):
        expected = np.array([2., .4, -1.5, .7], dtype=np.float32)
        encoder, head = nn.Linear(1, 1, device=device), nn.Linear(1, 1, device=device)
        encoder.load_state_dict({"weight": nn.tensor([[expected[0]]]), "bias": nn.tensor([expected[1]])})
        head.load_state_dict({"weight": nn.tensor([[expected[2]]]), "bias": nn.tensor([expected[3]])})
        parameters = tuple(chain(encoder.parameters(), head.parameters()))
        rate, beta1, beta2, epsilon, decay = .04, .5, .75, 1e-5, .02
        order = (0, 1, 2, 3) if kind is optim.SGD else (2, 1, 3, 0)
        settings = {} if kind is optim.SGD else dict(betas=(beta1, beta2), eps=epsilon, weight_decay=decay)
        optimizer = kind((parameters[index] for index in order), lr=rate, **settings)
        with np.testing.assert_raises_regex(ValueError, "one live optimizer"):
            optim.SGD(head.parameters())

        xs, targets = np.array([.25, -.75], np.float32), np.array([.3, .4], np.float32)
        x, target = (nn.tensor(values.reshape(-1, 1), device=device) for values in (xs, targets))
        first, second = np.zeros_like(expected), np.zeros_like(expected)
        counts = np.zeros(len(parameters), dtype=np.int32)
        versions = [encoder.parameter_version, head.parameter_version]
        execute_work = optim._execute_work

        def assert_not_published(states, moments, steps):
            assert encoder._state is states[0] and head._state is states[1]
            assert optimizer._state is moments and optimizer._committed_steps == steps
            assert (encoder.parameter_version, head.parameter_version) == tuple(versions)

        def fail_after_candidates(cursor, *args):
            execute_work(cursor, *args)
            raise RuntimeError("update failed after executing private candidates")

        def replace_gradient(cursor, *args):
            execute_work(cursor, *args)
            head.bias.grad = nn.ones_like(head.bias)

        for step in (1, 2):
            optimizer.zero_grad()
            hidden = encoder(x)
            prediction = head(hidden if step == 1 else hidden.detach())
            F.mse_loss(prediction, target).backward()
            latent = xs * expected[0] + expected[1]
            upstream = 2. * (latent * expected[2] + expected[3] - targets) / len(xs)
            gradients = np.array([(upstream * expected[2] * xs).sum(), (upstream * expected[2]).sum(),
                                  (upstream * latent).sum(), upstream.sum()], dtype=np.float32)
            active = np.array([True, True, True, True] if step == 1 else [False, False, True, True])
            for i, parameter in enumerate(parameters):
                if active[i]:
                    np.testing.assert_allclose(parameter.grad.item(), gradients[i], rtol=3e-5, atol=3e-6)
                else:
                    assert parameter.grad is None

            states, moments = (encoder._state, head._state), optimizer._state
            old_gradients = tuple(parameter.grad for parameter in parameters)
            with patch.object(optim, "_execute_work", side_effect=fail_after_candidates), \
                 np.testing.assert_raises_regex(RuntimeError, "private candidates"):
                optimizer.step()
            assert_not_published(states, moments, step - 1)
            assert all(parameter.grad is old for parameter, old in zip(parameters, old_gradients))

            if step == 1:
                with patch.object(optim, "_execute_work", side_effect=replace_gradient), \
                     np.testing.assert_raises_regex(RuntimeError, "changed before publication"):
                    optimizer.step()
                assert_not_published(states, moments, step - 1)
                head.bias.grad = old_gradients[3]

            with patch.object(optim, "_execute_work", wraps=execute_work) as execute:
                optimizer.step()
                assert execute.call_count == 1
                assert len(execute.call_args.args[0].operations) == active.sum() * (1 if kind is optim.SGD else 3)
            assert tuple(chain(encoder.parameters(), head.parameters())) == parameters
            assert all(parameter.grad is old for parameter, old in zip(parameters, old_gradients))
            versions[0] += int(step == 1)
            versions[1] += 1
            assert (encoder.parameter_version, head.parameter_version) == tuple(versions)
            if step == 2:
                assert encoder._state is states[0]

            counts[active] += 1
            if kind is optim.SGD:
                expected[active] -= rate * gradients[active]
            else:
                adjusted = gradients[active] + decay * expected[active]
                first[active] = beta1 * first[active] + (1. - beta1) * adjusted
                second[active] = beta2 * second[active] + (1. - beta2) * adjusted ** 2
                corrected_first = first[active] / (1. - beta1 ** counts[active])
                corrected_second = second[active] / (1. - beta2 ** counts[active])
                expected[active] -= rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
            np.testing.assert_allclose([parameter.item() for parameter in parameters], expected, rtol=3e-5, atol=3e-6)
            snapshot = optimizer.state_dict()
            assert snapshot["committed_steps"] == step
            if kind is optim.Adam:
                for index, slot in enumerate(order):
                    entry = snapshot["state"][index]
                    assert entry["step"] == counts[slot]
                    np.testing.assert_allclose([entry["exp_avg"].item(), entry["exp_avg_sq"].item()],
                                               [first[slot], second[slot]], rtol=3e-5, atol=3e-6)
            if step == 1:
                optimizer.load_state_dict(snapshot)

        optimizer.zero_grad(set_to_none=False)
        assert encoder.weight.grad is None and head.weight.grad.item() == 0.
        optimizer.zero_grad()
        states, moments = (encoder._state, head._state), optimizer._state
        optimizer.step()
        assert_not_published(states, moments, 2)
        print(f"Composed-model {kind.__name__} updates and failed-publication isolation passed on {device}", flush=True)


def check_functional_call(device):
    from infernux.nn import _plan

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 2, bias=False, device=device)
            self.second = nn.Linear(2, 2, bias=False, device=device)
            self.second.weight = self.first.weight

        def forward(self, value, residual):
            return self.second(self.first(value)) + residual

    model = Tied()
    root_weight = np.array([[.25, .5], [-.2, .6]], dtype=np.float32)
    first_weight = np.array([[1.5, -.3], [.4, -.25]], dtype=np.float32)
    second_weight = np.array([[0., -.75], [.6, 1.15]], dtype=np.float32)
    xs = np.array([[.75, -.4], [-.6, 1.25]], dtype=np.float32)
    rs = np.array([[.2, -.1], [.8, -.6]], dtype=np.float32)
    x, residual = (nn.tensor(value, device=device) for value in (xs, rs))
    model.load_state_dict({"first.weight": nn.tensor(root_weight)})
    param, = model.parameters()
    assert param is model.first.weight is model.second.weight
    param.grad = initial_grad = nn.ones_like(param)
    model.prepare(x, residual)
    first = {"first.weight": nn.tensor(first_weight, device=device)}
    second = {"first.weight": nn.tensor(second_weight, device=device)}
    original_state, original_version, plan = model._state, model.parameter_version, model._plan
    native_work = None if device == "cpu" else plan._native_forward.work
    owner, method = (inx.compute, "_execute_work_instance") if device == "gpu" else (_plan, "_execute_work")
    execute = getattr(owner, method)

    def evaluate(weight):
        return (xs @ weight.T) @ weight.T + rs

    with patch.object(owner, method, wraps=execute) as work, \
         patch.object(inx.compute.Buffer, "get_data", side_effect=AssertionError("forward read back a buffer")):
        left = nn.functional_call(model, first, (x, residual))
        right = nn.functional_call(model, second, (x, residual))
        assert work.call_count == 2
    for value, weight in ((left, first_weight), (right, second_weight)):
        assert value.device == device and not value.requires_grad
        np.testing.assert_allclose(value.cpu().numpy(), evaluate(weight), rtol=3e-5, atol=3e-6)
    assert model._plan is plan and len(model._plans) == 1
    if native_work is not None:
        assert plan._native_forward.work is native_work

    with np.testing.assert_raises_regex(ValueError, "named_parameters"):
        nn.functional_call(model, {}, (x, residual))
    with np.testing.assert_raises_regex(ValueError, "shape and dtype"):
        nn.functional_call(model, {"first.weight": x.sum()}, (x, residual))
    with np.testing.assert_raises_regex(ValueError, "detached Tensor"):
        nn.functional_call(model, dict(model.named_parameters()), (x, residual))

    def fail_after_work(*args, **kwargs):
        execute(*args, **kwargs)
        raise RuntimeError("test forward failure")

    with patch.object(owner, method, side_effect=fail_after_work), \
         np.testing.assert_raises_regex(RuntimeError, "test forward failure"):
        nn.functional_call(model, first, (x, residual))
    assert model._state is original_state and model.parameter_version == original_version
    assert param.grad is initial_grad
    with nn.no_grad():
        np.testing.assert_allclose(model(x, residual).cpu().numpy(), evaluate(root_weight), rtol=3e-5, atol=3e-6)
        retry = nn.functional_call(model, first, (x, residual))
        np.testing.assert_allclose(retry.cpu().numpy(), evaluate(first_weight), rtol=3e-5, atol=3e-6)

    x.requires_grad = residual.requires_grad = True
    bound = nn.functional_call(model, second, (x, residual))
    projection = np.array([[.8, -.45]], dtype=np.float32)
    head = nn.Linear(2, 1, bias=False, device=device)
    head.load_state_dict({"weight": nn.tensor(projection)})
    prediction = head(bound)
    model.load_state_dict({"first.weight": nn.tensor(root_weight + 1.)})
    cotangent = np.array([[.6], [-1.25]], dtype=np.float32)
    seed = nn.tensor(cotangent, device=device)
    dx, dr, untouched, dw = nn.autograd.grad(prediction, (x, residual, param, head.weight), seed, allow_unused=True)
    upstream = cotangent @ projection
    np.testing.assert_allclose(dx.cpu().numpy(), upstream @ second_weight @ second_weight, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(dr.cpu().numpy(), upstream, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(dw.cpu().numpy(), cotangent.T @ evaluate(second_weight), rtol=3e-5, atol=3e-6)
    assert untouched is None and param.grad is initial_grad

    direction = np.array([[-.3, .5], [.8, .15]], dtype=np.float32)
    _, jvp = nn.autograd.jvp(lambda values: nn.functional_call(model, first, (values, residual)),
                             x, nn.tensor(direction, device=device))
    np.testing.assert_allclose(jvp.cpu().numpy(), direction @ first_weight.T @ first_weight.T, rtol=3e-5, atol=3e-6)
    print("Functional calls reuse Work and isolate model state on", device, flush=True)


def main():
    workspace = tempfile.TemporaryDirectory(prefix="infernux-slang-network-")
    project = Path(workspace.name)
    (project / "Assets").mkdir(parents=True, exist_ok=True)
    (project / "ProjectSettings").mkdir(exist_ok=True)
    engine = inx.Engine()
    native = engine.get_native_engine()
    engine.init_renderer(64, 64, str(project))
    try:
        nn.manual_seed(23)
        cpu = nn.Sequential(nn.Linear(3, 5), nn.Sigmoid(), nn.Linear(5, 2))
        gpu = nn.Sequential(nn.Linear(3, 5, device="gpu"), nn.Sigmoid(), nn.Linear(5, 2, device="gpu"))
        gpu.load_state_dict(cpu.state_dict())
        rng = np.random.default_rng(23)
        xs = rng.normal(size=(4, 3)).astype(np.float32)
        ys = rng.normal(size=(4, 2)).astype(np.float32)
        xcpu, xgpu = nn.tensor(xs), nn.tensor(xs, device="gpu")
        tcpu, tgpu = nn.tensor(ys), nn.tensor(ys, device="gpu")
        optimizer_cpu = nn.optim.SGD(cpu.parameters(), lr=.01)
        optimizer_gpu = nn.optim.SGD(gpu.parameters(), lr=.01)
        for step in range(3):
            optimizer_cpu.zero_grad()
            optimizer_gpu.zero_grad()
            outcpu, outgpu = cpu(xcpu), gpu(xgpu)
            np.testing.assert_allclose(outgpu.detach().cpu().numpy(), outcpu.numpy(), rtol=2e-5, atol=2e-6)
            a, b = F.mse_loss(outcpu, tcpu), F.mse_loss(outgpu, tgpu)
            a.backward()
            b.backward()
            for left, right in zip(cpu.parameters(), gpu.parameters()):
                np.testing.assert_allclose(right.grad.detach().cpu().numpy(), left.grad.numpy(), rtol=3e-4, atol=3e-6)
            print("step", step, "CPU", a.item(), "Slang GPU", b.item(), flush=True)
            optimizer_cpu.step()
            optimizer_gpu.step()
        expected = gpu(xgpu).detach().cpu().numpy()
        nn.export(gpu, str(project / "exported.inm"))
        loaded = nn.load(str(project / "exported.inm"), device="gpu")
        np.testing.assert_allclose(loaded(xgpu).detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        nn.bake(gpu, str(project / "baked.inm"), optimizer=optimizer_gpu)
        baked = nn.load(str(project / "baked.inm"), device="gpu")
        np.testing.assert_allclose(baked(xgpu).detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        baked_parameters = tuple(baked.parameters())
        gradients = nn.autograd.grad(F.mse_loss(baked(xgpu), tgpu), baked_parameters)
        optimizer_cpu.zero_grad()
        F.mse_loss(cpu(xcpu), tcpu).backward()
        for gradient, parameter, baked_parameter in zip(gradients, cpu.parameters(), baked_parameters, strict=True):
            np.testing.assert_allclose(gradient.cpu().numpy(), parameter.grad.numpy(), rtol=3e-4, atol=3e-6)
            assert baked_parameter.grad is None
        nn.save_checkpoint(gpu, optimizer_gpu, str(project / "checkpoint.inm"))
        checkpoint, optimizer = nn.load_checkpoint(str(project / "checkpoint.inm"), device="gpu", programs=baked)
        np.testing.assert_allclose(checkpoint(xgpu).detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        print("MLP forward/backward/SGD/export/bake/checkpoint passed", flush=True)
        check_grad("cpu")
        check_grad("gpu")
        check_composition("cpu")
        check_composition("gpu")
        check_jvp("cpu")
        check_jvp("gpu")
        check_composed_optimizers("cpu")
        check_composed_optimizers("gpu")
        check_functional_call("cpu")
        check_functional_call("gpu")
    finally:
        inx.compute._release_engine_resources()
        native.cleanup()
        workspace.cleanup()


if __name__ == "__main__":
    main()
