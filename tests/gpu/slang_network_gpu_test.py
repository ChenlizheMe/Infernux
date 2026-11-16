"""A real Slang MLP retains training, model IO and the public NN syntax."""

from pathlib import Path
import tempfile

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
    finally:
        inx.compute._release_engine_resources()
        native.cleanup()
        workspace.cleanup()


if __name__ == "__main__":
    main()
