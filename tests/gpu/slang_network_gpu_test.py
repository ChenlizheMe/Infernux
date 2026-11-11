"""A real Slang MLP retains training, model IO and the public NN syntax."""

from pathlib import Path
import tempfile

import numpy as np

import infernux as inx
import infernux.nn as nn
from infernux.nn import functional as F


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
        nn.save_checkpoint(gpu, optimizer_gpu, str(project / "checkpoint.inm"))
        checkpoint, optimizer = nn.load_checkpoint(str(project / "checkpoint.inm"), device="gpu", programs=baked)
        np.testing.assert_allclose(checkpoint(xgpu).detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        print("MLP forward/backward/SGD/export/bake/checkpoint passed", flush=True)
    finally:
        inx.compute._release_engine_resources()
        native.cleanup()
        workspace.cleanup()


if __name__ == "__main__":
    main()
