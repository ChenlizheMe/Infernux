"""Exercise authored Slang through the NN graph and its portable model codec."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import infernux as inx
from infernux import nn
from infernux._compiler import slang
from infernux.lib import RuntimeMode, _Infernux
from infernux.nn import _jvp, _plan


SOURCE = """[Differentiable]
public float bend(float x) {
    float xx = x * x;
    return xx * x + 0.25f * sin(xx);
}
"""
WEIGHT, BIAS = 0.75, -0.125
LINKED_SOURCE = """import shape_curve;
[Differentiable]
public float bend(float x) { return shape_curve::bendCurve(x); }
"""
SHAPE_CURVE = """import curve_math;
namespace shape_curve {
    [Differentiable]
    public float bendCurve(float x) {
        float xx = x * x;
        return xx * x + curve_math::modulation(xx);
    }
}
"""
CURVE_MATH = """namespace curve_math {
    [Differentiable]
    public float modulation(float x) { return 0.25f * sin(x); }
}
"""


class _Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cubic = nn.SlangUnary(SOURCE, entry_point="bend")
        self.quadratic = nn.SlangUnary("[Differentiable] public float bend(float x) { return x * x + 2.0f * x; }",
                                       entry_point="bend")

    def forward(self, x):
        return self.cubic(x), self.quadratic(x)


def check_distinct_definitions(model):
    values = np.array([-.75, .2, 1.5], dtype=np.float32)
    x = nn.tensor(values, device="gpu", requires_grad=True)
    cubic, quadratic = model(x)
    np.testing.assert_allclose(cubic.detach().cpu().numpy(), values**3 + .25 * np.sin(values**2), atol=5e-6)
    np.testing.assert_allclose(quadratic.detach().cpu().numpy(), values**2 + 2. * values, atol=5e-6)
    seed = nn.ones_like(x)
    dx, = nn.autograd.grad((cubic, quadratic), x, grad_outputs=(seed, seed))
    np.testing.assert_allclose(dx.cpu().numpy(), 3. * values**2 + .5 * values * np.cos(values**2) + 2. * values + 2.,
                               atol=5e-6)
    directions = np.array([0.5, -0.25, 0.75], dtype=np.float32)
    _, (dcubic, dquadratic) = nn.autograd.jvp(model, x, nn.tensor(directions, device="gpu"))
    np.testing.assert_allclose(dcubic.cpu().numpy(), directions * (3. * values**2 + .5 * values * np.cos(values**2)),
                               atol=5e-6)
    np.testing.assert_allclose(dquadratic.cpu().numpy(), directions * (2. * values + 2.), atol=5e-6)


def check_gradients(model):
    values = np.linspace(-1.25, 1.5, 131, dtype=np.float32).reshape(-1, 1)
    seeds = np.linspace(0.4, -0.25, len(values), dtype=np.float32).reshape(-1, 1)
    x = nn.tensor(values, device="gpu", requires_grad=True)
    y = model(x)
    latent = values * WEIGHT + BIAS
    expected = latent ** 3 + .25 * np.sin(latent ** 2)
    pull = seeds * (3. * latent ** 2 + .5 * latent * np.cos(latent ** 2))
    with patch.object(_plan, "_execute_work", wraps=_plan._execute_work) as execute:
        dx, dw, db = nn.autograd.grad(y, (x, *model.parameters()), nn.tensor(seeds, device="gpu"))
        assert execute.call_count == 1
    assert all(value.device == "gpu" for value in (y, dx, dw, db))
    for observed, wanted in ((y, expected), (dx, WEIGHT * pull),
                              (dw, (values * pull).sum().reshape(1, 1)), (db, pull.sum().reshape(1))):
        np.testing.assert_allclose(observed.detach().cpu().numpy(), wanted, rtol=1e-4, atol=5e-6)


def check_directional_derivative(model):
    values = np.linspace(-1.25, 1.5, 131, dtype=np.float32).reshape(-1, 1)
    directions = np.linspace(-0.7, 0.5, len(values), dtype=np.float32).reshape(-1, 1)
    x = nn.tensor(values, device="gpu")
    dx = nn.tensor(directions, device="gpu")
    with patch.object(_jvp, "_execute_work", wraps=_jvp._execute_work) as execute:
        prediction, tangent = nn.autograd.jvp(model, x, dx)
        assert execute.call_count == 1
    assert all(value.device == "gpu" and not value.requires_grad for value in (prediction, tangent))
    latent = WEIGHT * values + BIAS
    expected = latent**3 + 0.25 * np.sin(latent**2)
    derivative = (3. * latent**2 + .5 * latent * np.cos(latent**2)) * WEIGHT * directions
    np.testing.assert_allclose(prediction.cpu().numpy(), expected, rtol=1e-4, atol=5e-6)
    np.testing.assert_allclose(tangent.cpu().numpy(), derivative, rtol=1e-4, atol=5e-6)


def main():
    with TemporaryDirectory(prefix="infernux-authored-nn-") as directory:
        project = Path(directory)
        for name in ("Assets", "ProjectSettings"):
            (project / name).mkdir()
        engine = inx.Engine(mode=RuntimeMode.Headless)
        engine.init_headless(str(project), gpu=True)
        try:
            libraries = {"shape_curve": "Assets/shape_curve.slang", "curve_math": "Assets/curve_math.slang"}
            entry, leaf = (project / path for path in libraries.values())
            entry.write_text(SHAPE_CURVE, encoding="utf-8")
            leaf.write_text(CURVE_MATH, encoding="utf-8")
            model = nn.Sequential(nn.Linear(1, 1, device="gpu"),
                                  nn.SlangUnary(LINKED_SOURCE, entry_point="bend", libraries=libraries))
            leaf.write_text(CURVE_MATH.replace("0.25f", "0.5f"), encoding="utf-8")
            variant = nn.SlangUnary(LINKED_SOURCE, entry_point="bend", libraries=libraries)
            changed = variant(nn.tensor([1.], device="gpu")).cpu().numpy()
            np.testing.assert_allclose(changed, [1.0 + 0.5 * np.sin(1.0)], atol=5e-6)
            model.load_state_dict({"0.weight": nn.tensor([[WEIGHT]]), "0.bias": nn.tensor([BIAS])})
            check_gradients(model)

            exported = str(project / "Assets" / "authored.inm")
            baked = str(project / "Assets" / "baked.inm")
            nn.export(model, exported)
            assert not model._plan._tangent_programs
            with patch.object(_jvp, "_execute_work", side_effect=AssertionError("bake executed tangent Work")):
                nn.bake(model, baked)
            leaf.write_text(CURVE_MATH.replace("0.25f * sin(x)", "missing(x)"), encoding="utf-8")
            invalid = nn.SlangUnary(LINKED_SOURCE, entry_point="bend", libraries=libraries)
            with np.testing.assert_raises(_Infernux.ShaderCompilationError) as failure:
                invalid(nn.tensor([1.], device="gpu"))
            assert any(row["path"] == libraries["curve_math"] and row["line"] == 3
                       for row in failure.exception.diagnostics)
            entry.unlink()
            leaf.unlink()
            with patch.object(slang, "compiler_cache_root", return_value=project / "cold-model-cache"):
                source_model = nn.load(exported, device="gpu")
                check_gradients(source_model)
                check_directional_derivative(source_model)
            with patch.object(slang, "compile_kernel", side_effect=AssertionError("baked code asked for compilation")):
                baked_model = nn.load(baked, device="gpu")
                check_gradients(baked_model)
                check_directional_derivative(baked_model)

            parallel = _Parallel()
            check_distinct_definitions(parallel)
            parallel_baked = str(project / "Assets" / "parallel.inm")
            nn.bake(parallel, parallel_baked)
            with patch.object(slang, "compile_kernel", side_effect=AssertionError("baked code asked for compilation")):
                check_distinct_definitions(nn.load(parallel_baked, device="gpu"))

            cpu = nn.tensor([1.])
            with np.testing.assert_raises_regex(inx.compute.ComputeCapabilityError, "requires a GPU tensor"):
                nn.SlangUnary(SOURCE, entry_point="bend")(cpu)
            with np.testing.assert_raises_regex(TypeError, "resource-free"):
                nn.SlangUnary("StructuredBuffer<float> table;\n"
                              "[Differentiable] public float forward(float x) { return x + table[0]; }")(
                                  nn.tensor([1.], device="gpu"))
            print("Authored Slang NN forward/JVP/VJP and portable/baked INM passed", flush=True)
        finally:
            inx.compute._release_engine_resources()
            engine.get_native_engine().cleanup()


if __name__ == "__main__":
    main()
