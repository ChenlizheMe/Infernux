"""Web work-item bounds must match the native source-declared execution domain."""
import importlib.machinery
import importlib.util
import linecache
import py_compile

import numpy as np
import pytest

import infernux as inx
from infernux._compiler.source_metadata import embed_compute_sources
from infernux._compiler.taichi import frontend
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.build.compute_cpu import build_cpu_compute_source
from infernux.engine.project_context import using_project_root


def source(order, decorator):
    names = ["output", "weights", "amount"]
    names.insert(order, "domain")
    imports = "import infernux as inx\n"
    if decorator == "gpu.kernel":
        imports += "import infernux.compute as gpu\n"
    elif decorator == "kernel":
        imports += "from infernux.compute import kernel\n"
    return (imports + f"@{decorator}\ndef solve({', '.join(names)}):\n"
            "    i = inx.compute.index(domain)\n    j = 0\n"
            "    while j < 1:\n        output[i] += weights[i] * amount\n        j += 1\n")


def source_less_kernel(path, text):
    path.write_text(text, encoding="utf-8")
    bytecode = path.with_suffix(".pyc")
    py_compile.compile(str(path), cfile=str(bytecode), doraise=True)
    path.unlink()
    linecache.clearcache()
    loader = importlib.machinery.SourcelessFileLoader("DomainProbe", str(bytecode))
    spec = importlib.util.spec_from_loader("DomainProbe", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.solve


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("domain_size", [2, 4])
@pytest.mark.parametrize("decorator", ["inx.compute.kernel", "gpu.kernel", "kernel"])
def test_source_less_web_and_native_dispatch_agree(tmp_path, engine, monkeypatch, order, domain_size, decorator):
    text = source(order, decorator)
    namespace = {"__name__": "DomainProbe"}
    exec(compile(embed_compute_sources(text), "<domain-native>", "exec"), namespace)
    descriptors = [_gpu_buffer_descriptor((5,), np.int32), _gpu_buffer_descriptor((5,), np.int32), 2]
    descriptors.insert(order, _gpu_buffer_descriptor((domain_size,), np.int32))
    with using_project_root(tmp_path):
        artifact = frontend.compile_kernel(namespace["solve"].function, tuple(descriptors))
    assert artifact.domain_parameter == order
    expected = np.zeros(5, dtype=np.int32)
    expected[:domain_size] = np.arange(1, domain_size + 1) * 2

    # Real native Vulkan dispatch is the independent numeric oracle.
    host = engine._acquire_compute_host()
    monkeypatch.setattr(inx.compute, "_native_compute_host", lambda: host)
    buffers = [inx.buffer(shape=5, dtype=np.int32, device="gpu"),
               inx.buffer(shape=5, dtype=np.int32, device="gpu", data=[1, 2, 3, 4, 5]),
               inx.buffer(shape=domain_size, dtype=np.int32, device="gpu")]
    params = [buffers[0], buffers[1], 2]
    params.insert(order, buffers[2])
    executable = inx.compute._KernelExecutable(artifact, host, tuple(params))
    try:
        executable.launch(tuple(params))
        readback = buffers[0].get_data()
        try:
            np.testing.assert_array_equal(readback.numpy(), expected)
        finally:
            readback.close()
    finally:
        executable.close()
        for buffer in buffers:
            buffer.close()
        host._release_lease()

    cooked = build_cpu_compute_source(text)
    declaration = source_less_kernel(tmp_path / "Cooked.py", cooked)
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")
    monkeypatch.setattr(frontend, "_function_source", lambda *_: pytest.fail("cooked domain must not parse source at runtime"))
    buffers = [inx.buffer(shape=5, dtype=np.int32, device="gpu"),
               inx.buffer(shape=5, dtype=np.int32, device="gpu", data=[1, 2, 3, 4, 5]),
               inx.buffer(shape=domain_size, dtype=np.int32, device="gpu")]
    params = [buffers[0], buffers[1], 2]
    params.insert(order, buffers[2])
    try:
        for iteration in (1, 2):
            inx.compute.launch(declaration, tuple(params))
            np.testing.assert_array_equal(buffers[0].numpy(), expected * iteration)
    finally:
        for buffer in buffers:
            buffer.close()


@pytest.mark.parametrize("bad_domain", ["scalar", "rank2"])
def test_web_rejects_invalid_domain_before_modifying_output(tmp_path, monkeypatch, bad_domain):
    declaration = source_less_kernel(tmp_path / "Cooked.py", build_cpu_compute_source(source(3, "inx.compute.kernel")))
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")
    output = inx.buffer(shape=5, dtype=np.int32, device="gpu")
    weights = inx.buffer(shape=5, dtype=np.int32, device="gpu", data=[1, 2, 3, 4, 5])
    domain = 3 if bad_domain == "scalar" else inx.buffer(shape=(2, 2), dtype=np.int32, device="gpu")
    try:
        with pytest.raises(TypeError, match="one-dimensional GPU"):
            inx.compute.launch(declaration, (output, weights, 2, domain))
        np.testing.assert_array_equal(output.numpy(), np.zeros(5, dtype=np.int32))
    finally:
        output.close()
        weights.close()
        if isinstance(domain, inx.compute.Buffer):
            domain.close()


def test_uncooked_engine_kernel_resolves_domain_once(monkeypatch):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    @inx.compute.kernel
    def solve(output, domain):
        i = inx.compute.index(domain)
        output[i] += 1

    output = inx.buffer(shape=5, dtype=np.int32, device="gpu")
    first = inx.buffer(shape=2, dtype=np.int32, device="gpu")
    second = inx.buffer(shape=3, dtype=np.int32, device="gpu")
    try:
        inx.compute.launch(solve, (output, first))
        monkeypatch.setattr(frontend, "_function_source", lambda *_: pytest.fail("domain parsed more than once"))
        inx.compute.launch(solve, (output, second))
        np.testing.assert_array_equal(output.numpy(), [2, 2, 1, 0, 0])
    finally:
        output.close()
        first.close()
        second.close()
