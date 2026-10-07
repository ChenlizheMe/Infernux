"""Execute the production culler GLSL and check visible-index conservation."""
from pathlib import Path
import re
import struct

import numpy as np
import pytest

from infernux.lib import _Infernux


@pytest.fixture(scope="module")
def cull_kernels(engine):
    path = Path(__file__).resolve().parents[2] / "cpp/infernux/function/renderer/particle/ParticleGpuCuller.cpp"
    source = path.read_text(encoding="utf-8")
    common = re.search(r'CommonBindings\s*=\s*R"glsl\((.*?)\)glsl";', source, re.S).group(1)
    sources = {}
    for name in ("Reset", "Cull", "Finalize"):
        match = re.search(
            rf'GpuParticleCullShaderSources::{name}\(\).*?BuildShader\("([^"]+)", R"glsl\((.*?)\)glsl"\)',
            source, re.S,
        )
        assert match is not None
        sources[name] = "#version 450\n" + match[1].replace(r"\n", "\n") + common + match[2]
    compiled = _Infernux._compile_compute_glsl_batch(sources, "test-particle-cull-conservation")
    host = engine._acquire_compute_host()
    kernels = {name: host.create_kernel(code, buffer_binding_count=9, push_constant_bytes=112)
               for name, code in compiled.items()}
    state = [host, kernels]
    yield state
    for kernel in kernels.values():
        kernel.wait()
    kernels.clear()
    state.clear()
    kernel = host = None


def _ordered_float(value):
    bits, = struct.unpack("<I", struct.pack("<f", value))
    return ~bits & 0xFFFFFFFF if bits & 0x80000000 else bits ^ 0x80000000


@pytest.mark.parametrize("count, case", [
    (0, "inside"), (1, "unbounded"), (255, "unbounded"), (256, "unbounded"),
    (257, "unbounded"), (4096, "inside"), (1048576, "unbounded"),
    (1048576, "inside"), (1048576, "partial"), (1048576, "outside"),
    (1, "ribbon"), (257, "ribbon"), (4096, "ribbon_breaks"),
    (1048576, "ribbon"), (4096, "invalid_indices"), (257, "tangent"),
])
def test_cull_preserves_every_visible_source_index(cull_kernels, count, case):
    host, kernels = cull_kernels
    capacity = max(count, 1)
    ribbon = case.startswith("ribbon")
    visibility = np.zeros((capacity, 4), np.float32)
    visibility[:, 3] = 0.25
    if case == "partial":
        visibility[1::2, 0] = 4
    elif case == "outside":
        visibility[:, 0] = 4
    elif case == "tangent":
        visibility[:, 0] = 1.25
    indices = np.arange(capacity, dtype=np.uint32)[::-1].copy()
    if case == "invalid_indices":
        indices[::3] = capacity + 5
    metadata = np.zeros((capacity if ribbon else 1, 28), np.uint32)
    if case == "ribbon_breaks":
        metadata[:, 16] = np.arange(capacity, dtype=np.uint32) // 17
        metadata[::11, 18] = 1
    bounds = np.zeros(8, np.uint32)
    if case not in ("unbounded", "invalid_indices", "tangent"):
        bounds[:3] = [_ordered_float(float(visibility[:, axis].min() - 0.25)) for axis in range(3)]
        bounds[3:6] = [_ordered_float(float(visibility[:, axis].max() + 0.25)) for axis in range(3)]
        bounds[6] = 1
    source_count = max(count - 1, 0) if ribbon else count
    if ribbon:
        expected = np.arange(source_count, dtype=np.uint32)
        if case == "ribbon_breaks":
            first, second = indices[:source_count], indices[1:source_count + 1]
            expected = expected[(first // 17 == second // 17) & (second % 11 != 0)]
    else:
        expected = indices[:count]
        expected = expected[expected < capacity]
        if case == "partial":
            expected = expected[expected % 2 == 0]
        elif case == "outside":
            expected = expected[:0]
    expected = np.sort(expected)
    arrays = [visibility.reshape(-1),
              np.array([6 * source_count if ribbon else 6, count, 7, 9], np.uint32),
              np.full(capacity, 0xFFFFFFFF, np.uint32), np.full(4, 0xBAD, np.uint32),
              np.full(5, 0xBAD, np.uint32), bounds, np.array([0, 1, 0, 0], np.uint32),
              indices, metadata.reshape(-1)]
    planes = [1, 0, 0, 1, -1, 0, 0, 1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 0, 1, 1, 0, 0, -1, 1]
    constants = struct.pack("<24f4I", *planes, capacity, 6, int(ribbon), 0)
    buffers = []
    dispatches = None
    buffer = kernel = None
    try:
        for array in arrays:
            buffer = host.create_buffer(array.size, str(array.dtype))
            buffer.set_bytes(array.tobytes())
            buffers.append(buffer)
        groups = 0 if case == "outside" else (source_count + 255) // 256
        stages = [("Reset", 1), *([("Cull", groups)] if groups else []), ("Finalize", 1)]
        accesses = ["read", "read", "read_write", "read_write", "read_write",
                    "read", "read_write", "read", "read"]
        dispatches = [(kernels[name], buffers, accesses, constants, groups, 1, 1)
                      for name, groups in stages]
        for iteration in range(4):
            host.dispatch_batch(dispatches)
            arguments = np.frombuffer(buffers[3].get_bytes(16), np.uint32)
            dispatch = np.frombuffer(buffers[4].get_bytes(20), np.uint32)
            assert arguments.tolist() == [6, len(expected), 7, 9], (iteration, case, arguments)
            assert dispatch[:4].tolist() == [(len(expected) + 255) // 256, 1, 1, source_count]
            actual = np.frombuffer(buffers[2].get_bytes(capacity * 4), np.uint32)[:len(expected)]
            np.testing.assert_array_equal(np.sort(actual), expected)
    finally:
        for kernel in kernels.values():
            kernel.wait()
        dispatches = None
        buffers.clear()
        buffer = kernel = host = None
