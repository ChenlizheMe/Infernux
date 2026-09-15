"""Real Vulkan round-trip for the engine-owned public compute buffer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import struct

import numpy as np

import Infernux as inx
from Infernux.lib import _Infernux as native_api


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="infernux-compute-buffer-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()

        frontend = inx.Engine()
        native = frontend.get_native_engine()
        values = None
        scalar_values = None
        kernel = None
        host = None
        dispatch = None
        try:
            try:
                frontend.init_renderer(64, 64, str(project))
            except (OSError, RuntimeError) as exception:
                print(f"Compute buffer Vulkan test skipped: {exception}")
                return 77

            expected = np.arange(36, dtype=np.float32).reshape(12, 3)
            values = inx.buffer(shape=12, dtype=inx.vector3, device="gpu", data=expected)
            assert values.nbytes == expected.nbytes
            description = values.description
            assert description.shape == (12,)
            assert description.dtype == "vector3"
            assert description.scalar_dtype == "float32"
            assert description.lanes == 3
            assert description.attribute_offsets == (0, 4, 8)
            assert description.capacity == 12
            assert description.element_stride == 12
            assert description.byte_offset == 0
            assert description.byte_length == expected.nbytes
            assert description.nbytes == expected.nbytes
            assert description.device == "gpu"
            assert description.ownership == "owned"
            assert description.resource_index is not None
            assert description.resource_index >= 0
            assert description.resource_generation is not None
            assert description.resource_generation > 0
            assert values.description == description
            np.testing.assert_array_equal(values.get_data().numpy(), expected)

            expected *= -0.5
            values.set_data(expected)
            reusable = inx.buffer(shape=12, dtype=inx.vector3, device="cpu")
            assert values.get_data(reusable) is reusable
            np.testing.assert_array_equal(reusable.numpy(), expected)

            replacement = np.array([[90, 91, 92], [93, 94, 95]], dtype=np.float32)
            values.set_data(replacement, offset=5)
            expected[5:7] = replacement
            partial = values.get_data(offset=5, count=2)
            np.testing.assert_array_equal(partial.numpy(), replacement)

            source = """#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) buffer Values { float values[]; };
layout(push_constant) uniform Constants { uint count; float scale; } pc;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < pc.count) values[index] *= pc.scale;
}
"""
            spirv = native_api._compile_compute_glsl_batch(
                {"scale": source}, "infernux-compute-kernel-test"
            )["scale"]
            scalar_source = np.arange(257, dtype=np.float32)
            scalar_values = inx.buffer(
                shape=len(scalar_source), dtype=np.float32, device="gpu", data=scalar_source
            )
            host = native._acquire_compute_host()
            kernel = host.create_kernel(spirv, buffer_binding_count=1, push_constant_bytes=8)
            dispatch = (
                kernel,
                [scalar_values._native],
                ["read_write"],
                struct.pack("If", len(scalar_source), 2.5),
                (len(scalar_source) + 63) // 64,
                1,
                1,
            )
            # Multiple solver launches share one command buffer and one queue
            # submission while retaining the same resident binding.
            host.dispatch_batch([dispatch, dispatch])
            np.testing.assert_array_equal(
                scalar_values.get_data().numpy(), scalar_source * np.float32(6.25)
            )

            sparse_source = """#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) buffer Constants { float scale; } args;
layout(std430, set = 0, binding = 2) buffer Values { float values[]; } payload;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < 257) payload.values[index] *= args.scale;
}
"""
            sparse_spirv = native_api._compile_compute_glsl_batch(
                {"sparse": sparse_source}, "infernux-compute-sparse-bindings-test"
            )["sparse"]
            scale = inx.buffer(shape=1, dtype=np.float32, device="gpu", data=np.array([4], np.float32))
            sparse_kernel = host.create_kernel_with_bindings(sparse_spirv, [0, 2])
            assert sparse_kernel.buffer_bindings == [0, 2]
            sparse_kernel.dispatch([scale._native, scalar_values._native], group_count_x=5)
            np.testing.assert_array_equal(
                scalar_values.get_data().numpy(), scalar_source * np.float32(25)
            )
            sparse_kernel.wait()
            sparse_kernel = None
            scale.close()
        finally:
            if kernel is not None:
                kernel.wait()
                dispatch = None
                kernel = None
            if scalar_values is not None:
                scalar_values.close()
            if values is not None:
                values.close()
                assert values.description == description
            host = None
            inx.compute._release_engine_resources()
            native.cleanup()

    print("Compute buffer Vulkan test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
