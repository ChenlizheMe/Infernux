"""Real Vulkan round-trip for the engine-owned public compute buffer."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import Infernux as inx
from Infernux.lib import _Infernux as native_api


def verify_async_readback() -> None:
    expected = np.arange(8192, dtype=np.float32)
    values = inx.buffer(shape=len(expected), dtype=np.float32, device="gpu", data=expected)
    try:
        for _ in range(12):
            abandoned = values.get_data_async()
            before = inx.compute.statistics()
            assert abandoned.cancel()
            assert abandoned.done and abandoned.cancelled
            assert not abandoned.cancel()
            after = inx.compute.statistics()
            assert after.wait_count == before.wait_count
            assert after.host_map_count == before.host_map_count
            try:
                abandoned.get_data()
            except RuntimeError as error:
                assert "cancelled" in str(error)
            else:
                raise AssertionError("Cancelled result remained readable")
        retained = values.get_data_async(offset=17, count=257)
        values.close()
        np.testing.assert_array_equal(retained.get_data().numpy(), expected[17:274])
        assert retained.done and not retained.cancelled
        assert not retained.cancel()
    finally:
        values.close()


def verify_transform_write_tracking() -> None:
    transform = SimpleNamespace(
        position=inx.vector3(0, 0, 0), rotation=inx.quaternion.identity,
        local_scale=inx.vector3(1, 1, 1),
    )
    owner = SimpleNamespace(game_object=SimpleNamespace(id=42, handle=object()))
    data = np.array([[3, 4, 5, 1], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    pose = inx.buffer(shape=3, dtype=inx.vector4, device="gpu", data=data)
    changes = []
    binding = inx.compute.bind_transform(
        owner, pose=pose, on_transform=lambda old, new: changes.append((old, new)),
    )
    try:
        with patch.object(inx.compute, "_resolve_bound_transform", return_value=transform):
            binding._poll()
            binding._readback.get_data()  # Complete the real Vulkan transfer.
            binding._poll()
            assert tuple(transform.position) == (3, 4, 5)
            before = inx.compute.statistics()
            for _ in range(100):
                binding._poll()
            after = inx.compute.statistics()
            assert after.submission_count == before.submission_count
            assert after.readback_request_count == before.readback_request_count
            assert after.staging_allocation_count == before.staging_allocation_count
            assert after.host_map_count == before.host_map_count
            assert after.wait_count == before.wait_count

            # A queued writer must become visible before the serial comparison.
            data[0, :3] = (7, 8, 9)
            with inx.compute.recording():
                pose.set_data(data)
                binding._poll()
            binding._readback.get_data()
            binding._poll()
            assert tuple(transform.position) == (7, 8, 9)
            assert not changes

            # Editing an idle anchor still updates position, rotation and scale.
            transform.position = inx.vector3(10, 11, 12)
            transform.rotation = inx.quaternion.euler(10, 20, 30)
            transform.local_scale = inx.vector3(2, 3, 4)
            binding._poll()
            binding._readback.get_data()
            binding._poll()
            assert len(changes) == 1
            np.testing.assert_allclose(pose.get_data().numpy(), inx.compute._pose_array(changes[0][1]))
            assert tuple(transform.position) == (10, 11, 12)
            assert tuple(transform.local_scale) == (2, 3, 4)

            # A new write arriving while an earlier snapshot is pending must
            # schedule another read instead of losing that newer revision.
            data[0, :3] = (20, 21, 22)
            pose.set_data(data)
            binding._poll()
            data[0, :3] = (30, 31, 32)
            pose.set_data(data)
            binding._readback.get_data()
            binding._poll()
            assert tuple(transform.position) == (20, 21, 22)
            binding._readback.get_data()
            binding._poll()
            assert tuple(transform.position) == (30, 31, 32)
    finally:
        binding.close()
        pose.close()


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

            verify_async_readback()
            verify_transform_write_tracking()
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
            write_serial = scalar_values._native.last_write_serial
            host.dispatch_batch([dispatch, dispatch])
            assert scalar_values._native.last_write_serial > write_serial
            write_serial = scalar_values._native.last_write_serial
            np.testing.assert_array_equal(
                scalar_values.get_data().numpy(), scalar_source * np.float32(6.25)
            )
            assert scalar_values._native.last_write_serial == write_serial

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
            scale_serial = scale._native.last_write_serial
            host.dispatch_batch([
                (sparse_kernel, [scale._native, scalar_values._native], ["read", "read_write"], b"", 5, 1, 1),
            ])
            assert scale._native.last_write_serial == scale_serial
            np.testing.assert_array_equal(
                scalar_values.get_data().numpy(), scalar_source * np.float32(25)
            )
            sparse_kernel.wait()
            sparse_kernel = None
            scale.close()
            # An abandoned request also retires correctly when shutdown, not a
            # subsequent frame, is the next queue completion boundary.
            shutdown_source = inx.buffer(shape=32, dtype=np.int32, device="gpu", data=np.arange(32, dtype=np.int32))
            shutdown_request = shutdown_source.get_data_async()
            shutdown_source.close()
            assert shutdown_request.cancel()
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
