"""Unified compute entry and real CPU compilation regressions."""

from __future__ import annotations

import ast
import numpy as np
import pytest

import infernux as inx
from infernux import _jit_kernels as kernels


def test_web_cpu_cook_vectorizes_scalar_range_and_atomic_reductions(monkeypatch):
    from infernux.engine.build.compute_cpu import build_cpu_compute_source

    source = """
import infernux as inx

@inx.compute.kernel
def solve(domain, positions, limits, shifts, bins, totals, contacts, count):
    i = inx.compute.index(domain)
    px = positions[i][0]
    for barrier in range(count):
        if px < limits[barrier]:
            px += shifts[barrier]
            inx.compute.atomic_add(totals[bins[i]][0], px)
            inx.compute.atomic_add(contacts[0], 1)
    positions[i][0] = px
"""
    cooked = build_cpu_compute_source(source)
    assert "@inx.compute._cpu_kernel" in cooked
    namespace = {}
    exec(compile(cooked, "<web-cpu-range-test>", "exec"), namespace)
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    domain = inx.buffer(shape=4, dtype=np.int32, device="gpu")
    positions = inx.buffer(
        shape=4,
        dtype=inx.vector3,
        device="gpu",
        data=np.array(((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)), np.float32),
    )
    limits = inx.buffer(shape=2, dtype=np.float32, device="gpu", data=(2.5, 4.0))
    shifts = inx.buffer(shape=2, dtype=np.float32, device="gpu", data=(1.0, 0.5))
    bins = inx.buffer(shape=4, dtype=np.int32, device="gpu", data=(0, 1, 0, 1))
    totals = inx.buffer(shape=2, dtype=inx.vector3, device="gpu")
    contacts = inx.buffer(shape=1, dtype=np.int32, device="gpu")

    inx.compute.launch(
        namespace["solve"],
        (domain, positions, limits, shifts, bins, totals, contacts, 2),
    )

    np.testing.assert_allclose(positions.numpy(copy=False)[:, 0], (1.5, 2.5, 3.5, 3.5))
    np.testing.assert_allclose(totals.numpy(copy=False)[:, 0], (9.0, 8.0))
    assert int(contacts.numpy(copy=False)[0]) == 7


def test_web_cpu_atomic_add_preserves_lane_and_scalar_reductions(monkeypatch):
    from infernux.engine.build.compute_cpu import build_cpu_compute_source

    source = """
import infernux as inx

@inx.compute.kernel
def accumulate(domain, bins, values, vectors, count):
    i = inx.compute.index(domain)
    inx.compute.atomic_add(vectors[bins[i]][0], values[i][0])
    inx.compute.atomic_add(vectors[bins[i]][1], values[i][1])
    inx.compute.atomic_add(vectors[bins[i]][2], values[i][2])
    inx.compute.atomic_add(count[0], 1)
"""
    cooked = build_cpu_compute_source(source)
    assert cooked.count("_cpu_atomic_add") == 4
    namespace = {}
    exec(compile(cooked, "<web-cpu-atomic-test>", "exec"), namespace)
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    domain = inx.buffer(shape=4, dtype=np.int32, device="gpu")
    bins = inx.buffer(shape=4, dtype=np.int32, device="gpu", data=(0, 1, 0, 1))
    values = inx.buffer(
        shape=4,
        dtype=inx.vector3,
        device="gpu",
        data=np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    vectors = inx.buffer(shape=2, dtype=inx.vector3, device="gpu")
    count = inx.buffer(shape=1, dtype=np.int32, device="gpu")

    inx.compute.launch(namespace["accumulate"], (domain, bins, values, vectors, count))

    np.testing.assert_allclose(vectors.numpy(copy=False), ((6.0, 8.0, 10.0), (12.0, 14.0, 16.0)))
    assert int(count.numpy(copy=False)[0]) == 4


def test_web_cpu_atomic_add_never_indexes_inactive_lanes(monkeypatch):
    from infernux.engine.build.compute_cpu import build_cpu_compute_source

    source = """
import infernux as inx

@inx.compute.kernel
def accumulate(domain, bins, totals):
    i = inx.compute.index(domain)
    if i < 2:
        inx.compute.atomic_add(totals[bins[i]], 1)
"""
    cooked = build_cpu_compute_source(source)
    namespace = {}
    exec(compile(cooked, "<web-cpu-masked-atomic-test>", "exec"), namespace)
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    domain = inx.buffer(shape=4, dtype=np.int32, device="gpu")
    bins = inx.buffer(shape=4, dtype=np.int32, device="gpu", data=(0, 1, 99, 99))
    totals = inx.buffer(shape=2, dtype=np.int32, device="gpu")

    inx.compute.launch(namespace["accumulate"], (domain, bins, totals))

    np.testing.assert_array_equal(totals.numpy(copy=False), (1, 1))


def _execute_pair_lowered(source: str, *args):
    from infernux._compiler.taichi.frontend import _lower_serial_loops

    module = ast.parse(source)
    definition = module.body[0]
    assert isinstance(definition, ast.FunctionDef)
    _lower_serial_loops(definition)
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "<pair-lowered-test>", "exec"), namespace)
    return namespace[definition.name](*args)


def test_gpu_dynamic_range_pair_lowering_preserves_control_and_else():
    source = """
def probe(limit):
    values = []
    for index in range(limit):
        if index == 1:
            continue
        values.append(index)
        if index == 4:
            break
    else:
        values.append(99)
    return values
"""
    for limit in (0, 1, 4, 8):
        namespace = {}
        exec(compile(source, "<ordinary-range-test>", "exec"), namespace)
        expected = namespace["probe"](limit)
        assert _execute_pair_lowered(source, limit) == expected


def test_gpu_dynamic_negative_range_pair_lowering_preserves_order():
    source = """
def probe(start, stop):
    values = []
    for index in range(start, stop, -2):
        values.append(index)
    return values
"""
    for arguments in ((7, -2), (2, 2), (-3, -8)):
        namespace = {}
        exec(compile(source, "<ordinary-negative-range-test>", "exec"), namespace)
        expected = namespace["probe"](*arguments)
        assert _execute_pair_lowered(source, *arguments) == expected


def test_gpu_dynamic_while_pair_lowering_preserves_control_and_else():
    source = """
def probe(limit):
    index = -1
    values = []
    while index < limit:
        index += 1
        if index == 1:
            continue
        values.append(index)
        if index == 4:
            break
    else:
        values.append(99)
    return values
"""
    for limit in (-2, 0, 3, 8):
        namespace = {}
        exec(compile(source, "<ordinary-while-test>", "exec"), namespace)
        expected = namespace["probe"](limit)
        assert _execute_pair_lowered(source, limit) == expected


def test_buffer_cpu_typed_storage_and_explicit_snapshot():
    values = inx.buffer(shape=4, dtype=inx.vector3, device="cpu")
    values.set_data(np.arange(12, dtype=np.float32).reshape(4, 3))
    assert values.shape == (4,)
    assert values.dtype == "vector3"
    assert values.nbytes == 48
    assert values[2].x == 6.0 and values[2].z == 8.0

    snapshot = values.get_data()
    values[0] = inx.vector3(20.0, 21.0, 22.0)
    np.testing.assert_array_equal(snapshot.numpy(), np.arange(12, dtype=np.float32).reshape(4, 3))
    assert snapshot[0].x == 0.0


def test_transform_binding_publishes_completed_buffer_pose(monkeypatch):
    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        def __init__(self):
            self._transform = Transform()

        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()

        @property
        def transform(self):
            return self._transform

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="cpu",
        data=np.array(
            [
                [3.0, 4.0, 5.0, 1.0],
                [0.0, 0.0, 0.7071068, 0.7071068],
                [2.0, 3.0, 4.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    # The first barrier starts a snapshot; the next one publishes its completed
    # value. GPU bindings follow the same non-blocking control-plane contract.
    binding._poll()
    assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    binding._poll()
    assert tuple(owner.transform.position) == (3.0, 4.0, 5.0)
    assert tuple(owner.transform.local_scale) == (2.0, 3.0, 4.0)
    assert (
        inx.quaternion.angle(
            owner.transform.rotation,
            inx.quaternion(0.0, 0.0, 0.7071068, 0.7071068),
        )
        < 0.01
    )
    # CPU authors may write through NumPy directly, without a GPU write ticket.
    pose.numpy(copy=False)[0, :3] = (6.0, 7.0, 8.0)
    binding._poll()
    binding._poll()
    assert tuple(owner.transform.position) == (6.0, 7.0, 8.0)
    binding.close()
    assert binding.closed


def test_transform_binding_requires_three_vector4_pose_rows():
    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()

    with pytest.raises(ValueError, match="exactly three vector4"):
        inx.compute.bind_transform(
            Owner(),
            pose=inx.buffer(shape=1, dtype=inx.vector3, device="cpu"),
        )


def test_web_mapped_transform_binding_publishes_current_pose_without_readback_lag(
    monkeypatch,
):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()
        transform = Transform()

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="gpu",
        data=np.array(
            [[3.0, 4.0, 5.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    binding._poll()

    assert pose._cpu_mapped
    assert tuple(owner.transform.position) == (3.0, 4.0, 5.0)


def test_web_mapped_transform_binding_coalesces_to_final_native_frame_pose(
    monkeypatch,
):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()
        transform = Transform()

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="gpu",
        data=np.array(
            [[1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    inx.compute._begin_cpu_mesh_frame()
    try:
        binding._poll()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
        pose.numpy(copy=False)[0, :3] = (4.0, 5.0, 6.0)
        binding._poll()
        assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    finally:
        inx.compute._end_cpu_mesh_frame()

    assert tuple(owner.transform.position) == (4.0, 5.0, 6.0)


def test_web_mapped_transform_authored_edit_wins_at_frame_boundary(monkeypatch):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()
        transform = Transform()

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="gpu",
        data=np.array(
            [[1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    authored = []
    binding = inx.compute.bind_transform(owner, pose=pose, on_transform=lambda old, new: authored.append((old, new)))

    binding._poll()
    inx.compute._begin_cpu_mesh_frame()
    try:
        pose.numpy(copy=False)[0, :3] = (4.0, 5.0, 6.0)
        binding._poll()
        owner.transform.position = inx.vector3(7.0, 8.0, 9.0)
    finally:
        inx.compute._end_cpu_mesh_frame()

    np.testing.assert_allclose(pose.numpy(copy=False)[0, :3], (7.0, 8.0, 9.0))
    assert len(authored) == 1
    assert authored[0][0].position == (1.0, 2.0, 3.0)
    assert authored[0][1].position == (7.0, 8.0, 9.0)


@pytest.mark.parametrize("retirement", ["destroyed", "collected"])
def test_transform_binding_retires_with_owner_not_only_game_object(monkeypatch, retirement):
    import gc
    from types import SimpleNamespace

    class Owner:
        game_object = SimpleNamespace(id=42, handle=object())
        _is_destroyed = False

    transform = SimpleNamespace(
        position=inx.vector3(0, 0, 0),
        rotation=inx.quaternion.identity,
        local_scale=inx.vector3(1, 1, 1),
    )
    resolved = []
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *args: resolved.append(args) or transform)
    data = np.array([[3, 4, 5, 1], [0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    pose = inx.buffer(shape=3, dtype=inx.vector4, device="cpu", data=data)
    owner = Owner()
    binding = inx.compute.bind_transform(owner, pose=pose)
    binding._poll()  # A completed snapshot is waiting to be published.
    resolved.clear()
    if retirement == "destroyed":
        owner._is_destroyed = True
    else:
        del owner
        gc.collect()
    binding._poll()
    assert binding.closed and not resolved
    assert tuple(transform.position) == (0, 0, 0)
    assert not pose.closed  # Binding does not own the author's buffer.
    # A new component on exactly the same object identity is unaffected.
    replacement = Owner()
    current = inx.compute.bind_transform(replacement, pose=pose)
    current._poll()
    current._poll()
    assert tuple(transform.position) == (3, 4, 5)
    current.close()
    pose.close()


def test_transform_binding_authored_trs_wins_once_and_updates_pose(monkeypatch):
    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()
        transform = Transform()

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="cpu",
        data=np.array(
            [
                [1.0, 2.0, 3.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    authored = []
    binding = inx.compute.bind_transform(owner, pose=pose, on_transform=lambda old, new: authored.append((old, new)))
    binding._poll()
    binding._poll()

    owner.transform.position = inx.vector3(7.0, 8.0, 9.0)
    owner.transform.rotation = inx.quaternion.euler(10.0, 20.0, 30.0)
    owner.transform.local_scale = inx.vector3(2.0, 3.0, 4.0)
    binding._poll()
    binding._poll()

    assert len(authored) == 1
    assert authored[0][0].position == (1.0, 2.0, 3.0)
    assert authored[0][1].position == (7.0, 8.0, 9.0)
    np.testing.assert_allclose(pose.numpy()[0, :3], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(pose.numpy()[2, :3], [2.0, 3.0, 4.0])
    binding.close()


def test_transform_binding_uses_authored_transform_to_initialize_simulation(monkeypatch):
    class Transform:
        position = inx.vector3(6.0, 7.0, 8.0)
        rotation = inx.quaternion.euler(10.0, 20.0, 30.0)
        local_scale = inx.vector3(2.0, 3.0, 4.0)

    class Owner:
        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()
        transform = Transform()

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: owner.transform)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="cpu",
        data=np.array(
            [
                [0.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    changes = []
    initial = inx.compute.TransformPose(
        position=(0.0, 2.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
    )
    binding = inx.compute.bind_transform(
        owner,
        pose=pose,
        initial_pose=initial,
        on_transform=lambda old, new: changes.append((old, new)),
    )

    # Initialization is synchronous at the binding boundary.  A FixedUpdate
    # cannot run once against the default pose before authored TRS is applied.
    assert len(changes) == 1
    assert changes[0][0] == initial
    assert changes[0][1].position == (6.0, 7.0, 8.0)
    np.testing.assert_allclose(pose.numpy()[0, :3], [6.0, 7.0, 8.0])
    np.testing.assert_allclose(pose.numpy()[2, :3], [2.0, 3.0, 4.0])
    binding.close()


def test_transform_binding_closes_before_touching_a_destroyed_owner(monkeypatch):
    class Transform:
        position = inx.vector3(0.0, 0.0, 0.0)
        rotation = inx.quaternion.identity
        local_scale = inx.vector3(1.0, 1.0, 1.0)

    class Owner:
        def __init__(self):
            self._transform = Transform()

        class GameObject:
            id = 42
            handle = object()

        game_object = GameObject()

        @property
        def transform(self):
            return self._transform

    owner = Owner()
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform", lambda *_: None)
    pose = inx.buffer(
        shape=3,
        dtype=inx.vector4,
        device="cpu",
        data=np.array(
            [
                [3.0, 4.0, 5.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    binding._poll()

    assert binding.closed
    assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("device,web_runtime", [("cpu", False), ("gpu", True)])
def test_owned_buffer_copies_contiguous_initial_storage(monkeypatch, device, web_runtime):
    if web_runtime:
        monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")
    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    first = inx.buffer(shape=4, dtype=inx.vector3, device=device, data=source)
    second = inx.buffer(shape=4, dtype=inx.vector3, device=device, data=source)

    first.numpy(copy=False)[0, 1] = 99.0

    assert source[0, 1] == 1.0
    assert second.numpy(copy=False)[0, 1] == 1.0


def test_cpu_buffer_views_share_ranges_and_survive_owner_close():
    values = inx.buffer(shape=8, dtype=np.int32, device="cpu", data=np.arange(8, dtype=np.int32))
    first = values.view(offset=2, count=4)
    overlap = values.view(offset=4, count=3)

    assert first.description.ownership == "borrowed"
    assert first.description.byte_offset == 8
    assert first.description.byte_length == 16
    assert first.description.generation != values.description.generation
    first.set_data(np.array([20, 21, 22, 23], dtype=np.int32))
    np.testing.assert_array_equal(overlap.numpy(), [22, 23, 6])

    values.close()
    overlap[2] = 60
    np.testing.assert_array_equal(first.numpy(), [20, 21, 22, 23])
    assert overlap[2] == 60


def test_cpu_buffer_async_readback_is_an_immediate_exact_snapshot():
    values = inx.buffer(shape=6, dtype=np.int32, device="cpu", data=np.arange(6, dtype=np.int32))
    readback = values.get_data_async(offset=2, count=3)
    values.set_data([20, 21, 22], offset=2)
    assert readback.done
    np.testing.assert_array_equal(readback.get_data().numpy(), [2, 3, 4])
    output = inx.buffer(shape=3, dtype=np.int32, device="cpu")
    assert readback.get_data(output) is output
    np.testing.assert_array_equal(output.numpy(), [2, 3, 4])
    assert not readback.cancel()
    assert not readback.cancelled


def test_readback_cancel_abandons_without_polling_or_fetching():
    import weakref
    from infernux.compute import Readback

    class Pending:
        @property
        def done(self):
            pytest.fail("cancel must not poll the GPU")

        def get_bytes(self):
            pytest.fail("cancel must not fetch the GPU result")

    task = Pending()
    reference = weakref.ref(task)
    values = inx.buffer(shape=1, dtype=np.int32, device="cpu")
    readback = Readback(values._dtype, (1,), task=task)
    del task
    assert readback.cancel()
    assert reference() is None
    assert readback.cancelled and readback.done
    assert not readback.cancel()
    with pytest.raises(RuntimeError, match="cancelled"):
        readback.get_data()


def test_engine_resource_release_cancels_unconsumed_native_readbacks():
    from infernux.compute import Readback

    class Pending:
        @property
        def done(self):
            pytest.fail("engine teardown must not poll the GPU")

        def get_bytes(self):
            pytest.fail("engine teardown must not fetch the GPU result")

    host = object()
    values = inx.buffer(shape=1, dtype=np.int32, device="cpu")
    readback = Readback(values._dtype, (1,), task=Pending(), host=host)

    assert readback in inx.compute._readbacks
    inx.compute._release_engine_resources()

    assert readback.cancelled
    assert readback._host is None
    assert readback not in inx.compute._readbacks


def test_native_readback_abandonment_and_source_release_preserve_gpu_data(engine):
    host = engine._acquire_compute_host()
    expected = np.arange(1024, dtype=np.int32).tobytes()
    for _ in range(12):
        source = host.create_buffer(1024, "int32")
        source.set_bytes(expected)
        abandoned = source.get_bytes_async(len(expected))
        before = host.get_statistics()
        del abandoned
        after = host.get_statistics()
        assert after["wait_count"] == before["wait_count"]
        assert after["host_map_count"] == before["host_map_count"]
        retained = source.get_bytes_async(len(expected))
        del source
        assert retained.get_bytes() == expected
        del retained


def test_buffer_element_ranges_fill_and_reusable_output():
    values = inx.buffer(shape=6, dtype=inx.vector3, device="cpu")
    values.fill(inx.vector3(1.0, 2.0, 3.0))
    values.set_data(np.array([[8, 9, 10], [11, 12, 13]], np.float32), offset=2)
    out = inx.buffer(shape=2, dtype=inx.vector3, device="cpu")
    assert values.get_data(out, offset=2, count=2) is out
    np.testing.assert_array_equal(out.numpy(), [[8, 9, 10], [11, 12, 13]])
    values.zero()
    np.testing.assert_array_equal(values.numpy(), np.zeros((6, 3), np.float32))

    with pytest.raises(ValueError, match="exceeds"):
        values.set_data(np.ones((2, 3), np.float32), offset=5)
    with pytest.raises(ValueError, match="exceeds"):
        values.get_data(offset=5, count=2)


def test_buffer_gpu_uses_engine_owned_storage_and_never_indexes(monkeypatch):
    from types import SimpleNamespace
    from infernux.application import Application

    class NativeBuffer:
        def __init__(self, byte_size):
            self.payload = bytes(byte_size)
            self.resource_index = 7
            self.resource_generation = 13

        def set_bytes(self, value, offset=0):
            mutable = bytearray(self.payload)
            mutable[offset : offset + len(value)] = value
            self.payload = bytes(mutable)

        def get_bytes(self, byte_size, offset=0):
            return self.payload[offset : offset + byte_size]

    class Host:
        def __init__(self):
            self.created = []
            self.identity = 11

        def create_buffer(self, element_count, scalar_type, lanes=1):
            assert scalar_type == "float32" and lanes == 1
            byte_size = element_count * lanes * 4
            result = NativeBuffer(byte_size)
            self.created.append(result)
            return result

        def dispatch_batch_and_read(self, dispatches, updates, reads):
            assert dispatches == [] and updates == []
            return [native.payload[offset : offset + byte_size] for native, offset, byte_size in reads]

    host = Host()
    acquisitions = []
    native = SimpleNamespace(
        has_renderer=True,
        _acquire_compute_host=lambda: acquisitions.append(host) or host,
    )
    frontend = SimpleNamespace(get_native_engine=lambda: native)
    monkeypatch.setattr(Application, "_current_engine", staticmethod(lambda: frontend))

    values = inx.buffer(shape=4, dtype=np.float32, device="gpu", data=[1, 2, 3, 4])
    other = inx.buffer(shape=1, dtype=np.float32, device="gpu")
    assert len(host.created) == 2
    assert values._host is other._host
    assert acquisitions == [host]
    with pytest.raises(RuntimeError, match="cannot be indexed"):
        _ = values[0]
    with pytest.raises(RuntimeError, match="cannot be indexed"):
        values[0] = 8
    snapshot = values.get_data()
    np.testing.assert_array_equal(snapshot.numpy(), np.array([1, 2, 3, 4], np.float32))
    assert values.description.resource_index == 7
    assert values.description.resource_generation == 13
    assert values.description.usage == ("storage", "uniform", "vertex", "transfer_source", "transfer_destination")


def test_gpu_kernel_rejects_cpu_buffers_without_fallback():
    values = inx.buffer(shape=4, dtype=np.float32, device="cpu")

    @inx.compute.kernel
    def declared(payload):
        i = inx.compute.index(payload)
        payload[i] = 1.0

    with pytest.raises(TypeError, match="transfer CPU data explicitly"):
        inx.compute.launch(declared, params=(values,))


def test_engine_resource_release_closes_module_level_gpu_objects(monkeypatch):
    from types import SimpleNamespace
    from infernux.application import Application

    class NativeBuffer:
        resource_index = 17
        resource_generation = 1

        def set_bytes(self, value, offset=0):
            pass

    host = SimpleNamespace(identity=12, create_buffer=lambda *args: NativeBuffer())
    native = SimpleNamespace(has_renderer=True, _acquire_compute_host=lambda: host)
    monkeypatch.setattr(
        Application,
        "_current_engine",
        staticmethod(lambda: SimpleNamespace(get_native_engine=lambda: native)),
    )
    values = inx.buffer(shape=4, device="gpu")

    @inx.compute.kernel
    def declared(payload):
        i = inx.compute.index(payload)
        payload[i] = 1.0

    class Executable:
        closed = False

        def close(self):
            self.closed = True

    executable = Executable()
    declared._executables[(1, ())] = executable
    inx.compute._release_engine_resources()
    assert values.closed
    assert executable.closed
    assert declared._executables == {}


def test_gpu_kernel_specializations_are_bounded_and_retire_lru(monkeypatch):
    from infernux._compiler.taichi import frontend

    class Host:
        identity = 31

    class CompilerBuffer:
        device = "gpu"
        _host = Host()
        _cpu_mapped = False

    class Executable:
        def __init__(self, artifact, host, _params):
            self.artifact = artifact
            self.host = host
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(inx.compute, "Buffer", CompilerBuffer)
    monkeypatch.setattr(inx.compute, "_KernelExecutable", Executable)
    monkeypatch.setattr(frontend, "parameter_key", lambda params: params[1])
    monkeypatch.setattr(frontend, "compile_kernel", lambda _fn, _params: object())

    declaration = inx.compute.Kernel(lambda *_params: None)
    first = declaration._executable((CompilerBuffer(), (0,)))
    for index in range(1, 65):
        declaration._executable((CompilerBuffer(), (index,)))

    assert len(declaration._executables) == 64
    assert first.closed
    assert (31, (0,)) not in declaration._executables


def test_engine_phase_records_launches_into_one_host_submission():
    submissions = []

    class Host:
        identity = 17

        def dispatch_batch(self, dispatches, updates):
            submissions.append((tuple(dispatches), tuple(updates)))

    host = Host()

    class Executable:
        def __init__(self):
            self.host = host
            self.sequence = 0

        def prepare(self, params):
            self.sequence += 1
            native = object()
            return [(native, bytes([self.sequence]), 0)], [(f"dispatch-{self.sequence}", params)]

    executable = Executable()
    declaration = inx.compute.Kernel(lambda value: None)
    declaration._executable = lambda params: executable

    with inx.compute._record_commands():
        inx.compute.launch(declaration, params=(1,))
        inx.compute.launch(declaration, params=(2,))
        assert submissions == []

    assert len(submissions) == 1
    dispatches, updates = submissions[0]
    assert dispatches == (("dispatch-1", (1,)), ("dispatch-2", (2,)))
    assert [payload for _, payload, _ in updates] == [b"\x01", b"\x02"]


def test_public_recording_scope_batches_ordered_launches():
    submissions = []

    class Host:
        identity = 18

        def dispatch_batch(self, dispatches, updates):
            submissions.append((tuple(dispatches), tuple(updates)))

    host = Host()

    class Executable:
        def __init__(self):
            self.host = host
            self.sequence = 0

        def prepare(self, params):
            self.sequence += 1
            native = object()
            return [(native, bytes([self.sequence]), 0)], [(f"dispatch-{self.sequence}", params)]

    executable = Executable()
    declaration = inx.compute.Kernel(lambda value: None)
    declaration._executable = lambda params: executable

    with inx.compute.recording():
        inx.compute.launch(declaration, params=(1,))
        with inx.compute.recording():
            inx.compute.launch(declaration, params=(2,))
        assert submissions == []

    assert len(submissions) == 1
    dispatches, updates = submissions[0]
    assert dispatches == (("dispatch-1", (1,)), ("dispatch-2", (2,)))
    assert [payload for _, payload, _ in updates] == [b"\x01", b"\x02"]


def test_explicit_readback_joins_pending_compute_submission():
    submissions = []

    class NativeBuffer:
        pass

    class Host:
        identity = 19

        def dispatch_batch_and_read(self, dispatches, updates, reads):
            submissions.append((tuple(dispatches), tuple(updates), tuple(reads)))
            return [np.array([7.0, 8.0], np.float32).tobytes()]

    host = Host()

    class Executable:
        def __init__(self):
            self.host = host

        def prepare(self, params):
            return [(NativeBuffer(), b"input", 0)], [("kernel", params)]

    declaration = inx.compute.Kernel(lambda value: None)
    declaration._executable = lambda params: Executable()
    output = object.__new__(inx.compute.Buffer)
    output._shape = (2,)
    output._dtype = inx.compute._buffer_dtype(np.float32)
    output._storage_shape = (2,)
    output._device = "gpu"
    output._closed = False
    output._host = host
    output._native = NativeBuffer()
    output._array = None
    output._byte_offset = 0
    output._mesh_attribute_state = None
    output._dependent_resources = []

    with inx.compute.recording():
        inx.compute.launch(declaration, params=(1,))
        snapshot = output.get_data()

    assert len(submissions) == 1
    dispatches, updates, reads = submissions[0]
    assert dispatches == (("kernel", (1,)),)
    assert updates[0][1] == b"input"
    assert reads == ((output._native, 0, 8),)
    np.testing.assert_array_equal(snapshot.numpy(), [7.0, 8.0])
