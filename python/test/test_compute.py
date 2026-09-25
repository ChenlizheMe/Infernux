"""Unified compute entry and real CPU compilation regressions."""

from __future__ import annotations

import ast
import numpy as np
import pytest

import Infernux as inx
from Infernux import jit
from Infernux import _jit_kernels as kernels


def test_gpu_index_declaration_accepts_public_and_imported_spellings():
    from Infernux._compiler.taichi.frontend import _index_declaration

    for source in (
        "i = inx.compute.index(domain)",
        "i = compute.index(domain)",
        "i = index(domain)",
    ):
        statement = ast.parse(source).body[0]
        assert _index_declaration(statement) == ("i", "domain")


def _execute_pair_lowered(source: str, *args):
    from Infernux._compiler.taichi.frontend import _lower_serial_loops

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


def test_gpu_dynamic_range_step_must_be_compile_time_nonzero():
    from Infernux._compiler.taichi.frontend import _lower_serial_loops

    for source, message in (
        ("def probe(stop, step):\n    for i in range(0, stop, step):\n        pass\n",
         "compile-time integer"),
        ("def probe(stop):\n    for i in range(0, stop, 0):\n        pass\n", "cannot be zero"),
    ):
        definition = ast.parse(source).body[0]
        with pytest.raises(TypeError, match=message):
            _lower_serial_loops(definition)


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
        data=np.array([
            [3.0, 4.0, 5.0, 1.0],
            [0.0, 0.0, 0.7071068, 0.7071068],
            [2.0, 3.0, 4.0, 1.0],
        ], dtype=np.float32),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    # The first barrier starts a snapshot; the next one publishes its completed
    # value. GPU bindings follow the same non-blocking control-plane contract.
    binding._poll()
    assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)
    binding._poll()
    assert tuple(owner.transform.position) == (3.0, 4.0, 5.0)
    assert tuple(owner.transform.local_scale) == (2.0, 3.0, 4.0)
    assert inx.quaternion.angle(
        owner.transform.rotation,
        inx.quaternion(0.0, 0.0, 0.7071068, 0.7071068),
    ) < 0.01
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


@pytest.mark.parametrize("retirement", ["destroyed", "collected"])
def test_transform_binding_retires_with_owner_not_only_game_object(monkeypatch, retirement):
    import gc
    from types import SimpleNamespace

    class Owner:
        game_object = SimpleNamespace(id=42, handle=object())
        _is_destroyed = False

    transform = SimpleNamespace(
        position=inx.vector3(0, 0, 0), rotation=inx.quaternion.identity,
        local_scale=inx.vector3(1, 1, 1),
    )
    resolved = []
    monkeypatch.setattr(inx.compute, "_resolve_bound_transform",
                        lambda *args: resolved.append(args) or transform)
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
        data=np.array([
            [1.0, 2.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32),
    )
    authored = []
    binding = inx.compute.bind_transform(
        owner, pose=pose, on_transform=lambda old, new: authored.append((old, new))
    )
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
        data=np.array([
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32),
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
        data=np.array([
            [3.0, 4.0, 5.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=np.float32),
    )
    binding = inx.compute.bind_transform(owner, pose=pose)

    binding._poll()

    assert binding.closed
    assert tuple(owner.transform.position) == (0.0, 0.0, 0.0)


def test_buffer_description_freezes_layout_capacity_and_ownership():
    values = inx.buffer(shape=(3, 5), dtype=inx.vector3, device="cpu")
    description = values.description

    assert description.shape == (3, 5)
    assert description.dtype == "vector3"
    assert description.scalar_dtype == "float32"
    assert description.lanes == 3
    assert description.attribute_offsets == (0, 4, 8)
    assert description.capacity == 15
    assert description.element_stride == 12
    assert description.byte_offset == 0
    assert description.byte_length == 180
    assert description.nbytes == 180
    assert description.device == "cpu"
    assert description.usage == ("host_read", "host_write")
    assert description.access == "read_write"
    assert description.ownership == "owned"
    assert description.generation > 0
    assert description.resource_index is None
    assert description.resource_generation is None

    values.close()
    assert values.description == description


def test_buffer_is_runtime_storage_not_a_serialized_asset_value():
    from Infernux.components.value_codec import ValueCodecRegistry

    values = inx.buffer(shape=4, dtype=np.float32, device="cpu")
    with pytest.raises(TypeError, match="unsupported serialized value type"):
        ValueCodecRegistry().encode(values, "Probe.runtime_state")

    assert values.description.resource_index is None
    assert values.description.resource_generation is None


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


def test_buffer_empty_shape_and_read_only_view_are_explicit():
    with pytest.raises(ValueError, match="positive integers"):
        inx.buffer(shape=0, dtype=np.float32, device="cpu")

    values = inx.buffer(shape=4, dtype=np.int32, device="cpu", data=[1, 2, 3, 4])
    readonly = values.view(offset=1, count=2, readonly=True)
    assert readonly.readonly
    assert readonly.description.access == "read_only"
    np.testing.assert_array_equal(readonly.numpy(), [2, 3])
    with pytest.raises(RuntimeError, match="read-only"):
        readonly[0] = 9
    with pytest.raises(RuntimeError, match="read-only"):
        readonly.set_data([8, 9])


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
    from Infernux.compute import Readback

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
    from Infernux.compute import Readback

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
    from Infernux.application import Application

    class NativeBuffer:
        def __init__(self, byte_size):
            self.payload = bytes(byte_size)
            self.resource_index = 7
            self.resource_generation = 13
        def set_bytes(self, value, offset=0):
            mutable = bytearray(self.payload)
            mutable[offset:offset + len(value)] = value
            self.payload = bytes(mutable)
        def get_bytes(self, byte_size, offset=0):
            return self.payload[offset:offset + byte_size]

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
            return [native.payload[offset:offset + byte_size]
                    for native, offset, byte_size in reads]

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
    assert values.description.usage == (
        "storage", "uniform", "vertex", "transfer_source", "transfer_destination"
    )


def test_buffer_rejects_implicit_layout_or_device_guessing():
    with pytest.raises(TypeError, match="device"):
        inx.buffer(shape=4)
    with pytest.raises(ValueError, match="explicitly"):
        inx.buffer(shape=4, device="automatic")
    with pytest.raises(ValueError, match="data shape"):
        inx.buffer(shape=4, dtype=float, device="cpu", data=[1, 2])


def test_gpu_kernel_declaration_is_inert_until_explicit_launch():
    calls = []

    @inx.compute.kernel
    def declared(values):
        calls.append(values)

    assert isinstance(declared, inx.compute.Kernel)
    assert declared.__name__ == "declared"
    assert calls == []
    with pytest.raises(RuntimeError, match="only inside"):
        inx.compute.index(inx.buffer(shape=1, device="cpu"))
    with pytest.raises(TypeError, match="tuple"):
        inx.compute.launch(declared, params=[])
    assert calls == []


def test_source_less_player_metadata_preserves_kernel_and_helper_source(monkeypatch):
    from Infernux._compiler.source_metadata import embed_compute_sources
    from Infernux._compiler.taichi import frontend

    source = """\
from Infernux import compute as gpu

@gpu.function
def twice(value):
    return value * 2

@gpu.kernel
def scale(values):
    i = gpu.index(values)
    values[i] = twice(values[i])
"""
    cooked = embed_compute_sources(source)
    assert embed_compute_sources(cooked) == cooked
    namespace = {"__name__": "source_less_compute_fixture"}
    exec(compile(cooked, "<source-less-compute-fixture>", "exec"), namespace)
    monkeypatch.setattr(
        frontend.inspect,
        "getsource",
        lambda _value: (_ for _ in ()).throw(OSError("source is absent")),
    )
    assert "def scale(values)" in frontend._function_source(
        namespace["scale"].function
    )
    assert "def twice(value)" in frontend._function_source(
        namespace["twice"].function
    )


def test_source_less_metadata_preserves_engine_owned_bare_kernel(monkeypatch):
    from Infernux._compiler.source_metadata import embed_compute_sources
    from Infernux._compiler.taichi import frontend

    source = """\
def kernel(fn):
    return fn

@kernel
def internal(values):
    values[0] = 1
"""
    cooked = embed_compute_sources(source)
    namespace = {"__name__": "source_less_bare_kernel"}
    exec(compile(cooked, "<source-less-bare-kernel>", "exec"), namespace)
    monkeypatch.setattr(
        frontend.inspect,
        "getsource",
        lambda _value: (_ for _ in ()).throw(OSError("source is absent")),
    )
    assert "def internal(values)" in frontend._function_source(namespace["internal"])


def test_class_kernel_diagnostics_identify_receiver_and_rewrite():
    from Infernux._compiler.taichi import frontend

    class InvalidKernel:
        @inx.compute.kernel
        def step(self, domain):
            i = inx.compute.index(domain)
            domain[i] = 1

    with pytest.raises(TypeError) as error:
        frontend.compile_kernel(InvalidKernel.step.function, (None, None))
    message = str(error.value)
    assert "InvalidKernel.step" in message
    assert "implicit instance receiver 'self'" in message
    assert "@staticmethod" in message
    assert "test_compute.py:" in message


def test_kernel_closure_diagnostic_identifies_captured_value():
    from Infernux._compiler.taichi import frontend

    factor = 2

    @inx.compute.kernel
    def invalid(domain):
        i = inx.compute.index(domain)
        domain[i] = factor

    with pytest.raises(TypeError) as error:
        frontend.compile_kernel(invalid.function, (None,))
    message = str(error.value)
    assert "invalid" in message
    assert "arbitrary Python closure capture" in message
    assert "factor" in message
    assert "test_compute.py:" in message


def test_kernel_rejects_staticmethod_descriptor_in_wrong_decorator_order():
    with pytest.raises(TypeError, match="put @staticmethod above @inx.compute.kernel"):
        class WrongOrder:
            @inx.compute.kernel
            @staticmethod
            def step(domain):
                i = inx.compute.index(domain)
                domain[i] = 1


def test_gpu_kernel_rejects_cpu_buffers_without_fallback():
    values = inx.buffer(shape=4, dtype=np.float32, device="cpu")

    @inx.compute.kernel
    def declared(payload):
        i = inx.compute.index(payload)
        payload[i] = 1.0

    with pytest.raises(TypeError, match="transfer CPU data explicitly"):
        inx.compute.launch(declared, params=(values,))


def test_gpu_kernel_prepare_is_explicit_and_never_dispatches():
    calls = []

    declaration = inx.compute.Kernel(lambda value: None)
    executable = object()

    def resolve(params):
        calls.append(params)
        return executable

    declaration._executable = resolve
    inx.compute.prepare(declaration, params=("domain", 2.0))
    assert calls == [("domain", 2.0)]
    with pytest.raises(TypeError, match="tuple"):
        inx.compute.prepare(declaration, params=[])
    with pytest.raises(TypeError, match="kernel declaration"):
        inx.compute.prepare(object(), params=())


def test_gpu_compiler_artifact_cache_is_engine_owned_binary(tmp_path, monkeypatch):
    from Infernux._compiler.taichi import frontend

    artifact = frontend.CompilerArtifact(
        spirv_tasks=(b"\x03\x02\x23\x07", b"task-two"),
        task_metadata=({"name": "first", "buffer_bindings": []},),
        domain_parameter=1,
        argument_layout={
            "size": 16,
            "parameters": [
                {"kind": "int32", "offset": 0},
                {"kind": "float32", "offset": 4},
            ],
        },
        required_capabilities={"spirv_version": 0x10300},
        diagnostic_locations=({
            "entry_point": "first",
            "path": "Assets/Scripts/Kernel.py",
            "line": 17,
            "column": 0,
            "function": "game.Kernel.integrate",
        },),
    )
    monkeypatch.setattr(frontend, "_cache_root", lambda: tmp_path)
    frontend._store_artifact("a" * 64, artifact)
    path = tmp_path / (("a" * 64) + ".inxgpu")
    assert path.read_bytes().startswith(b"INXGPU\x01")

    restored = frontend._load_artifact("a" * 64)
    assert restored is not None
    assert restored.spirv_tasks == artifact.spirv_tasks
    assert restored.domain_parameter == 1
    assert restored.argument_layout == artifact.argument_layout
    assert restored.required_capabilities == artifact.required_capabilities
    assert restored.diagnostic_locations == artifact.diagnostic_locations


def test_gpu_compiler_artifact_cache_uses_project_library_and_player_data(
    tmp_path, monkeypatch
):
    from Infernux.application import Application
    from Infernux._compiler.taichi import frontend

    project_root = tmp_path / "project"
    player_root = tmp_path / "player-data"
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "data_path", staticmethod(lambda: str(project_root)))
    assert frontend._cache_root() == project_root / "Library" / "Artifacts" / "Compute"

    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(Application, "data_path", staticmethod(lambda: str(player_root)))
    assert frontend._cache_root() == player_root / "Library" / "Artifacts" / "Compute"


def test_gpu_compute_function_is_an_explicit_kernel_only_declaration():
    @inx.compute.function
    def double(value):
        return value * 2

    assert isinstance(double, inx.compute.Function)
    assert double.__name__ == "double"
    with pytest.raises(RuntimeError, match="only inside"):
        double(4)


def test_gpu_compute_errors_distinguish_capability_compile_and_execution(monkeypatch):
    from Infernux.application import Application
    from Infernux._compiler.taichi import CompilerInstallationError
    from Infernux._compiler.taichi import frontend

    monkeypatch.setattr(Application, "_current_engine", staticmethod(lambda: None))
    with pytest.raises(inx.compute.ComputeCapabilityError, match="running graphical"):
        inx.buffer(shape=1, dtype=np.float32, device="gpu")

    class CompilerBuffer:
        device = "gpu"
        _host = type("CompilerHost", (), {"identity": 91})()

    monkeypatch.setattr(inx.compute, "Buffer", CompilerBuffer)
    monkeypatch.setattr(frontend, "parameter_key", lambda _params: ("layout",))

    declaration = inx.compute.Kernel(lambda value: None)
    monkeypatch.setattr(
        frontend,
        "compile_kernel",
        lambda _function, _params: (_ for _ in ()).throw(
            CompilerInstallationError("private compiler payload is incomplete")
        ),
    )
    with pytest.raises(inx.compute.ComputeCompilerError, match="initialization failed"):
        declaration._executable((CompilerBuffer(),))

    authored = inx.compute.Kernel(lambda value: None)
    monkeypatch.setattr(
        frontend,
        "compile_kernel",
        lambda _function, _params: (_ for _ in ()).throw(
            TypeError("unsupported authored expression")
        ),
    )
    with pytest.raises(inx.compute.KernelCompilationError, match="authored expression"):
        authored._executable((CompilerBuffer(),))

    class Host:
        identity = 73

        def dispatch_batch(self, _dispatches, _updates):
            raise RuntimeError("device lost")

    class Executable:
        host = Host()

        @staticmethod
        def prepare(_params):
            return [], [(object(), (), (), b"", 1, 1, 1)]

    declaration = inx.compute.Kernel(lambda value: None)
    declaration._executable = lambda _params: Executable()
    with pytest.raises(inx.compute.ComputeExecutionError, match="submission failed"):
        inx.compute.launch(declaration, params=(object(),))


def test_engine_resource_release_closes_module_level_gpu_objects(monkeypatch):
    from types import SimpleNamespace
    from Infernux.application import Application

    class NativeBuffer:
        resource_index = 17
        resource_generation = 1

        def set_bytes(self, value, offset=0):
            pass

    host = SimpleNamespace(identity=12, create_buffer=lambda *args: NativeBuffer())
    native = SimpleNamespace(has_renderer=True, _acquire_compute_host=lambda: host)
    monkeypatch.setattr(
        Application, "_current_engine",
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
    from Infernux._compiler.taichi import frontend

    class Host:
        identity = 31

    class CompilerBuffer:
        device = "gpu"
        _host = Host()

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


def test_compiled_gpu_launch_preparation_is_reused_for_stable_parameters(monkeypatch):
    from types import SimpleNamespace
    from Infernux.application import Application
    from Infernux.compute import _KernelExecutable

    class NativeBuffer:
        resource_index = 19
        resource_generation = 1

        def set_bytes(self, _value, _offset=0):
            pass

    class Host:
        identity = 23

        def create_buffer(self, *_args):
            return NativeBuffer()

        def _create_kernel_from_compiler(self, spirv, layout):
            return (spirv, tuple(layout))

    host = Host()
    native = SimpleNamespace(has_renderer=True, _acquire_compute_host=lambda: host)
    monkeypatch.setattr(
        Application, "_current_engine",
        staticmethod(lambda: SimpleNamespace(get_native_engine=lambda: native)),
    )
    domain = inx.buffer(shape=257, dtype=np.int32, device="gpu")
    values = inx.buffer(shape=257, dtype=np.float32, device="gpu")

    class Artifact:
        spirv_tasks = (b"spirv",)
        task_metadata = ({
            "buffer_bindings": (
                {"binding": 0, "binding_type": "uniform", "resource_kind": "arguments"},
                {"binding": 1, "binding_type": "storage", "resource_kind": "external_buffer",
                 "argument_indices": (1,), "access": 3},
            ),
            "threads_per_group": 64,
        },)
        domain_parameter = 0

        def __init__(self):
            self.argument_calls = 0

        def argument_bytes(self, _params):
            self.argument_calls += 1
            return b"stable-arguments"

    artifact = Artifact()
    executable = _KernelExecutable(artifact, host, (domain, values, 2.0))
    first_updates, first_dispatches = executable.prepare((domain, values, 2.0))
    second_updates, second_dispatches = executable.prepare((domain, values, 2.0))
    assert artifact.argument_calls == 1
    assert len(first_updates) == 1 and second_updates == []
    assert second_dispatches is first_dispatches

    changed_updates, changed_dispatches = executable.prepare((domain, values, 3.0))
    assert artifact.argument_calls == 2
    assert len(changed_updates) == 1
    assert changed_dispatches is not first_dispatches


def test_public_compute_surface_has_one_gpu_execution_model():
    import infernux

    assert infernux.compute is inx.compute
    assert not callable(inx.compute)
    assert not hasattr(jit, "compute")
    for removed in ("hpc", "resident", "Resident", "batch", "nn"):
        assert not hasattr(inx.compute, removed)


def test_compute_has_no_in_process_installer_or_second_device_array_api():
    assert not hasattr(jit, "ensure_jit_runtime")
    assert not hasattr(jit, "_install_numba")
    assert not hasattr(jit, "create_array")
    from Infernux.core import parallel_backend
    assert not hasattr(parallel_backend, "ResidentArray")


@pytest.mark.parametrize("imports,decorator", [
    ("import infernux as inx", "inx.jit.compile"),
    ("import Infernux", "Infernux.jit.compile"),
    ("from Infernux import jit as cpu", "cpu.compile"),
    ("import Infernux.jit as cpu", "cpu.compile"),
    ("from Infernux.jit import compile as optimize", "optimize"),
])
def test_cpu_jit_declarations_are_recognized_without_importing_authored_code(imports, decorator):
    source = f"{imports}\n@{decorator}\ndef fill(values):\n    return values\n"
    assert kernels.cpu_jit_declarations(source) == ("fill",)


@pytest.mark.parametrize(
    "source",
    [
        (
            "import infernux as inx\n"
            "@inx.jit.compile(parallel_policy='required')\n"
            "def fill(values):\n"
            "    inx.jit.warmup(fill, values)\n"
            "    for index in range(len(values)):\n"
            "        values[index] = index\n"
        ),
        (
            "from Infernux import jit as cpu\n"
            "class Solver:\n"
            "    @cpu.compile(\n"
            "        cache=True,\n"
            "    )\n"
            "    def fill(self, values):\n"
            "        cpu.warmup(self.fill, values)\n"
            "        return [value + 1 for value in values]\n"
        ),
        (
            "from Infernux.jit import compile as optimize, warmup as prepare\n"
            "@optimize\n"
            "def fill(values):\n"
            "    marker = prepare(fill, values)\n"
            "    return marker, [value * 2 for value in values]\n"
        ),
    ],
)
def test_no_jit_build_freezes_cpu_decorators_and_warmup_as_ordinary_python(source):
    cooked = kernels.build_interpreted_cpu_source(source)

    assert cooked.count("\n") == source.count("\n")
    original_lines = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    cooked_lines = [
        node.lineno
        for node in ast.walk(ast.parse(cooked))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert cooked_lines == original_lines
    assert "@inx.jit.compile" not in cooked
    assert "@cpu.compile" not in cooked
    assert "@optimize" not in cooked
    compile(cooked, "Assets/Scripts/Cpu.py", "exec")

    namespace = {}
    exec(cooked, namespace)
    target = namespace.get("fill")
    if target is None:
        target = namespace["Solver"]().fill
    values = [8, 9, 10]
    result = target(values)
    if result is None:
        assert values == [0, 1, 2]
    elif isinstance(result, tuple):
        assert result == (None, [16, 18, 20])
    else:
        assert result == [9, 10, 11]


def test_no_jit_build_follows_assigned_cpu_jit_aliases():
    source = (
        "import infernux as inx\n"
        "cpu = inx.jit\n"
        "decorator, prepare = cpu.compile, cpu.warmup\n"
        "optimize = decorator\n"
        "@optimize(parallel_policy='required')\n"
        "def fill(values):\n"
        "    prepared = prepare(fill, values)\n"
        "    return prepared, [value + 1 for value in values]\n"
    )

    cooked = kernels.build_interpreted_cpu_source(source)

    assert "@optimize" not in cooked
    assert "prepared = prepare(" not in cooked
    namespace = {}
    exec(cooked, namespace)
    assert namespace["fill"]([1, 2, 3]) == (None, [2, 3, 4])


def test_no_jit_build_follows_public_jit_star_imports():
    source = (
        "from Infernux.jit import *\n"
        "@compile(parallel_policy='required')\n"
        "def fill(values):\n"
        "    prepared = warmup(fill, values)\n"
        "    return prepared, [value * 2 for value in values]\n"
    )

    cooked = kernels.build_interpreted_cpu_source(source)

    assert "@compile" not in cooked
    assert "prepared = warmup(" not in cooked
    namespace = {}
    exec(cooked, namespace)
    assert namespace["fill"]([2, 4]) == (None, [4, 8])


@pytest.mark.parametrize(
    "source",
    [
        (
            "from Infernux import jit as cpu\n"
            "events = []\n"
            "class Ordinary:\n"
            "    def compile(self, function):\n"
            "        events.append('decorate')\n"
            "        return function\n"
            "    def warmup(self, *args):\n"
            "        events.append('warmup')\n"
            "        return 17\n"
            "cpu = Ordinary()\n"
            "@cpu.compile\n"
            "def fill(values):\n"
            "    return cpu.warmup(fill, values)\n"
        ),
        (
            "from Infernux.jit import compile as optimize, warmup as prepare\n"
            "events = []\n"
            "def ordinary_decorator(function):\n"
            "    events.append('decorate')\n"
            "    return function\n"
            "def ordinary_warmup(*args):\n"
            "    events.append('warmup')\n"
            "    return 17\n"
            "optimize = ordinary_decorator\n"
            "prepare = ordinary_warmup\n"
            "@optimize\n"
            "def fill(values):\n"
            "    return prepare(fill, values)\n"
        ),
    ],
)
def test_no_jit_build_preserves_imported_names_rebound_to_ordinary_code(source):
    cooked = kernels.build_interpreted_cpu_source(source)

    assert cooked == source
    namespace = {}
    exec(cooked, namespace)
    assert namespace["events"] == ["decorate"]
    assert namespace["fill"]([]) == 17
    assert namespace["events"] == ["decorate", "warmup"]


def test_no_jit_build_applies_import_rebinding_in_source_order():
    source = (
        "from Infernux import jit as cpu\n"
        "@cpu.compile(parallel_policy='required')\n"
        "def cooked(values):\n"
        "    return cpu.warmup(cooked, values)\n"
        "events = []\n"
        "class Ordinary:\n"
        "    def compile(self, function):\n"
        "        events.append('decorate')\n"
        "        return function\n"
        "    def warmup(self, *args):\n"
        "        events.append('warmup')\n"
        "        return 23\n"
        "cpu = Ordinary()\n"
        "@cpu.compile\n"
        "def ordinary(values):\n"
        "    return cpu.warmup(ordinary, values)\n"
    )

    cooked = kernels.build_interpreted_cpu_source(source)

    assert "@cpu.compile(parallel_policy='required')" not in cooked
    assert "return cpu.warmup(cooked, values)" not in cooked
    assert "@cpu.compile\ndef ordinary" in cooked
    assert "return cpu.warmup(ordinary, values)" in cooked
    namespace = {}
    exec(cooked, namespace)
    assert namespace["cooked"]([]) is None
    assert namespace["events"] == ["decorate"]
    assert namespace["ordinary"]([]) == 23
    assert namespace["events"] == ["decorate", "warmup"]


def test_no_jit_build_rejects_control_flow_ambiguous_jit_binding():
    source = (
        "from Infernux import jit as cpu\n"
        "if use_ordinary:\n"
        "    cpu = ordinary_cpu\n"
        "@cpu.compile\n"
        "def fill(values):\n"
        "    return values\n"
    )

    with pytest.raises(ValueError, match="ambiguous CPU JIT binding at line 4"):
        kernels.build_interpreted_cpu_source(source)


def test_no_jit_build_rejects_conditional_jit_alias_assignment():
    source = (
        "from Infernux import jit as cpu\n"
        "optimize = cpu.compile if enable_jit else ordinary_decorator\n"
        "@optimize\n"
        "def fill(values):\n"
        "    return values\n"
    )

    with pytest.raises(ValueError, match="ambiguous CPU JIT binding at line 3"):
        kernels.build_interpreted_cpu_source(source)


def test_no_jit_build_rejects_unknown_star_import_over_public_jit_alias():
    source = (
        "from Infernux.jit import compile as optimize\n"
        "from authored_helpers import *\n"
        "@optimize\n"
        "def fill(values):\n"
        "    return values\n"
    )

    with pytest.raises(ValueError, match="ambiguous CPU JIT binding at line 3"):
        kernels.build_interpreted_cpu_source(source)


@pytest.mark.parametrize(
    "source",
    [
        (
            "class Ordinary:\n"
            "    def compile(self, function):\n"
            "        return function\n"
            "jit = Ordinary()\n"
            "from Infernux import *\n"
            "@jit.compile\n"
            "def fill(values):\n"
            "    return values\n"
        ),
        (
            "from .Infernux.jit import compile as optimize\n"
            "@optimize\n"
            "def fill(values):\n"
            "    return values\n"
        ),
    ],
)
def test_no_jit_build_does_not_infer_non_public_star_or_relative_imports(source):
    assert kernels.build_interpreted_cpu_source(source) == source


def test_no_jit_build_does_not_rewrite_gpu_execution_model():
    source = (
        "import infernux as inx\n"
        "@inx.compute.kernel\n"
        "def fill(values):\n"
        "    values[inx.compute.index(values)] = 1\n"
    )
    assert kernels.build_interpreted_cpu_source(source) == source


def test_gpu_kernel_is_not_rewritten_as_cpu_parallel_work():
    source = (
        "from Infernux import compute\n"
        "@compute.kernel\n"
        "def fill(values):\n"
        "    i = compute.index(values)\n"
        "    values[i] = i\n"
    )
    assert kernels.cpu_jit_declarations(source) == ()
    assert kernels.auto_parallel_declarations(source) == ()
    assert kernels.build_auto_parallel_embedded_source(source) is None
