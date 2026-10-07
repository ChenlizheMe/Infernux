"""Borrowed Buffer permissions at NumPy, copy, readback and cooked boundaries."""
from contextlib import ExitStack

import numpy as np
import pytest

import infernux as inx
from infernux.engine.build.compute_cpu import build_cpu_compute_source


@pytest.mark.parametrize("web", [False, True])
@pytest.mark.parametrize("vector", [False, True])
def test_readonly_numpy_borrows_preserve_owner_updates(monkeypatch, web, vector):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1" if web else "0")
    data = np.arange(18 if vector else 6, dtype=np.float32).reshape((6, 3) if vector else (6,))
    with ExitStack() as stack:
        owner = stack.enter_context(inx.buffer(shape=6, dtype=inx.vector3 if vector else np.float32,
                                               device="gpu" if web else "cpu", data=data))
        read = stack.enter_context(owner.view(offset=1, count=4, readonly=True))
        nested = stack.enter_context(read.view(offset=1, count=2, readonly=False))
        write = stack.enter_context(owner.view(offset=2, count=2))
        assert read.readonly and nested.readonly
        for borrowed in (read.numpy(copy=False), read[:], nested.numpy(copy=False), nested[:]):
            assert np.shares_memory(borrowed, owner.numpy(copy=False))
            with pytest.raises(ValueError, match="read.only"):
                borrowed[...] = 99
        # Detached copies are writable and cannot mutate the original.
        independent = read.numpy(copy=True)
        independent[...] = -1
        np.testing.assert_array_equal(owner.numpy(), data)
        write.fill(17)
        np.testing.assert_array_equal(nested.numpy(), np.full(nested.numpy().shape, 17))
        assert owner.numpy(copy=False).flags.writeable
        assert not read.numpy(copy=False).flags.writeable
        owner.close()
        write.fill(23)  # Each borrowed wrapper retains the storage independently.
        assert np.all(nested.numpy() == 23)


@pytest.mark.parametrize("device", ["cpu", "gpu", "web"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_copy_out_rejects_readonly_or_closed_destination_before_consuming(
        engine, monkeypatch, device, asynchronous, closed):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1" if device == "web" else "0")
    host = engine._acquire_compute_host() if device == "gpu" else None
    if host is not None:
        monkeypatch.setattr(inx.compute, "_native_compute_host", lambda: host)
    try:
        with ExitStack() as stack:
            source = stack.enter_context(inx.buffer(shape=3, dtype=np.int32, device="cpu" if device == "cpu" else "gpu",
                                                    data=[7, 8, 9]))
            owner = stack.enter_context(inx.buffer(shape=5, dtype=np.int32, device="cpu", data=[1, 2, 3, 4, 5]))
            rejected = stack.enter_context(owner.view(offset=1, count=3, readonly=not closed))
            if closed:
                rejected.close()
            request = source.get_data_async() if asynchronous else source
            task_before = request._task if asynchronous else None
            with pytest.raises(RuntimeError, match="closed" if closed else "read-only"):
                request.get_data(out=rejected)
            if asynchronous:
                assert request._task is task_before  # Bad out must not consume a pending GPU result.
            np.testing.assert_array_equal(owner.numpy(), [1, 2, 3, 4, 5])
            accepted = stack.enter_context(owner.view(offset=1, count=3))
            assert request.get_data(out=accepted) is accepted
            np.testing.assert_array_equal(owner.numpy(), [1, 7, 8, 9, 5])
            if asynchronous:
                request.get_data().close()
    finally:
        if host is not None:
            host._release_lease()


@pytest.mark.parametrize("operation", ["construct", "set_data", "readback_cached", "copy_range"])
def test_closed_source_is_rejected_at_public_copy_boundary(operation):
    with ExitStack() as stack:
        source = stack.enter_context(inx.buffer(shape=3, dtype=np.float32, device="cpu", data=[1, 2, 3]))
        target = stack.enter_context(inx.buffer(shape=3, dtype=np.float32, device="cpu"))
        request = source.get_data_async()
        source.close()
        if operation == "readback_cached":
            request.get_data().close()
        with pytest.raises(RuntimeError, match="closed"):
            if operation == "construct":
                inx.buffer(shape=3, dtype=np.float32, device="cpu", data=source)
            elif operation == "set_data":
                target.set_data(source)
            elif operation == "copy_range":
                source.get_data(out=target, offset=0, count=3)
            else:
                request.get_data(out=target)
        np.testing.assert_array_equal(target.numpy(), [0, 0, 0])
        if operation != "readback_cached":
            request.get_data().close()


@pytest.mark.parametrize("atomic", ["none", "scalar", "indexed"])
def test_cooked_web_kernel_cannot_write_readonly_mapping(monkeypatch, atomic):
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")
    assignment = {"none": "output[i] = values[i] + 1",
                  "scalar": "inx.compute.atomic_add(output[0], values[i])",
                  "indexed": "inx.compute.atomic_add(output[i], values[i])"}[atomic]
    text = ("import infernux as inx\n@inx.compute.kernel\ndef solve(values, output):\n"
            f"    i = inx.compute.index(values)\n    {assignment}\n")
    namespace = {}
    exec(compile(build_cpu_compute_source(text), "<readonly-cooked>", "exec"), namespace)
    with ExitStack() as stack:
        values = stack.enter_context(inx.buffer(shape=3, dtype=np.int32, device="gpu", data=[1, 2, 3]))
        inputs = stack.enter_context(values.view(readonly=True))
        output = stack.enter_context(inx.buffer(shape=3, dtype=np.int32, device="gpu"))
        readonly = stack.enter_context(output.view(readonly=True))
        with pytest.raises((ValueError, RuntimeError), match="read.only"):
            inx.compute.launch(namespace["solve"], (inputs, readonly))
        np.testing.assert_array_equal(output.numpy(), [0, 0, 0])
        inx.compute.launch(namespace["solve"], (inputs, output))
        np.testing.assert_array_equal(output.numpy(), {"none": [2, 3, 4], "scalar": [6, 0, 0],
                                                       "indexed": [1, 2, 3]}[atomic])


@inx.compute.kernel
def _copy_readonly_input(values, output):
    i = inx.compute.index(values)
    output[i] = values[i] + 1


def test_native_gpu_kernel_preserves_readonly_binding_contract(engine, monkeypatch, tmp_path):
    from infernux.engine.project_context import using_project_root

    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "0")
    host = engine._acquire_compute_host()
    monkeypatch.setattr(inx.compute, "_native_compute_host", lambda: host)
    try:
        with using_project_root(tmp_path), ExitStack() as stack:
            values = stack.enter_context(inx.buffer(shape=3, dtype=np.int32, device="gpu", data=[1, 2, 3]))
            inputs = stack.enter_context(values.view(readonly=True))
            output = stack.enter_context(inx.buffer(shape=3, dtype=np.int32, device="gpu"))
            readonly = stack.enter_context(output.view(readonly=True))
            with pytest.raises(ValueError, match="read-only"):
                inx.compute.launch(_copy_readonly_input, (inputs, readonly))
            inx.compute.launch(_copy_readonly_input, (inputs, output))
            with output.get_data() as result:
                np.testing.assert_array_equal(result.numpy(), [2, 3, 4])
            # A cached writable launch must not authorize a different wrapper.
            with pytest.raises(ValueError, match="read-only"):
                inx.compute.launch(_copy_readonly_input, (inputs, readonly))
    finally:
        # Public declarations retain retired native bindings until the normal
        # compute frame boundary, which this manually owned host must run.
        inx.compute._flush_commands(host)
        _copy_readonly_input._release_engine_resources()
        host._release_lease()
