"""Fullscreen graph storage-buffer input contract."""

import weakref
from types import SimpleNamespace

import pytest

from infernux.lib import GraphBufferAccessType, GraphCommandType
from infernux.rendergraph.graph import RenderGraph


def test_fullscreen_buffer_binding_declares_read_and_serializes_resource_order():
    graph = RenderGraph("StorageInput")
    color = graph.create_texture("color", camera_target=True)
    values = graph.create_buffer("values", 64)
    with graph.add_pass("PresentValues") as render_pass:
        render_pass.set_buffer("_Values", values)
        render_pass.write_color(color)
        render_pass.set_clear(color=(0, 0, 0, 1))
        render_pass.fullscreen_quad("Buffer Viewer")
    graph.set_output(color)

    description = graph.build()
    authored = next(item for item in description.passes if item.name == "PresentValues")
    assert authored.commands[0].type == GraphCommandType.FULLSCREEN_QUAD
    assert authored.commands[0].input_bindings == [("_Values", "values")]
    assert [(item.resource, item.type) for item in authored.buffer_accesses] == [
        ("values", GraphBufferAccessType.STORAGE_READ)
    ]


class _ComputeStorage:
    byte_size = 64
    last_write_serial = 1

    def has_initialized_range(self, offset, size):
        return 0 <= offset < self.byte_size and 0 < size <= self.byte_size - offset


def _gpu_buffer(*, device="gpu", scalar="uint32", lanes=1, offset=0, size=64):
    return SimpleNamespace(
        device=device, closed=False, nbytes=size, _native=_ComputeStorage(),
        description=SimpleNamespace(scalar_dtype=scalar, lanes=lanes, byte_offset=offset),
    )


def test_import_buffer_rejects_non_gpu_or_wrong_scalar_type():
    graph = RenderGraph("Import")
    with pytest.raises(ValueError, match="scalar float32 or uint32 GPU"):
        graph.import_buffer("values", _gpu_buffer(device="cpu"))
    with pytest.raises(ValueError, match="scalar float32 or uint32 GPU"):
        graph.import_buffer("values", _gpu_buffer(scalar="int32"))


def test_import_buffer_requires_full_allocation():
    # The existing case now exercises full initialization of the imported
    # logical range. A view may cover part of an allocation.
    graph = RenderGraph("Import")
    with pytest.raises(ValueError, match="initialized logical range"):
        graph.import_buffer("invalid", _gpu_buffer(offset=32, size=64))
    source = _gpu_buffer(offset=32, size=32)
    view = graph.import_buffer("view", source)
    assert (view.byte_offset, view.byte_size) == (32, 32)
    assert view.compute_buffer is source._native
    assert graph.import_buffer("values", _gpu_buffer()).byte_size == 64


def test_imported_buffer_is_read_only_in_graph():
    graph = RenderGraph("ImportedReadOnly")
    imported = graph.import_buffer("values", _gpu_buffer())
    scratch = graph.create_buffer("scratch", 64)
    with graph.add_pass("TryWrite") as render_pass:
        with pytest.raises(ValueError, match="read-only"):
            render_pass.write_buffer(imported)
    with graph.add_copy_pass("TryCopy") as copy_pass:
        with pytest.raises(ValueError, match="copy destination"):
            copy_pass.copy_buffer(scratch, imported)


def test_logical_import_does_not_keep_native_allocation_alive_after_owner_close():
    owner = _gpu_buffer()
    allocation = weakref.ref(owner._native)
    graph = RenderGraph("ImportedLifetime")
    handle = graph.import_buffer("values", owner)
    assert handle.compute_buffer is owner._native

    owner.closed = True
    owner._native = None
    assert allocation() is None, "Logical graph retains native storage beyond Buffer.close()"
    with pytest.raises(ValueError, match="closed"):
        _ = handle.compute_buffer
