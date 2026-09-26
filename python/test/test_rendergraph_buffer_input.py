"""Fullscreen graph storage-buffer input contract."""

import pytest

from Infernux.lib import GraphBufferAccessType, GraphCommandType
from Infernux.rendergraph.graph import RenderGraph


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


def test_import_buffer_rejects_non_gpu_or_wrong_scalar_type():
    class Buffer:
        device = "cpu"
        dtype = "uint32"
        closed = False
        _native = object()

    graph = RenderGraph("Import")
    with pytest.raises(ValueError, match="uint32 GPU"):
        graph.import_buffer("values", Buffer())
    Buffer.device = "gpu"
    Buffer.dtype = "float32"
    with pytest.raises(ValueError, match="uint32 GPU"):
        graph.import_buffer("values", Buffer())


def test_import_buffer_requires_full_allocation():
    class Native:
        byte_size = 64

    class Buffer:
        device = "gpu"
        dtype = "uint32"
        closed = False
        _native = Native()
        _byte_offset = 4
        nbytes = 60

    graph = RenderGraph("Import")
    with pytest.raises(ValueError, match="complete GPU allocation"):
        graph.import_buffer("view", Buffer())
    Buffer._byte_offset = 0
    Buffer.nbytes = 64
    assert graph.import_buffer("values", Buffer()).byte_size == 64


def test_imported_buffer_is_read_only_in_graph():
    class Native:
        byte_size = 64

    class Buffer:
        device = "gpu"
        dtype = "uint32"
        closed = False
        _native = Native()
        _byte_offset = 0
        nbytes = 64

    graph = RenderGraph("ImportedReadOnly")
    imported = graph.import_buffer("values", Buffer())
    scratch = graph.create_buffer("scratch", 64)
    with graph.add_pass("TryWrite") as render_pass:
        with pytest.raises(ValueError, match="read-only"):
            render_pass.write_buffer(imported)
    with graph.add_copy_pass("TryCopy") as copy_pass:
        with pytest.raises(ValueError, match="copy destination"):
            copy_pass.copy_buffer(scratch, imported)
