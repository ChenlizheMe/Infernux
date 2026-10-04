from __future__ import annotations

import numpy as np
import pytest

from infernux.core.parallel_backend import (
    ParallelBackend,
    ParallelBufferView,
    ParallelCapabilities,
    ParallelTaskState,
)


PARTICLE_SCHEMA = (
    ("position", "float32x3"),
    ("size", "float32"),
    ("color", "float32x4"),
    ("rotation", "float32"),
)


def test_parallel_buffer_view_requires_stable_identity_and_contiguous_array():
    array = np.zeros((8, 9), dtype=np.float32)
    view = ParallelBufferView("particles:1", array, PARTICLE_SCHEMA, (8, 9), "float32")

    assert view.handle == "particles:1"
    assert view.buffer is array
    with pytest.raises(ValueError, match="C-contiguous"):
        ParallelBufferView(
            "particles:2",
            np.zeros((8, 18), dtype=np.float32)[:, ::2],
            PARTICLE_SCHEMA,
            (8, 9),
            "float32",
        )


def test_parallel_backend_protocol_stays_renderer_external():
    class Backend:
        def capabilities(self):
            return ParallelCapabilities("test", asynchronous=True)

        def create_buffer_view(self, schema, shape, dtype):
            array = np.zeros(tuple(shape), dtype=dtype)
            return ParallelBufferView("buffer:1", array, tuple(schema), tuple(shape), dtype)

        def submit(self, kernel, inputs, outputs):
            return "task:1"

        def poll(self, task_handle):
            return ParallelTaskState.COMPLETED

        def wait(self, task_handle, timeout=None):
            return ParallelTaskState.COMPLETED

        def cancel(self, task_handle):
            return True

        def commit(self, output, destination_handle):
            assert destination_handle == "particle-batch:1"

    backend = Backend()
    assert isinstance(backend, ParallelBackend)
    output = backend.create_buffer_view(PARTICLE_SCHEMA, (4, 9), "float32")
    task = backend.submit("integrate", (), (output,))
    assert task == "task:1"
    assert backend.poll(task) == ParallelTaskState.COMPLETED
    backend.commit(output, "particle-batch:1")
