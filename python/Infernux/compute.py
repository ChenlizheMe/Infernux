"""Engine-owned GPU buffers and single-work-item Vulkan kernels.

Use :func:`buffer` for explicit CPU or GPU storage and declare GPU work with
``@inx.compute.kernel``.  :func:`launch` derives dispatch size from the
kernel's ``inx.compute.index(buffer)`` domain and records work into the current
engine phase.  CPU function compilation is deliberately separate at
``inx.jit.compile``.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
import math
import threading
from typing import Literal
import weakref

import numpy as np

_gpu_buffers = weakref.WeakSet()
_kernel_declarations = weakref.WeakSet()
_transform_bindings = weakref.WeakSet()
_command_recording = threading.local()
_buffer_generations = count(1)
_retiring_native_kernels = {}


class ComputeCapabilityError(RuntimeError):
    """The running engine/device cannot provide the requested compute capability."""


class ComputeCompilerError(RuntimeError):
    """The installed engine GPU compiler cannot initialize."""


class KernelCompilationError(TypeError):
    """Authored GPU kernel source cannot be lowered for the current contract."""


class ComputeExecutionError(RuntimeError):
    """A valid, compiled compute command failed during device execution."""


@dataclass(frozen=True, slots=True)
class BufferDescription:
    """Immutable public layout and ownership contract for one ``inx.buffer``.

    ``capacity`` is the number of logical elements in the allocation; vector
    lanes are part of one element and therefore do not multiply it. GPU
    ``resource_index``/``resource_generation`` are opaque RHI identity words,
    not Vulkan handles or serializable asset identities.
    """

    shape: tuple[int, ...]
    dtype: str
    scalar_dtype: str
    lanes: int
    attribute_offsets: tuple[int, ...]
    capacity: int
    element_stride: int
    byte_offset: int
    byte_length: int
    nbytes: int
    device: Literal["cpu", "gpu"]
    usage: tuple[str, ...]
    access: Literal["read_write", "read_only"]
    ownership: Literal["owned", "borrowed"]
    generation: int
    resource_index: int | None
    resource_generation: int | None


@dataclass(frozen=True, slots=True)
class Statistics:
    submission_count: int
    dispatch_count: int
    upload_request_count: int
    upload_bytes: int
    readback_request_count: int
    readback_bytes: int
    staging_allocation_count: int
    host_map_count: int
    native_boundary_count: int
    wait_count: int
    cpu_submit_ms: float
    wait_ms: float
    pending_submission_count: int
    gpu_profile_available: bool
    gpu_profile_serial: int
    gpu_time_ms: float | None


@dataclass(frozen=True, slots=True)
class TransformPose:
    """World position/rotation and local scale exchanged by a Transform anchor."""

    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float]


def _recording_depth() -> int:
    return int(getattr(_command_recording, "depth", 0))


def _pending_batches() -> dict[int, dict]:
    batches = getattr(_command_recording, "batches", None)
    if batches is None:
        batches = {}
        _command_recording.batches = batches
    return batches


def _pending_batch(host) -> dict:
    identity = int(host.identity)
    batches = _pending_batches()
    batch = batches.get(identity)
    if batch is None:
        batch = {
            "host": host,
            "updates": [],
            "dispatches": [],
            "updated": set(),
            "mesh_attributes": {},
        }
        batches[identity] = batch
    return batch


def _append_mesh_attribute_dispatches(batch: dict) -> None:
    """Append one derived-vertex rebuild for each resident mesh touched."""
    for vertex_buffer in batch["mesh_attributes"].values():
        state = vertex_buffer._mesh_attribute_state
        if state is None or vertex_buffer.closed:
            continue
        params = (
            state.domain,
            vertex_buffer,
            state.triangles,
            state.adjacency,
            state.counts,
            state.width,
            int(state.normals),
            int(state.tangents),
        )
        executable = _mesh_attribute_kernel._executable(params)
        updates, dispatches = executable.prepare(params)
        batch["updates"].extend(updates)
        batch["dispatches"].extend(dispatches)


def _flush_commands(host=None) -> None:
    """Submit recorded work in order; explicit readback uses this boundary."""
    batches = _pending_batches()
    identities = tuple(batches) if host is None else (int(host.identity),)
    for identity in identities:
        batch = batches.pop(identity, None)
        if batch is None:
            continue
        _append_mesh_attribute_dispatches(batch)
        if batch["updates"] or batch["dispatches"]:
            try:
                batch["host"].dispatch_batch(batch["dispatches"], batch["updates"])
            except Exception as exception:
                raise ComputeExecutionError("GPU compute submission failed") from exception
    _collect_retiring_native_kernels()


def _collect_retiring_native_kernels() -> None:
    """Poll only kernels that own bindings for wrappers already closed."""
    for identity, native in tuple(_retiring_native_kernels.items()):
        native.collect()
        if int(native.pending_dispatch_count) == 0:
            _retiring_native_kernels.pop(identity, None)


def _discard_buffer_launches(value: "Buffer") -> None:
    for declaration in tuple(_kernel_declarations):
        declaration._discard_buffer_launches(value)


def _submit_and_read(host, reads) -> list[bytes]:
    """Submit pending work and its requested readbacks with one queue wait."""
    batch = _pending_batches().pop(int(host.identity), None)
    if batch is None:
        batch = {"host": host, "updates": [], "dispatches": [], "mesh_attributes": {}}
    else:
        _append_mesh_attribute_dispatches(batch)
    native_reads = []
    for value, byte_offset, byte_size in reads:
        if not isinstance(value, Buffer) or value.device != "gpu":
            raise TypeError("Compute readback requires GPU inx.buffer values")
        value._require_open()
        if int(value._host.identity) != int(host.identity):
            raise ValueError("Compute readback buffers must use the requested compute host")
        native_reads.append((value._native, int(byte_offset), int(byte_size)))
    try:
        return host.dispatch_batch_and_read(batch["dispatches"], batch["updates"], native_reads)
    except Exception as exception:
        raise ComputeExecutionError("GPU compute readback submission failed") from exception


@contextmanager
def _record_commands():
    """Engine lifecycle scope that records public launches into one submission."""
    _command_recording.depth = _recording_depth() + 1
    try:
        yield
    finally:
        _command_recording.depth -= 1
        if _recording_depth() == 0:
            _flush_commands()


@contextmanager
def recording():
    """Record ordered :func:`launch` calls into one GPU submission.

    This is a command-recording boundary only. It does not create a second
    device or imply a readback; explicit buffer reads and native consumers
    flush pending work before observing it. Nested scopes preserve the outer
    submission so fixed-step solvers can group dependent kernels without
    knowing about the Vulkan queue.
    """
    with _record_commands():
        yield


def _queue_buffer_update(value: "Buffer", payload: bytes, byte_offset: int) -> None:
    if _recording_depth() == 0:
        try:
            value._native.set_bytes(payload, byte_offset)
        except Exception as exception:
            raise ComputeExecutionError("GPU buffer upload failed") from exception
        return
    batch = _pending_batch(value._host)
    native_identity = id(value._native)
    # Native batches place uploads before dispatches and reuse one staging
    # allocation per destination. Preserve authored ordering by ending the
    # current submission at either conflict.
    if batch["dispatches"] or native_identity in batch["updated"]:
        _flush_commands(value._host)
        batch = _pending_batch(value._host)
    batch["updates"].append((value._native, payload, byte_offset))
    batch["updated"].add(native_identity)


@dataclass(frozen=True, slots=True)
class _BufferDType:
    name: str
    scalar: np.dtype
    lanes: int = 1

    @property
    def itemsize(self) -> int:
        return int(self.scalar.itemsize) * self.lanes


def _buffer_dtype(value) -> _BufferDType:
    if isinstance(value, _BufferDType):
        return value
    from Infernux.math import vector2, vector3, vector4

    vectors = {
        vector2: _BufferDType("vector2", np.dtype(np.float32), 2),
        vector3: _BufferDType("vector3", np.dtype(np.float32), 3),
        vector4: _BufferDType("vector4", np.dtype(np.float32), 4),
    }
    if value in vectors:
        return vectors[value]
    if value is float:
        return _BufferDType("float", np.dtype(np.float32))
    if value is int:
        return _BufferDType("int", np.dtype(np.int32))
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint32)}:
        raise TypeError("inx.buffer scalar dtype must be float32, int32, or uint32")
    return _BufferDType(dtype.name, dtype)


def _logical_shape(value) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, int) else tuple(value)
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("inx.buffer shape must contain positive integers")
    return shape


def _native_compute_host():
    from Infernux.application import Application

    engine = Application._current_engine()
    native = engine.get_native_engine() if engine is not None else None
    if native is None or not bool(getattr(native, "has_renderer", False)):
        raise ComputeCapabilityError("GPU buffers require a running graphical Infernux engine")
    host = getattr(engine, "_infernux_compute_host", None)
    if host is None:
        try:
            host = native._acquire_compute_host()
        except Exception as exception:
            raise ComputeCapabilityError("The active renderer cannot provide GPU compute") from exception
        engine._infernux_compute_host = host
    return host


class Buffer:
    """Engine-owned typed storage shared by compute, mesh, and rendering.

    CPU buffers are directly indexable. GPU buffers remain device resident and
    require explicit :meth:`set_data` / :meth:`get_data` transfers; ordinary
    Python indexing never triggers a hidden readback.
    """

    def __init__(self, *, shape, dtype=float, device: Literal["cpu", "gpu"], data=None) -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError("inx.buffer device must be explicitly 'cpu' or 'gpu'")
        self._shape = _logical_shape(shape)
        self._dtype = _buffer_dtype(dtype)
        self._storage_shape = self._shape + ((self._dtype.lanes,) if self._dtype.lanes > 1 else ())
        self._device = device
        self._closed = False
        self._host = None
        self._native = None
        self._array = None
        self._mesh_attribute_state = None
        self._dependent_resources = []
        self._generation = next(_buffer_generations)
        self._byte_offset = 0
        self._ownership = "owned"
        self._readonly = False
        self._description = None

        initial = np.zeros(self._storage_shape, dtype=self._dtype.scalar)
        if data is not None:
            initial = self._coerce_data(data)
        if device == "cpu":
            self._array = initial
        else:
            self._host = _native_compute_host()
            self._native = self._host.create_buffer(
                self.element_count, self._dtype.scalar.name, self._dtype.lanes
            )
            _gpu_buffers.add(self)
            _queue_buffer_update(self, initial.tobytes(order="C"), 0)
        self._description = self._build_description()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype.name

    @property
    def device(self) -> Literal["cpu", "gpu"]:
        return self._device

    @property
    def nbytes(self) -> int:
        return int(np.prod(self._storage_shape, dtype=np.int64)) * int(self._dtype.scalar.itemsize)

    @property
    def element_count(self) -> int:
        return int(np.prod(self._shape, dtype=np.int64))

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def readonly(self) -> bool:
        return self._readonly

    @property
    def description(self) -> BufferDescription:
        """Return the authoritative typed allocation description.

        The returned value contains no raw host/GPU address and is not an
        asset reference. Closing the buffer invalidates use of the resource
        but does not mutate this immutable description snapshot.
        """
        return self._description

    def _build_description(self) -> BufferDescription:
        resource_index = None
        resource_generation = None
        if self._native is not None:
            resource_index = int(self._native.resource_index)
            resource_generation = int(self._native.resource_generation)
        usage = (
            ("storage", "uniform", "vertex", "transfer_source", "transfer_destination")
            if self._device == "gpu"
            else ("host_read", "host_write")
        )
        return BufferDescription(
            shape=self._shape,
            dtype=self._dtype.name,
            scalar_dtype=self._dtype.scalar.name,
            lanes=self._dtype.lanes,
            attribute_offsets=tuple(
                lane * int(self._dtype.scalar.itemsize) for lane in range(self._dtype.lanes)
            ),
            capacity=self.element_count,
            element_stride=self._dtype.itemsize,
            byte_offset=self._byte_offset,
            byte_length=self.nbytes,
            nbytes=self.nbytes,
            device=self._device,
            usage=usage,
            access="read_only" if self._readonly else "read_write",
            ownership=self._ownership,
            generation=self._generation,
            resource_index=resource_index,
            resource_generation=resource_generation,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("inx.buffer is closed")

    def _require_writable(self) -> None:
        self._require_open()
        if self._readonly:
            raise RuntimeError("inx.buffer view is read-only")

    def _coerce_data(self, data) -> np.ndarray:
        if isinstance(data, Buffer):
            if data.device != "cpu":
                raise TypeError("GPU-to-GPU copies require the compute copy operation")
            source = data._array
        else:
            source = data
        try:
            result = np.asarray(source, dtype=self._dtype.scalar)
        except (TypeError, ValueError):
            if self._dtype.lanes <= 1:
                raise
            result = np.asarray(
                [[getattr(item, axis) for axis in "xyzw"[: self._dtype.lanes]] for item in source],
                dtype=self._dtype.scalar,
            )
        if result.shape != self._storage_shape:
            raise ValueError(f"inx.buffer data shape must be {self._storage_shape}, got {result.shape}")
        return np.ascontiguousarray(result)

    def _coerce_element_range(self, data) -> np.ndarray:
        if isinstance(data, Buffer):
            if data.device != "cpu":
                raise TypeError("GPU-to-GPU copies require the compute copy operation")
            if data._dtype != self._dtype:
                raise ValueError("inx.buffer range dtype does not match the destination")
            source = data._array
        else:
            source = data
        try:
            result = np.asarray(source, dtype=self._dtype.scalar)
        except (TypeError, ValueError):
            if self._dtype.lanes <= 1:
                raise
            result = np.asarray(
                [[getattr(item, axis) for axis in "xyzw"[: self._dtype.lanes]] for item in source],
                dtype=self._dtype.scalar,
            )
        if self._dtype.lanes > 1:
            if result.ndim == 0 or result.shape[-1] != self._dtype.lanes:
                raise ValueError(f"inx.buffer range data must end in {self._dtype.lanes} vector lanes")
            result = result.reshape(-1, self._dtype.lanes)
        else:
            result = result.reshape(-1)
        return np.ascontiguousarray(result)

    def set_data(self, data, *, offset: int = 0) -> None:
        """Write logical elements at ``offset``; GPU work is queued in order."""
        self._require_writable()
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("inx.buffer set_data offset must be a non-negative element index")
        values = self._coerce_element_range(data)
        count = int(values.shape[0])
        if count == 0 or offset > self.element_count or count > self.element_count - offset:
            raise ValueError("inx.buffer set_data range exceeds the destination")
        flat = self._array.reshape((-1, self._dtype.lanes)) if self._dtype.lanes > 1 and self._array is not None \
            else self._array.reshape(-1) if self._array is not None else None
        if self._device == "cpu":
            np.copyto(flat[offset:offset + count], values, casting="no")
        else:
            _queue_buffer_update(
                self,
                values.tobytes(order="C"),
                self._byte_offset + offset * self._dtype.itemsize,
            )

    def get_data(self, out: "Buffer | None" = None, *, offset: int = 0, count: int | None = None) -> "Buffer":
        """Return a CPU element-range snapshot; GPU readback completes before return."""
        self._require_open()
        if not isinstance(offset, int) or offset < 0 or offset > self.element_count:
            raise ValueError("inx.buffer get_data offset must address the buffer")
        count = self.element_count - offset if count is None else count
        if not isinstance(count, int) or count <= 0 or count > self.element_count - offset:
            raise ValueError("inx.buffer get_data range exceeds the source")
        if out is None:
            output_shape = self._shape if offset == 0 and count == self.element_count else (count,)
            out = Buffer(shape=output_shape, dtype=self._dtype, device="cpu")
        if not isinstance(out, Buffer) or out.device != "cpu":
            raise TypeError("inx.buffer get_data out must be a CPU Buffer")
        if out.element_count != count or out._dtype != self._dtype:
            raise ValueError("inx.buffer get_data out layout does not match the source")
        destination = out._array.reshape((-1, self._dtype.lanes)) if self._dtype.lanes > 1 else out._array.reshape(-1)
        if self._device == "cpu":
            source = self._array.reshape((-1, self._dtype.lanes)) if self._dtype.lanes > 1 else self._array.reshape(-1)
            np.copyto(destination, source[offset:offset + count], casting="no")
        else:
            byte_size = count * self._dtype.itemsize
            payload, = _submit_and_read(
                self._host,
                ((self, self._byte_offset + offset * self._dtype.itemsize, byte_size),),
            )
            values = np.frombuffer(payload, dtype=self._dtype.scalar).reshape(destination.shape)
            np.copyto(destination, values, casting="no")
        return out

    def get_data_async(self, *, offset: int = 0, count: int | None = None) -> "Readback":
        """Start an explicit snapshot without waiting for GPU completion."""
        self._require_open()
        if not isinstance(offset, int) or offset < 0 or offset > self.element_count:
            raise ValueError("inx.buffer get_data_async offset must address the buffer")
        count = self.element_count - offset if count is None else count
        if not isinstance(count, int) or count <= 0 or count > self.element_count - offset:
            raise ValueError("inx.buffer get_data_async range exceeds the source")
        output_shape = self._shape if offset == 0 and count == self.element_count else (count,)
        if self._device == "cpu":
            return Readback(self._dtype, output_shape, immediate=self.get_data(offset=offset, count=count))
        _flush_commands(self._host)
        byte_size = count * self._dtype.itemsize
        task = self._native.get_bytes_async(
            byte_size, self._byte_offset + offset * self._dtype.itemsize)
        return Readback(self._dtype, output_shape, task=task, host=self._host)

    def fill(self, value) -> None:
        """Fill every element and explicitly publish it to the selected device."""
        if self._dtype.lanes > 1 and hasattr(value, "x"):
            value = tuple(getattr(value, axis) for axis in "xyzw"[: self._dtype.lanes])
        shape = (self.element_count, self._dtype.lanes) if self._dtype.lanes > 1 else (self.element_count,)
        values = np.empty(shape, dtype=self._dtype.scalar)
        values[...] = value
        self.set_data(values)

    def zero(self) -> None:
        """Set every byte in the typed element range to zero."""
        self.fill(0)

    def numpy(self, *, copy: bool = True) -> np.ndarray:
        """Expose a CPU buffer as NumPy; GPU callers must use get_data first."""
        self._require_open()
        if self._device != "cpu":
            raise RuntimeError("GPU buffers require get_data() before NumPy access")
        return self._array.copy() if copy else self._array

    def view(self, *, offset: int = 0, count: int | None = None, readonly: bool = False) -> "Buffer":
        """Borrow a one-dimensional element range without copying storage.

        The view has its own logical shape and lifetime while retaining the
        same CPU allocation or engine RHI resource. Closing either wrapper is
        therefore safe; storage is released only after its final reference.
        """
        self._require_open()
        if not isinstance(offset, int) or offset < 0 or offset > self.element_count:
            raise ValueError("inx.buffer view offset must address the buffer")
        count = self.element_count - offset if count is None else count
        if not isinstance(count, int) or count <= 0 or count > self.element_count - offset:
            raise ValueError("inx.buffer view range exceeds the source")
        if not isinstance(readonly, bool):
            raise TypeError("inx.buffer view readonly must be a bool")

        result = object.__new__(Buffer)
        result._shape = (count,)
        result._dtype = self._dtype
        result._storage_shape = (count,) + ((self._dtype.lanes,) if self._dtype.lanes > 1 else ())
        result._device = self._device
        result._closed = False
        result._host = self._host
        result._native = self._native
        result._mesh_attribute_state = None
        result._dependent_resources = []
        result._generation = next(_buffer_generations)
        result._byte_offset = self._byte_offset + offset * self._dtype.itemsize
        result._ownership = "borrowed"
        result._readonly = self._readonly or readonly
        if self._array is None:
            result._array = None
        else:
            flat = (
                self._array.reshape((-1, self._dtype.lanes))
                if self._dtype.lanes > 1
                else self._array.reshape(-1)
            )
            result._array = flat[offset:offset + count]
        result._description = result._build_description()
        if result._device == "gpu":
            _gpu_buffers.add(result)
        return result

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index):
        self._require_open()
        if self._device != "cpu":
            raise RuntimeError("GPU buffers cannot be indexed from Python; use get_data()")
        value = self._array[index]
        if self._dtype.lanes == 1:
            return value.item() if np.isscalar(value) else value
        if getattr(value, "ndim", 0) != 1:
            return value
        from Infernux.math import vector2, vector3, vector4
        constructor = {2: vector2, 3: vector3, 4: vector4}[self._dtype.lanes]
        return constructor(*(float(component) for component in value))

    def __setitem__(self, index, value) -> None:
        self._require_writable()
        if self._device != "cpu":
            raise RuntimeError("GPU buffers cannot be indexed from Python; use set_data()")
        if self._dtype.lanes > 1 and hasattr(value, "x"):
            value = tuple(getattr(value, axis) for axis in "xyzw"[: self._dtype.lanes])
        self._array[index] = value

    def close(self) -> None:
        if self._closed:
            return
        attribute_state = self._mesh_attribute_state
        self._mesh_attribute_state = None
        if attribute_state is not None:
            attribute_state.close()
        dependents, self._dependent_resources = self._dependent_resources, []
        for dependent in dependents:
            dependent.close()
        if self._device == "gpu" and self._host is not None:
            _flush_commands(self._host)
            _discard_buffer_launches(self)
        _gpu_buffers.discard(self)
        self._native = None
        self._host = None
        self._array = None
        self._closed = True

    def _retain_dependent(self, resource) -> None:
        """Tie an engine-side derived resource to this buffer's lifetime."""
        self._require_open()
        self._dependent_resources.append(resource)

    def __enter__(self) -> "Buffer":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class Readback:
    """An explicit CPU snapshot of one buffer range.

    ``done`` only polls the exact GPU submission. ``get_data`` is the single
    completion boundary and returns a CPU Buffer; no stale frame is exposed.
    """

    def __init__(self, dtype: _BufferDType, shape: tuple[int, ...], *, task=None,
                 immediate: Buffer | None = None, host=None) -> None:
        self._dtype = dtype
        self._shape = shape
        self._task = task
        self._result = immediate
        self._host = host

    @property
    def done(self) -> bool:
        return self._result is not None or bool(self._task.done)

    def get_data(self, out: Buffer | None = None) -> Buffer:
        if self._result is None:
            payload = self._task.get_bytes()
            result = Buffer(shape=self._shape, dtype=self._dtype, device="cpu")
            values = np.frombuffer(payload, dtype=self._dtype.scalar).reshape(result._array.shape)
            np.copyto(result._array, values, casting="no")
            self._result = result
            self._task = None
            self._host = None
        if out is None:
            return self._result
        if not isinstance(out, Buffer) or out.device != "cpu":
            raise TypeError("inx.compute.Readback get_data out must be a CPU Buffer")
        if out.element_count != self._result.element_count or out._dtype != self._dtype:
            raise ValueError("inx.compute.Readback get_data out layout does not match the snapshot")
        np.copyto(out._array.reshape(self._result._array.shape), self._result._array, casting="no")
        return out


def _vector_tuple(value, lanes: int) -> tuple[float, ...]:
    return tuple(float(getattr(value, axis)) for axis in "xyzw"[:lanes])


def _transform_pose(transform) -> TransformPose:
    return TransformPose(
        position=_vector_tuple(transform.position, 3),
        rotation=_vector_tuple(transform.rotation, 4),
        scale=_vector_tuple(transform.local_scale, 3),
    )


def _pose_changed(left: TransformPose, right: TransformPose) -> bool:
    return not np.allclose(
        (*left.position, *left.rotation, *left.scale),
        (*right.position, *right.rotation, *right.scale),
        rtol=0.0,
        atol=1.0e-5,
    )


def _pose_array(value: TransformPose) -> np.ndarray:
    return np.asarray(
        (
            (*value.position, 1.0),
            value.rotation,
            (*value.scale, 1.0),
        ),
        dtype=np.float32,
    )


def _buffer_pose(value: Buffer) -> TransformPose:
    rows = value.numpy(copy=False)
    return TransformPose(
        position=tuple(float(item) for item in rows[0, :3]),
        rotation=tuple(float(item) for item in rows[1, :4]),
        scale=tuple(float(item) for item in rows[2, :3]),
    )


def _publish_transform_pose(transform, value: TransformPose) -> None:
    from Infernux.math import quaternion, vector3

    transform.position = vector3(*value.position)
    transform.rotation = quaternion(*value.rotation)
    transform.local_scale = vector3(*value.scale)


class TransformBinding:
    """Exchange one simulation pose with a component Transform in both directions.

    ``pose`` is three ``vector4`` elements: world position, quaternion rotation,
    and local scale. Simulation publication uses one asynchronous snapshot. An
    authored Transform edit wins the current physics boundary, is uploaded to
    the same pose buffer, and is delivered once to ``on_transform`` so the
    simulation can apply the TRS delta to its resident state.
    """

    def __init__(self, owner, pose: Buffer, *, initial_pose: TransformPose | None = None,
                 domain: Buffer | None = None, points=(), vectors=(), on_transform=None) -> None:
        if owner is None or not hasattr(owner, "game_object"):
            raise TypeError("inx.compute.bind_transform expects an Infernux component owner")
        try:
            game_object = owner.game_object
            object_id = int(game_object.id)
            object_handle = game_object.handle
        except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError) as exc:
            raise TypeError("inx.compute.bind_transform expects a live Infernux component owner") from exc
        if not isinstance(pose, Buffer):
            raise TypeError("inx.compute.bind_transform pose must be an inx.buffer")
        pose._require_open()
        if pose.dtype != "vector4" or pose.shape != (3,):
            raise ValueError(
                "inx.compute.bind_transform pose must contain exactly three vector4 values"
            )
        if on_transform is not None and not callable(on_transform):
            raise TypeError("inx.compute.bind_transform on_transform must be callable")
        if initial_pose is not None and not isinstance(initial_pose, TransformPose):
            raise TypeError("inx.compute.bind_transform initial_pose must be a TransformPose")
        points = tuple(points)
        vectors = tuple(vectors)
        if points or vectors:
            if pose.device != "gpu":
                raise TypeError(
                    "inx.compute.bind_transform resident state requires a GPU pose buffer"
                )
            if not isinstance(domain, Buffer):
                raise TypeError(
                    "inx.compute.bind_transform domain is required for resident state buffers"
                )
            domain._require_open()
            for value in (*points, *vectors):
                if not isinstance(value, Buffer) or value.device != pose.device:
                    raise TypeError(
                        "inx.compute.bind_transform state values must be inx.buffer values on the pose device"
                    )
                value._require_open()
                if value.dtype == "vector3":
                    valid = value.element_count == domain.element_count
                else:
                    valid = value.dtype == "float32" and value.shape == (domain.element_count, 3)
                if not valid:
                    raise ValueError(
                        "inx.compute.bind_transform state values must contain one vector3 per domain item"
                    )
        self._object_id = object_id
        self._object_handle = object_handle
        self._pose = pose
        self._domain = domain
        self._points = points
        self._vectors = vectors
        self._on_transform = on_transform
        self._initial_pose = initial_pose
        self._readback = None
        self._published_pose = None
        self._closed = False
        _transform_bindings.add(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def _poll(self) -> None:
        if self._closed:
            return
        if self._pose.closed:
            self.close()
            return
        transform = _resolve_bound_transform(self._object_id, self._object_handle)
        if transform is None:
            self.close()
            return
        current_pose = _transform_pose(transform)
        if self._initial_pose is not None:
            previous_pose = self._initial_pose
            self._initial_pose = None
            self._pose.set_data(_pose_array(current_pose))
            if _pose_changed(current_pose, previous_pose):
                _apply_transform_delta(
                    self._domain,
                    self._points,
                    self._vectors,
                    previous_pose,
                    current_pose,
                )
                if self._on_transform is not None:
                    self._on_transform(previous_pose, current_pose)
            self._published_pose = current_pose
            self._readback = self._pose.get_data_async()
            return
        if self._readback is not None:
            if not self._readback.done:
                return
            snapshot = self._readback.get_data()
            self._readback = None
            if self._published_pose is not None and _pose_changed(
                current_pose, self._published_pose
            ):
                previous_pose = self._published_pose
                self._pose.set_data(_pose_array(current_pose))
                _apply_transform_delta(
                    self._domain,
                    self._points,
                    self._vectors,
                    previous_pose,
                    current_pose,
                )
                if self._on_transform is not None:
                    self._on_transform(previous_pose, current_pose)
                self._published_pose = current_pose
                self._readback = self._pose.get_data_async()
                return
            value = _buffer_pose(snapshot)
            try:
                _publish_transform_pose(transform, value)
            except (AttributeError, ReferenceError, RuntimeError):
                self.close()
                return
            self._published_pose = value
        self._readback = self._pose.get_data_async()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _transform_bindings.discard(self)
        self._readback = None
        self._object_handle = None
        self._pose = None
        self._domain = None
        self._points = ()
        self._vectors = ()
        self._on_transform = None
        self._initial_pose = None
        self._published_pose = None


def bind_transform(owner, *, pose: Buffer, initial_pose: TransformPose | None = None,
                   domain: Buffer | None = None, points=(), vectors=(), on_transform=None) -> TransformBinding:
    """Create one deterministic two-way simulation/Transform pose binding.

    When ``initial_pose`` is supplied, the authored Transform initializes the
    resident simulation state before this call returns.  FixedUpdate therefore
    never observes the model-space/default pose for an authored runtime object.
    """
    binding = TransformBinding(
        owner,
        pose,
        initial_pose=initial_pose,
        domain=domain,
        points=points,
        vectors=vectors,
        on_transform=on_transform,
    )
    if initial_pose is not None:
        try:
            binding._poll()
        except Exception:
            binding.close()
            raise
    return binding


def _resolve_bound_transform(object_id: int, object_handle):
    """Resolve a binding through the scene handle; never retain a native pointer."""
    from Infernux.lib import SceneManager

    game_object = SceneManager.instance().find_runtime_object_by_id(object_id)
    if game_object is None or game_object.handle != object_handle:
        return None
    return game_object.transform


def _poll_transform_bindings() -> None:
    """Exchange authored/simulated poses at the physics Transform barrier."""
    with recording():
        for binding in tuple(_transform_bindings):
            binding._poll()


def buffer(*, shape, dtype=float, device: Literal["cpu", "gpu"], data=None) -> Buffer:
    """Create typed engine storage on an explicit CPU or GPU device."""
    return Buffer(shape=shape, dtype=dtype, device=device, data=data)


def statistics(*, reset: bool = False) -> Statistics:
    """Read cumulative compute transfer/submission counters without logging."""
    host = _native_compute_host()
    raw = dict(host.get_statistics())
    values = Statistics(
        submission_count=int(raw["submission_count"]),
        dispatch_count=int(raw["dispatch_count"]),
        upload_request_count=int(raw["upload_request_count"]),
        upload_bytes=int(raw["upload_bytes"]),
        readback_request_count=int(raw["readback_request_count"]),
        readback_bytes=int(raw["readback_bytes"]),
        staging_allocation_count=int(raw["staging_allocation_count"]),
        host_map_count=int(raw["host_map_count"]),
        native_boundary_count=int(raw["native_boundary_count"]),
        wait_count=int(raw["wait_count"]),
        cpu_submit_ms=float(raw["cpu_submit_ms"]),
        wait_ms=float(raw["wait_ms"]),
        pending_submission_count=int(raw["pending_submission_count"]),
        gpu_profile_available=bool(raw["gpu_profile_available"]),
        gpu_profile_serial=int(raw["gpu_profile_serial"]),
        gpu_time_ms=None if raw["gpu_time_ms"] is None else float(raw["gpu_time_ms"]),
    )
    if reset:
        host.reset_statistics()
    return values


def set_profiling_enabled(enabled: bool) -> None:
    """Enable optional GPU timestamps; changing query resources drains compute."""
    if not isinstance(enabled, bool):
        raise TypeError("inx.compute profiling enabled must be a bool")
    if not _native_compute_host().set_profiling_enabled(enabled):
        raise ComputeCapabilityError("GPU compute timestamp profiling is unavailable")


class _KernelExecutable:
    def __init__(self, artifact, host, params) -> None:
        self.artifact = artifact
        self.host = host
        # A launch specialization is immutable: resident buffer identities and
        # scalar values fully determine both the compiler argument block and
        # the native resource bindings. Fixed-step solvers commonly repeat the
        # same launch tens of times, so prepare it once instead of rebuilding a
        # Taichi launch context and Python binding list on every dispatch.
        self._prepared_launches = OrderedDict()
        self.tasks = []
        for spirv, metadata in zip(artifact.spirv_tasks, artifact.task_metadata):
            bindings = sorted(metadata["buffer_bindings"], key=lambda binding: binding["binding"])
            layout = [(binding["binding"], binding["binding_type"]) for binding in bindings]
            native = host._create_kernel_from_compiler(spirv, layout)
            self.tasks.append((native, bindings, int(metadata["threads_per_group"])))

    @staticmethod
    def _argument_index(binding) -> int:
        indices = binding.get("argument_indices", ())
        if isinstance(indices, int):
            return indices
        if len(indices) != 1:
            raise RuntimeError("GPU kernel buffer bindings require one top-level parameter index")
        return int(indices[0])

    @staticmethod
    def _launch_key(params):
        key = []
        for value in params:
            if isinstance(value, Buffer):
                key.append(("buffer", id(value)))
            elif isinstance(value, (bool, int, np.integer)):
                key.append(("int32", int(value)))
            elif isinstance(value, (float, np.floating)):
                key.append(("float32", float(value)))
            else:
                raise TypeError("GPU kernel parameters must be inx.buffer values or numeric scalars")
        return tuple(key)

    def prepare(self, params):
        launch_key = self._launch_key(params)
        prepared = self._prepared_launches.pop(launch_key, None)
        if prepared is not None:
            self._prepared_launches[launch_key] = prepared
            return [], prepared

        arguments = self.artifact.argument_bytes(params)
        argument_buffer = self.host.create_buffer(max(1, (len(arguments) + 3) // 4), "uint32")
        updates = [(argument_buffer, arguments, 0)] if arguments else []
        domain = params[self.artifact.domain_parameter]
        dispatches = []
        for native, bindings, threads_per_group in self.tasks:
            resources = []
            accesses = []
            for binding in bindings:
                kind = binding["resource_kind"]
                if kind == "arguments":
                    resources.append(argument_buffer)
                    accesses.append("read")
                elif kind == "external_buffer":
                    value = params[self._argument_index(binding)]
                    if not isinstance(value, Buffer):
                        raise TypeError("Compiled GPU buffer binding does not match the launch parameters")
                    access = int(binding.get("access", 0))
                    if value.readonly and access in {2, 3}:
                        raise ValueError("GPU kernel writes a read-only inx.buffer view")
                    resources.append(value._native)
                    if access == 1:
                        accesses.append("read")
                    elif access == 2:
                        accesses.append("write")
                    elif access == 3:
                        accesses.append("read_write")
                    else:
                        raise RuntimeError("GPU compiler omitted an external buffer access declaration")
                else:
                    raise RuntimeError(f"Unsupported GPU compiler resource kind: {kind}")
            groups = (domain.element_count + threads_per_group - 1) // threads_per_group
            dispatches.append((native, resources, accesses, b"", groups, 1, 1))

        if len(self._prepared_launches) >= 64:
            self._prepared_launches.popitem(last=False)
        self._prepared_launches[launch_key] = dispatches
        return updates, dispatches

    def launch(self, params) -> None:
        updates, dispatches = self.prepare(params)
        self.host.dispatch_batch(dispatches, updates)

    def discard_buffer_launches(self, value: Buffer) -> None:
        token = ("buffer", id(value))
        for key in tuple(self._prepared_launches):
            if token not in key:
                continue
            self._prepared_launches.pop(key, None)
        # The Python LRU may already have evicted this wrapper while the native
        # kernel still caches its weak binding identity. Schedule one bounded
        # native collection regardless of whether a Python entry remained.
        for native, _, _ in self.tasks:
            _retiring_native_kernels[id(native)] = native

    def close(self) -> None:
        self.tasks.clear()
        self._prepared_launches.clear()
        self.artifact = None
        self.host = None


class Kernel:
    """A GPU single-work-item declaration prepared or compiled on first launch."""

    def __init__(self, function) -> None:
        if not callable(function):
            raise TypeError("inx.compute.kernel requires a callable")
        self.function = function
        self._executables = {}
        self._lock = threading.RLock()
        for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(self, attribute, getattr(function, attribute, None))
        _kernel_declarations.add(self)

    def _executable(self, params):
        from Infernux._compiler.taichi import CompilerInstallationError
        from Infernux._compiler.taichi.frontend import compile_kernel, parameter_key

        buffers = [value for value in params if isinstance(value, Buffer)]
        if not buffers:
            raise TypeError("GPU kernel launch requires at least one GPU inx.buffer")
        if any(value.device != "gpu" for value in buffers):
            raise TypeError(
                "GPU kernel launch requires GPU inx.buffer parameters; "
                "transfer CPU data explicitly with set_data"
            )
        host = buffers[0]._host
        host_identity = int(host.identity)
        if any(int(value._host.identity) != host_identity for value in buffers):
            raise ValueError("GPU kernel buffers must belong to the same Infernux compute host")
        key = (host_identity, parameter_key(params))
        with self._lock:
            executable = self._executables.get(key)
            if executable is None:
                try:
                    artifact = compile_kernel(self.function, params)
                except CompilerInstallationError as exception:
                    raise ComputeCompilerError(
                        f"Infernux GPU compiler initialization failed: {exception}"
                    ) from exception
                except KernelCompilationError:
                    raise
                except Exception as exception:
                    raise KernelCompilationError(
                        f"GPU kernel '{self.__qualname__}' compilation failed: {exception}"
                    ) from exception
                executable = _KernelExecutable(artifact, host, params)
                self._executables[key] = executable
            return executable

    def _release_engine_resources(self) -> None:
        with self._lock:
            executables = tuple(self._executables.values())
            self._executables.clear()
        for executable in executables:
            executable.close()

    def _discard_buffer_launches(self, value: Buffer) -> None:
        with self._lock:
            for executable in self._executables.values():
                executable.discard_buffer_launches(value)


class Function:
    """A reusable numeric helper compiled only as part of a GPU kernel."""

    def __init__(self, value) -> None:
        if not callable(value):
            raise TypeError("inx.compute.function requires a callable")
        self.function = value
        for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(self, attribute, getattr(value, attribute, None))

    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("@inx.compute.function helpers are valid only inside a GPU kernel")


def _release_engine_resources() -> None:
    """Release module-level GPU declarations before native renderer teardown."""
    _flush_commands()
    for binding in tuple(_transform_bindings):
        binding.close()
    for declaration in tuple(_kernel_declarations):
        declaration._release_engine_resources()
    for value in tuple(_gpu_buffers):
        value.close()
    _retiring_native_kernels.clear()
    from Infernux.application import Application

    engine = Application._current_engine()
    if engine is not None:
        engine._infernux_compute_host = None


def kernel(function) -> Kernel:
    """Declare a GPU kernel; the function describes one logical work item."""
    return Kernel(function)


def function(value) -> Function:
    """Declare a reusable numeric helper for :func:`kernel` functions."""
    return Function(value)


def index(domain: Buffer) -> int:
    """Declare and return the current work-item index inside a GPU kernel."""
    raise RuntimeError("inx.compute.index is valid only inside @inx.compute.kernel")


def atomic_add(target, value):
    """Atomically add ``value`` to one buffer element inside a GPU kernel."""
    raise RuntimeError("inx.compute.atomic_add is valid only inside @inx.compute.kernel")


def prepare(declaration: Kernel, params) -> None:
    """Compile one specialization and create its native pipelines without dispatching it.

    Call this from an owning component's preparation/start boundary when the
    first gameplay interaction must not pay compiler or pipeline creation
    cost. Argument values are not uploaded and authored kernel code is not
    executed; the following :func:`launch` remains the first data operation.
    """
    if not isinstance(declaration, Kernel):
        raise TypeError("inx.compute.prepare requires an @inx.compute.kernel declaration")
    if not isinstance(params, tuple):
        raise TypeError("inx.compute.prepare params must be a tuple")
    declaration._executable(params)


@kernel
def _transform_anchor_points(domain, values,
                             opx, opy, opz, oqx, oqy, oqz, oqw, osx, osy, osz,
                             npx, npy, npz, nqx, nqy, nqz, nqw, nsx, nsy, nsz):
    """Apply ``new TRS * inverse(old TRS)`` to resident world-space points."""
    i = index(domain)
    dx = values[i, 0] - opx
    dy = values[i, 1] - opy
    dz = values[i, 2] - opz
    or00 = 1.0 - 2.0 * (oqy * oqy + oqz * oqz)
    or01 = 2.0 * (oqx * oqy - oqz * oqw)
    or02 = 2.0 * (oqx * oqz + oqy * oqw)
    or10 = 2.0 * (oqx * oqy + oqz * oqw)
    or11 = 1.0 - 2.0 * (oqx * oqx + oqz * oqz)
    or12 = 2.0 * (oqy * oqz - oqx * oqw)
    or20 = 2.0 * (oqx * oqz - oqy * oqw)
    or21 = 2.0 * (oqy * oqz + oqx * oqw)
    or22 = 1.0 - 2.0 * (oqx * oqx + oqy * oqy)
    lx = (or00 * dx + or10 * dy + or20 * dz) / osx
    ly = (or01 * dx + or11 * dy + or21 * dz) / osy
    lz = (or02 * dx + or12 * dy + or22 * dz) / osz
    lx *= nsx
    ly *= nsy
    lz *= nsz
    nr00 = 1.0 - 2.0 * (nqy * nqy + nqz * nqz)
    nr01 = 2.0 * (nqx * nqy - nqz * nqw)
    nr02 = 2.0 * (nqx * nqz + nqy * nqw)
    nr10 = 2.0 * (nqx * nqy + nqz * nqw)
    nr11 = 1.0 - 2.0 * (nqx * nqx + nqz * nqz)
    nr12 = 2.0 * (nqy * nqz - nqx * nqw)
    nr20 = 2.0 * (nqx * nqz - nqy * nqw)
    nr21 = 2.0 * (nqy * nqz + nqx * nqw)
    nr22 = 1.0 - 2.0 * (nqx * nqx + nqy * nqy)
    values[i, 0] = npx + nr00 * lx + nr01 * ly + nr02 * lz
    values[i, 1] = npy + nr10 * lx + nr11 * ly + nr12 * lz
    values[i, 2] = npz + nr20 * lx + nr21 * ly + nr22 * lz


@kernel
def _transform_anchor_vectors(domain, values,
                              oqx, oqy, oqz, oqw, osx, osy, osz,
                              nqx, nqy, nqz, nqw, nsx, nsy, nsz):
    """Apply the rotational/scale part of an authored TRS delta to vectors."""
    i = index(domain)
    dx = values[i, 0]
    dy = values[i, 1]
    dz = values[i, 2]
    or00 = 1.0 - 2.0 * (oqy * oqy + oqz * oqz)
    or01 = 2.0 * (oqx * oqy - oqz * oqw)
    or02 = 2.0 * (oqx * oqz + oqy * oqw)
    or10 = 2.0 * (oqx * oqy + oqz * oqw)
    or11 = 1.0 - 2.0 * (oqx * oqx + oqz * oqz)
    or12 = 2.0 * (oqy * oqz - oqx * oqw)
    or20 = 2.0 * (oqx * oqz - oqy * oqw)
    or21 = 2.0 * (oqy * oqz + oqx * oqw)
    or22 = 1.0 - 2.0 * (oqx * oqx + oqy * oqy)
    lx = (or00 * dx + or10 * dy + or20 * dz) * nsx / osx
    ly = (or01 * dx + or11 * dy + or21 * dz) * nsy / osy
    lz = (or02 * dx + or12 * dy + or22 * dz) * nsz / osz
    nr00 = 1.0 - 2.0 * (nqy * nqy + nqz * nqz)
    nr01 = 2.0 * (nqx * nqy - nqz * nqw)
    nr02 = 2.0 * (nqx * nqz + nqy * nqw)
    nr10 = 2.0 * (nqx * nqy + nqz * nqw)
    nr11 = 1.0 - 2.0 * (nqx * nqx + nqz * nqz)
    nr12 = 2.0 * (nqy * nqz - nqx * nqw)
    nr20 = 2.0 * (nqx * nqz - nqy * nqw)
    nr21 = 2.0 * (nqy * nqz + nqx * nqw)
    nr22 = 1.0 - 2.0 * (nqx * nqx + nqy * nqy)
    values[i, 0] = nr00 * lx + nr01 * ly + nr02 * lz
    values[i, 1] = nr10 * lx + nr11 * ly + nr12 * lz
    values[i, 2] = nr20 * lx + nr21 * ly + nr22 * lz


def _apply_transform_delta(domain, points, vectors, old: TransformPose,
                           new: TransformPose) -> None:
    if not points and not vectors:
        return
    if any(abs(value) <= 1.0e-8 for value in (*old.scale, *new.scale)):
        raise ValueError("Transform anchor scale components must be non-zero")
    old_point = (*old.position, *old.rotation, *old.scale)
    new_point = (*new.position, *new.rotation, *new.scale)
    for value in points:
        launch(_transform_anchor_points, params=(domain, value, *old_point, *new_point))
    old_vector = (*old.rotation, *old.scale)
    new_vector = (*new.rotation, *new.scale)
    for value in vectors:
        launch(_transform_anchor_vectors, params=(domain, value, *old_vector, *new_vector))


@kernel
def _mesh_attribute_kernel(domain, vertices, triangles, adjacency, counts, width,
                           rebuild_normals, rebuild_tangents):
    """Rebuild geometric normals and tangent frames on a resident mesh."""
    i = index(domain)
    nx = 0.0
    ny = 0.0
    nz = 0.0
    tx = 0.0
    ty = 0.0
    tz = 0.0
    bx = 0.0
    by = 0.0
    bz = 0.0
    for slot in range(width):
        if slot < counts[i]:
            triangle = adjacency[i, slot]
            a = triangles[triangle, 0]
            b = triangles[triangle, 1]
            c = triangles[triangle, 2]
            abx = vertices[b, 0] - vertices[a, 0]
            aby = vertices[b, 1] - vertices[a, 1]
            abz = vertices[b, 2] - vertices[a, 2]
            acx = vertices[c, 0] - vertices[a, 0]
            acy = vertices[c, 1] - vertices[a, 1]
            acz = vertices[c, 2] - vertices[a, 2]
            nx += aby * acz - abz * acy
            ny += abz * acx - abx * acz
            nz += abx * acy - aby * acx
            if rebuild_tangents != 0:
                du1 = vertices[b, 13] - vertices[a, 13]
                dv1 = vertices[b, 14] - vertices[a, 14]
                du2 = vertices[c, 13] - vertices[a, 13]
                dv2 = vertices[c, 14] - vertices[a, 14]
                determinant = du1 * dv2 - dv1 * du2
                if abs(determinant) > 1.0e-12:
                    inverse = 1.0 / determinant
                    tx += (abx * dv2 - acx * dv1) * inverse
                    ty += (aby * dv2 - acy * dv1) * inverse
                    tz += (abz * dv2 - acz * dv1) * inverse
                    bx += (acx * du1 - abx * du2) * inverse
                    by += (acy * du1 - aby * du2) * inverse
                    bz += (acz * du1 - abz * du2) * inverse

    if rebuild_normals != 0:
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1.0e-12:
            nx /= length
            ny /= length
            nz /= length
        else:
            nx = 0.0
            ny = 0.0
            nz = 0.0
        vertices[i, 3] = nx
        vertices[i, 4] = ny
        vertices[i, 5] = nz
    else:
        nx = vertices[i, 3]
        ny = vertices[i, 4]
        nz = vertices[i, 5]
        # Authored normals stay untouched in the vertex stream, but derived
        # tangent frames always operate in a unit normal frame.  This matches
        # the CPU tangent builder and avoids lighting drift for imported meshes
        # that carry scaled normals while deliberately disabling regeneration.
        normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_length > 1.0e-12:
            nx /= normal_length
            ny /= normal_length
            nz /= normal_length

    if rebuild_tangents != 0:
        projection = nx * tx + ny * ty + nz * tz
        tx -= nx * projection
        ty -= ny * projection
        tz -= nz * projection
        tangent_length = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tangent_length > 1.0e-12:
            tx /= tangent_length
            ty /= tangent_length
            tz /= tangent_length
        elif abs(nx) > abs(nz):
            tangent_length = math.sqrt(nx * nx + ny * ny)
            tx = -ny / max(tangent_length, 1.0e-12)
            ty = nx / max(tangent_length, 1.0e-12)
            tz = 0.0
        else:
            tangent_length = math.sqrt(ny * ny + nz * nz)
            tx = 0.0
            ty = -nz / max(tangent_length, 1.0e-12)
            tz = ny / max(tangent_length, 1.0e-12)
        tangent_length = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tangent_length <= 1.0e-12:
            tx = 1.0
            ty = 0.0
            tz = 0.0
        handedness = 1.0
        if (ny * tz - nz * ty) * bx + (nz * tx - nx * tz) * by + (nx * ty - ny * tx) * bz < 0.0:
            handedness = -1.0
        vertices[i, 6] = tx
        vertices[i, 7] = ty
        vertices[i, 8] = tz
        vertices[i, 9] = handedness


@dataclass(slots=True)
class _MeshAttributeState:
    domain: Buffer
    triangles: Buffer
    adjacency: Buffer
    counts: Buffer
    width: int
    normals: bool
    tangents: bool

    def close(self) -> None:
        for value in (self.domain, self.triangles, self.adjacency, self.counts):
            value.close()


def _enable_automatic_mesh_attributes(
    vertex_buffer: Buffer,
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    normals: bool,
    tangents: bool,
) -> None:
    """Attach immutable topology used by pre-render attribute rebuilding."""
    positions = np.asarray(positions, dtype=np.float32)
    triangles = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(triangles):
        raise ValueError("automatic mesh attributes require triangle geometry")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(positions):
        raise ValueError("automatic mesh attributes require valid triangle indices")

    # Match authored topology. UV seams and hard edges deliberately stay
    # split; guessing welds from equal floating-point positions changes the
    # asset's smoothing semantics.
    incident_sets = [set() for _ in range(len(positions))]
    for face, triangle in enumerate(triangles):
        for vertex in triangle:
            incident_sets[int(vertex)].add(face)
    incident = [sorted(faces) for faces in incident_sets]
    width = max(map(len, incident))
    adjacency = np.zeros((len(positions), width), dtype=np.int32)
    counts = np.empty(len(positions), dtype=np.int32)
    for vertex, faces in enumerate(incident):
        counts[vertex] = len(faces)
        adjacency[vertex, : len(faces)] = faces

    vertex_buffer._mesh_attribute_state = _MeshAttributeState(
        domain=buffer(shape=len(positions), dtype=np.int32, device="gpu"),
        triangles=buffer(shape=triangles.shape, dtype=np.int32, device="gpu", data=triangles),
        adjacency=buffer(shape=adjacency.shape, dtype=np.int32, device="gpu", data=adjacency),
        counts=buffer(shape=counts.shape, dtype=np.int32, device="gpu", data=counts),
        width=width,
        normals=normals,
        tangents=tangents,
    )
    state = vertex_buffer._mesh_attribute_state
    prepare(_mesh_attribute_kernel, (
        state.domain,
        vertex_buffer,
        state.triangles,
        state.adjacency,
        state.counts,
        state.width,
        int(state.normals),
        int(state.tangents),
    ))


def launch(declaration: Kernel, params) -> None:
    """Compile if needed and record a GPU kernel for the current engine phase."""
    if not isinstance(declaration, Kernel):
        raise TypeError("inx.compute.launch requires an @inx.compute.kernel declaration")
    if not isinstance(params, tuple):
        raise TypeError("inx.compute.launch params must be a tuple")
    executable = declaration._executable(params)
    updates, dispatches = executable.prepare(params)
    batch = _pending_batch(executable.host)
    for native, payload, offset in updates:
        native_identity = id(native)
        if native_identity in batch["updated"]:
            _flush_commands(executable.host)
            batch = _pending_batch(executable.host)
        batch["updates"].append((native, payload, offset))
        batch["updated"].add(native_identity)
    batch["dispatches"].extend(dispatches)
    for value in params:
        if isinstance(value, Buffer) and value._mesh_attribute_state is not None:
            batch["mesh_attributes"][id(value)] = value
    if _recording_depth() == 0:
        _flush_commands(executable.host)


__all__ = [
    "ComputeCapabilityError", "ComputeCompilerError", "KernelCompilationError",
    "ComputeExecutionError",
    "buffer", "Buffer", "BufferDescription", "Readback", "Statistics", "statistics",
    "set_profiling_enabled",
    "kernel", "Kernel", "function", "Function", "index", "atomic_add", "prepare", "launch",
    "recording", "bind_transform", "TransformBinding", "TransformPose",
]
