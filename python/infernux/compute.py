"""Engine-owned compute buffers and single-work-item kernels.

Use :func:`buffer` for explicit CPU or GPU storage and declare GPU work with
``@inx.compute.kernel``.  :func:`launch` derives dispatch size from the
kernel's ``inx.compute.index(buffer)`` domain and records work into the current
engine phase.  CPU function compilation is deliberately separate at
``inx.jit.compile``.
"""

from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from itertools import count
import math
import os
import threading
from time import perf_counter, perf_counter_ns
from typing import Literal
import weakref

import numpy as np

from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
from infernux.compute_source import BufferSource

_gpu_buffers = weakref.WeakSet()
_gpu_works = weakref.WeakSet()
_deferred_prepare_state = threading.local()
_readbacks = weakref.WeakSet()
_kernel_declarations = weakref.WeakSet()
_transform_bindings = weakref.WeakSet()
_command_recording = threading.local()
_cpu_work_item = threading.local()
_cpu_dirty_mesh_buffers = weakref.WeakSet()
_cpu_dirty_transform_bindings = weakref.WeakSet()
_buffer_generations = count(1)
_retiring_native_kernels = {}
_cpu_statistics = {
    "submission_count": 0,
    "dispatch_count": 0,
    "cpu_submit_ms": 0.0,
}


def _web_cpu_compute_enabled() -> bool:
    """Return the build-selected Web compute execution model.

    Web Player sets this before importing cooked project modules. Authored
    ``device='gpu'`` remains the public contract; only its physical execution
    storage changes for the Web target.
    """
    return os.environ.get("INFERNUX_WEB_RUNTIME", "") == "1"


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
    copy_request_count: int
    copy_bytes: int
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


def _explicit_recording_depth() -> int:
    return int(getattr(_command_recording, "explicit_depth", 0))


def _record_activity() -> None:
    if _recording_depth():
        _command_recording.activity = getattr(_command_recording, "activity", 0) + 1


def _automatic_cost(function):
    """Charge compute-owned host intervals, never the surrounding callback."""
    @wraps(function)
    def measured(*args, **kwargs):
        if not getattr(_command_recording, "automatic_depth", 0) or getattr(
            _command_recording, "cost_depth", 0
        ):
            return function(*args, **kwargs)
        from infernux.application import Application

        engine = Application._current_engine()
        if engine is None:
            return function(*args, **kwargs)
        started = perf_counter_ns()
        frame = engine._frame_work
        frame.begin_scope()
        owner = None
        _command_recording.cost_depth = 1
        try:
            owner = engine._cpu_work()
            return function(*args, **kwargs)
        finally:
            _command_recording.cost_depth = 0
            try:
                if owner is not None:
                    owner._charge_host_interval(started, automatic=True)
                else:
                    frame.end_scope()
            finally:
                owner = engine = None
    return measured


def _cpu_mesh_frame_depth() -> int:
    return int(getattr(_command_recording, "cpu_mesh_frame_depth", 0))


def _begin_cpu_mesh_frame() -> None:
    """Defer Web CPU mesh publication until the native frame is complete."""
    if _web_cpu_compute_enabled():
        _command_recording.cpu_mesh_frame_depth = _cpu_mesh_frame_depth() + 1


def _end_cpu_mesh_frame() -> None:
    if not _web_cpu_compute_enabled():
        return
    depth = _cpu_mesh_frame_depth()
    if depth <= 0:
        raise RuntimeError("Web CPU mesh frame boundary is unbalanced")
    _command_recording.cpu_mesh_frame_depth = depth - 1
    if depth == 1 and (_cpu_dirty_transform_bindings or _cpu_dirty_mesh_buffers):
        with _record_commands():
            _flush_cpu_transform_bindings()
            if _cpu_dirty_mesh_buffers:
                _flush_cpu_meshes()


@dataclass(slots=True, eq=False)
class _PreparedCommands:
    # Keep the owner behind a mutable field: a failed drain's traceback may
    # retain this record after engine teardown cancels its native work.
    native: object
    host_identity: int
    completion: object = None
    storage_blocks: tuple = ()
    payload_bytes: int = 0
    execution_signature: tuple = ()
    accepted_transfer_bytes: int = 0
    preservation: object = None

    def __post_init__(self):
        try:
            self.storage_blocks = tuple(_native_storage_blocks(self.host_identity, self.native))
            self.payload_bytes = int(self.native.payload_bytes)
            self.execution_signature = tuple(self.native.execution_signature)
        except BaseException:
            self._cancel_unaccepted()
            raise

    def _release_accepted(self, on_accepted) -> None:
        try:
            self.accepted_transfer_bytes = int(self.native.transfer_bytes)
            self.completion = self.native.completion()
            self.payload_bytes = int(self.native.payload_bytes)
            readbacks = tuple(_native_storage_blocks(self.host_identity, self.native))
            if readbacks:
                blocks = dict(self.storage_blocks)
                blocks.update(readbacks)
                self.storage_blocks = tuple(blocks.items())
        finally:
            self.native = None
            if self.completion is not None:
                on_accepted(self)

    @property
    def transfer_bytes(self):
        return self.accepted_transfer_bytes if self.native is None else int(self.native.transfer_bytes)

    @property
    def needs_preparation(self):
        return (self.native.requires_preparation or
                self.preservation is not None and self.preservation.completion is not None or
                self.preservation is None and self.native.has_preservation_work)

    @property
    def submission(self):
        return self if self.preservation is None else self.preservation

    def prepare_storage(self, chunk_bytes, *, managed):
        ready, native = self.native.prepare_preservation(chunk_bytes, managed)
        if not ready:
            return False
        if self.preservation is not None and (self.preservation.completion is not None or
                                             self.preservation.native is not native):
            self.preservation._cancel_unaccepted()
            self.preservation = None
        if native is not None and self.preservation is None:
            self.preservation = _PreparedCommands(native, self.host_identity)
        self.storage_blocks = tuple(_native_storage_blocks(self.host_identity, self.native))
        return True

    def _cancel_unaccepted(self) -> None:
        try:
            if self.preservation is not None:
                self.preservation._cancel_unaccepted()
                self.preservation = None
            if self.native is not None:
                self.native.cancel()
        finally:
            self.native = None
            self.storage_blocks = ()
            self.payload_bytes = 0


def _command_streams() -> dict[int, dict]:
    streams = getattr(_command_recording, "streams", None)
    if streams is None:
        streams = {}
        _command_recording.streams = streams
    return streams


def _command_stream(host) -> dict:
    assert_runtime_owner_thread()
    identity = int(host.identity)
    streams = _command_streams()
    stream = streams.get(identity)
    if stream is None:
        stream = {
            "host": host,
            "updates": [],
            "dispatches": [],
            "updated": set(),
            "mesh_attributes": {},
            "pending": deque(),
            "failure": None,
        }
        streams[identity] = stream
    if stream["failure"] is not None:
        raise ComputeExecutionError(stream["failure"])
    return stream


def _append_kernel_commands(commands: dict, executable, params, *, gradients=None, bounds=None, cache=True):
    """Merge only the kernel's immutable private argument initializations."""
    updates = dispatches = update = None
    recorded = False
    try:
        if bounds is None and cache:
            updates, dispatches = executable.prepare(params, gradients=gradients)
        else:
            begin, end = (None, None) if bounds is None else bounds
            updates, dispatches = executable.prepare(
                params, gradients=gradients, begin=begin, end=end, cache=False,
            )
        for update in updates:
            identity = id(update[0])
            if identity not in commands["updated"]:
                commands["updates"].append(update)
                commands["updated"].add(identity)
                recorded = True
        commands["dispatches"].extend(dispatches)
        return recorded or bool(dispatches)
    finally:
        # An exception traceback must not independently retain native owners.
        updates = dispatches = update = None


def _append_mesh_attribute_dispatches(stream: dict, commands: dict) -> None:
    """Derive into temporary commands, leaving the open builder retryable."""
    for vertex_buffer in stream["mesh_attributes"].values():
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
        _append_kernel_commands(commands, executable, params)


def _prepare_commands(host, source: dict, reads=(), *, copies=()) -> _PreparedCommands | None:
    """Prepare one packet; its caller owns admission and ordering."""
    if not (source["updates"] or source["dispatches"] or source["mesh_attributes"] or reads or copies):
        return None
    commands = {
        "updates": list(source["updates"]),
        "dispatches": list(source["dispatches"]),
        "updated": set(source["updated"]),
    }
    try:
        _append_mesh_attribute_dispatches(source, commands)
        packet = None
        if commands["updates"] or commands["dispatches"] or reads or copies:
            packet = _PreparedCommands(host.prepare_batch(
                commands["dispatches"], commands["updates"], reads, copies,
            ), int(host.identity))
        for key in ("updates", "dispatches", "updated", "mesh_attributes"):
            source[key].clear()
        return packet
    except Exception as exception:
        raise ComputeExecutionError("GPU compute batch preparation failed") from exception
    finally:
        for values in commands.values():
            values.clear()
        commands.clear()
        reads = copies = None


def _seal_commands(stream: dict, reads=(), *, source=None, copies=()) -> _PreparedCommands | None:
    """Seal ordinary imperative work at its original FIFO position."""
    try:
        packet = _prepare_commands(stream["host"], stream if source is None else source, reads, copies=copies)
        if packet is not None:
            stream["pending"].append(packet)
        return packet
    finally:
        reads = copies = None


def _native_storage_blocks(host_identity, native):
    """Native retained buffer capacities, including accepted readback storage.

    Duplicate handles are intentional; consumers deduplicate across owners.
    This is neither VMA block occupancy nor command-upload page residency.
    """
    try:
        blocks = native.storage_blocks
    finally:
        native = None  # Metadata allocation failure must not pin an unregistered batch in its traceback.
    return ((("gpu", host_identity, int(index), int(generation)), int(size))
            for index, generation, size in blocks)


def _storage_blocks(values):
    """Backing allocation identities/capacities shared by graph lifetime and budgets."""
    blocks = {}
    for value in values:
        if value is None:
            continue
        array = value if isinstance(value, np.ndarray) else value._array
        if array is None:
            if value._native is not None:
                blocks.update(_native_storage_blocks(int(value._host.identity), value._native))
            continue
        base = root = array
        while True:
            if isinstance(base, np.ndarray):
                root = base
            owner = base.obj if isinstance(base, memoryview) else getattr(base, "base", None)
            if owner is None:
                break
            base = owner
        if isinstance(base, np.ndarray):
            size = base.nbytes
        else:
            try:
                size = memoryview(base).nbytes
            except TypeError:
                base, size = root, root.nbytes
        blocks[id(base)] = int(size)
    return blocks


def _has_pending_write(stream, native) -> bool:
    """Query actual writer declarations without sealing or admitting the FIFO."""
    if any(packet.native.has_pending_write(native) for packet in stream["pending"]):
        return True
    if any(buffer is native for buffer, _, _ in stream["updates"]):
        return True
    for _, buffers, accesses, *_ in stream["dispatches"]:
        if any(buffer is native and access in ("write", "read_write")
               for buffer, access in zip(buffers, accesses)):
            return True
    return any(value._native is native for value in stream["mesh_attributes"].values())


def _prepare_work_commands(launches):
    """Prepare one detached, ordered dispatch segment on the original queue."""
    assert_runtime_owner_thread()
    if _recording_depth():
        raise RuntimeError("managed GPU work cannot prepare inside compute.recording")
    commands = {"updates": [], "dispatches": [], "updated": set(), "mesh_attributes": {}}
    executable = params = gradients = None
    try:
        host = launches[0][0].host
        for executable, params, bounds, gradients in launches:
            if executable.host is not host:
                raise ValueError("one managed GPU segment requires one compute host")
            _append_kernel_commands(
                commands, executable, params, gradients=gradients, bounds=bounds, cache=False,
            )
        packet = _prepare_commands(host, commands)
        if packet is None:
            raise ComputeExecutionError("managed GPU operation produced no work")
        return packet
    finally:
        for values in commands.values():
            values.clear()
        commands.clear()
        launches = executable = params = gradients = host = None


def _cancel_work_commands(host, packet):
    """Cancel this unaccepted packet only; accepted work keeps its exact ticket."""
    assert_runtime_owner_thread()
    if packet.completion is not None:
        return
    stream = _command_streams().get(int(host.identity))
    if stream is not None and packet in stream["pending"]:
        stream["pending"].remove(packet)
    packet._cancel_unaccepted()


def _submit_commands(packet, *, block: bool, on_accepted) -> bool:
    """The one acceptance path for an imperative head or detached managed work."""
    try:
        serial = packet.native.submit() if block else packet.native.try_submit()
    except Exception as exception:
        # Native acceptance is final even if host reporting subsequently fails.
        if int(packet.native.submission_serial) != 0:
            packet._release_accepted(on_accepted)
        raise ComputeExecutionError("GPU compute submission failed") from exception
    if serial is None:
        return False
    packet._release_accepted(on_accepted)
    return True


def _drain_commands(stream: dict, *, block: bool, until=None, max_submissions=None) -> bool:
    if stream["failure"] is not None:
        raise ComputeExecutionError(stream["failure"])
    pending = stream["pending"]
    if max_submissions is not None and max_submissions <= 0:
        return not pending
    submissions = 0
    from infernux.application import Application
    engine = Application._current_engine()
    owner = None if engine is None else engine._cpu_work()
    while pending:
        packet = pending[0]
        if not packet.native.has_work:
            packet._cancel_unaccepted()
            pending.popleft()
            if packet is until:
                return True
            continue
        if packet.needs_preparation:
            if owner is not None and not block and owner._frame.active:
                return False  # The common owner prepares this head in its host lane.
            if owner is not None and block:
                owner._collect_background_receipt()
            if owner is None:
                ready = packet.prepare_storage(65536, managed=not block)
            else:
                ready = owner._prepare_fifo(packet, owner.transfer_chunk_bytes, block=block)
            if not ready:
                return False
        work = packet.submission
        try:
            if owner is not None and not block and not owner._can_admit_fifo(work):
                return False
            accepted = (lambda value: owner._record_fifo_packet(value)) if owner is not None else (lambda value: None)
            if not _submit_commands(work, block=block, on_accepted=accepted):
                return False
        finally:
            if packet.native is None:
                pending.popleft()
        submissions += 1
        if packet is until and packet.native is None:
            return True
        if max_submissions is not None and submissions >= max_submissions:
            return not pending
    return True


def _remove_empty_stream(stream: dict) -> None:
    if not (stream["failure"] or stream["pending"] or stream["updates"] or stream["dispatches"] or
            stream["mesh_attributes"]):
        _command_streams().pop(int(stream["host"].identity), None)


@_automatic_cost
def _flush_commands(host=None, *, block: bool = True, max_submissions=None) -> bool:
    """Seal then admit work in host FIFO order; True means accepted, not done.

    Only explicit block=False callers may leave work pending. Ordinary public
    reads, writes, launches and close retain their strict admission boundary.
    """
    assert_runtime_owner_thread()
    streams = _command_streams()
    identities = tuple(streams) if host is None else (int(host.identity),)
    accepted = True
    for identity in identities:
        stream = streams.get(identity)
        if stream is None:
            continue
        if stream["failure"] is not None:
            raise ComputeExecutionError(stream["failure"])
        _seal_commands(stream)
        if _drain_commands(stream, block=block, max_submissions=max_submissions):
            streams.pop(identity)
        else:
            accepted = False
    _collect_retiring_native_kernels()
    return accepted


def _fail_command_stream(stream: dict, exception: Exception) -> None:
    if stream["failure"] is not None:
        return
    from infernux.debug import Debug

    cause = exception.__cause__ or exception
    stream["failure"] = (
        f"GPU compute stopped: {cause}. Call infernux.compute.discard_pending() "
        "before recording replacement work."
    )
    # Keep text, not an exception/traceback that owns the unaccepted payload.
    Debug.log_error(stream["failure"])


def _admit_commands(host) -> None:
    """Try one sealed FIFO head after the common owner grants admission credit."""
    assert_runtime_owner_thread()
    stream = _command_streams().get(int(host.identity))
    if stream is not None and stream["failure"] is None:
        try:
            _drain_commands(stream, block=False, max_submissions=1)
        except Exception as exception:
            _fail_command_stream(stream, exception)


def _collect_commands(host) -> bool:
    """Retire without admission, preparation or mapping; report further upkeep."""
    assert_runtime_owner_thread()
    stream = _command_streams().get(int(host.identity))
    try:
        native_pending = int(host.collect())
        _collect_retiring_native_kernels()
    except Exception as exception:
        if stream is None:
            stream = _command_stream(host)
        _fail_command_stream(stream, exception)
        return False
    if stream is not None:
        _remove_empty_stream(stream)
    return native_pending > 0 or bool(
        stream is not None and stream["failure"] is None and stream["pending"]
    )


def _request_compute_frame() -> None:
    """Wake the existing frame owner only while its compute work needs upkeep."""
    from infernux.application import Application

    engine = Application._current_engine()
    host = getattr(engine, "_infernux_compute_host", None) if engine is not None else None
    if host is None:
        return
    stream = _command_streams().get(int(host.identity))
    if (stream is not None and stream["failure"] is None and stream["pending"]) or int(host.collect()) > 0:
        engine.get_native_engine().request_full_speed_frame()


def discard_pending() -> None:
    """Discard unaccepted/open compute work, including a stopped FIFO.

    Accepted GPU commands still complete normally. Recreate the desired work
    after this explicit recovery boundary; discarded reads remain cancelled.
    """
    _discard_commands()


def _discard_commands() -> None:
    """Cancel unaccepted authoring work before releasing the engine host lease."""
    assert_runtime_owner_thread()
    streams = _command_streams()
    for stream in streams.values():
        for packet in stream["pending"]:
            packet._cancel_unaccepted()
        for key in ("pending", "updates", "dispatches", "updated", "mesh_attributes"):
            stream[key].clear()
        stream["failure"] = None
        stream["host"] = None
    streams.clear()


@_automatic_cost
def _flush_cpu_meshes() -> None:
    """Publish CPU-lowered compute vertex streams once per native frame."""
    for value in tuple(_cpu_dirty_mesh_buffers):
        _cpu_dirty_mesh_buffers.discard(value)
        binding = getattr(value, "_cpu_mesh_binding", None)
        if binding is None or value.closed:
            continue
        renderer, vertex_count, indices, name, space = binding
        renderer._publish_cpu_vertex_buffer(
            value, vertex_count, indices, name, space
        )


def _flush_cpu_transform_bindings() -> None:
    """Publish the final CPU-lowered simulation pose for the native frame."""
    for binding in tuple(_cpu_dirty_transform_bindings):
        _cpu_dirty_transform_bindings.discard(binding)
        binding._flush_cpu_pose()


def _collect_retiring_native_kernels() -> None:
    """Poll only kernels that own bindings for wrappers already closed."""
    for identity, native in tuple(_retiring_native_kernels.items()):
        native.collect()
        if int(native.pending_dispatch_count) == 0:
            _retiring_native_kernels.pop(identity, None)


def _discard_buffer_launches(value: "Buffer") -> None:
    for declaration in tuple(_kernel_declarations):
        declaration._discard_buffer_launches(value)


def _validate_reads(host, reads) -> None:
    for value, byte_offset, byte_size in reads:
        if not isinstance(value, Buffer) or value.device != "gpu":
            raise TypeError("Compute readback requires GPU inx.buffer values")
        value._require_open()
        if int(value._host.identity) != int(host.identity):
            raise ValueError("Compute readback buffers must use the requested compute host")
        if not isinstance(byte_offset, int) or not isinstance(byte_size, int):
            raise TypeError("Compute readback byte offset and size must be integers")
        if (byte_size <= 0 or byte_offset < value._byte_offset or
                byte_offset - value._byte_offset > value.nbytes or
                byte_size > value.nbytes - (byte_offset - value._byte_offset)):
            raise ValueError("Compute readback range exceeds the buffer view")
        if byte_offset % 4 or byte_size % 4:
            raise ValueError("Compute readback byte offset and size must be four-byte aligned")


def _prepare_readbacks(host, reads, layouts):
    """Seal reads at this FIFO position; result wrappers own teardown cleanup."""
    reads = tuple(reads)
    _validate_reads(host, reads)
    native_reads = []
    try:
        native_reads.extend((value._native, offset, size) for value, offset, size in reads)
        stream = _command_stream(host)
        packet = _seal_commands(stream, native_reads)
        results = [Readback(dtype, shape, task=packet.native.readback(index), host=host, packet=packet)
                   for index, (dtype, shape) in enumerate(layouts)]
        return stream, packet, results
    finally:
        native_reads.clear()


def _readback_destination(dtype, shape, out):
    """Validate before admission; every new destination is fully overwritten."""
    if out is None:
        return Buffer._empty(shape=shape, dtype=dtype, device="cpu")
    if not isinstance(out, Buffer) or out.device != "cpu":
        raise TypeError("Compute readback out must be a CPU Buffer")
    out._require_writable()
    if out.element_count != int(np.prod(shape)) or out._dtype != dtype:
        raise ValueError("Compute readback out layout does not match the source")
    return out


@_automatic_cost
def _submit_and_read(host, reads) -> list[bytes]:
    """Fuse open work and tail reads, then explicitly wait for their results."""
    reads = tuple(reads)
    if not reads:
        return []
    results = []
    try:
        # Raw native byte ranges need no second typed CPU materialization.
        layouts = ((_buffer_dtype(np.uint32), (size // 4,)) for _, _, size in reads)
        _, _, results = _prepare_readbacks(host, reads, layouts)
        return [result._take_bytes() for result in results]
    except (TypeError, ValueError):
        raise
    except Exception as exception:
        raise ComputeExecutionError("GPU compute readback failed") from exception
    finally:
        for result in results:
            result.cancel()
        results.clear()
        result = None


class _RecordingScope:
    """Allocation-free reentrant command-recording scope."""

    __slots__ = ("_block", "_automatic")

    def __init__(self, block=True, *, automatic=False):
        self._block = block
        self._automatic = automatic

    def __enter__(self):
        outermost = _recording_depth() == 0
        if self._automatic:
            _command_recording.automatic_depth = getattr(_command_recording, "automatic_depth", 0) + 1
        if outermost:
            _command_recording.boundary_activity = getattr(_command_recording, "activity", 0)
        if not self._automatic:
            explicit_depth = _explicit_recording_depth()
            if explicit_depth == 0:
                _command_recording.block = self._block
            _command_recording.explicit_depth = explicit_depth + 1
        if outermost and _web_cpu_compute_enabled():
            _command_recording.cpu_batch_started = perf_counter()
            _command_recording.cpu_batch_dispatches = int(_cpu_statistics["dispatch_count"])
        _command_recording.depth = _recording_depth() + 1
        return None

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._exit(exc_type, exc, tb)
        finally:
            if self._automatic:
                _command_recording.automatic_depth -= 1

    def _exit(self, exc_type, exc, tb):
        _command_recording.depth -= 1
        outermost = _recording_depth() == 0
        if self._automatic:
            boundary = outermost and getattr(_command_recording, "activity", 0) != (
                _command_recording.boundary_activity
            )
            block = True
        else:
            _command_recording.explicit_depth -= 1
            boundary = _command_recording.explicit_depth == 0
            block = _command_recording.block
        if boundary:
            # Explicit scopes retain their admission boundary even inside an
            # automatic phase. Empty automatic phases leave earlier work alone.
            # A real frame consumer still needs the ordered FIFO prefix.
            try:
                if _command_streams():
                    _flush_commands(block=block, max_submissions=None if block else 1)
                    if not block:
                        _request_compute_frame()
            finally:
                # An enclosing automatic scope must not retry an explicit
                # failure or turn an explicit nonblocking boundary into a wait.
                _command_recording.boundary_activity = getattr(_command_recording, "activity", 0)
        if outermost:
            if _cpu_mesh_frame_depth() == 0 and _cpu_dirty_mesh_buffers:
                _flush_cpu_meshes()
            if _web_cpu_compute_enabled():
                dispatches = int(_cpu_statistics["dispatch_count"]) - int(
                    _command_recording.cpu_batch_dispatches
                )
                if dispatches > 0:
                    _cpu_statistics["submission_count"] += 1
                    _cpu_statistics["cpu_submit_ms"] += (
                        perf_counter() - _command_recording.cpu_batch_started
                    ) * 1000.0
                _command_recording.cpu_batch_started = None
                _command_recording.cpu_batch_dispatches = 0
        return False


_recording_scope = _RecordingScope()
_nonblocking_recording_scope = _RecordingScope(False)
_automatic_recording_scope = _RecordingScope(automatic=True)


def recording(*, block: bool = True):
    """Record ordered :func:`launch` calls at a shared admission boundary.

    This is a command-recording boundary only. It does not create a second
    device or imply a readback; explicit buffer reads and native consumers
    flush pending work before observing it. Nested scopes preserve the outer
    boundary so fixed-step solvers can group dependent kernels. With
    ``block=False``, try one FIFO head and let the engine's frame owner
    continue admission without waiting for a free slot. The outermost scope
    controls admission; explicit reads and strict consumers may still wait.
    """
    if not isinstance(block, bool):
        raise TypeError("inx.compute.recording block must be a bool")
    return _recording_scope if block else _nonblocking_recording_scope


def _record_commands():
    """Batch an automatic engine phase, submitting only its actual activity."""
    return _automatic_recording_scope


def _queue_buffer_update(value: "Buffer", payload: bytes, byte_offset: int) -> None:
    stream = _command_stream(value._host)
    native_identity = id(value._native)
    # Native batches place uploads before dispatches and allow one update per
    # buffer. Seal at either conflict to preserve the authored write order.
    if stream["dispatches"] or native_identity in stream["updated"]:
        _seal_commands(stream)
    stream["updates"].append((value._native, payload, byte_offset))
    stream["updated"].add(native_identity)
    _record_activity()
    if _recording_depth() == 0:
        _flush_commands(value._host)


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
    from infernux.math import vector2, vector3, vector4

    vectors = {
        vector2: _BufferDType("vector2", np.dtype(np.float32), 2),
        vector3: _BufferDType("vector3", np.dtype(np.float32), 3),
        vector4: _BufferDType("vector4", np.dtype(np.float32), 4),
    }
    if value in vectors:
        return vectors[value]
    if value is float:
        value = np.float32
    elif value is int:
        value = np.int32
    dtype = np.dtype(value)
    if dtype not in {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.int64)}:
        raise TypeError("inx.buffer scalar dtype must be float32, int32, uint32, or int64")
    return _BufferDType(dtype.name, dtype)


def _logical_shape(value) -> tuple[int, ...]:
    shape = (value,) if isinstance(value, int) else tuple(value)
    if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("inx.buffer shape must contain positive integers")
    return shape


def _native_compute_host():
    assert_runtime_owner_thread()
    from infernux.application import Application

    engine = Application._current_engine()
    native = engine.get_native_engine() if engine is not None else None
    if native is None or not native.has_gpu_compute:
        raise ComputeCapabilityError("GPU buffers require graphical startup or init_headless(..., gpu=True)")
    host = getattr(engine, "_infernux_compute_host", None)
    if host is None:
        try:
            host = native._acquire_compute_host()
        except Exception as exception:
            raise ComputeCapabilityError("The active engine cannot provide GPU compute") from exception
        engine._infernux_compute_host = host
    return host


class Buffer:
    """Engine-owned typed storage shared by compute, mesh, and rendering.

    CPU buffers are directly indexable. GPU buffers remain device resident and
    require explicit :meth:`set_data` / :meth:`get_data` transfers; ordinary
    Python indexing never triggers a hidden readback.
    """

    def __init__(self, *, shape, dtype=float, device: Literal["cpu", "gpu"], data=None) -> None:
        self._initialize(shape, dtype, device, data, initialize=True)

    @classmethod
    def _empty(cls, *, shape, dtype=float, device, managed=False):
        """Private storage for a proven full writer; no initial upload/readable value."""
        result = object.__new__(cls)
        if result._initialize(shape, dtype, device, None, initialize=False, managed=managed):
            return result

    @classmethod
    def _take_cpu_array(cls, array):
        """Adopt a completed private allocation without copying its storage."""
        result = object.__new__(cls)
        result._configure(array.shape, array.dtype, "cpu")
        result._array = array
        result._description = result._build_description()
        return result

    def _configure(self, shape, dtype, device):
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
        self._cpu_mapped = bool(device == "gpu" and _web_cpu_compute_enabled())
        self._cpu_mesh_binding = None
        self._mesh_attribute_state = None
        self._dependent_resources = []
        self._generation = next(_buffer_generations)
        self._byte_offset = 0
        self._ownership = "owned"
        self._readonly = False
        self._description = None


    @classmethod
    def _from_native(cls, native, *, shape, byte_offset=0):
        """Adopt a real accepted scalar value; no allocation, upload or copy."""
        result = object.__new__(cls)
        result._configure(shape, native.scalar_type, "gpu")
        host = _native_compute_host()
        if not native.belongs_to(host):
            raise ValueError("GPU value belongs to another or retired engine")
        if native.lanes != 1 or byte_offset < 0 or result.nbytes > native.byte_size - byte_offset:
            raise ValueError("GPU value layout exceeds its actual scalar storage")
        result._host, result._native = host, native
        result._byte_offset, result._readonly, result._ownership = byte_offset, True, "borrowed"
        result._description = result._build_description()
        _gpu_buffers.add(result)
        return result

    @_automatic_cost
    def _initialize(self, shape, dtype, device, data, *, initialize, managed=False):
        self._configure(shape, dtype, device)

        initial = None
        if data is not None:
            # Every owned buffer has independent storage.  A contiguous input
            # array may otherwise be returned unchanged by ``ascontiguousarray``;
            # that would make multiple Web CPU mappings alias one authored GPU
            # upload even though native GPU buffers always copy the payload.
            initial = self._coerce_data(data).copy(order="C")
        elif initialize:
            initial = np.zeros(self._storage_shape, dtype=self._dtype.scalar)
        if device == "cpu" or self._cpu_mapped:
            self._array = np.empty(self._storage_shape, dtype=self._dtype.scalar) if initial is None else initial
        else:
            self._host = _native_compute_host()
            layout = self.element_count, self._dtype.scalar.name, self._dtype.lanes
            self._native = self._host.create_buffer(*layout, managed=True) if managed else self._host.create_buffer(*layout)
            if self._native is None:
                return False
            _gpu_buffers.add(self)
            if initial is not None:
                _queue_buffer_update(self, initial.tobytes(order="C"), 0)
        self._description = self._build_description()
        return True

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
        return self.element_count * self._dtype.itemsize

    @property
    def element_count(self) -> int:
        return math.prod(self._shape)

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
        if self._native is not None:
            self._description = self._build_description()
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
        byte_length = self.nbytes
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
            byte_length=byte_length,
            nbytes=byte_length,
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
        if self._native is not None:
            assert_runtime_owner_thread()

    def _require_writable(self) -> None:
        self._require_open()
        if self._readonly:
            raise RuntimeError("inx.buffer view is read-only")

    def _input_array(self, data) -> np.ndarray:
        if isinstance(data, Buffer):
            data._require_open()
            if data.device != "cpu":
                raise TypeError("GPU-to-GPU copies require the compute copy operation")
            source = data._array
        else:
            source = data
        from infernux.math import vector2, vector3, vector4
        if isinstance(source, (list, tuple)) and source and isinstance(source[0], (vector2, vector3, vector4)):
            vector_type = {2: vector2, 3: vector3, 4: vector4}.get(self._dtype.lanes)
            if vector_type is None or any(not isinstance(item, vector_type) for item in source):
                raise TypeError("vector input must match the buffer's vector width")
            source = [[getattr(item, axis) for axis in "xyzw"[:self._dtype.lanes]] for item in source]
        return np.asarray(source, dtype=self._dtype.scalar)

    def _coerce_data(self, data) -> np.ndarray:
        result = self._input_array(data)
        if result.shape != self._storage_shape:
            raise ValueError(f"inx.buffer data shape must be {self._storage_shape}, got {result.shape}")
        return np.ascontiguousarray(result)

    def _coerce_element_range(self, data) -> np.ndarray:
        if isinstance(data, Buffer):
            if data._dtype != self._dtype:
                raise ValueError("inx.buffer range dtype does not match the destination")
        result = self._input_array(data)
        if self._dtype.lanes > 1:
            if result.ndim == 0 or result.shape[-1] != self._dtype.lanes:
                raise ValueError(f"inx.buffer range data must end in {self._dtype.lanes} vector lanes")
            result = result.reshape(-1, self._dtype.lanes)
        else:
            result = result.reshape(-1)
        return np.ascontiguousarray(result)

    @_automatic_cost
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
        if self._array is not None:
            np.copyto(flat[offset:offset + count], values, casting="no")
        else:
            _queue_buffer_update(
                self,
                values.tobytes(order="C"),
                self._byte_offset + offset * self._dtype.itemsize,
            )

    @_automatic_cost
    def get_data(self, out: "Buffer | None" = None, *, offset: int = 0, count: int | None = None) -> "Buffer":
        """Return a CPU element-range snapshot; GPU readback completes before return."""
        self._require_open()
        if not isinstance(offset, int) or offset < 0 or offset > self.element_count:
            raise ValueError("inx.buffer get_data offset must address the buffer")
        count = self.element_count - offset if count is None else count
        if not isinstance(count, int) or count <= 0 or count > self.element_count - offset:
            raise ValueError("inx.buffer get_data range exceeds the source")
        output_shape = self._shape if offset == 0 and count == self.element_count else (count,)
        if self._array is not None:
            out = _readback_destination(self._dtype, output_shape, out)
            destination = out._array.reshape((-1, self._dtype.lanes)) if self._dtype.lanes > 1 else out._array.reshape(-1)
            source = self._array.reshape((-1, self._dtype.lanes)) if self._dtype.lanes > 1 else self._array.reshape(-1)
            np.copyto(destination, source[offset:offset + count], casting="no")
        else:
            if out is not None:
                _readback_destination(self._dtype, output_shape, out)
            byte_size = count * self._dtype.itemsize
            _, _, (request,) = _prepare_readbacks(
                self._host,
                ((self, self._byte_offset + offset * self._dtype.itemsize, byte_size),),
                ((self._dtype, output_shape),),
            )
            try:
                return request.get_data(out)
            except (TypeError, ValueError):
                raise
            except Exception as exception:
                raise ComputeExecutionError("GPU compute readback failed") from exception
            finally:
                request.cancel()
        return out

    @_automatic_cost
    def get_data_async(self, *, offset: int = 0, count: int | None = None) -> "Readback":
        """Seal a snapshot and try one FIFO head without waiting for a slot.

        Busy requests progress through the engine's early owner maintenance;
        get_data() is an explicit strict completion boundary.
        """
        self._require_open()
        if not isinstance(offset, int) or offset < 0 or offset > self.element_count:
            raise ValueError("inx.buffer get_data_async offset must address the buffer")
        count = self.element_count - offset if count is None else count
        if not isinstance(count, int) or count <= 0 or count > self.element_count - offset:
            raise ValueError("inx.buffer get_data_async range exceeds the source")
        output_shape = self._shape if offset == 0 and count == self.element_count else (count,)
        if self._array is not None:
            return Readback(self._dtype, output_shape, immediate=self.get_data(offset=offset, count=count))
        byte_size = count * self._dtype.itemsize
        stream, _, results = _prepare_readbacks(
            self._host,
            ((self, self._byte_offset + offset * self._dtype.itemsize, byte_size),),
            ((self._dtype, output_shape),),
        )
        _drain_commands(stream, block=False, max_submissions=1)
        _remove_empty_stream(stream)
        _request_compute_frame()
        return results[0]

    @_automatic_cost
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

    @_automatic_cost
    def numpy(self, *, copy: bool = True) -> np.ndarray:
        """Expose a CPU buffer as NumPy; GPU callers must use get_data first."""
        self._require_open()
        if self._array is None:
            raise RuntimeError("GPU buffers require get_data() before NumPy access")
        return self._array.copy() if copy else self._array

    def view(self, *, offset: int = 0, count: int | None = None, readonly: bool = False,
             shape: int | tuple[int, ...] | None = None) -> "Buffer":
        """Borrow a contiguous element range without copying storage.

        By default the logical shape is flat. An explicit shape describes
        the range and supplies its count when omitted. The view retains the
        allocation independently of the original wrapper.
        """
        self._require_open()
        capacity = self.element_count
        if not isinstance(offset, int) or offset < 0 or offset > capacity:
            raise ValueError("inx.buffer view offset must address the buffer")
        if shape is not None:
            shape = _logical_shape(shape)
            size = math.prod(shape)
            if count is None:
                count = size
            elif count != size:
                raise ValueError("inx.buffer view shape must match the element count")
        elif count is None:
            count = capacity - offset
        if not isinstance(count, int) or count <= 0 or count > capacity - offset:
            raise ValueError("inx.buffer view range exceeds the source")
        if not isinstance(readonly, bool):
            raise TypeError("inx.buffer view readonly must be a bool")

        result = object.__new__(Buffer)
        result._shape = (count,) if shape is None else shape
        result._dtype = self._dtype
        result._storage_shape = result._shape + ((self._dtype.lanes,) if self._dtype.lanes > 1 else ())
        result._device = self._device
        result._closed = False
        result._host = self._host
        result._native = self._native
        result._mesh_attribute_state = None
        result._cpu_mapped = self._cpu_mapped
        result._cpu_mesh_binding = None
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
            if shape is not None:
                result._array = result._array.reshape(result._storage_shape)
            if result._readonly:
                # Permissions belong to this borrowed ndarray, not its owner.
                # NumPy slices and cooked CPU kernels inherit the same flag
                # without checks or copies on each access.
                result._array.flags.writeable = False
        result._description = result._build_description()
        if result._device == "gpu":
            _gpu_buffers.add(result)
        return result

    def __len__(self) -> int:
        return self._shape[0]

    def __getitem__(self, index):
        self._require_open()
        if self._array is None:
            raise RuntimeError("GPU buffers cannot be indexed from Python; use get_data()")
        value = self._array[index]
        if self._cpu_mapped:
            return value
        if self._dtype.lanes == 1:
            return value.item() if np.isscalar(value) else value
        if getattr(value, "ndim", 0) != 1:
            return value
        from infernux.math import vector2, vector3, vector4
        constructor = {2: vector2, 3: vector3, 4: vector4}[self._dtype.lanes]
        return constructor(*(float(component) for component in value))

    def __setitem__(self, index, value) -> None:
        self._require_writable()
        if self._array is None:
            raise RuntimeError("GPU buffers cannot be indexed from Python; use set_data()")
        if self._dtype.lanes > 1 and hasattr(value, "x"):
            value = tuple(getattr(value, axis) for axis in "xyzw"[: self._dtype.lanes])
        self._array[index] = value

    def close(self) -> None:
        self._close(flush=True)

    def _publish_storage(self):
        if self._native is not None:
            self._native.publish_storage()

    def _close(self, *, flush: bool) -> None:
        """Share cleanup; only a retired private value may skip public FIFO flush."""
        if self._closed:
            return
        self._require_open()
        if not flush and (self._mesh_attribute_state is not None or self._cpu_mesh_binding is not None
                          or self._dependent_resources):
            raise ValueError("private retirement requires plain numerical storage")
        if self._device == "gpu" and self._host is not None:
            # Derived mesh work still needs its topology while sealing.
            if flush:
                _flush_commands(self._host)
            _discard_buffer_launches(self)
        attribute_state = self._mesh_attribute_state
        self._mesh_attribute_state = None
        if attribute_state is not None:
            attribute_state.close()
        dependents, self._dependent_resources = self._dependent_resources, []
        for dependent in dependents:
            dependent.close()
        _gpu_buffers.discard(self)
        self._native = None
        self._host = None
        self._array = None
        self._cpu_mesh_binding = None
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


class _BufferCapacityPending(Exception):
    """A managed preparation can continue once the device has room."""


def _managed_empty(*, shape, dtype, device):
    value = Buffer._empty(shape=shape, dtype=dtype, device=device, managed=device == "gpu")
    if value is None:
        raise _BufferCapacityPending
    return value


@dataclass(frozen=True)
class _TransferStep:
    kind: str
    begin: int
    end: int
    allocate: bool = False

    @property
    def domain(self):
        return "gpu" if self.kind in ("upload", "device_copy", "readback", "submit") else "host"


def _transfer_chunk_bytes(chunk_bytes, itemsize):
    result = chunk_bytes // itemsize * itemsize
    if not result:
        raise ValueError("transfer_bytes must fit at least one complete element")
    return result


def _transfer_prepare_bytes(nbytes, source_storage, target_storage):
    if source_storage == target_storage:
        return 0
    if target_storage == "gpu":
        from infernux.lib import _Infernux

        return nbytes + _Infernux._compute_upload_staging_bytes(nbytes)
    return 2 * nbytes


def _fit_aligned_range(extent, itemsize, fits):
    maximum = max(0, extent // itemsize * itemsize)
    if not maximum or fits(maximum):
        return maximum
    # Staging pages change size in steps. Find the largest whole-element range
    # within the actual working set, including at a page-class boundary.
    lo, hi = 0, maximum // itemsize
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if fits(middle * itemsize):
            lo = middle
        else:
            hi = middle - 1
    return lo * itemsize


def _fit_transfer_chunk(chunk_bytes, itemsize, available, source_storage, target_storage):
    return _fit_aligned_range(chunk_bytes, itemsize,
        lambda size: _transfer_prepare_bytes(size, source_storage, target_storage) <= available)


def _transfer_work(nbytes, chunk_bytes, *, source_storage, target_storage, begin=0,
                   pending_end=0, allocate=True, prepared=False, accepted=False, readback_ready=False):
    """Numerical recipe shared by live cursors and future model-entry layouts.

    Storage names describe physical CPU/GPU storage; a web CPU fallback still
    has CPU storage. This function owns and allocates no Buffer or device work.
    """
    blocks = []

    def block(first, last, kind=None):
        action = (kind if kind is not None else "copy_host" if source_storage == target_storage == "cpu" else
                  "device_copy" if source_storage == target_storage == "gpu" else
                  "upload" if target_storage == "gpu" else "readback")
        steps = (_TransferStep(action, first, last, allocate),)
        if action in ("readback", "submit") and target_storage == "cpu":
            steps += (_TransferStep("materialize", first, last),)
        return steps

    if readback_ready or accepted and target_storage == "cpu":
        blocks.append(((_TransferStep("materialize", begin, pending_end, allocate),), 1))
        begin, allocate = pending_end, False
    elif accepted:
        begin = pending_end
    elif prepared:
        blocks.append((block(begin, pending_end, "submit"), 1))
        begin, allocate = pending_end, False
    if begin < nbytes and allocate:
        end = min(nbytes, begin+chunk_bytes)
        blocks.append((block(begin, end), 1))
        begin, allocate = end, False
    full, tail = divmod(nbytes-begin, chunk_bytes)
    middle = full if tail else max(0, full-1)
    if middle:
        blocks.append((block(begin, begin+chunk_bytes), middle))
        begin += middle*chunk_bytes
    if begin < nbytes:
        blocks.append((block(begin, nbytes), 1))
    return tuple(blocks)


def _transfer_host_key(identity, nbytes, step, *, finalizing=False):
    return ("transfer-host", *identity, step.kind, step.end-step.begin,
            nbytes if step.allocate else None, finalizing)


def _freeze_cpu_buffer(source, frame_work):
    """One atomic host snapshot shared by transfer and numerical admission."""
    reservation = frame_work.try_reserve_cpu_storage(source.nbytes)
    if reservation is None:
        raise ValueError("CPU source capture exceeds host storage capacity")
    snapshot = None
    try:
        snapshot = Buffer._empty(shape=source.shape, dtype=source.dtype, device="cpu")
        # Borrowed Buffer views and NumPy slices retain the owning ndarray.
        # Request retirement alone does not prove the allocation was freed.
        weakref.finalize(snapshot._array, lambda claim: None, reservation)
        np.copyto(snapshot._array, source._array.reshape(snapshot._storage_shape), casting="no")
        snapshot._array.flags.writeable = False
        snapshot._readonly = True
        snapshot._description = snapshot._build_description()
        return snapshot
    except BaseException:
        if snapshot is not None:
            snapshot._close(flush=False)
        raise
    finally:
        # An exception traceback can retain this frame after a failed allocation.
        reservation = None


def _freeze_gpu_buffer(source):
    """Retain one accepted GPU value without copying it at capture time."""
    _accepted_buffer_version(source)
    value = source.view(readonly=True)
    try:
        value._native = source._native.freeze()
        value._description = value._build_description()
        return value
    except BaseException:
        value._close(flush=False)
        raise


def _accepted_buffer_version(source):
    if int(source._host.identity) != int(_native_compute_host().identity):
        raise ValueError("source must belong to the active engine")
    if not int(source._native.last_write_serial) or not source._native.has_initialized_range(source._byte_offset, source.nbytes):
        raise ValueError("source requires an accepted producer for its full range")
    version = tuple(source._native.last_write_submission)
    if not _buffer_version_current(source, version):
        raise ValueError("source has an unaccepted writer")
    return version


def _buffer_version_current(source, version):
    if source.closed or tuple(source._native.last_write_submission) != version:
        return False
    stream = _command_streams().get(int(source._host.identity))
    return stream is None or not _has_pending_write(stream, source._native)


class _TransferCursor:
    """One private copied value, one packet or completed readback at a time."""

    def __init__(self, source, device, chunk_bytes):
        self.shape, self.dtype, self.nbytes = source.shape, source._dtype, source.nbytes
        # Larger Vulkan uploads use the same capacity-checked staging pool.
        self.chunk_bytes = _transfer_chunk_bytes(chunk_bytes, self.dtype.itemsize)
        self.device = device
        self.host = source._host
        if device == "gpu" and not _web_cpu_compute_enabled():
            self.host = _native_compute_host()
        if source._native is not None:
            self.source_version = _accepted_buffer_version(source)
        else:
            self.source_version = None
        self.source = _freeze_gpu_buffer(source) if source._native is not None else source.view(readonly=True)
        self._owned_source = False
        self.target = self.snapshot = self.readback = None
        self.completed_bytes = self.pending_end = 0
        self.readback_ready = False

    @classmethod
    def _take_owned_cpu(cls, source, device, chunk_bytes):
        """Consume an unpublished CPU allocation after its writer has finished.

        The engine caller retains no writable NumPy aliases. The original
        wrapper closes only after construction succeeds; this cursor keeps the
        readonly source allocation, without another full-size snapshot.
        """
        source._require_open()
        if source.device != "cpu" or source._ownership != "owned" or source._readonly:
            raise ValueError("private transfer requires an owned writable CPU allocation")
        result = cls(source, device, chunk_bytes)
        try:
            source._close(flush=False)
        except BaseException:
            result.close()
            raise
        result._owned_source = True
        return result

    @property
    def done(self):
        return self.completed_bytes == self.nbytes

    @property
    def execution_domain(self):
        return "gpu" if self.next_step().domain == "gpu" else "cpu"

    def remaining_work(self, *, prepared=False, accepted=False):
        """Pure, bounded-size recipe from this cursor's actual byte position.

        Each pair contains a small tuple of actions and its repeat count. A
        readback alternates device copy and host materialization per block;
        long buffers never require a descriptor per byte range. Packet facts
        belong to the caller: a prepared range has not necessarily executed.
        The recipe completes data movement, not its consumer's publication.
        """
        return _transfer_work(self.nbytes, self.chunk_bytes,
            source_storage=self.source_storage, target_storage=self.target_storage,
            begin=self.completed_bytes, pending_end=self.pending_end, allocate=self.target is None,
            prepared=prepared, accepted=accepted, readback_ready=self.readback_ready)

    def next_step(self, *, prepared=False):
        work = self.remaining_work(prepared=prepared)
        return (work[0][0][0] if work else
                _TransferStep("complete", self.nbytes, self.nbytes, self.target is None))

    @property
    def cost_identity(self):
        return self.source.device, self.device, self.source.dtype, 0 if self.host is None else int(self.host.identity)

    def host_cost_key(self, step, *, finalizing=False):
        return _transfer_host_key(self.cost_identity, self.nbytes, step, finalizing=finalizing)

    @property
    def reserved_bytes(self):
        return self.nbytes * (2 if self.source_version is None and not self._owned_source else 1)

    @property
    def source_storage(self):
        return "cpu" if self.source_version is None else "gpu"

    @property
    def target_storage(self):
        return self.device if self.host is not None else "cpu"

    @property
    def prepare_bytes(self):
        return _transfer_prepare_bytes(min(self.chunk_bytes, self.nbytes - self.completed_bytes),
                                       self.source_storage, self.target_storage)

    def limit_chunk(self, available):
        if self.prepare_bytes <= available:
            return
        chunk = _fit_transfer_chunk(min(self.chunk_bytes, self.nbytes - self.completed_bytes),
                                    self.dtype.itemsize, available, self.source_storage, self.target_storage)
        if not chunk:
            raise ValueError("transfer preparation cannot fit one element in numerical capacity")
        self.chunk_bytes = chunk

    @property
    def native_blocks(self):
        return (() if self.readback is None else
                tuple(_native_storage_blocks(int(self.host.identity), self.readback)))

    def capture_source(self, frame_work):
        """Freeze CPU input once; a read-only alias alone is not a snapshot."""
        if self.source_version is not None or self._owned_source:
            return
        old = self.source
        snapshot = _freeze_cpu_buffer(old, frame_work)
        self.source = self.snapshot = snapshot
        old._close(flush=False)

    def is_current(self):
        if self.source.closed:
            return False
        if self.source_version is None:
            return True
        if not _buffer_version_current(self.source, self.source_version):
            return False
        return (self.readback is None or not int(self.readback.submission_serial)
                or tuple(self.readback.source_write_submission) == self.source_version)

    def ensure_target(self):
        if self.target is None:
            self.target = _managed_empty(shape=self.shape, dtype=self.dtype, device=self.device)

    def next_range(self):
        step = self.next_step()
        return step.begin, step.end

    def copy_host_range(self, begin, end):
        np.copyto(self.target._array.view(np.uint8).reshape(-1)[begin:end],
                  self.source._array.view(np.uint8).reshape(-1)[begin:end], casting="no")
        self.completed_bytes = end

    def prepare(self, begin, end):
        assert_runtime_owner_thread()
        if _recording_depth():
            raise RuntimeError("managed GPU work cannot prepare inside compute.recording")
        commands = {"updates": [], "dispatches": [], "updated": set(), "mesh_attributes": {}}
        reads = ()
        copies = ()
        packet = None
        try:
            kind = self.next_step().kind
            if kind == "device_copy":
                copies = ((self.source._native, self.target._native, self.source._byte_offset + begin,
                           self.target._byte_offset + begin, end - begin),)
            elif kind == "upload":
                data = memoryview(self.source._array).cast("B")[begin:end]
                commands["updates"].append((self.target._native, data, self.target._byte_offset + begin))
            else:
                reads = ((self.source._native, self.source._byte_offset + begin, end - begin),)
            packet = _prepare_commands(self.host, commands, reads, copies=copies)
            if self.device == "cpu":
                self.readback = packet.native.readback(0)
            self.pending_end = end
            return packet
        except BaseException:
            if packet is not None:
                _cancel_work_commands(self.host, packet)
            raise
        finally:
            for values in commands.values():
                values.clear()
            commands.clear()
            data = reads = copies = None

    def consume_execution(self):
        if self.device == "gpu":
            self.completed_bytes = self.pending_end
            self.pending_end = 0
        else:
            self.readback_ready = True

    def materialize(self):
        """Only called with owner credit after the exact copy completion."""
        scalar_bytes = int(self.dtype.scalar.itemsize)
        target = self.target._array.reshape(-1)[
            self.completed_bytes // scalar_bytes:self.pending_end // scalar_bytes]
        self.readback.copy_into(target)
        self.completed_bytes = self.pending_end
        self.pending_end = 0
        self.readback = None
        self.readback_ready = False

    def close(self):
        if self.readback is not None:
            self.readback.cancel()
            self.readback = None
        for value in (self.source, self.target):
            if value is not None:
                value._close(flush=False)
        self.source = self.target = self.snapshot = self.host = None


def transfer(source: Buffer | BufferSource, *, device: Literal["cpu", "gpu"], finish_within_frames=None, finish_within_ms=None, budget=None):
    """Release a budgeted copy; result stays private until complete.

    CPU input is captured once during release and must not be concurrently
    modified during that capture. GPU input retains its accepted value. A
    BufferSource may still await its producer; binding and transfers then use
    the same request, host budget and original completion window. Ordinary
    buffer read/write methods retain their synchronous semantics.
    """
    from infernux.application import Application

    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError("managed transfers require an active Engine")
    owner = engine._cpu_work(budget)
    request = owner.release_transfer(source, device=device, window_opportunities=finish_within_frames,
                                     finish_within_ms=finish_within_ms)
    return engine._wake_cpu_request(owner, request)


def _copy_gpu_into(source, target):
    """Record an ordered byte copy into existing storage, including view offsets."""
    source._require_open()
    target._require_writable()
    if (source._native is None or target._native is None or source.nbytes != target.nbytes
            or int(source._host.identity) != int(target._host.identity)):
        raise ValueError("GPU copy requires equal byte ranges on one compute host")
    stream = _command_stream(source._host)
    _seal_commands(stream)
    packet = _seal_commands(stream, source=dict(updates=[], dispatches=[], updated=set(), mesh_attributes={}),
        copies=((source._native, target._native, source._byte_offset, target._byte_offset, source.nbytes),))
    _record_activity()
    return packet


@_automatic_cost
def _copy_gpu_buffer(source):
    """Explicit same-device copy, in author order, without a copy-only shader."""
    source._require_open()
    if _recording_depth():
        raise RuntimeError("owned GPU copies cannot start inside compute.recording")
    host = source._host
    target = Buffer._empty(shape=source.shape, dtype=source._dtype, device="gpu")
    packet = None
    try:
        packet = _copy_gpu_into(source, target)
        _flush_commands(host, block=True)
        return target
    except BaseException:
        if packet is not None:
            _cancel_work_commands(host, packet)
        target._close(flush=False)
        raise
    finally:
        source = target = host = packet = None


class Readback:
    """An explicit CPU snapshot of one buffer range.

    ``done`` only polls the exact GPU submission. ``get_data`` first admits
    this request in FIFO order, then waits and returns its CPU snapshot.
    ``cancel`` abandons the result without waiting; submitted GPU work completes
    normally and its storage retires with the engine queue.

    The first GPU ``get_data(out)`` writes directly into ``out`` and retains it
    as the result. Later ``get_data()`` returns that same mutable CPU Buffer;
    editing or closing it affects subsequent access. CPU requests already own
    their immediate snapshot before ``get_data`` is called.
    """

    def __init__(self, dtype: _BufferDType, shape: tuple[int, ...], *, task=None,
                 immediate: Buffer | None = None, host=None, packet=None) -> None:
        if host is not None or task is not None:
            assert_runtime_owner_thread()
        self._dtype = dtype
        self._shape = shape
        self._task = task
        self._result = immediate
        self._host = host
        self._packet = packet
        self._cancelled = False
        self._source_write_submission = None
        if host is not None or task is not None:
            _readbacks.add(self)

    @property
    def done(self) -> bool:
        if self._host is not None or self._task is not None:
            assert_runtime_owner_thread()
        return self.cancelled or self._result is not None or bool(self._task.done)

    @property
    def cancelled(self) -> bool:
        return self._cancelled or (self._task is not None and bool(self._task.cancelled))

    @property
    def source_write_submission(self) -> tuple[int, int, int] | None:
        """Actual (device, queue, serial), retained after CPU materialization.

        None means no GPU copy has been accepted, or this is a CPU snapshot.
        Producer acceptance is distinct from completion of the GPU copy.
        """
        self._capture_source_write_submission()
        return self._source_write_submission

    @property
    def source_write_serial(self) -> int | None:
        """Queue-local diagnostic counter; use source_write_submission for identity."""
        value = self.source_write_submission
        return None if value is None else value[2]

    def _capture_source_write_submission(self) -> None:
        if self._task is not None:
            assert_runtime_owner_thread()
            if self._source_write_submission is None and int(self._task.submission_serial) != 0:
                self._source_write_submission = tuple(self._task.source_write_submission)

    def cancel(self) -> bool:
        """Abandon an unconsumed result, not the submitted GPU commands.

        Returns False if already cancelled or if a CPU result is available.
        No polling, waiting or device-to-host mapping is performed.
        """
        if self._cancelled or self._result is not None:
            return False
        if self._host is not None or self._task is not None:
            assert_runtime_owner_thread()
        if self._task is not None:
            self._capture_source_write_submission()
            self._task.cancel()
        if self._packet is not None and self._packet.native is not None and not self._packet.native.has_work:
            # Empty read-only packets have no producer to preserve. Remove
            # them without recording a GPU no-op or admitting unrelated work.
            self._packet.native.cancel()
            stream = _command_streams()[int(self._host.identity)]
            stream["pending"].remove(self._packet)
            self._packet.native = None
            _remove_empty_stream(stream)
        self._cancelled = True
        self._task = None
        self._host = None
        self._packet = None
        _readbacks.discard(self)
        return True

    @_automatic_cost
    def get_data(self, out: Buffer | None = None) -> Buffer:
        if self._host is not None or self._task is not None:
            assert_runtime_owner_thread()
        if self.cancelled:
            raise RuntimeError("Readback result was cancelled")
        if self._result is not None:
            self._result._require_open()
            if out is None or out is self._result:
                return self._result
            out = _readback_destination(self._dtype, self._shape, out)
            np.copyto(out._array.reshape(self._result._array.shape), self._result._array, casting="no")
            return out
        result = _readback_destination(self._dtype, self._shape, out)
        try:
            self._admit()
            self._task.copy_into(result._array)
        except BaseException:
            if out is None:
                result.close()
            raise
        self._result = result
        self._task = None
        self._host = None
        self._packet = None
        _readbacks.discard(self)
        return result

    def _admit(self) -> None:
        """Admit only through this snapshot's FIFO position, then keep identity."""
        if self._packet is not None and self._packet.native is not None:
            stream = _command_streams()[int(self._host.identity)]
            _drain_commands(stream, block=True, until=self._packet)
            _remove_empty_stream(stream)
        self._capture_source_write_submission()

    @_automatic_cost
    def _take_bytes(self) -> bytes:
        if self._host is not None or self._task is not None:
            assert_runtime_owner_thread()
        if self.cancelled:
            raise RuntimeError("Readback result was cancelled")
        self._admit()
        return self._task.get_bytes()


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
    from infernux.math import quaternion, vector3

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

    An authored synchronization error closes this binding and stops its GPU
    stream. Discard unaccepted work and explicitly rebind after resolving the
    partial operation; callbacks and accepted GPU writes cannot be rolled back.
    """

    def __init__(self, owner, pose: Buffer, *, initial_pose: TransformPose | None = None,
                 domain: Buffer | None = None, points=(), vectors=(), on_transform=None) -> None:
        assert_runtime_owner_thread()
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
        self._owner = weakref.ref(owner)
        from infernux.execution import capture_execution_scope, component_scope
        from infernux.components import InxComponent
        self._execution_scope = capture_execution_scope()
        if self._execution_scope is None and isinstance(owner, InxComponent):
            self._execution_scope = component_scope(owner)
        self._pose = pose
        self._domain = domain
        self._points = points
        self._vectors = vectors
        self._on_transform = on_transform
        self._initial_pose = initial_pose
        self._readback = None
        self._published_pose = None
        self._pending_cpu_pose = None
        self._readback_source = None
        self._closed = False
        _transform_bindings.add(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def _request_readback(self) -> None:
        if self._pose.device == "gpu" and not self._pose._cpu_mapped:
            stream = _command_streams().get(int(self._pose._host.identity))
            version = tuple(self._pose._native.last_write_submission)
            # An unaccepted writer has no native write serial yet. Preserve
            # its FIFO position instead of mistaking the old serial for it.
            if version == self._readback_source and (
                stream is None or not _has_pending_write(stream, self._pose._native)
            ):
                return
            self._readback = self._pose.get_data_async()
        else:
            # CPU buffers expose writable NumPy views, so a queue serial cannot
            # represent their mutations.
            self._readback = self._pose.get_data_async()

    def _apply_authored_pose(self, previous_pose, current_pose) -> None:
        if self._readback is not None:
            self._readback.cancel()
            self._readback = None
        host = self._pose._host if self._pose.device == "gpu" and not self._pose._cpu_mapped else None
        try:
            with recording():
                self._pose.set_data(_pose_array(current_pose))
                if _pose_changed(current_pose, previous_pose):
                    _apply_transform_delta(
                        self._domain, self._points, self._vectors, previous_pose, current_pose,
                    )
                    if self._on_transform is not None:
                        from infernux.execution import component_execution
                        with component_execution(self._execution_scope):
                            self._on_transform(previous_pose, current_pose)
            if host is not None:
                # An outer nonblocking scope must not weaken authored Transform
                # authority or initial-pose visibility at a strict consumer.
                _flush_commands(host)
        except Exception as exception:
            if host is not None:
                stream = _command_streams().get(int(host.identity))
                if stream is None:
                    stream = _command_stream(host)
                _fail_command_stream(stream, exception)
            # A callback or an accepted GPU write may already have side effects.
            # Stop this binding; retry requires discarding pending work and an
            # explicit rebind, never replaying a guessed previous/current delta.
            self.close()
            raise
        self._published_pose = current_pose
        self._pending_cpu_pose = None
        _cpu_dirty_transform_bindings.discard(self)

    @_automatic_cost
    def _poll(self) -> None:
        if self._closed:
            return
        scope = self._execution_scope
        if scope is not None and scope.state == 'closed':
            self.close()
            return
        owner = self._owner()
        # Play-domain replacement preserves the GameObject handle but retires
        # its old Python component. That component must not keep publishing a
        # second pose into the replacement's still-live Transform.
        if owner is None or getattr(owner, "_is_destroyed", False):
            self.close()
            return
        if self._pose.closed:
            self.close()
            return
        transform = _resolve_bound_transform(self._object_id, self._object_handle)
        if transform is None:
            self.close()
            return
        current_pose = _transform_pose(transform)
        if scope is not None and scope.state == 'paused':
            if self._published_pose is not None and _pose_changed(current_pose, self._published_pose):
                self._apply_authored_pose(self._published_pose, current_pose)
            return
        if self._initial_pose is not None:
            previous_pose = self._initial_pose
            self._apply_authored_pose(previous_pose, current_pose)
            self._initial_pose = None
            if self._pose._cpu_mapped:
                self._readback = None
                return
            self._request_readback()
            return
        if self._pose._cpu_mapped:
            if self._published_pose is not None and _pose_changed(
                current_pose, self._published_pose
            ):
                self._apply_authored_pose(self._published_pose, current_pose)
                return
            self._pending_cpu_pose = _buffer_pose(self._pose)
            _cpu_dirty_transform_bindings.add(self)
            if _cpu_mesh_frame_depth() == 0:
                _cpu_dirty_transform_bindings.discard(self)
                self._flush_cpu_pose()
            return
        # Authored TRS wins even while an older snapshot is still Busy/in flight.
        if self._published_pose is not None and _pose_changed(current_pose, self._published_pose):
            self._apply_authored_pose(self._published_pose, current_pose)
            self._request_readback()
            return
        snapshot = None
        if self._readback is not None:
            if self._readback.cancelled:
                self._readback = None
            else:
                source_write_submission = self._readback.source_write_submission
                if source_write_submission is not None:
                    self._readback_source = source_write_submission
                if not self._readback.done:
                    return
                snapshot = self._readback.get_data()
                self._readback = None
        if snapshot is not None:
            value = _buffer_pose(snapshot)
            try:
                _publish_transform_pose(transform, value)
            except (AttributeError, ReferenceError, RuntimeError):
                self.close()
                raise
            self._published_pose = value
        self._request_readback()

    @_automatic_cost
    def _flush_cpu_pose(self) -> None:
        """Resolve CPU simulation/Transform ownership once at frame end."""
        if self._closed or self._pending_cpu_pose is None:
            return
        scope = self._execution_scope
        if scope is not None and scope.state == 'closed':
            self.close()
            return
        if scope is not None and scope.state == 'paused':
            return
        owner = self._owner()
        if owner is None or getattr(owner, "_is_destroyed", False) or self._pose.closed:
            self.close()
            return
        transform = _resolve_bound_transform(self._object_id, self._object_handle)
        if transform is None:
            self.close()
            return
        current_pose = _transform_pose(transform)
        if self._published_pose is not None and _pose_changed(
            current_pose, self._published_pose
        ):
            self._apply_authored_pose(self._published_pose, current_pose)
            return
        value = self._pending_cpu_pose
        self._pending_cpu_pose = None
        try:
            _publish_transform_pose(transform, value)
        except (AttributeError, ReferenceError, RuntimeError):
            self.close()
            raise
        self._published_pose = value

    def close(self) -> None:
        if self._closed:
            return
        assert_runtime_owner_thread()
        self._closed = True
        _transform_bindings.discard(self)
        _cpu_dirty_transform_bindings.discard(self)
        if self._readback is not None:
            self._readback.cancel()
            self._readback = None
        self._owner = None
        self._object_handle = None
        self._pose = None
        self._domain = None
        self._points = ()
        self._vectors = ()
        self._on_transform = None
        self._initial_pose = None
        self._published_pose = None
        self._pending_cpu_pose = None


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
    from infernux.lib import SceneManager

    game_object = SceneManager.instance().find_runtime_object_by_id(object_id)
    if game_object is None or game_object.handle != object_handle:
        return None
    return game_object.transform


def _poll_transform_bindings() -> None:
    """Exchange authored/simulated poses at the physics Transform barrier."""
    with _record_commands():
        for binding in tuple(_transform_bindings):
            binding._poll()


def buffer(*, shape, dtype=float, device: Literal["cpu", "gpu"], data=None) -> Buffer:
    """Create typed engine storage on an explicit CPU or GPU device."""
    return Buffer(shape=shape, dtype=dtype, device=device, data=data)


def statistics(*, reset: bool = False) -> Statistics:
    """Read current GPU/Web transfer counters without creating a GPU host.

    Native CPU range execution is reported by its request metrics. A CPU-only
    engine has no device transfers, submissions or waits to report here.
    """
    from infernux.application import Application
    host = getattr(Application._current_engine(), "_infernux_compute_host", None)
    if host is None or _web_cpu_compute_enabled():
        values = Statistics(
            submission_count=int(_cpu_statistics["submission_count"]),
            dispatch_count=int(_cpu_statistics["dispatch_count"]),
            upload_request_count=0,
            upload_bytes=0,
            copy_request_count=0,
            copy_bytes=0,
            readback_request_count=0,
            readback_bytes=0,
            staging_allocation_count=0,
            host_map_count=0,
            native_boundary_count=0,
            wait_count=0,
            cpu_submit_ms=float(_cpu_statistics["cpu_submit_ms"]),
            wait_ms=0.0,
            pending_submission_count=0,
            gpu_profile_available=False,
            gpu_profile_serial=0,
            gpu_time_ms=None,
        )
        if reset:
            _cpu_statistics.update(
                submission_count=0, dispatch_count=0, cpu_submit_ms=0.0
            )
        return values
    assert_runtime_owner_thread()
    raw = dict(host.get_statistics())
    values = Statistics(
        submission_count=int(raw["submission_count"]),
        dispatch_count=int(raw["dispatch_count"]),
        upload_request_count=int(raw["upload_request_count"]),
        upload_bytes=int(raw["upload_bytes"]),
        copy_request_count=int(raw["copy_request_count"]),
        copy_bytes=int(raw["copy_bytes"]),
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
    if _web_cpu_compute_enabled():
        raise ComputeCapabilityError(
            "GPU timestamp profiling is unavailable for Web CPU compute"
        )
    if not _native_compute_host().set_profiling_enabled(enabled):
        raise ComputeCapabilityError("GPU compute timestamp profiling is unavailable")


@dataclass(slots=True)
class _PreparedLaunch:
    arguments: bytes
    argument_buffer: object
    dispatches: list

    def clear(self) -> None:
        self.arguments = b""
        self.argument_buffer = None
        self.dispatches.clear()


class _KernelExecutable:
    def __init__(self, artifact, host, params) -> None:
        import hashlib
        from infernux._compiler.gpu_artifact import _specialization_identity

        if any(isinstance(value, Buffer) and value._dtype.scalar == np.dtype(np.int64) for value in params) and not host.shader_int64:
            raise ComputeCapabilityError("this Vulkan device has no enabled shaderInt64 capability")

        self.artifact = artifact
        self.host = host
        self.specialization = _specialization_identity(params, autodiff_mode=artifact.autodiff_mode,
                                                      gradient_parameters=artifact.gradient_parameters)
        self.cost_identity = hashlib.sha256(b"".join(artifact.spirv_tasks)).hexdigest()
        # A launch specialization is immutable: resident buffer identities and
        # scalar values fully determine both the compiler argument block and
        # the native resource bindings. Fixed-step solvers commonly repeat the
        # same launch tens of times, so prepare it once instead of rebuilding a
        # Taichi launch context and Python binding list on every dispatch.
        self._prepared_launches = OrderedDict()
        self.tasks = []
        widths = tuple(int(task["threads_per_group"]) for task in artifact.task_metadata)
        self.dispatch_grain = min(widths, default=1)
        self.max_dispatch_rows = host.max_dispatch_groups_x * self.dispatch_grain if widths else None
        native = None
        try:
            for spirv, metadata in zip(artifact.spirv_tasks, artifact.task_metadata):
                bindings = sorted(metadata["buffer_bindings"], key=lambda binding: binding["binding"])
                layout = [(binding["binding"], binding["binding_type"]) for binding in bindings]
                native = host._create_kernel_from_compiler(spirv, layout)
                self.tasks.append((native, bindings, int(metadata["threads_per_group"])))
        except Exception:
            # A failed constructor is not registered in the declaration cache.
            self.close()
            raise
        finally:
            native = None

    @staticmethod
    def _argument_index(binding) -> int:
        indices = binding.get("argument_indices", ())
        if isinstance(indices, int):
            return indices
        if len(indices) != 1:
            raise RuntimeError("GPU kernel buffer bindings require one top-level parameter index")
        return int(indices[0])

    @staticmethod
    def _launch_key(params, bounds=None, gradients=None):
        import struct

        key = []
        for value in params:
            if isinstance(value, Buffer):
                key.append(("buffer", value._generation))
            elif isinstance(value, (bool, int, np.integer)):
                key.append(("int32", struct.pack("<i", int(value))))
            elif isinstance(value, (float, np.floating)):
                key.append(("float32", struct.pack("<f", float(value))))
            else:
                raise TypeError("GPU kernel parameters must be inx.buffer values or numeric scalars")
        if bounds is not None:
            key.append(("range", *bounds))
        if gradients:
            key.append(("gradients", *sorted(gradients)))
            key.extend(("buffer", gradients[index]._generation) for index in sorted(gradients))
        return tuple(key)

    def prepare(self, params, *, gradients=None, begin=None, end=None, cache=True):
        bounds = self.artifact.range_bounds(params, begin=begin, end=end, host=self.host)
        # Gradient permissions, host and physical aliasing must also be checked
        # on an LRU hit. The frontend owns the same ABI validation as AOT use.
        arguments = None
        if self.artifact.gradient_parameters or gradients is not None:
            arguments = self.artifact.argument_bytes(
                params, bounds=bounds, gradients=gradients, host=self.host,
            )
        if bounds is not None and bounds[0] == bounds[1]:
            return (), ()
        launch_key = self._launch_key(params, bounds, gradients) if cache else None
        prepared = self._prepared_launches.pop(launch_key, None) if cache else None
        if prepared is None:
            if arguments is None:
                arguments = self.artifact.argument_bytes(params, bounds=bounds)
            prepared = self._prepare_launch(params, bounds, arguments, gradients=gradients)
            if cache and len(self._prepared_launches) >= 64:
                self._prepared_launches.popitem(last=False)[1].clear()
        if cache:
            self._prepared_launches[launch_key] = prepared
        # The private argument buffer is immutable after initialization. Native
        # AttachWrite publishes this serial only after the upload is accepted.
        updates = []
        if prepared.arguments and int(prepared.argument_buffer.last_write_serial) == 0:
            updates.append((prepared.argument_buffer, prepared.arguments, 0))
        return updates, prepared.dispatches

    def _prepare_launch(self, params, bounds, arguments, *, gradients=None) -> _PreparedLaunch:
        prepared = _PreparedLaunch(arguments, None, [])
        native = resources = None
        try:
            prepared.argument_buffer = self.host.create_buffer(max(1, (len(prepared.arguments) + 3) // 4), "uint32")
            domain = params[self.artifact.domain_parameter]
            contract = self.artifact.continuation
            for native, bindings, threads_per_group in self.tasks:
                resources = []
                accesses = []
                reads, writes = [], []
                for position, binding in enumerate(bindings):
                    kind = binding["resource_kind"]
                    if kind == "arguments":
                        resources.append(prepared.argument_buffer)
                        accesses.append("read")
                        if contract.eligible:
                            reads.append((position, 0, int(prepared.argument_buffer.byte_size)))
                    elif kind == "external_buffer":
                        index = self._argument_index(binding)
                        value = gradients[index] if binding.get("is_grad", False) else params[index]
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
                        if contract.eligible:
                            if index in contract.input_parameters:
                                reads.append((position, value._byte_offset, value.nbytes))
                            if index in contract.fully_written_output_parameters:
                                first, last = bounds
                                writes.append((position, value._byte_offset + first * value._dtype.itemsize,
                                               (last - first) * value._dtype.itemsize))
                    else:
                        raise RuntimeError(f"Unsupported GPU compiler resource kind: {kind}")
                count = domain.element_count if bounds is None else bounds[1] - bounds[0]
                groups = (count + threads_per_group - 1) // threads_per_group
                proof = (reads, writes) if contract.eligible else None
                prepared.dispatches.append((native, resources, accesses, b"", groups, 1, 1, proof))
            return prepared
        except Exception:
            prepared.clear()
            raise
        finally:
            native = resources = None

    def discard_buffer_launches(self, value: Buffer) -> None:
        token = ("buffer", value._generation)
        for key in tuple(self._prepared_launches):
            if token not in key:
                continue
            self._prepared_launches.pop(key).clear()
        # The Python LRU may already have evicted this wrapper while the native
        # kernel still caches its weak binding identity. Schedule one bounded
        # native collection regardless of whether a Python entry remained.
        for native, _, _ in self.tasks:
            _retiring_native_kernels[id(native)] = native

    def close(self) -> None:
        self.tasks.clear()
        for prepared in self._prepared_launches.values():
            prepared.clear()
        self._prepared_launches.clear()
        self.artifact = None
        self.host = None


class _CpuKernelExecutable:
    """Cook-selected sequential execution of one GPU work-item function."""

    host = None

    def __init__(self, function, *, vectorized: bool = False, domain_parameter: int | None = None) -> None:
        self.function = function
        self.vectorized = bool(vectorized)
        if domain_parameter is None:
            # Uncooked engine declarations are analyzed once on preparation.
            # Cooked project kernels carry the same slot explicitly.
            import ast
            from infernux._compiler.kernel_contract import execution_domain
            from infernux._compiler.taichi.frontend import _function_source

            domain_parameter = execution_domain(ast.parse(_function_source(function)).body[0])[3]
        self.domain_parameter = domain_parameter
        self._last_params = None
        self._last_buffers = ()
        self._last_mapped = ()
        self._last_domain = None
        self._last_mesh_buffers = ()

    def prepare(self, params):
        return (), ()

    def execute(self, params) -> None:
        if self._last_params == params:
            buffers = self._last_buffers
            mapped = self._last_mapped
            domain = self._last_domain
            mesh_buffers = self._last_mesh_buffers
        else:
            buffers = tuple(value for value in params if isinstance(value, Buffer))
            if not buffers:
                raise TypeError("Compute kernel launch requires at least one inx.buffer")
            if any(not value._cpu_mapped for value in buffers):
                raise TypeError(
                    "Web CPU compute requires build-mapped device='gpu' buffers"
                )
            if not 0 <= self.domain_parameter < len(params):
                raise TypeError("Compute kernel execution-domain parameter is missing")
            domain = params[self.domain_parameter]
            if not isinstance(domain, Buffer) or len(domain.shape) != 1:
                raise TypeError("inx.compute.index currently requires a one-dimensional GPU inx.buffer")
            mapped = tuple(
                value._array if isinstance(value, Buffer) else value
                for value in params
            )
            mesh_buffers = tuple(
                value for value in buffers if value._cpu_mesh_binding is not None
            )
            self._last_params = params
            self._last_buffers = buffers
            self._last_mapped = mapped
            self._last_domain = domain
            self._last_mesh_buffers = mesh_buffers
        if any(value.closed for value in buffers):
            raise RuntimeError("Compute buffer is closed")
        started = perf_counter() if _recording_depth() == 0 else None
        if self.vectorized:
            self.function(*mapped)
        else:
            _cpu_work_item.active = True
            try:
                for work_item in range(domain.element_count):
                    _cpu_work_item.index = work_item
                    self.function(*params)
            finally:
                _cpu_work_item.active = False
                _cpu_work_item.index = None
        _cpu_statistics["dispatch_count"] += 1
        if started is not None:
            _cpu_statistics["cpu_submit_ms"] += (perf_counter() - started) * 1000.0
        for value in mesh_buffers:
            _cpu_dirty_mesh_buffers.add(value)


def _cpu_atomic_add(target, indices, values, mask=None) -> None:
    """Apply one cooked CPU atomic reduction without a Python work-item loop."""
    target = np.asarray(target)
    # NumPy's indexed add.at can write through a non-writeable ndarray, unlike
    # ordinary assignment. Enforce the same permission before that operation.
    if not target.flags.writeable:
        raise RuntimeError("CPU compute atomic target is read-only")
    if target.ndim != 1:
        raise ValueError("CPU compute atomic targets must expose one scalar lane")
    indices = np.asarray(indices)
    values = np.asarray(values)
    active_count = None
    if mask is not None:
        mask = np.asarray(mask, dtype=np.bool_)
        active_count = int(np.count_nonzero(mask))
        if indices.ndim:
            indices = indices[mask]
            if values.ndim:
                values = values[mask]
        elif values.ndim:
            values = values[mask]
    if indices.ndim == 0:
        if values.ndim:
            delta = np.sum(values, dtype=target.dtype)
        elif active_count is None:
            delta = values
        else:
            delta = values * active_count
        target[int(indices)] += delta
        return
    if values.ndim == 0:
        values = np.full(indices.shape, values, dtype=target.dtype)
    np.add.at(target, indices.astype(np.intp, copy=False), values)


class Kernel:
    """A GPU single-work-item declaration prepared or compiled on first launch."""

    def __init__(self, function, *, cpu_vectorized: bool = False, cpu_domain_parameter: int | None = None) -> None:
        if not callable(function):
            raise TypeError("inx.compute.kernel requires a callable")
        self._init_runtime()
        self.function = function
        self._cpu_vectorized = bool(cpu_vectorized)
        self._cpu_domain_parameter = cpu_domain_parameter
        for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(self, attribute, getattr(function, attribute, None))

    def _init_runtime(self):
        self.function = None
        self._sealed_artifacts = None
        self._lowered_work_item = None
        self._inner_variants = {}
        self._cpu_vectorized = False
        self._cpu_domain_parameter = None
        self._executables = OrderedDict()
        self._lock = threading.RLock()
        # Receiver source analysis is a declaration property.  Bound launches
        # still snapshot live field values every time, but do not re-parse the
        # method source on every frame.
        self._receiver_field_names = None
        self._cpu_executable = None
        _kernel_declarations.add(self)

    def __get__(self, instance, owner=None):
        # Keep class access (`Component.step`) stable for source inspection and
        # AOT discovery, while an instance access carries only the explicit,
        # engine-owned receiver fields into the GPU ABI.
        if instance is None or self.function is None:
            return self
        return _BoundKernel(self, instance)

    @classmethod
    def _from_work_item(cls, function, lowered):
        declaration = cls(function)
        declaration._lowered_work_item = lowered
        return declaration

    @classmethod
    def _from_artifacts(cls, artifacts, identity):
        """A compiled program has contracts and code, without an author AST."""
        declaration = object.__new__(cls)
        declaration._init_runtime()
        declaration._sealed_artifacts = dict(artifacts)
        declaration.__name__ = declaration.__qualname__ = identity
        declaration.__module__ = "infernux.nn"
        declaration.__doc__ = "Built GPU program on the common compute runtime"
        return declaration

    def _executable(self, params, *, receiver_fields=None, autodiff_mode="none", gradient_parameters=()):
        from infernux._compiler.gpu_artifact import _compilation_request, parameter_key, _specialization_identity

        gradient_parameters = tuple(gradient_parameters)
        differentiated = autodiff_mode != "none" or bool(gradient_parameters)
        if differentiated and receiver_fields is not None:
            raise KernelCompilationError("Local VJP requires a module-level or static kernel, without a bound receiver")
        gradient_parameters = _compilation_request(params, autodiff_mode, gradient_parameters)

        buffers = [value for value in params if isinstance(value, Buffer)]
        if not buffers:
            raise TypeError("GPU kernel launch requires at least one GPU inx.buffer")
        if any(value._cpu_mapped for value in buffers):
            if self._sealed_artifacts is not None:
                raise ComputeCapabilityError("built Vulkan programs require the native GPU backend")
            if self._lowered_work_item is not None:
                raise ComputeCapabilityError("ordered GPU work items require the native GPU backend")
            if differentiated:
                raise ComputeCapabilityError("Local VJP requires native GPU execution; Web CPU compute has no reverse fallback")
            if receiver_fields is not None:
                raise KernelCompilationError(
                    "Web CPU compute requires module-level or static kernels"
                )
            if any(not value._cpu_mapped for value in buffers):
                raise TypeError("A compute launch cannot mix Web CPU and native GPU buffers")
            if self._cpu_executable is None:
                self._cpu_executable = _CpuKernelExecutable(
                    self.function, vectorized=self._cpu_vectorized,
                    domain_parameter=self._cpu_domain_parameter,
                )
            return self._cpu_executable
        if any(value.device != "gpu" for value in buffers):
            raise TypeError(
                "GPU kernel launch requires GPU inx.buffer parameters; "
                "transfer CPU data explicitly with set_data"
            )
        assert_runtime_owner_thread()
        if differentiated:
            for value in buffers:
                value._require_open()
        host = buffers[0]._host
        host_identity = int(host.identity)
        if any(int(value._host.identity) != host_identity for value in buffers):
            raise ValueError("GPU kernel buffers must belong to the same Infernux compute host")
        key = (host_identity, parameter_key(params), autodiff_mode, gradient_parameters)
        with self._lock:
            executable = self._executables.pop(key, None)
            if executable is None:
                if self._sealed_artifacts is not None:
                    identity = _specialization_identity(params, autodiff_mode=autodiff_mode,
                                                        gradient_parameters=gradient_parameters)
                    try:
                        artifact = self._sealed_artifacts[identity]
                    except KeyError:
                        raise ComputeCapabilityError("built GPU program does not contain the requested argument/gradient role") from None
                else:
                    artifact = self._compile_artifact(params, receiver_fields, autodiff_mode, gradient_parameters)
                executable = _KernelExecutable(artifact, host, params)
                if len(self._executables) >= 64:
                    _, retired = self._executables.popitem(last=False)
                    retired.close()
                self._executables[key] = executable
            else:
                self._executables[key] = executable
            return executable

    def _compile_artifact(self, params, receiver_fields, autodiff_mode, gradient_parameters):
        from infernux._compiler.taichi import CompilerInstallationError
        from infernux._compiler.taichi.frontend import compile_kernel

        try:
            return compile_kernel(
                self.function, params, receiver_fields=receiver_fields,
                autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters,
                **({"_lowered": self._lowered_work_item} if self._lowered_work_item is not None else {}),
            )
        except CompilerInstallationError as exception:
            raise ComputeCompilerError(f"Infernux GPU compiler initialization failed: {exception}") from exception
        except KernelCompilationError:
            raise
        except Exception as exception:
            raise KernelCompilationError(f"GPU kernel '{self.__qualname__}' compilation failed: {exception}") from exception

    def _get_receiver_field_names(self):
        if self._receiver_field_names is None:
            from infernux._compiler.taichi.frontend import receiver_fields

            self._receiver_field_names = receiver_fields(self.function)
        return self._receiver_field_names

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


def _reduction_kernels(function, params, *, receiver_fields=None, _prepared=None):
    """Declare the shared finite reduction variants; allocate/dispatch nothing."""
    from infernux._compiler.work_item import normalize_work_item
    from infernux._compiler.kernel_contract import ARRAY_KINDS
    from infernux._compiler.taichi.frontend import _continuation_contract, _work_item_proof

    item = (normalize_work_item(function, len(params), receiver_fields=receiver_fields)
            if _prepared is None else _prepared[0])
    domain = params[item.domain_parameter]
    if not isinstance(domain, Buffer) or domain.device != "gpu" or len(domain.shape) != 1:
        raise TypeError("ordered GPU work items require a one-dimensional GPU buffer domain")
    proof = _work_item_proof(item, params) if _prepared is None else _prepared[1]
    contract = _continuation_contract(item, params, autodiff_mode="none", gradient_parameters=(), proof=proof)
    if not contract.eligible:
        raise TypeError("GPU work item cannot continue: " + "; ".join(contract.diagnostics))
    kinds = tuple(ARRAY_KINDS[value.dtype] if isinstance(value, Buffer) else "float32" for value in params)
    return {kind: Kernel._from_work_item(function, lowered)
            for kind, lowered in item.reduction_variants(proof, kinds).items()}


class _BoundKernel(Kernel):
    """Descriptor view that lowers a narrow instance receiver explicitly."""

    def __init__(self, declaration: Kernel, receiver) -> None:
        # Do not register a second declaration: the unbound declaration owns
        # executable lifetime and is already tracked by the engine teardown.
        self._declaration = declaration
        self._receiver = receiver
        self.function = declaration.function
        for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(self, attribute, getattr(declaration, attribute, None))

    def _effective_params(self, params):
        from infernux._compiler.taichi.frontend import receiver_bindings

        fields = self._declaration._get_receiver_field_names()
        bindings = receiver_bindings(self.function, self._receiver, fields)
        values = tuple(value for _, value in bindings)
        supported = (Buffer, bool, int, float, np.integer, np.floating)
        for name, value in bindings:
            if not isinstance(value, supported):
                raise TypeError(
                    f"GPU kernel receiver field '{name}' has unsupported type "
                    f"{type(value).__name__}; bind an inx.buffer or numeric scalar explicitly"
                )
        return bindings, (*values, *tuple(params))

    def _executable(self, params, **request):
        return self._resolve(params, **request)[0]

    def _resolve(self, params, **request):
        bindings, effective = self._effective_params(params)
        executable = self._declaration._executable(
            effective,
            receiver_fields=bindings,
            **request,
        )
        return executable, effective

    def _release_engine_resources(self) -> None:
        self._declaration._release_engine_resources()

    def _discard_buffer_launches(self, value: Buffer) -> None:
        self._declaration._discard_buffer_launches(value)


class Function:
    """A reusable numeric helper compiled only as part of a GPU kernel."""

    def __init__(self, value) -> None:
        if not callable(value):
            raise TypeError("inx.compute.function requires a callable")
        self.function = value
        for attribute in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(self, attribute, getattr(value, attribute, None))

    def __call__(self, *_args, **_kwargs):
        if bool(getattr(_cpu_work_item, "active", False)):
            return self.function(*_args, **_kwargs)
        raise RuntimeError("@inx.compute.function helpers are valid only inside a GPU kernel")


def _release_engine_resources() -> None:
    """Release module-level GPU declarations before native renderer teardown."""
    assert_runtime_owner_thread()
    from infernux.application import Application

    engine = Application._current_engine()
    host = getattr(engine, "_infernux_compute_host", None) if engine is not None else None
    # Engine shutdown cancels the entire unaccepted authoring stream. Accepted
    # GPU work already belongs to the native queue and retires during cleanup.
    _discard_commands()
    _flush_cpu_meshes()
    # An asynchronous readback owns the compute-host lease until its result is
    # consumed or abandoned. Components may intentionally leave a completed
    # telemetry readback unread on the final frame, so teardown must retire
    # those leases independently of component object lifetime.
    for readback in tuple(_readbacks):
        readback.cancel()
    for binding in tuple(_transform_bindings):
        binding.close()
    for work in tuple(_gpu_works):
        work.close()
    for declaration in tuple(_kernel_declarations):
        declaration._release_engine_resources()
    for value in tuple(_gpu_buffers):
        value.close()
    _retiring_native_kernels.clear()
    if host is not None:
        host._release_lease()
    if engine is not None:
        engine._infernux_compute_host = None


def kernel(function) -> Kernel:
    """Declare a GPU kernel; the function describes one logical work item."""
    if isinstance(function, staticmethod):
        raise TypeError(
            "@inx.compute.kernel received a staticmethod descriptor; put "
            "@staticmethod above @inx.compute.kernel so the kernel decorator "
            "receives the underlying function"
        )
    return Kernel(function)


def _cpu_kernel(*, domain_parameter: int, vectorized: bool):
    """Internal decorator carrying the cook-selected execution contract."""
    def decorate(function):
        return Kernel(function, cpu_vectorized=vectorized, cpu_domain_parameter=domain_parameter)
    return decorate


def function(value) -> Function:
    """Declare a reusable numeric helper for :func:`kernel` functions."""
    return Function(value)


def index(domain: Buffer) -> int:
    """Declare and return the current work-item index inside a GPU kernel."""
    if bool(getattr(_cpu_work_item, "active", False)):
        return int(_cpu_work_item.index)
    raise RuntimeError("inx.compute.index is valid only inside @inx.compute.kernel")


def atomic_add(target, value):
    """Atomically add ``value`` to one buffer element inside a GPU kernel."""
    if bool(getattr(_cpu_work_item, "active", False)):
        raise KernelCompilationError(
            "Web CPU cook did not lower compute.atomic_add to an assignment"
        )
    raise RuntimeError("inx.compute.atomic_add is valid only inside @inx.compute.kernel")


@contextmanager
def defer_prepares():
    """Capture prepare requests so a host can materialize them over later frames."""
    stack = getattr(_deferred_prepare_state, "stack", None)
    if stack is None:
        stack = []
        _deferred_prepare_state.stack = stack
    pending = []
    stack.append(pending)
    try:
        yield pending
    finally:
        if not stack or stack[-1] is not pending:
            raise RuntimeError("inx.compute deferred prepare scopes must be nested")
        stack.pop()


def prepare(declaration: Kernel, params, *, autodiff_mode="none", gradient_parameters=()) -> None:
    """Compile one specialization and create its native pipelines without dispatching it.

    Call this from an owning component's preparation/start boundary when the
    first gameplay interaction must not pay compiler or pipeline creation
    cost. Argument values are not uploaded and authored kernel code is not
    executed; the following :func:`launch` remains the first data operation.
    Local VJP preparation uses ``autodiff_mode='reverse'`` and positional
    ``gradient_parameters``; it neither initializes gradients nor consumes seeds.
    """
    if not isinstance(declaration, Kernel):
        raise TypeError("inx.compute.prepare requires an @inx.compute.kernel declaration")
    if not isinstance(params, tuple):
        raise TypeError("inx.compute.prepare params must be a tuple")
    gradient_parameters = tuple(gradient_parameters)
    stack = getattr(_deferred_prepare_state, "stack", None)
    if stack:
        if any(isinstance(value, Buffer) and getattr(value, "_native", None) is not None for value in params):
            assert_runtime_owner_thread()
        # Keep the exact declaration and buffer identities alive until the
        # owner-thread startup queue materializes this specialization.
        stack[-1].append((declaration, params, autodiff_mode, gradient_parameters))
        return
    _prepare_now(declaration, params, autodiff_mode, gradient_parameters)


def prepare_continuation(declaration: Kernel, params):
    """Cold prepare a CPU/Vulkan work item for budgeted, cross-frame execution.

    The returned fixed-signature program exposes submit(params,
    finish_within_frames=..., budget=...). Output parameters are layout
    prototypes: completed readonly Buffers appear by name in request.result.
    CPU inputs are snapshotted at submission. GPU inputs retain their accepted
    value without copying; later writers preserve it through the common FIFO.
    """
    from ._work import Continuation
    return Continuation(declaration, params)


def _prepare_now(declaration: Kernel, params, autodiff_mode="none", gradient_parameters=()) -> None:
    request = {"autodiff_mode": autodiff_mode, "gradient_parameters": gradient_parameters}
    if isinstance(declaration, _BoundKernel):
        declaration._resolve(params, **request)
    else:
        declaration._executable(params, **request)


class Work:
    """A fixed native resource/program recipe, not an executor or ready result.

    Create with :func:`prepare_work`. Private storage is cloned per native
    instance without copying prototype contents. External resources are pinned;
    input ports require complete writers supplied by the eventual caller.
    """

    def __init__(self, native, host):
        self._native, self._host = native, host
        self._ports = tuple(native.ports)
        _gpu_works.add(self)

    @property
    def closed(self) -> bool:
        return self._native is None

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(port[0] for port in self._ports if port[4])

    @property
    def outputs(self) -> tuple[str, ...]:
        return tuple(port[0] for port in self._ports if not port[4])

    def _require_open(self):
        if self.closed:
            raise RuntimeError("compute work is closed")
        assert_runtime_owner_thread()
        return self._native

    def close(self) -> None:
        if self.closed:
            return
        assert_runtime_owner_thread()
        self._native = self._host = None
        _gpu_works.discard(self)


def _admit_work_arguments(host, updates) -> None:
    """Use the original FIFO for one cold upload-only packet, never inference."""
    if not updates:
        return
    stream = _command_stream(host)
    packet = None
    try:
        _seal_commands(stream)
        packet = _PreparedCommands(host.prepare_batch((), updates), int(host.identity))
        stream["pending"].append(packet)
        _drain_commands(stream, block=True, until=packet)
        _remove_empty_stream(stream)
    except Exception as error:
        _fail_command_stream(stream, error)
        raise
    finally:
        packet = host = updates = None


def prepare_work(operations, *, private_buffers, inputs, outputs) -> Work:
    """Freeze finite ``(Kernel, params_tuple)`` operations without running them.

    The contract is native GPU, fixed-shape scalar f32/i32/u32/i64 whole-domain work
    with the compiler's shared row proof. ``private_buffers`` declares complete
    allocations to clone; logical input/output mappings name their exact views.
    External resources retain their actual owners and must have accepted
    producers. Their valid contents remain a caller/producer precondition.

    Cold argument initialization is ordered through the existing strict FIFO.
    This can wait for admission when that queue is full, but does not wait for
    the prepared work to execute, dispatch inference or read GPU results.
    """
    assert_runtime_owner_thread()
    operations = tuple(operations)
    private_buffers = tuple(private_buffers)
    inputs, outputs = dict(inputs), dict(outputs)
    if not operations or not private_buffers or not outputs:
        raise ValueError("prepare_work requires operations, private storage and an output port")
    if set(inputs) & set(outputs) or any(not isinstance(name, str) or not name for name in (*inputs, *outputs)):
        raise ValueError("compute work port names must be nonempty and unique")
    for value in (*private_buffers, *inputs.values(), *outputs.values()):
        if not isinstance(value, Buffer):
            raise TypeError("compute work storage and ports must be inx.compute.Buffer values")
        value._require_open()
        if value.device != "gpu" or value._cpu_mapped or value._native is None:
            raise ComputeCapabilityError("compute work requires native GPU storage")
    host = private_buffers[0]._host
    if any(int(value._host.identity) != int(host.identity)
           for value in (*private_buffers, *inputs.values(), *outputs.values())):
        raise ValueError("compute work storage must belong to one host")

    def identity(native):
        return int(native.resource_index), int(native.resource_generation)

    private_keys = {identity(value._native) for value in private_buffers}
    if any(identity(value._native) not in private_keys for value in (*inputs.values(), *outputs.values())):
        raise ValueError("compute work ports must name declared per-instance allocations")
    slots, ports, work_dispatches, argument_updates = [], [], [], []
    private_views = list(private_buffers)
    slot_indices, updated = {}, set()
    executable = native_recipe = declaration = effective = task = dispatch = resource = resources = None
    updates = dispatches = None

    def slot_for(native):
        key = identity(native)
        if key not in slot_indices:
            slot_indices[key] = len(slots)
            slots.append((native, key in private_keys))
        return slot_indices[key]

    try:
        # Admit older authored writes before inspecting immutable argument
        # initialization serials. Preparation never jumps ahead of that FIFO.
        _flush_commands(host, block=True)
        for declaration, params in operations:
            if not isinstance(declaration, Kernel) or not isinstance(params, tuple):
                raise TypeError("compute work operations must contain a Kernel and a params tuple")
            if isinstance(declaration, _BoundKernel):
                executable, effective = declaration._resolve(params)
            else:
                effective = params
                executable = declaration._executable(effective)
            for value in effective:
                if isinstance(value, Buffer) and value._native is not None and identity(value._native) in private_keys:
                    private_views.append(value)
            if not isinstance(executable, _KernelExecutable) or int(executable.host.identity) != int(host.identity):
                raise ComputeCapabilityError("compute work requires compiled kernels on the same native GPU host")
            artifact = executable.artifact
            if not artifact.continuation.eligible or artifact.autodiff_mode != "none" or artifact.gradient_parameters:
                raise TypeError("compute work requires the shared forward row contract")
            domain_count = int(effective[artifact.domain_parameter].element_count)
            if domain_count <= 0:
                raise ValueError("compute work execution domain must be positive")
            updates, dispatches = executable.prepare(effective, begin=0, end=domain_count)
            for update in updates:
                key = identity(update[0])
                if key not in updated:
                    argument_updates.append(update)
                    updated.add(key)
            for task, dispatch in zip(executable.tasks, dispatches, strict=True):
                native_kernel, resources, accesses, constants, gx, gy, gz, proof = dispatch
                bindings = []
                for position, (metadata, resource, access) in enumerate(zip(task[1], resources, accesses, strict=True)):
                    slot = slot_for(resource)
                    kind = metadata["resource_kind"]
                    if kind == "arguments":
                        offset, length = 0, int(resource.byte_size)
                    elif kind == "external_buffer":
                        parameter = executable._argument_index(metadata)
                        value = effective[parameter]
                        offset, length = int(value.description.byte_offset), value.nbytes
                    else:
                        raise TypeError(f"compute work does not support compiler resource kind {kind!r}")
                    bindings.append((slot, offset, length, access))
                work_dispatches.append((native_kernel, bindings, proof[1], constants, gx, gy, gz))
        if private_keys - set(slot_indices):
            raise ValueError("compute work private allocations must participate in its operations")
        for is_input, values in ((True, inputs), (False, outputs)):
            for name, value in values.items():
                ports.append((name, slot_indices[identity(value._native)], int(value.description.byte_offset),
                              value.nbytes, is_input))
        _admit_work_arguments(host, argument_updates)
        native_recipe = host.prepare_work(slots, ports, work_dispatches)
        return Work(native_recipe, host)
    finally:
        # The recipe owns only real native kernels/external resources, not a
        # prototype Run or its private launch-cache pins. Do not close user data.
        for value in private_views:
            if not value.closed:
                _discard_buffer_launches(value)
        private_views.clear()
        slots.clear()
        ports.clear()
        work_dispatches.clear()
        argument_updates.clear()
        slot_indices.clear()
        updated.clear()
        inputs.clear()
        outputs.clear()
        operations = private_buffers = ()
        params = value = host = artifact = None
        executable = native_recipe = declaration = effective = task = dispatch = resource = resources = None
        updates = dispatches = native_kernel = update = None


@kernel
def _transform_anchor_points(domain, values,
                             opx, opy, opz, oqx, oqy, oqz, oqw, osx, osy, osz,
                             npx, npy, npz, nqx, nqy, nqz, nqw, nsx, nsy, nsz):
    """Apply ``new TRS * inverse(old TRS)`` to resident world-space points."""
    i = index(domain)
    dx = values[i][0] - opx
    dy = values[i][1] - opy
    dz = values[i][2] - opz
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
    values[i][0] = npx + nr00 * lx + nr01 * ly + nr02 * lz
    values[i][1] = npy + nr10 * lx + nr11 * ly + nr12 * lz
    values[i][2] = npz + nr20 * lx + nr21 * ly + nr22 * lz


@kernel
def _transform_anchor_vectors(domain, values,
                              oqx, oqy, oqz, oqw, osx, osy, osz,
                              nqx, nqy, nqz, nqw, nsx, nsy, nsz):
    """Apply the rotational/scale part of an authored TRS delta to vectors."""
    i = index(domain)
    dx = values[i][0]
    dy = values[i][1]
    dz = values[i][2]
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
    values[i][0] = nr00 * lx + nr01 * ly + nr02 * lz
    values[i][1] = nr10 * lx + nr11 * ly + nr12 * lz
    values[i][2] = nr20 * lx + nr21 * ly + nr22 * lz


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


@_automatic_cost
def launch(declaration: Kernel, params, *, autodiff_mode="none", gradients=None) -> None:
    """Compile if needed and record a GPU kernel in the existing host FIFO.

    For a supported local VJP, pass ``autodiff_mode='reverse'`` and a mapping
    from positional parameter indices to GPU gradient buffers. Input gradients
    accumulate; output gradients are upstream seeds consumed and cleared by
    the generated reverse kernel. Initialize them explicitly with buffer writes.
    The primal values must still describe that forward evaluation. No tape,
    host fallback or cross-frame range splitting is implied by this call.
    """
    if not isinstance(declaration, Kernel):
        raise TypeError("inx.compute.launch requires an @inx.compute.kernel declaration")
    if not isinstance(params, tuple):
        raise TypeError("inx.compute.launch params must be a tuple")
    gradients = None if gradients is None else dict(gradients)
    request = {}
    if autodiff_mode != "none" or gradients:
        request = {"autodiff_mode": autodiff_mode, "gradient_parameters": tuple(gradients or ())}
    if isinstance(declaration, _BoundKernel):
        executable, effective_params = declaration._resolve(params, **request)
    elif declaration._cpu_executable is not None and not request:
        executable = declaration._cpu_executable
        effective_params = params
    else:
        executable = declaration._executable(params, **request)
        effective_params = params
    if isinstance(executable, _CpuKernelExecutable):
        executable.execute(effective_params)
        if _recording_depth() == 0:
            _cpu_statistics["submission_count"] += 1
            _flush_cpu_meshes()
        return
    batch = _command_stream(executable.host)
    if _append_kernel_commands(batch, executable, effective_params, gradients=gradients):
        _record_activity()
    for value in effective_params:
        if isinstance(value, Buffer) and value._mesh_attribute_state is not None:
            batch["mesh_attributes"][id(value)] = value
            _record_activity()
    if _recording_depth() == 0:
        _flush_commands(executable.host)


def _finish_kernels(declarations) -> None:
    """Complete already admitted kernels, without submitting, mapping or device-idle.

    Synchronous offline clients call this after their recording scope. It is
    not a per-run completion ticket or a budgeted execution boundary.
    """
    assert_runtime_owner_thread()
    if _recording_depth():
        raise RuntimeError("synchronous compute completion cannot run inside recording")
    seen = set()
    for declaration in declarations:
        for executable in declaration._executables.values():
            for native, _, _ in executable.tasks:
                if id(native) not in seen:
                    seen.add(id(native))
                    native.wait()


__all__ = [
    "ComputeCapabilityError", "ComputeCompilerError", "KernelCompilationError",
    "ComputeExecutionError",
    "buffer", "Buffer", "BufferDescription", "Readback", "Statistics", "statistics", "transfer",
    "set_profiling_enabled",
    "kernel", "Kernel", "function", "Function", "index", "atomic_add", "prepare", "prepare_continuation", "prepare_work", "Work", "launch",
    "recording", "discard_pending", "bind_transform", "TransformBinding", "TransformPose",
]
