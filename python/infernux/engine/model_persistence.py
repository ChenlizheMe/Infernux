"""Model input/snapshots on the common transfer cursor and native IO tasks."""
from dataclasses import dataclass, field
import math
import os
from time import perf_counter_ns

import numpy as np

from infernux import compute
from .cpu_continuation import CpuRequest, _RequestStorage, _advance_transfer, _prepare_transfer, _plan_storage
from .service_windows import (ServiceDemand, _service_availability, combine_services, cursor_transfer_service,
                              host_service, value_transfer_service)
from .io_service import recipe_service


@dataclass(eq=False, kw_only=True)
class ModelSaveRequest(CpuRequest):
    model: object
    optimizer: object
    plan: object
    path: str
    chunk_bytes: int
    file_reservation: int
    metadata_reservation: int
    write_scope: object
    values: dict = field(default_factory=dict)
    offsets: dict = field(default_factory=dict)
    file: object = None
    cursor: object = None
    source_name: str | None = None
    saved_steps: int | None = None
    entry_layout: tuple = ()

    def _prepared_transfer(self):
        return self.cursor

    @classmethod
    def create(cls, owner, facts, model, optimizer, path):
        from infernux.nn.modules import Module
        from infernux.nn.optim import Optimizer
        from infernux.nn.model import _model_path
        from infernux import lib

        if not isinstance(model, Module) or (optimizer is not None and not isinstance(optimizer, Optimizer)):
            raise TypeError("model persistence requires a Module and its optimizer")
        root = model._bind()
        with root._lock:
            plan = root._plan
            if root is not model or plan is None or root._dirty:
                raise ValueError("prepare a clean model before requesting a save")
            if optimizer is not None:
                if optimizer._root is not root:
                    raise ValueError("checkpoint optimizer belongs to another model")
                optimizer._check_domain()
            path = os.path.normcase(_model_path(path, writing=True))
            if any(isinstance(request, cls) and request.path == path for request in owner.requests):
                raise RuntimeError("a model save already owns this output path")
            if len(plan.definition_bytes) > owner.budget.snapshot_bytes:
                raise ValueError("fixed model metadata exceeds the atomic snapshot_bytes limit")
            # A descriptor estimates the very same native writer without opening
            # a file, allocating its IO blocks, or starting a compression context.
            # Training can publish any gradient slot before this snapshot freezes.
            specs = []
            for index, value in enumerate(root._state):
                specs.append((f"parameters/{index}.f32", value.nbytes, False))
                if optimizer is not None:
                    specs.append((f"gradients/{index}.f32", value.nbytes, False))
            if optimizer is not None:
                specs.extend(optimizer._checkpoint_specs())
            largest = max((size for _, size, _ in specs), default=0)
            chunk = min(max(64 * 1024, largest), 1024 * 1024)
            if plan.device == "gpu":
                chunk = min(chunk, owner.transfer_chunk_bytes)
            chunk = chunk // 4 * 4
            if not chunk:
                raise ValueError("model IO requires space for one float32 element")
            write_scope = lib.PathWriteScope.begin_write([path])
            descriptor = None
            try:
                descriptor = lib._Infernux._InxPackWriteTask(path, specs, block_bytes=chunk, write_scope=write_scope)
                # The future text prelude is not a fake binary entry. Reserve its
                # bounded header plus the short INM/length delimiters explicitly.
                reservation = descriptor.reservation_bytes + lib._Infernux._inxpack_header_reservation(owner.budget.snapshot_bytes)
                return cls(**facts, model=root, optimizer=optimizer, plan=plan, path=path,
                           chunk_bytes=chunk, file_reservation=reservation, write_scope=write_scope,
                           metadata_reservation=owner.budget.snapshot_bytes, reserved_bytes=0)
            except BaseException:
                write_scope.close()
                raise
            finally:
                # This descriptor never accepts an IO step. A retained error
                # traceback must not keep its child publication right alive.
                descriptor = None

    def _model_access(self, root):
        return "snapshot" if self.model is root else None

    def _blocked(self, owner):
        return (self._predecessor(owner) is not None
                or (self.file is not None and self.file.pending))

    def _predecessor(self, owner):
        return None if self.model is None else owner._training_writer(self.model)

    def _dependencies(self, owner):
        writer = self._predecessor(owner)
        return () if writer is None else (writer,)

    def _io_tasks(self):
        return () if self.file is None else (self.file,)

    def _io_ready(self, owner):
        return (self.cursor is None and not self._blocked(owner)
                and not self._complete() and self._domain() == "cpu")

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        if self.model is not None:
            return ServiceDemand(reason="model snapshot entries are not yet frozen",
                                 prerequisites=self._dependencies(owner))
        availability = _service_availability(self, owner, opportunity)
        if availability.reason:
            return availability
        observation = owner._io_observations.get(self.file)
        recipe = observation.recipe if self.file.pending else self.file.describe_remaining_work()
        offsets, used_cursor = dict(self.offsets), False

        def input_service(span, repeats):
            nonlocal used_cursor
            name, total, device, dtype, host = self.entry_layout[span.entry_index]
            count = span.input_bytes
            offsets[name] += count * repeats
            final_input = offsets[name] == total
            if device == "cpu":
                return host_service(owner, ("model-save", "input", final_input), count)
            identity = device, "cpu", dtype, host
            tag = ("model-save-transfer", final_input)
            if self.cursor is not None and not used_cursor:
                used_cursor = True
                return cursor_transfer_service(self, owner, self.cursor, tag, append=True)
            blocks = compute._transfer_work(count, self.chunk_bytes,
                source_storage="gpu" if host else "cpu", target_storage="cpu")
            return value_transfer_service(owner, blocks, identity, count, tag, append=True)

        demand = recipe_service(recipe, owner, self.chunk_bytes, observation=observation,
                                pending=self.file.pending, input_service=input_service)
        return combine_services(availability, (demand,))

    def _service_host_key(self):
        if self.cursor is None:
            if self.file is None or self.file.state != "needs_input":
                return None
            name, source, offset, count = self._input()
            if source.device != "gpu":
                return "model-save", "input", offset+count == source.nbytes
            identity = source.device, "cpu", source.dtype, 0 if source._host is None else int(source._host.identity)
            step = compute._transfer_work(count, self.chunk_bytes,
                source_storage="gpu" if source._native is not None else "cpu", target_storage="cpu")[0][0][0]
            return ("model-save-transfer", offset+count == source.nbytes,
                *compute._transfer_host_key(identity, count, step,
                    finalizing=step.domain == "host" and step.end == count))
        c = self.cursor
        step = c.next_step(prepared=self.packet is not None)
        final_input = self.offsets[self.source_name] + c.nbytes == self.values[self.source_name].nbytes
        return ("model-save-transfer", final_input,
                *c.host_cost_key(step, finalizing=step.domain == "host" and step.end == c.nbytes))

    def _service_host_units(self, key):
        return self.file.input_bytes if key[0] == "model-save" else None

    def _observed_host_key(self, key, progress):
        if key[0] == "model-save":
            return key if self.metrics.get("model_io_actions", 0) != progress[4] else None
        if key[-1] and self.metrics.get("model_io_actions", 0) == progress[4]:
            return None
        return key

    def _domain(self):
        if self.cursor is not None:
            return self.cursor.execution_domain
        if self.file is not None and self.file.state == "needs_input":
            value = self.values[self.file.input_path]
            if isinstance(value, compute.Buffer) and value.device == "gpu":
                return "gpu"
        return "cpu"

    def _resources(self):
        retained = list(self.values.values())
        if self.model is not None:
            retained.extend(_plan_storage(self.plan))
        private, peak, reserve = [], 0, self.metadata_reservation if self.model is not None else 0
        # This is one total native reservation, before and after construction.
        # It already contains the live context/input/output memory.
        native = [(("inxpack-write", id(self)), self.file_reservation if self.file is None
                   else self.file.reservation_bytes)]
        if self.plan is not None:
            native.extend(self.plan.native_blocks)
        gpu = [value for value in self.values.values()
               if isinstance(value, compute.Buffer) and value.device == "gpu"]
        if self.model is not None and self.plan.device == "gpu":
            gpu.extend(self.model._state)
        if gpu:
            transfer = min(self.chunk_bytes, max(value.nbytes for value in gpu))
            reserve += transfer
            peak = 2 * transfer
        if self.cursor is not None:
            c = self.cursor
            retained.extend((c.source, c.target, c.snapshot))
            private.extend((c.target, c.snapshot))
            native.extend(c.native_blocks)
            reserve = max(reserve, c.reserved_bytes)
            peak = c.prepare_bytes if self.packet is None and not c.readback_ready and not c.done else 0
        return _RequestStorage(retained, compute._storage_blocks(private), reserve, prepare_bytes=peak, native_blocks=native)

    def _is_current(self, owner):
        if self.model is not None:
            return (self.plan._is_current()
                    and (self.optimizer is None or self.model._optimizer() is self.optimizer))
        return self.cursor is None or self.cursor.is_current()

    def _freeze(self, owner):
        from infernux import lib
        from infernux.nn.model import _snapshot

        started = perf_counter_ns()
        metadata, values, step, copied_bytes = _snapshot(self.model, self.optimizer, plan=self.plan, work_owner=owner)
        try:
            if len(metadata) > self.metadata_reservation:
                raise ValueError("complete model metadata exceeds the atomic snapshot_bytes limit")
            self.values = dict(values)
            self.entry_layout = tuple((name, value.nbytes, value.device, value.dtype,
                0 if value._host is None else int(value._host.identity)) for name, value in sorted(values))
            self.offsets = {name: 0 for name in self.values}
            self.saved_steps = step
            self.metrics["metadata_bytes"] = len(metadata)
            self.metrics["snapshot_bytes"] = copied_bytes
            self.metrics["snapshot_retained_bytes"] = sum(value.nbytes for _, value in values)
            self.metrics["snapshot_opportunity"] = owner._last_opportunity
            specs = [(name, value.nbytes, False) for name, value in self.values.items()]
            resident = tuple(value.numpy(copy=False) if value.device == "cpu" and np.little_endian else None
                             for value in self.values.values())
            self.metrics["resident_input_bytes"] = sum(view.nbytes for view in resident if view is not None)
            self.file = lib._Infernux._InxPackWriteTask(self.path, specs, block_bytes=self.chunk_bytes,
                                                      text_header=metadata, write_scope=self.write_scope,
                                                      resident_inputs=resident)
        except BaseException:
            for _, value in values:
                value._close(flush=False)
            self.values.clear()
            raise
        finally:
            # Each buffer is now owned by self.values, or is closed when an
            # exception prevents adoption. Never keep a duplicate Python list.
            values.clear()
            self.metrics["snapshot_ns"] += perf_counter_ns() - started
        self.model = self.optimizer = self.plan = None
        # Native construction only allocated metadata. Refuse before Open can
        # allocate its IO blocks/codec context or create a temporary file.
        if owner._capacity() > owner.budget.capacity_bytes:
            raise ValueError("model snapshot and file workspace exceed numerical capacity")

    def _append(self, name, array):
        self.file.append(array)
        count = array.nbytes
        self.offsets[name] += count
        self.metrics["pack_traffic_bytes"] += 2 * count  # CPU read + native input copy.
        self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
        if self.offsets[name] == self.values[name].nbytes:
            value = self.values.pop(name)
            self.offsets.pop(name)
            if isinstance(value, compute.Buffer):
                value._close(flush=False)

    def _input(self):
        name = self.file.input_path
        return name, self.values[name], self.offsets[name], self.file.input_bytes

    def _complete(self):
        return self.file is not None and self.file.state == "published"

    def _work(self, owner, opportunity):
        if not owner._available(background=True):
            return None, False
        if self.model is not None:
            if owner._training_writer(self.model) is None:
                self._freeze(owner)
                return None, True
            return None, False
        progressed = False
        state = self.file.state
        if self.cursor is None and state == "needs_input":
            name, source, offset, count = self._input()
            if isinstance(source, compute.Buffer) and source.device == "gpu":
                view = source.view(offset=offset // 4, count=count // 4, readonly=True)
                try:
                    self.cursor = compute._TransferCursor(view, "cpu", self.chunk_bytes)
                finally:
                    view._close(flush=False)
                self.host, self.source_name = self.cursor.host, name
                progressed = True
        if self.cursor is not None:
            work, advanced = _advance_transfer(self.cursor, owner, self)
            progressed |= advanced
            if self.cursor.done and owner._available(background=True):
                self._append(self.source_name, self.cursor.target.numpy(copy=False))
                self.cursor.close()
                self.cursor = self.source_name = None
                progressed = True
            return work, progressed
        if state == "needs_input":
            if isinstance(source, compute.Buffer):
                array = source.numpy(copy=False).reshape(-1)
                data = (array.view(np.uint8)[offset:offset+count] if np.little_endian else
                        array[offset//4:(offset+count)//4].astype("<f4"))
            else:
                data = source[offset:offset+count]
            self._append(name, data)
            return None, True  # Data preparation precedes the next IO step.
        if state in ("working", "ready"):
            resident = self.metrics["resident_input_bytes"]
            batch_bytes = resident if resident == self.metrics["snapshot_retained_bytes"] else 0
            return None, owner._advance_io(self, self.file, self.chunk_bytes, opportunity,
                                            publish=state == "ready", batch="write" if state == "working" else None,
                                            batch_bytes=batch_bytes)
        return None, progressed

    def _prepare_gpu(self, owner, work):
        return _prepare_transfer(self, owner, self.cursor, work)

    def _consume_gpu(self):
        self.metrics["device_completed_bytes"] = (self.metrics.get("device_completed_bytes", 0)
            + self.cursor.pending_end - self.cursor.completed_bytes)
        self.cursor.consume_execution()

    def _publish(self, owner, opportunity):
        self.result = self.path
        self.status = "published"
        self.metrics["terminal_opportunity"] = opportunity
        self.metrics["saved_committed_steps"] = self.saved_steps
        self.metrics["observation_late"] = self.deadline is not None and opportunity > self.deadline

    def _poll_external(self, owner, opportunity):
        if self.file is None or self.file.pending:
            return
        self.metrics["file_execution_ns"] = self.file.execution_ns
        self.metrics["file_submissions"] = self.file.submissions
        self.metrics.update({"file_"+key: value for key, value in self.file.progress.items()})
        if self.file.state == "published":
            self._publish(owner, opportunity)
        elif self.file.state == "failed":
            self.status, self.reason = "failed", self.file.error
            self.metrics["terminal_opportunity"] = opportunity

    def _cancel_external(self):
        return self.file is None or self.file.cancel()

    def _retire_external(self):
        return self.file is None or self.file.done

    def _wait_external(self, owner):
        if self.file is not None:
            self.file.wait()
            self._poll_external(owner, owner._last_opportunity)
            if not self.file.done:
                # A transient replace can have reopened the gate after the
                # first shutdown Cancel found a commit already in progress.
                owner._cancel(self, "cancelled", "engine exiting")
                self.file.wait()
                self._poll_external(owner, owner._last_opportunity)

    def _release_private(self):
        if self.cursor is not None:
            self.cursor.close()
        for value in self.values.values():
            if isinstance(value, compute.Buffer):
                value._close(flush=False)
        self.values.clear()
        self.offsets.clear()
        self.entry_layout = ()
        self.model = self.optimizer = self.plan = self.file = self.cursor = self.source_name = None
        self.write_scope.close()
        self.write_scope = None


@dataclass(eq=False, kw_only=True)
class ModelLoadRequest(CpuRequest):
    """A private model/state candidate, published only with complete code and data."""
    _IO_BATCH_BYTES = 1024 * 1024
    path: str
    device: str
    kind: str
    source: object
    program_definition: object
    file: object
    chunk_bytes: int
    metadata_limit: int
    metadata_bytes: int
    stage: str = "directory"
    manifest: dict = field(default_factory=dict)
    definition: object = None
    record: object = None
    entries: tuple = ()
    index: int = 0
    tensor_bytes: int = 0
    reading: bool = False
    cpu_reservation: object = None
    values: dict = field(default_factory=dict)
    cursor: object = None
    gpu_value: object = None
    uploaded_bytes: int = 0
    reserved_window_bytes: int = 0
    program_input: object = None
    read_recipes: tuple = ()

    def _prepared_transfer(self):
        return self.cursor

    def _io_tasks(self):
        tasks = () if self.file is None else (self.file,)
        if self.program_input is not None and self.program_input.task is not None:
            tasks += (self.program_input.task,)
        return tasks

    def _io_ready(self, owner):
        if self.cursor is not None:
            return self._parallel_io_ready(owner)
        if self._complete() or self._blocked(owner):
            return False
        needed = 0
        if self.stage == "programs":
            needed = self.program_input._next_cpu_input_bytes()
        elif self.stage == "values" and not self.reading:
            if self.device == "cpu" and self.cpu_reservation is None:
                needed = self.tensor_bytes
        if self.gpu_value is not None and self.file.state == "needs_buffer":
            return bool(self._gpu_window_size(owner))
        return not needed or owner._frame.can_reserve_cpu_storage(needed)

    @classmethod
    def create(cls, owner, facts, path, device, kind, programs):
        from infernux import lib
        from infernux.nn.model import _checkpoint_programs, _model_path

        if device not in ("cpu", "gpu"):
            raise ValueError("model input device must be 'cpu' or 'gpu'")
        definition, source = _checkpoint_programs(programs, device)
        chunk = owner.transfer_chunk_bytes // 4 * 4
        if not chunk:
            raise ValueError("model input requires space for one float32 element")
        path = _model_path(path)
        file = lib._Infernux._InxPackReadTask(path, metadata_bytes=owner.budget.snapshot_bytes)
        return cls(**facts, path=path, device=device, kind=kind, source=source, program_definition=definition,
                   file=file, chunk_bytes=chunk, metadata_limit=owner.budget.snapshot_bytes,
                   metadata_bytes=owner.budget.snapshot_bytes, reserved_bytes=0)

    def _input_pending(self):
        return ((self.file is not None and self.file.pending)
                or (self.program_input is not None and self.program_input.pending))

    def _blocked(self, owner):
        return self.cursor is None and self._input_pending()

    def _domain(self):
        return "cpu" if self.cursor is None else self.cursor.execution_domain

    def _host_key(self, action, dimensions=0):
        return "model-load", action, self.device, dimensions

    def _service_host_key(self):
        if self.stage == "programs":
            work = self.program_input.host_work(self.file)
            return None if work is None else self.program_input.host_key(work[0])
        if self.cursor is not None:
            c = self.cursor
            step = c.next_step(prepared=self.packet is not None)
            return ("model-load-transfer", *c.host_cost_key(step,
                finalizing=step.domain == "host" and step.end == c.nbytes))
        if self.stage == "complete" or self.stage == "values" and self.index == len(self.entries):
            return self._host_key("publish")
        if self.stage != "values":
            return None
        if self.device == "gpu" and self.gpu_value is not None:
            if self.file.state == "needs_buffer":
                return self._host_key("reserve-window")
            return self._host_key("adopt") if self.file.state in ("output", "ready") else None
        if not self.reading:
            return self._host_key("allocate", len(self.entries[self.index][3]))
        return self._host_key("adopt") if self.file.state == "ready" else None

    def _service_host_units(self, key):
        if key[0] == "model-program":
            return self.program_input.host_work(self.file)[1]
        if key[0] != "model-load":
            return None
        if key[1] == "allocate":
            return math.prod(self.entries[self.index][3]) * 4
        if key[1] == "publish":
            return len(self.entries) + len(self.definition.layers)
        return 1

    def _observed_host_key(self, key, progress):
        return key if self.metrics.get("model_io_actions", 0) != progress[4] else None

    def _service_demand(self, owner, opportunity, *, capacity_bytes=None):
        if self.stage not in ("programs", "values", "complete"):
            return ServiceDemand(reason="model directory/structure/program facts are not yet ready")
        availability = _service_availability(self, owner, opportunity)
        if availability.reason:
            return availability
        demands = []
        if self.stage == "programs":
            program = self.program_input
            demands.extend(program.remaining_services(self.file, self._IO_BATCH_BYTES, owner,
                lambda action, units: host_service(owner, program.host_key(action), units)))
        observation = owner._io_observations.get(self.file)
        for index in range(self.index, len(self.entries)):
            _, _, _, shape = self.entries[index]
            nbytes, current = math.prod(shape) * 4, index == self.index
            cursor = self.cursor if current else None
            if cursor is not None:
                demands.append(cursor_transfer_service(self, owner, cursor, ("model-load-transfer",)))
            started = current and (self.reading or self.gpu_value is not None)
            if not started:
                demands.append(host_service(owner, self._host_key("allocate", len(shape)), nbytes))
                recipe, active, pending = self.read_recipes[index], None, False
            else:
                pending, active = self.file.pending, observation
                recipe = observation.recipe if pending else self.file.describe_remaining_work()
            maximum = (max(nbytes, self._IO_BATCH_BYTES) if self.device == "cpu" else
                       (self.reserved_window_bytes if started else 0) or self._IO_BATCH_BYTES)
            demands.append(recipe_service(recipe, owner, maximum, observation=active, pending=pending))
            if self.device == "cpu":
                demands.append(host_service(owner, self._host_key("adopt"), 1))
            else:
                remaining = nbytes - (self.uploaded_bytes if current else 0) - (cursor.nbytes if cursor else 0)
                if remaining:
                    demands.append(self._gpu_window_service(owner, remaining, current))
        demands.append(host_service(owner, self._host_key("publish"), len(self.entries)+len(self.definition.layers)))
        return combine_services(availability, demands)

    def _gpu_window_service(self, owner, nbytes, current):
        if not current or nbytes != self.reserved_window_bytes:
            return ServiceDemand(reason="a remaining model input window has not been admitted")
        host = getattr(owner._engine(), "_infernux_compute_host", None)
        if host is None:
            return ServiceDemand(reason="model input has not prepared its GPU host")
        identity = "cpu", "gpu", "float32", int(host.identity)
        blocks = compute._transfer_work(nbytes, min(self.chunk_bytes, nbytes),
                                        source_storage="cpu", target_storage="gpu", allocate=False)
        return combine_services(ServiceDemand(), (
            host_service(owner, self._host_key("adopt"), 1),
            value_transfer_service(owner, blocks, identity, nbytes, ("model-load-transfer",)),
        ))

    def _next_cpu_input_bytes(self):
        if self.stage == "values" and self.index < len(self.entries):
            return math.prod(self.entries[self.index][3]) * 4
        return 0

    def _upload_workspace(self, incoming=0):
        left = 0 if self.cursor is None else self.cursor.nbytes - self.cursor.completed_bytes
        span = min(self.chunk_bytes, max(4, left, self.reserved_window_bytes, incoming))
        return compute._transfer_prepare_bytes(span, "cpu", "gpu")

    def _unclaimed_window_bytes(self):
        # Multiple candidate models must each leave space for one float32
        # element until they own a decoded range.
        return 4 if self.cursor is None and not self.reserved_window_bytes else 0

    def _unclaimed_upload_workspace(self):
        # The active and prefetched windows share one upload workspace. The
        # packet and queue already account for the portion that they own.
        workspace = self._upload_workspace()
        packet = self.packet
        if packet is not None:
            if packet.native is not None:
                workspace -= packet.payload_bytes + int(packet.native.pending_staging_bytes)
            elif packet.completion is not None and not packet.completion.retired:
                size = packet.transfer_bytes
                workspace -= compute._transfer_prepare_bytes(size, "cpu", "gpu") - size
        return workspace

    def _resources(self):
        retained = list(self.values.values())
        private = list(retained)
        native = [] if self.source is None else list(self.source.native_blocks)
        reservation = self.tensor_bytes + self.metadata_bytes
        if self.file is not None:
            native.append((("inxpack-read", id(self)), self.file.reservation_bytes))
        if self.reading and self.device == "cpu":
            # The IO cursor already owns this final CPU tensor's reservation.
            reservation -= self._next_cpu_input_bytes()
        peak = 0
        if self.device == "gpu" and self.index < len(self.entries):
            peak = self._unclaimed_upload_workspace() + self._unclaimed_window_bytes()
        if self.gpu_value is not None:
            retained.append(self.gpu_value)
            private.append(self.gpu_value)
        if self.cursor is not None:
            retained.append(self.cursor.source)
        if self.stage == "metadata":
            peak += self.metadata_bytes  # Bounded header/JSON handoff, not a whole model copy.
        if self.program_input is not None:
            program_retained, program_native, program_peak = self.program_input.resources()
            retained.extend(program_retained)
            native.extend(program_native)
            peak += program_peak
        return _RequestStorage(retained, compute._storage_blocks(private), reservation, prepare_bytes=peak, native_blocks=native)

    def _is_current(self, owner):
        return self.cursor is None or self.cursor.is_current()

    def _complete(self):
        return self.stage == "complete"

    def _parse_metadata(self, owner):
        from infernux.nn.model import _read_facts, _model_entries, _check_entries, _read_header

        started = perf_counter_ns()
        record = _read_header(self.file.manifest)
        definition, flags = _read_facts(record, self.kind)
        if self.program_definition is not None and self.program_definition != definition:
            raise ValueError("checkpoint programs require a built model with the same fixed definition")
        entries = _model_entries(definition, flags, record.get("optimizer"))
        if "programs" in record:
            from .model_program_input import ModelProgramInput
            self.program_input = ModelProgramInput(self.path, record["programs"], self.device, self.metadata_limit, self.file)
        _check_entries(self.manifest, entries, () if self.program_input is None else self.program_input.source.file_sizes.items())
        self.definition, self.entries = definition, entries
        self.read_recipes = tuple(self.file.describe_read_work(name, windowed=self.device == "gpu")
                                  for name, _, _, _ in entries)
        self.record = {"kind": self.kind}
        if self.kind == "checkpoint":
            self.record["optimizer"] = record["optimizer"]
        self.tensor_bytes = sum(math.prod(shape) * 4 for _, _, _, shape in entries)
        self.metrics.update(metadata_bytes=self.metadata_bytes, tensor_bytes=self.tensor_bytes,
                            metadata_parse_ns=perf_counter_ns() - started)
        self.metadata_bytes = 0
        self.manifest.clear()
        self.stage = "values" if self.program_input is None else "programs"
        cpu_bytes = self.tensor_bytes if self.device == "cpu" else 4 if entries else 0
        if cpu_bytes > owner._frame.cpu_storage_limit:
            raise ValueError("model input working set exceeds the CPU storage limit")
        # Account for the final model before opening its first value.
        if owner._capacity() > owner.budget.capacity_bytes:
            raise ValueError("model input candidates and IO workspace exceed numerical capacity")

    def _work(self, owner, opportunity):
        if not owner._available(background=True):
            return None, False
        if self.stage == "complete":
            return None, False
        if self.stage == "programs":
            progressed = self.program_input.advance(self.file, self._IO_BATCH_BYTES, owner, opportunity, self)
            if self.status == "pending" and self.program_input.complete:
                self.source = self.program_input.source
                self.stage = "values"
                progressed = True
            return None, progressed
        if self.stage == "values" and self.device == "gpu":
            return self._gpu_work(owner, opportunity)
        state = self.file.state
        if self.stage == "directory" and state == "ready":
            manifest = self.file.manifest
            self.manifest = {row["path"]: row["raw_bytes"] for row in manifest["files"]}
            header = manifest["text_header"]
            if header is None:
                raise ValueError("an INM model requires a text header followed by binary ckpt data")
            self.metadata_bytes = len(header)
            if self.metadata_bytes > self.metadata_limit:
                raise ValueError("INM text header exceeds the managed metadata limit")
            self.metrics["directory_opportunity"] = opportunity
            self.stage = "metadata"
            return None, True
        if self.stage == "metadata":
            if state == "ready":
                self._parse_metadata(owner)
                return None, True
        if self.stage == "values":
            if self.index == len(self.entries):
                self.stage = "complete"
                return None, True
            name, slot, gradient, _ = self.entries[self.index]
            if not self.reading:
                if self.device == "cpu" and self.cpu_reservation is None:
                    # Reserve the whole unpublished model before holding any
                    # of its tensors. Two partial models cannot pin each other.
                    if self.tensor_bytes > owner._frame.cpu_storage_limit:
                        raise ValueError("model input working set exceeds the CPU storage limit")
                    self.cpu_reservation = owner._frame.try_reserve_cpu_storage(self.tensor_bytes)
                    if self.cpu_reservation is None:
                        self.metrics["io_storage_deferrals"] = self.metrics.get("io_storage_deferrals", 0) + 1
                        return None, False
                if not self.file.begin_read_buffer(name, owner._frame, self.cpu_reservation):
                    self.metrics["io_storage_deferrals"] = self.metrics.get("io_storage_deferrals", 0) + 1
                    return None, False
                self.reading = True
                self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
                return None, True
            if state == "ready":
                self.values[slot, gradient] = self._take_read_value()
                self.reading = False
                self.index += 1
                self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
                return None, True
        # CPU entries already own their final allocation. Native range size
        # follows that storage, independent of the small GPU staging windows.
        maximum = max(self._IO_BATCH_BYTES, self._next_cpu_input_bytes()) if self.reading else self._IO_BATCH_BYTES
        return None, owner._advance_io(self, self.file, maximum, opportunity,
                                        batch="directory" if state == "opening" else None)

    def _take_read_value(self):
        array = np.frombuffer(self.file.take_read_buffer(), dtype=np.float32)
        if not np.little_endian:
            array.byteswap(inplace=True)
            self.metrics["pack_traffic_bytes"] += 2 * array.nbytes
        return compute.Buffer._take_cpu_array(array)

    def _gpu_window_size(self, owner):
        frame = owner._frame
        desired = min(self.file.read_buffer_remaining_bytes, self._IO_BATCH_BYTES)
        prefetch = self.cursor is not None
        used = frame.cpu_storage_bytes if prefetch else frame.cpu_storage_reserved_bytes
        maximum = min(desired, frame.cpu_storage_limit - used)
        if maximum < 4:
            return 0
        available = (owner.budget.capacity_bytes - owner._capacity() + self._upload_workspace()
                     + self._unclaimed_window_bytes())
        window = compute._fit_aligned_range(maximum, 4, lambda size: size + self._upload_workspace(size) <= available)
        if prefetch:
            # Speculation uses free storage. It neither evicts assets nor
            # forces smaller batches while the current window is in use.
            return window if window >= min(desired, self.cursor.nbytes) else 0
        return compute._fit_aligned_range(window, 4, frame.can_reserve_cpu_storage)

    def _stream_read(self, owner, opportunity):
        if self.file.state == "needs_buffer":
            if owner._frame.cpu_storage_limit < 4:
                raise ValueError("model input requires one element in the CPU storage limit")
            size = self._gpu_window_size(owner)
            if not size or not self.file.reserve_read_window(owner._frame, size):
                if (self.tensor_bytes + self.file.reservation_bytes + 4 + compute._transfer_prepare_bytes(4, "cpu", "gpu")
                        > owner.budget.capacity_bytes):
                    raise ValueError("model input and one upload range exceed numerical capacity")
                self.metrics["io_storage_deferrals"] = self.metrics.get("io_storage_deferrals", 0) + 1
                return False
            self.reserved_window_bytes = size
            self.metrics["file_window_peak_bytes"] = max(self.metrics.get("file_window_peak_bytes", 0), size)
            self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
            return True
        return owner._advance_io(self, self.file, self.reserved_window_bytes or self._IO_BATCH_BYTES, opportunity)

    def _parallel_io_ready(self, owner):
        if self.cursor is None or self.cursor.done or self.file.pending:
            return False
        state = self.file.state
        if state == "needs_buffer":
            return bool(self._gpu_window_size(owner))
        return state == "reading" and bool(owner._frame.resource_remaining(owner._resource("io")))

    def _advance_parallel_io(self, owner, opportunity):
        submitted = self.file.submissions
        progressed = self._stream_read(owner, opportunity)
        if (self.file.submissions != submitted and self.packet is not None
                and self.packet.completion is not None and not self.packet.completion.retired):
            self.metrics["file_submissions_with_gpu_in_flight"] = (
                self.metrics.get("file_submissions_with_gpu_in_flight", 0) + 1)
        return progressed

    def _gpu_work(self, owner, opportunity):
        if self.index == len(self.entries):
            self.stage = "complete"
            return None, True
        name, slot, gradient, shape = self.entries[self.index]
        if self.gpu_value is None:
            self.gpu_value = compute._managed_empty(shape=shape, dtype=np.float32, device="gpu")
            self.file.begin_read_stream(name)
        elif self.cursor is not None:
            if not self.cursor.done:
                return _advance_transfer(self.cursor, owner, self)
            self.uploaded_bytes += self.cursor.nbytes
            self.cursor.close()
            self.cursor = self.host = None
            if self.uploaded_bytes == self.gpu_value.nbytes:
                self.values[slot, gradient], self.gpu_value = self.gpu_value, None
                self.index += 1
                self.uploaded_bytes = 0
        elif self.file.state in ("output", "ready"):
            source = self._take_read_value()
            self.reserved_window_bytes = 0
            self.cursor = compute._TransferCursor._take_owned_cpu(source, "gpu", self.chunk_bytes)
            self.cursor.target = self.gpu_value.view(offset=self.uploaded_bytes // 4, count=self.cursor.nbytes // 4)
            self.host = self.cursor.host
            self.metrics["file_stream_windows"] = self.metrics.get("file_stream_windows", 0) + 1
        else:
            return None, self._stream_read(owner, opportunity)
        self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
        return None, True

    def _prepare_gpu(self, owner, work):
        if _prepare_transfer(self, owner, self.cursor, work):
            self.metrics["pack_traffic_bytes"] += 2 * (self.cursor.pending_end - work[0])
            return True
        return False

    def _consume_gpu(self):
        before = self.cursor.completed_bytes
        self.cursor.consume_execution()
        self.metrics["device_completed_bytes"] = self.metrics.get("device_completed_bytes", 0) + self.cursor.completed_bytes - before

    def _publish(self, owner, opportunity):
        from infernux.nn.model import _loaded_result

        result = _loaded_result(self.definition, self.values, self.device, self.record, self.source)
        if self.program_input is not None:
            self.program_input.commit()
        for value in self.values.values():
            value._publish_storage()
        self.result = result
        self.metrics["model_io_actions"] = self.metrics.get("model_io_actions", 0) + 1
        self.status = "published"
        self.metrics["terminal_opportunity"] = opportunity
        if self.kind == "checkpoint":
            self.metrics["loaded_committed_steps"] = self.record["optimizer"]["committed_steps"]

    def _poll_external(self, owner, opportunity):
        if self.program_input is not None:
            self.program_input.poll(owner, self)
        if self.file is None or self.file.pending:
            return
        self.metrics["file_execution_ns"] = self.file.execution_ns
        self.metrics["file_submissions"] = self.file.submissions
        progress = self.file.progress
        if self.stage in ("values", "complete"):
            self.metrics["materialized_bytes"] = progress["buffered_bytes"] - self.metrics.get("program_materialized_bytes", 0)
        self.metrics.update({"file_" + key: value for key, value in progress.items()})
        if self.file.state == "failed":
            self.metrics["file_error"] = self.file.error
            if self.status == "pending":
                owner._cancel(self, "failed", self.file.error)

    def _cancel_external(self):
        if self.file is not None:
            self.file.cancel()
        if self.program_input is not None:
            self.program_input.cancel()
        return True  # No read-side commit prevents withdrawing publication rights.

    def _retire_external(self):
        return not self._input_pending()

    def _wait_external(self, owner):
        if self.file is not None:
            self.file.wait()
        if self.program_input is not None and self.program_input.task is not None:
            self.program_input.task.wait()
        self._poll_external(owner, owner._last_opportunity)

    def _release_private(self):
        if self.program_input is not None:
            self.program_input.close()
        if self.file is not None:
            self.file.close()
        if self.cursor is not None:
            self.cursor.close()
        if self.gpu_value is not None:
            self.gpu_value._close(flush=False)
        for value in self.values.values():
            value._close(flush=False)
        self.values.clear()
        self.cpu_reservation = None
        self.manifest.clear()
        self.file = self.cursor = self.gpu_value = self.source = self.program_definition = None
        self.program_input = None
        self.definition = self.record = None
        self.entries = ()
        self.read_recipes = ()
        self.tensor_bytes = self.metadata_bytes = 0
