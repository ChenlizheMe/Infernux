"""Program IO on the model request's existing archive and original IO workers."""
from pathlib import Path

import numpy as np

from infernux.cpu_programs import NativeCpuLibrary, library_directory, _library_path
from infernux._compiler.gpu_artifact import _artifact_header_bytes
from infernux.nn._programs import create_source
from .io_service import recipe_service
from .service_windows import ServiceDemand


class ModelProgramInput:
    def __init__(self, path, record, device, metadata_limit, archive):
        self.source = create_source(record)
        self.source.require_device(device)
        self.directory = Path(path).parent
        self.metadata_limit = metadata_limit
        self.stage = "declare"
        self.task = None
        self.code_path = None
        self.payload = None
        self.index = 0
        self.header_bytes = 0
        self.code_bytes = sum(self.source.file_sizes.values()) if device == "gpu" else 0
        self.committed = False
        self.read_recipes = (tuple(archive.describe_read_work(row["path"]) for row in record["entries"])
                             if device == "gpu" else ())

    @property
    def pending(self):
        return self.task is not None and self.task.pending

    @property
    def complete(self):
        return self.stage == "complete"

    def _next_cpu_input_bytes(self):
        entries = self.source.record["entries"]
        return (entries[self.index]["bytes"] if self.stage == "gpu_allocate"
                and self.index < len(entries) else 0)

    def resources(self):
        retained, native, peak = [], list(self.source.native_blocks), 0
        if self.source.device == "cpu":
            if self.task is not None:
                task = self.task
                native.append((("native-module-input", id(self)), task.reservation_bytes))
                if task.image_bytes:
                    key = (("cpu-code", task.module_identity) if task.module_identity else
                           ("cpu-code-pending", id(self)))
                    native.append((key, task.image_bytes))
            elif self.stage == "materialize":
                peak = self.metadata_limit  # Bounded native descriptor/symbol handoff.
        else:
            native.append((("gpu-code-future", id(self)), self.code_bytes))
            if self.payload is not None:
                retained.append(self.payload)
            peak = 2 * self.header_bytes if self.stage == "gpu_decode" else self._next_cpu_input_bytes()
        return retained, native, peak

    def host_key(self, action):
        return "model-program", self.source.device, action

    def host_work(self, archive):
        if self.stage in ("declare", "bind"):
            return next(self.source.host_work(self.stage))
        if self.stage == "materialize" and archive.state == "ready":
            return "materialize_adopt", len(self.source.record["entries"])
        if self.stage == "cpu_module" and self.task.state == "loaded":
            return "module_adopt", 1
        if self.stage == "gpu_allocate":
            entries = self.source.record["entries"]
            return ("gpu_allocate", entries[self.index]["bytes"]) if self.index < len(entries) else ("gpu_allocate_end", 1)
        if self.stage == "gpu_read" and archive.state == "ready":
            return "gpu_header", 1
        if self.stage == "gpu_decode":
            return "gpu_decode", self.source.record["entries"][self.index]["bytes"]
        return None

    def remaining_services(self, archive, chunk, owner, host_service):
        """Pure projection of this original input and its existing source."""
        source, demands = self.source, []
        if self.complete:
            return demands
        if source.device == "cpu":
            if self.stage not in ("cpu_module", "bind"):
                return [ServiceDemand(reason="CPU model image has not yet been inspected")]
            if self.stage == "cpu_module":
                task = self.task
                observation = owner._io_observations.get(task)
                recipe = observation.recipe if task.pending and observation is not None else (
                    None if task.pending else task.describe_remaining_work())
                if recipe is None:
                    return [ServiceDemand(reason="accepted CPU program has no remaining-work observation")]
                demands.append(recipe_service(recipe, owner, chunk, observation=observation, pending=task.pending))
                demands.append(host_service("module_adopt", 1))
        else:
            if self.stage == "declare":
                demands.extend(host_service(action, units) for action, units in source.host_work("declare"))
            if self.stage != "bind":
                entries = source.record["entries"]
                observation = owner._io_observations.get(archive)
                for index in range(self.index, len(entries)):
                    row, current = entries[index], index == self.index
                    decoding = current and self.stage == "gpu_decode"
                    if not decoding:
                        reading = current and self.stage == "gpu_read"
                        if reading:
                            pending, active = archive.pending, observation
                            recipe = observation.recipe if pending and observation is not None else (
                                None if pending else archive.describe_remaining_work())
                            if recipe is None:
                                return [ServiceDemand(reason="accepted GPU program has no remaining-work observation")]
                        else:
                            demands.append(host_service("gpu_allocate", row["bytes"]))
                            recipe, active, pending = self.read_recipes[index], None, False
                        demands.append(recipe_service(recipe, owner, chunk, observation=active, pending=pending))
                        demands.append(host_service("gpu_header", 1))
                    # Includes copying and validating all SPIR-V words, not only
                    # the JSON header. Kernel binding creates no driver pipeline.
                    demands.append(host_service("gpu_decode", row["bytes"]))
                demands.append(host_service("gpu_allocate_end", 1))
        demands.extend(host_service(action, units) for action, units in source.host_work("bind"))
        return demands

    def advance(self, archive, chunk, owner, opportunity, request):
        host = self.host_work(archive)
        progressed = self._advance(archive, chunk, owner, opportunity, request)
        if progressed and host is not None and request.status == "pending":
            request.metrics["model_io_actions"] = request.metrics.get("model_io_actions", 0) + 1
        return progressed

    def _advance(self, archive, chunk, owner, opportunity, request):
        source, record = self.source, self.source.record
        if self.stage == "declare":
            if not source.declare_step():
                return True
            if source.device == "cpu":
                target = _library_path(library_directory(self.directory), record["library"])
                archive.begin_materialize("programs/" + record["library"]["file"], str(target), record["library"]["sha256"])
                self.stage = "materialize"
            else:
                self.stage = "gpu_allocate"
            return True
        if self.stage == "materialize":
            if archive.state == "ready":
                from infernux import lib
                file = archive.take_materialized_file()
                self.code_path = file.path
                self.task = lib._Infernux._NativeModuleTask(file, [row["symbol"] for row in record["entries"]])
                self.stage = "cpu_module"
            else:
                return owner._advance_io(request, archive, chunk, opportunity)
            return True
        if self.stage == "cpu_module":
            if self.task.state == "loaded":
                source.library = NativeCpuLibrary(self.task.module, self.code_path)
                self.stage = "bind"
            else:
                # Inspected image_bytes is already in resources(), so the
                # owner's next admission precedes this separate mapping step.
                return owner._advance_io(request, self.task, chunk, opportunity)
            return True
        if self.stage == "gpu_allocate":
            if self.index == len(record["entries"]):
                self.stage = "bind"
            else:
                row = record["entries"][self.index]
                if not archive.begin_read_buffer(row["path"], owner._frame):
                    request.metrics["io_storage_deferrals"] = request.metrics.get("io_storage_deferrals", 0) + 1
                    return False
                self.stage = "gpu_read"
            return True
        if self.stage == "gpu_read":
            if archive.state == "ready":
                row = record["entries"][self.index]
                self.payload = np.frombuffer(archive.take_read_buffer(), dtype=np.uint8)
                request.metrics["program_materialized_bytes"] = (
                    request.metrics.get("program_materialized_bytes", 0) + self.payload.nbytes)
                self.header_bytes = _artifact_header_bytes(self.payload, path=row["path"])
                if self.header_bytes > self.metadata_limit:
                    raise ValueError("GPU artifact metadata exceeds the managed metadata limit")
                self.payload.flags.writeable = False
                self.stage = "gpu_decode"
            else:
                return owner._advance_io(request, archive, chunk, opportunity)
            return True
        if self.stage == "gpu_decode":
            row = record["entries"][self.index]
            source.decode_entry(row, self.payload)
            self.payload = None
            self.code_bytes -= row["bytes"]
            self.header_bytes = 0
            self.index += 1
            self.stage = "gpu_allocate"
            return True
        if self.stage == "bind":
            if source.bind_step():
                self.stage = "complete"
            return True
        return False

    def commit(self):
        if self.task is not None:
            self.task.commit()
        else:
            self.source._publish()
        self.committed = True

    def cancel(self):
        if self.task is not None:
            self.task.cancel()

    def close(self):
        if not self.committed:
            self.source._discard_unpublished()
        self.task = self.payload = self.source = None

    def poll(self, owner, request):
        task = self.task
        if task is None or task.pending:
            return
        request.metrics.update(program_execution_ns=task.execution_ns, program_submissions=task.submissions,
                               program_read_bytes=task.read_bytes, program_load_ns=task.load_ns)
        if task.state == "failed" and request.status == "pending":
            owner._cancel(request, "failed", task.error)
