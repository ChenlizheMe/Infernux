"""Self-contained INM text headers and binary checkpoint data on native IO.

Ordinary calls are explicit synchronous boundaries. A completion window selects
the same codec through the Engine's managed snapshot, transfer and IO pipeline.
Model code uses the existing CPU ABI or GPU artifact on the same Plan.
"""
import json
import math
import os

import numpy as np

from . import _primitives as p
from ._plan import _Definition, _InputSpec, _Layer, _ParameterSpec
from ._tensor import Parameter, Tensor, _empty, _copy_storage, _check_options, _check_storage
from .modules import Module
from .optim import Optimizer, SGD, Adam


class _FixedModule(Module):
    """A captured value graph with canonical named parameters, no author class."""
    def __init__(self, definition, parameters, gradients, device):
        super().__init__()
        self._frozen_definition = definition
        if not definition.parameters:
            self._parameterless_device = device
        object.__setattr__(self, "training", definition.training)
        temporary = []
        try:
            for spec, value, gradient in zip(definition.parameters, parameters, gradients):
                if value.device != device:
                    value = _copy_storage(value, device)
                    temporary.append(value)
                parameter = Parameter._from_buffer(value, spec.shape, requires_grad=spec.requires_grad)
                self._parameters[spec.name] = parameter
                if gradient is not None:
                    if gradient.device != device:
                        gradient = _copy_storage(gradient, device)
                        temporary.append(gradient)
                    parameter._grad = Tensor._from_buffer(gradient, spec.shape)
            self._bind()
        except BaseException:
            p.retire((parameter._data for parameter in self._parameters.values()), flush=False)
            p.retire((parameter.grad._data for parameter in self._parameters.values() if parameter.grad is not None), flush=False)
            self._parameters.clear()
            self._state = None
            raise
        finally:
            p.retire(temporary, flush=False)

    def _capture_definition(self, inputs):
        definition = self._frozen_definition
        if tuple(_InputSpec.from_value(x) for x in inputs) != definition.inputs:
            raise ValueError("loaded model requires its exported input shape and gradient signature")
        if self.training != definition.training:
            raise ValueError("loaded model only contains its exported capture mode")
        return _Definition.from_module(self, inputs, definition.layers, definition.result)

    def train(self, mode=True):
        if type(mode) is not bool:
            raise TypeError("training mode must be bool")
        if mode != self._frozen_definition.training:
            raise ValueError("loaded model only contains its exported capture mode")
        return super().train(mode)


def _snapshot(model, optimizer, *, plan=None, work_owner=None):
    from infernux.application import Application

    if not isinstance(model, Module) or (optimizer is not None and not isinstance(optimizer, Optimizer)):
        raise TypeError("model IO requires a Module and its optimizer")
    root = model._bind()
    with root._lock:
        plan = root._plan if plan is None else plan
        if root is not model or plan is None or plan.entry is not root or not plan._is_current():
            raise ValueError("prepare a clean model before exporting its fixed graph")
        _check_options(None, plan.device)
        engine = Application._current_engine()
        owner = None if engine is None else engine._cpu_work_owner
        if owner is not None and owner._training_writer(root) is not None:
            raise RuntimeError("cold model save requires the active training writer to retire")
        if optimizer is not None:
            if optimizer._root is not root:
                raise ValueError("checkpoint optimizer belongs to another model")
            optimizer._check_domain()
        record = dict(kind="model" if optimizer is None else "checkpoint", dtype="float32-le")
        gradients = [parameter.grad for _, parameter in root._named] if optimizer is not None else ()
        shared = {index for index, grad in enumerate(gradients)
                  if grad is not None and grad._shared_input and grad.device == "cpu"}
        copied_bytes = sum(gradients[index]._data.nbytes for index in shared)
        if work_owner is not None and copied_bytes:
            if copied_bytes > work_owner.budget.snapshot_bytes:
                raise ValueError("model gradient capture exceeds the atomic snapshot_bytes limit")
            if work_owner._capacity() + copied_bytes > work_owner.budget.capacity_bytes:
                raise ValueError("model gradient snapshot exceeds numerical capacity")
        if optimizer is not None:
            record["optimizer"] = optimizer._record()
            record["gradients"] = [grad is not None for grad in gradients]
        values, pinned = [], []
        try:
            for index, storage in enumerate(root._state):
                value = storage.view(readonly=True)
                pinned.append(value)
                values.append((f"parameters/{index}.f32", value))
            for index, grad in enumerate(gradients):
                if grad is None:
                    continue
                if index not in shared:
                    value = grad._data.view(readonly=True)
                elif work_owner is not None:
                    from infernux.compute import _freeze_cpu_buffer

                    value = _freeze_cpu_buffer(grad._data, work_owner._frame)
                else:
                    value = _copy_storage(grad._data, "cpu")
                pinned.append(value)
                values.append((f"gradients/{index}.f32", value))
            if optimizer is not None:
                for name, storage in optimizer._checkpoint_values():
                    value = storage.view(readonly=True)
                    pinned.append(value)
                    values.append((name, value))
            if pinned:
                _check_storage(pinned)
            facts = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            metadata = facts[:-1] + b',"definition":' + plan.definition_bytes + b'}'
            return metadata, values, None if optimizer is None else optimizer._committed_steps, copied_bytes
        except BaseException:
            p.retire(pinned, flush=False)
            values.clear()
            pinned.clear()
            raise


def _save(model, optimizer, path, *, programs=None, program_files=(), plan=None):
    from infernux import lib

    path = _model_path(path, writing=True)
    from infernux.core.document_store import PathWriteScope

    with PathWriteScope.begin_write([path]):
        metadata, values, _, _ = _snapshot(model, optimizer, plan=plan)
        temporary, sources = [], []
        try:
            if programs is not None:
                metadata = metadata[:-1] + b',"programs":' + json.dumps(
                    programs, allow_nan=False, separators=(",", ":")).encode("utf-8") + b'}'
            sources.extend(program_files)
            for name, value in values:
                if value.device == "gpu":
                    value = value.get_data()  # Existing explicit readback; cold, not a managed frame operation.
                    temporary.append(value)
                sources.append((name, np.asarray(value.numpy(copy=False), dtype="<f4")))
            # Borrow contiguous bytes; no full-model string or tensor temporary files.
            lib._Infernux._inxpack_write(sources, path, text_header=metadata)
        finally:
            sources.clear()
            p.retire((*temporary, *(value for _, value in values)), flush=False)
            temporary.clear()
            values.clear()


def bake(model, path, *, optimizer=None, objective="mse"):
    """Cold-build a fixed CPU/Vulkan model and its declared program closure.

    With its optimizer, include loss/backward/update and zero_grad(False) programs without
    executing training. This is a model asset, not optimizer/checkpoint state.
    CPU and GPU programs are stored in the same model container. Loading either
    needs no source compiler; CPU code materializes in the common program cache.
    Player cook is a separate step; this explicit cold call is not frame work.
    """
    return _bake(model, path, optimizer=optimizer, objective=objective)


def _bake(model, path, *, optimizer=None, objective="mse", plan=None, output_index=None):
    from pathlib import Path
    from ._plan import _objective, _target_layout
    objective = _objective(objective)

    path = _model_path(path, writing=True)
    from infernux.core.document_store import PathWriteScope

    with PathWriteScope.begin_write([path]):
        root = model._bind()
        with root._lock:
            plan = root._plan if plan is None else plan
            _, pinned, _, _ = _snapshot(model, optimizer, plan=plan)  # Also rejects an active managed writer.
            p.retire((value for _, value in pinned), flush=False)
            if optimizer is not None:
                if plan.input_grad:
                    raise ValueError("training bake requires a non-differentiable input signature")
                # Cold shapes only; prepare performs no forward/backward/state update.
                inputs = tuple(Tensor._from_buffer(_empty(spec.shape, plan.device, spec.dtype), spec.shape)
                               for spec in plan.inputs)
                result = plan._result_spec(output_index)
                target_layout = _target_layout(objective, result.shape)
                target = (None if target_layout is None else
                          Tensor._from_buffer(_empty(target_layout[0], plan.device, target_layout[1]), target_layout[0]))
                zero_sample = _empty((1,), plan.device)
                try:
                    optimizer._prepare_training(inputs, target, record_loss=True, objective=objective,
                                                output_index=result.index)
                    optimizer._prepare_zero(zero_sample)
                finally:
                    p.retire((*(x._data for x in inputs), None if target is None else target._data, zero_sample), flush=False)
            directory = Path(os.fsdecode(path)).resolve().parent
            programs = (*plan.programs, *(() if optimizer is None else (*optimizer._update_programs, optimizer._zero)))
            if root._program_source is not None:
                record, files = root._program_source.stage()
            elif plan.device == "gpu":
                from ._gpu_programs import build
                plan._gpu_entries(programs, reverse_programs=plan._gpu_reverse_programs)
                record, files = build(programs)
            else:
                from infernux._compiler.cpu_native import build
                record, files = build(programs, directory)
            _save(model, None, path, programs=record, program_files=files, plan=plan)


def _managed_save(model, optimizer, path, window, budget, finish_within_ms):
    from infernux.application import Application

    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError("managed model saves require an active Engine")
    owner = engine._cpu_work(budget)
    request = owner.release_model_save(model, optimizer, path, window_opportunities=window,
                                       finish_within_ms=finish_within_ms)
    return engine._wake_cpu_request(owner, request)


def export(model, path, *, finish_within_frames=None, finish_within_ms=None, budget=None):
    """Export the prepared fixed graph and current weights.

    The file contains one capture mode/signature. It retains registered unused
    parameters and shared slots, but not gradients or optimizer state. A
    budget, host-frame window or millisecond target selects managed snapshot/IO; otherwise
    this call blocks. An ``Assets/...`` destination is relative to the active
    authoring project. The Editor's asset path becomes readable through its
    catalogue after the asset is imported. In a Player, save to a writable path such as
    :meth:`Application.persistent_data_path` instead.
    """
    if any(value is not None for value in (finish_within_frames, finish_within_ms, budget)):
        return _managed_save(model, None, path, finish_within_frames, budget, finish_within_ms)
    _save(model, None, path)


def save_checkpoint(model, optimizer, path, *, finish_within_frames=None, finish_within_ms=None, budget=None):
    """Save published weights, gradients and optimizer state at an idle writer boundary.

    A budget, host-frame window or millisecond target returns a managed request. It waits for an outstanding
    training writer, freezes published generations, then advances transfer/IO
    under the common budget. No implicit zero_grad or extra step runs. External
    data order, random generators and world state remain outside this file.
    """
    if any(value is not None for value in (finish_within_frames, finish_within_ms, budget)):
        return _managed_save(model, optimizer, path, finish_within_frames, budget, finish_within_ms)
    _save(model, optimizer, path)


def _definition(record):
    parameters = tuple(_ParameterSpec(row["name"], tuple(row["shape"]), row["requires_grad"])
                       for row in record["parameters"])
    layers = tuple(_Layer(**{key: tuple(value) if isinstance(value, list) else value for key, value in row.items()})
                   for row in record["layers"])
    inputs = tuple(_InputSpec(tuple(row["shape"]), row["requires_grad"], row["dtype"]) for row in record["inputs"])
    result = record.get("result")
    return _Definition(inputs, record["training"], parameters, layers,
                       tuple(result) if isinstance(result, list) else result)


def _read_facts(record, kind):
    """One metadata/graph contract for cold and managed input drivers."""
    if record["kind"] != kind or record["dtype"] != "float32-le":
        raise ValueError("model file kind or float32 encoding differs from the requested load")
    definition = _definition(record["definition"])
    if kind == "checkpoint" and "programs" in record:
        raise ValueError("checkpoint files contain state; built programs belong to a model asset")
    flags = tuple(record["gradients"]) if kind == "checkpoint" else (False,) * len(definition.parameters)
    if len(flags) != len(definition.parameters) or any(type(value) is not bool for value in flags):
        raise ValueError("checkpoint gradient slots differ from the parameter table")
    if kind == "checkpoint":
        facts = record["optimizer"]
        if (not definition.parameters or type(facts["committed_steps"]) is not int or facts["committed_steps"] < 0
                or isinstance(facts["lr"], bool) or not isinstance(facts["lr"], (int, float))
                or not np.isfinite(facts["lr"]) or not 0 <= facts["lr"] <= np.finfo(np.float32).max):
            raise ValueError("checkpoint optimizer state is invalid")
        if facts["kind"] not in ("sgd", "adam"):
            raise ValueError("unknown checkpoint optimizer")
        if facts["kind"] == "adam" and (len(facts["steps"]) != len(definition.parameters)
                or any(type(step) is not int or step < 0 for step in facts["steps"])):
            raise ValueError("Adam counters must match the model parameter table")
    return definition, flags


def _checkpoint_programs(programs, device):
    if programs is None:
        return None, None
    if not isinstance(programs, _FixedModule) or programs._program_source is None:
        raise ValueError("checkpoint programs require a built fixed model")
    source = programs._program_source
    source.require_device(device)
    return programs._frozen_definition, source


def _model_entries(definition, flags, optimizer=None):
    entries = tuple((f"{kind}/{index}.f32", index, kind, spec.shape)
                 for index, spec in enumerate(definition.parameters)
                 for kind in (("parameters", "gradients") if flags[index] else ("parameters",)))
    if optimizer is not None and optimizer["kind"] == "adam":
        entries += tuple((f"optimizer/{index}/{name}.f32", index, name, spec.shape)
                         for index, (spec, step) in enumerate(zip(definition.parameters, optimizer["steps"]))
                         if step for name in ("first", "second"))
    return entries


def _check_entries(manifest, entries, program_files=()):
    expected = dict(program_files)
    expected.update((name, math.prod(shape) * 4) for name, _, _, shape in entries)
    if expected != manifest:
        raise ValueError("model tensor entries or sizes differ from the fixed definition")


def _loaded_result(definition, values, device, record, source):
    parameters = [values[index, "parameters"] for index in range(len(definition.parameters))]
    gradients = [values.get((index, "gradients")) for index in range(len(parameters))]
    model = _FixedModule(definition, parameters, gradients, device)
    model._program_source = source
    if record["kind"] == "model":
        return model
    facts = record["optimizer"]
    if facts["kind"] == "sgd":
        optimizer = SGD(model.parameters(), lr=facts["lr"])
    else:
        optimizer = Adam(model.parameters(), lr=facts["lr"], betas=facts["betas"],
                         eps=facts["eps"], weight_decay=facts["weight_decay"])
        state, temporary = [], []
        try:
            for index, step in enumerate(facts["steps"]):
                if not step:
                    state.append(None)
                    continue
                moments = []
                for name in ("first", "second"):
                    storage = values[index, name]
                    if storage.device != device:
                        storage = _copy_storage(storage, device)
                        temporary.append(storage)
                    moments.append(storage.view(readonly=True))
                state.append((step, *moments))
            optimizer._state = tuple(state)
        finally:
            p.retire(temporary, flush=False)
    optimizer._committed_steps = facts["committed_steps"]
    return model, optimizer


def _model_path(path, *, writing=False):
    from infernux.application import Application
    from infernux.engine.path_utils import (
        is_lexical_path_within, resolve_destination_path, resolve_project_path, resolved_path,
    )

    path = os.fsdecode(path)
    if os.path.splitext(path)[1].casefold() != ".inm":
        raise ValueError("Infernux neural models and checkpoints require the .inm suffix")
    project = Application.data_path()
    root, separator, relative = path.replace("\\", "/").partition("/")
    if project and separator and root.casefold() == "assets":
        authored = "Assets/" + relative
        if not writing:
            return os.path.abspath(Application.asset_path(authored))
        if Application.is_player():
            raise ValueError("Player model output requires a writable path outside packaged Assets")
        from infernux.core.assets import AssetManager

        AssetManager._validate_author_asset_pattern(authored)
        destination = resolve_project_path(authored, project, destination=True)
        if not is_lexical_path_within(destination, resolved_path(os.path.join(project, "Assets")), allow_root=False):
            raise ValueError("model asset destination must stay within the project's Assets directory")
        return destination
    return resolve_destination_path(path) if writing else os.path.abspath(path)


def _read_header(manifest):
    header = manifest["text_header"]
    if header is None:
        raise ValueError("an INM model requires a text header followed by binary ckpt data")
    return json.loads(header)


def _read(path, device, kind, programs=None):
    from infernux import lib

    _check_options(None, device)
    path = _model_path(path)
    reader = lib._Infernux._InxPackReader(path)
    buffers, values = [], {}
    array = None
    source = owned_source = None
    complete = False
    try:
        facts = reader.manifest
        record = _read_header(facts)
        definition, flags = _read_facts(record, kind)
        manifest = {row["path"]: row["raw_bytes"] for row in facts["files"]}
        if "programs" in record:
            from pathlib import Path
            from ._programs import create_source, load_programs
            source = owned_source = create_source(record["programs"])
            source.require_device(device)
        entries = _model_entries(definition, flags, record.get("optimizer"))
        _check_entries(manifest, entries,
                       () if owned_source is None else owned_source.file_sizes.items())
        if owned_source is not None:
            load_programs(Path(os.fsdecode(path)).resolve().parent, owned_source, reader)
        if programs is not None:
            expected, source = _checkpoint_programs(programs, device)
            if kind != "checkpoint" or expected != definition:
                raise ValueError("checkpoint programs require a built model with the same fixed definition")
        if source is not None:
            source.require_device(device)
        for name, index, kind, shape in entries:
            storage = _empty(shape, "cpu")
            buffers.append(storage)
            array = storage.numpy(copy=False)
            reader.read_into(name, array)
            if not np.little_endian:
                array.byteswap(inplace=True)
            values[index, kind] = storage
        result = _loaded_result(definition, values, device, record, source)
        if owned_source is not None:
            owned_source._publish()
        complete = True
        return result
    finally:
        if not complete and owned_source is not None:
            owned_source._discard_unpublished()  # Never release a borrowed provider on load failure.
        array = None  # A retained IO exception must not retain the last numerical allocation.
        reader.close()
        p.retire(buffers, flush=False)
        buffers.clear()
        values.clear()


def _managed_load(path, device, kind, programs, window, budget, finish_within_ms):
    from infernux.application import Application

    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError("managed model input requires an active Engine")
    owner = engine._cpu_work(budget)
    request = owner.release_model_load(path, device=device, kind=kind, programs=programs,
                                      window_opportunities=window, finish_within_ms=finish_within_ms)
    return engine._wake_cpu_request(owner, request)


def load(path, *, device="cpu", finish_within_frames=None, finish_within_ms=None, budget=None):
    """Load a fixed graph/weights without importing its author class.

    A baked file uses its built programs on the ordinary Plan. An export without
    code uses the compiler on first prepare. A budget, host-frame window or millisecond target schedules
    program input and parameter transfers on the same managed owner; result
    publishes the complete Module. Player cook is a separate step.
    """
    if any(value is not None for value in (finish_within_frames, finish_within_ms, budget)):
        return _managed_load(path, device, "model", None, finish_within_frames, budget, finish_within_ms)
    return _read(path, device, "model")


def load_checkpoint(path, *, device="cpu", programs=None, finish_within_frames=None, finish_within_ms=None, budget=None):
    """Return (fixed Module, optimizer), retaining published gradient accumulation.

    Pass a loaded baked model as programs to reuse its code without a compiler.
    The checkpoint remains portable state, independent of the code directory.
    A budget, host-frame window or millisecond target returns a managed request: directory reads, decoding and GPU
    transfers use the common owner, then result publishes the complete tuple.
    It borrows existing programs and never prepares or compiles a model during
    loading. Without built programs, the caller explicitly prepares later.
    """
    if any(value is not None for value in (finish_within_frames, finish_within_ms, budget)):
        return _managed_load(path, device, "checkpoint", programs, finish_within_frames, budget, finish_within_ms)
    return _read(path, device, "checkpoint", programs)
