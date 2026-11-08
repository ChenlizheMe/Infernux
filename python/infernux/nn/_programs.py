"""Fixed NN recipe selection; execution remains in Plan and the common owner."""
import json
import numbers

from infernux import compute
from infernux._compiler.kernel_contract import ARRAY_KINDS
from infernux.cpu_programs import (NativeCpuLibrary, NativeCpuProgram, library_directory,
    _library_path, _require_cpu_target)


def recipe(function, params, facts):
    constants = {name: value for name in function.__code__.co_names
                 if isinstance(value := facts.get(name, function.__globals__.get(name)), numbers.Real)
                 and not isinstance(value, bool)}
    constants = {name: int(value) if isinstance(value, numbers.Integral) else float(value)
                 for name, value in constants.items()}
    buffers = [value for value in params if isinstance(value, compute.Buffer)]
    return dict(operation=function.__name__, constants=constants, device=buffers[0].device,
                kinds=[ARRAY_KINDS[value.dtype] if isinstance(value, compute.Buffer) else
                       "int32" if isinstance(value, numbers.Integral) else "float32" for value in params])


def _key(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _structure_units(value):
    if isinstance(value, dict):
        return 1 + sum(_structure_units(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return 1 + sum(_structure_units(item) for item in value)
    return max(1, len(value)) if isinstance(value, str) else 1


def _bind_family(program, variants):
    """Check only the finite relationship that the existing cursor consumes."""
    plan = program.continuation_plan
    region = plan.ordered_reduction
    expected = ({"initial", "final"} | ({"resume"} if region.count > 2 else set())
                if region is not None and region.eligible and region.count > 1 else set())
    if (plan.inner_range is not None or not plan.outputs or plan.outputs != plan.fully_written_outputs
            or set(variants) != expected):
        raise ValueError("native NN program has an incomplete continuation family")
    count = len(program.parameter_names)
    for kind, child in variants.items():
        proof, control = child.continuation_plan, child.continuation_plan.inner_range
        if (control is None or (control.kind, control.count, control.begin_parameter, control.end_parameter)
                != (kind, region.count, count + 2, count + 3)
                or child.domain_parameter != program.domain_parameter or proof.ordered_reduction != region
                or child.parameter_names[:count] != program.parameter_names
                or child.parameter_kinds != (*program.parameter_kinds, "array_f32", "array_f32", "int32", "int32")
                or len(child.parameter_names) != count + 4):
            raise ValueError("native NN continuation layout differs from its whole program")
        carry_in, carry_out = child.parameter_names[count:count+2]
        outputs = plan.outputs if kind == "final" else (carry_out,)
        if (proof.outputs != outputs or proof.fully_written_outputs != outputs
                or (carry_in in proof.inputs) != (kind != "initial") or carry_out in proof.inputs
                or set(proof.inputs) - {carry_in} - set(plan.inputs)):
            raise ValueError("native NN continuation read/write roles differ from its whole program")
    program._inner_variants = variants


def _catalogue(record):
    """Validate the serialized catalogue before any code or numerical IO."""
    device = record["device"]
    identities = [row["identity"] for row in record["entries"]]
    if len(set(identities)) != len(identities):
        raise ValueError("model program entries must have unique identities")
    if device == "cpu":
        _require_cpu_target(record["library"]["target"])
        path = "programs/" + _library_path(".", record["library"]).name
        files = {path: record["library"]["bytes"]}
    elif device == "gpu":
        files = {}
        for row in record["entries"]:
            if row["path"] != "programs/" + row["identity"] + ".inxgpu":
                raise ValueError("GPU model program path differs from its identity")
            files[row["path"]] = row["bytes"]
    else:
        raise ValueError("model programs require a CPU or GPU target")
    if any(type(count) is not int or count <= 0 for count in files.values()):
        raise ValueError("model program files require positive byte counts")
    return files, frozenset(identities)


class ProgramSource:
    def __init__(self, record):
        self.record = record
        self.device = record["device"]
        self.programs = {}
        self.file_sizes, self._identities = _catalogue(record)
        self._recipe_index = 0
        self._recipe_keys = set()
        # Count the real repeated structural work once, rather than serialize
        # the whole catalogue during each opportunity's numerical forecast.
        self._entry_units = tuple(_structure_units(row) for row in record["entries"])
        self._recipe_units = tuple(_structure_units(row) for row in record["recipes"])

    def host_work(self, stage):
        """Remaining real declare/bind calls, including their terminal call."""
        if stage == "declare" and self.device == "cpu":
            for index in range(self._entry_index, len(self._entry_units)):
                yield "declare_entry", self._entry_units[index]
        entries = stage == "bind" and self.device == "cpu"
        units = self._entry_units if entries else self._recipe_units
        start = self._recipe_index if stage == "declare" else self._bind_index
        for index in range(start, len(units)):
            yield stage + ("_entry" if entries else "_recipe"), units[index]
        yield stage + "_end", max(1, len(self._entry_units)) if entries else 1

    def _check_recipe(self, row):
        if row["recipe"]["device"] != self.device:
            raise ValueError("model recipe device differs from its program catalogue")
        key = _key(row["recipe"])
        if key in self._recipe_keys:
            raise ValueError("model recipes must have unique bindings")
        self._recipe_keys.add(key)
        return key

    def require_device(self, device):
        if device != self.device:
            raise ValueError(f"model contains {self.device} programs; build {device} programs before moving it")

    def prepare(self, function, params, **facts):
        self.require_device(next(value.device for value in params if isinstance(value, compute.Buffer)))
        key = _key(recipe(function, params, facts))
        try:
            return self.programs[key]
        except KeyError:
            raise ValueError(f"model was not built with the required {function.__name__} recipe") from None


class CpuProgramSource(ProgramSource):
    def __init__(self, record):
        super().__init__(record)
        self.library = None
        self._entries = {}
        self._entry_index = self._bind_index = 0

    def declare_step(self):
        """One entry or relationship; the complete catalogue precedes OS IO."""
        if self._entry_index < len(self.record["entries"]):
            row = self.record["entries"][self._entry_index]
            self._entries[row["identity"]] = NativeCpuProgram(row)
            self._entry_index += 1
            return False
        if self._recipe_index < len(self.record["recipes"]):
            row = self.record["recipes"][self._recipe_index]
            key = self._check_recipe(row)
            if row["whole"] not in self._entries or any(i not in self._entries for i in row["variants"].values()):
                raise ValueError("CPU model recipe references an absent program")
            program = self._entries[row["whole"]]
            _bind_family(program, {kind: self._entries[i] for kind, i in row["variants"].items()})
            program._nn_recipe = row["recipe"]
            self.programs[key] = program
            self._recipe_index += 1
            return False
        return True

    def bind_step(self):
        if self._bind_index < len(self.record["entries"]):
            row = self.record["entries"][self._bind_index]
            self._entries[row["identity"]].bind(self.library)
            self._bind_index += 1
            return False
        self._entries.clear()
        return True

    @property
    def native_blocks(self):
        return () if self.library is None else self.library.native_blocks

    def stage(self):
        """Reuse already-built code without importing the compiler."""
        return self.record, (("programs/"+self.record["library"]["file"], str(self.library.path)),)

    def _discard_unpublished(self):
        if self.library is not None:
            self.library._discard_unpublished()
        self.programs.clear()
        self._entries.clear()

    def _publish(self):
        self.library._publish()


def load_programs(directory, source, reader):
    record = source.record
    try:
        while not source.declare_step():
            pass
        if source.device == "cpu":
            path = _library_path(library_directory(directory), record["library"])
            file = reader.extract_entry("programs/" + record["library"]["file"], str(path), record["library"]["sha256"])
            source.library = NativeCpuLibrary.load(file, [r["symbol"] for r in record["entries"]])
            while not source.bind_step():
                pass
        else:
            for row in record["entries"]:
                payload = reader.read(row["path"])
                source.decode_entry(row, payload)
            while not source.bind_step():
                pass
        return source
    except BaseException:
        source._discard_unpublished()
        raise


def create_source(record):
    if record["device"] == "gpu":
        from ._gpu_programs import GpuProgramSource
        return GpuProgramSource(record)
    return CpuProgramSource(record)
