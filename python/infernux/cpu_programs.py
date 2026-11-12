"""Target-native CPU programs on the common range binding and JobSystem ABI."""
from functools import cache
import os
from pathlib import Path

from .cpu_execution import CpuRangeProgram, _range_payload
from ._compiler.kernel_contract import ARRAY_DTYPES
from .jit_hir import ContinuationPlan, InnerRangeContract, OrderedReductionRegion, ResourceExtent

_SUFFIX = {"windows-x86_64": ".dll", "linux-x86_64": ".so"}


@cache
def _call_type():
    # Model recipes and GPU programs also import this module. Loading an OS
    # call ABI belongs to the native CPU entry, not to recipe declaration.
    import ctypes

    return ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
                           ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)


def library_directory(model_directory):
    """Programs use the project's common cache, or the explicit offline root."""
    from .application import Application

    return Path(Application.persistent_data_path() or model_directory) / "Library/Artifacts/Compute/CPU/Native"


def _library_path(directory, record):
    filename = record["file"]
    if filename != library_filename(record["target"], record["sha256"]) or Path(filename).name != filename:
        raise ValueError("CPU program library filename must match its code identity")
    return Path(directory) / filename


def library_filename(target, digest):
    return "nn-" + digest + _SUFFIX[target]


def native_cpu_target():
    from . import lib

    return lib._Infernux._native_cpu_program_target


def _require_cpu_target(target):
    host = native_cpu_target()
    if host is None or target != host:
        raise ValueError(f"CPU program target {target} cannot run on {host or 'this host'}")


def stage_library(source, directory, record):
    """Publish immutable code in the common cache before packing its model."""
    destination = _library_path(directory, record)
    if destination.exists():
        return destination
    from . import lib
    lib._Infernux._copy_file_verified(os.fspath(source), os.fspath(destination), record["sha256"])
    return destination


def _continuation(record):
    facts = dict(record)
    facts["resource_extents"] = tuple(ResourceExtent.from_record(value) for value in facts["resource_extents"])
    for name, cls in (("ordered_reduction", OrderedReductionRegion), ("inner_range", InnerRangeContract)):
        value = facts[name]
        facts[name] = None if value is None else cls(**value)
    for name in ("diagnostics", "inputs", "outputs", "fixed_parameters", "cross_slice_state", "fully_written_outputs"):
        facts[name] = tuple(facts[name])
    plan = ContinuationPlan(**facts)
    plan.require()
    return plan


class NativeCpuLibrary:
    """One OS reference, retained by entries until their real work retires."""
    def __init__(self, module, path):
        self.path = Path(path)
        self.image_bytes = module.image_bytes
        self._module = module

    @classmethod
    def load(cls, file, symbols):
        from . import lib

        module = lib._Infernux._NativeModule.load(file, symbols)
        try:
            return cls(module, file.path)
        except BaseException:
            module.discard_unpublished()
            raise

    @property
    def native_blocks(self):
        # Loaded image size, not working set, file-cache residency or process RSS.
        return (() if self._module is None or not self._module.identity else
                ((("cpu-code", self._module.identity), self.image_bytes),))

    def __del__(self):
        self._close()

    def _close(self):
        self._module = None

    def _publish(self):
        self._module.publish()

    def _discard_unpublished(self):
        module = getattr(self, "_module", None)
        if module is not None:
            module.discard_unpublished()
        self._close()


class _NativeEntry:
    def __init__(self, library, symbol):
        self.library = library
        self.module = library._module
        self.address = self.module.address(symbol)
        self.call = _call_type()(self.address)


class NativeCpuProgram(CpuRangeProgram):
    def __init__(self, record):
        super().__init__(record["parameter_names"], record["parameter_kinds"],
                         record["domain_parameter"], _continuation(record["continuation"]))
        if (len(self.parameter_names) != len(self.parameter_kinds)
                or len(set(self.parameter_names)) != len(self.parameter_names)
                or set(self.parameter_kinds) - {*ARRAY_DTYPES, "float32", "int32"}
                or self.domain_parameter not in self.array_parameters
                or record["symbol"] != "inx_cpu_" + record["identity"]):
            raise ValueError("native CPU program layout differs from its entry contract")
        self.cost_identity = record["identity"]
        self._symbol = record["symbol"]
        self._c_entry = None
        names = self.parameter_names
        self._work_access = (tuple(names.index(name) for name in self.continuation_plan.inputs),
                           tuple(names.index(name) for name in self.continuation_plan.outputs))

    def bind(self, library):
        self._c_entry = _NativeEntry(library, self._symbol)

    def _native_entry(self):
        if self._c_entry is None:
            raise RuntimeError("bind the declared native CPU program before execution")
        self._c_entry.module.require_published()
        return self._c_entry

    @property
    def native_blocks(self):
        if self._c_entry is None:
            return ()
        owner = self._c_entry.module
        return () if not owner.identity else ((("cpu-code", owner.identity), owner.image_bytes),)

    def __call__(self, *params, begin, end):
        import ctypes

        self._c_entry.module.require_published()
        storage, owners, slots, count, begin, end = _range_payload(self, params, begin, end)
        pointers = (ctypes.c_void_p * len(slots))(*(slot[0] for slot in slots))
        lengths = (ctypes.c_int32 * len(storage))(*(value.size for value in storage))
        status = self._c_entry.call(pointers, lengths, count, begin, end)
        if status:
            # Native errors may follow writes. Never retry or publish that output.
            raise RuntimeError(f"native CPU range failed with status {status}; output is invalid")
