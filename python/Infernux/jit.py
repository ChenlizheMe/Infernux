"""Engine-owned CPU compilation and warmup.

Use ``@inx.jit.compile`` for ordinary CPU functions. The compiler chooses a
legal serial or parallel implementation from the existing typed-HIR analysis;
``warmup`` prepares the selected implementation on isolated inputs before
gameplay, and decisions and diagnostics remain queryable.

GPU kernels are a separate ``inx.compute.kernel`` execution model and never
route through Numba CUDA. CPU compilation does not install, repair, or select a
GPU backend at runtime.

Numba remains an internal CPU code-generation backend; its decorator and
``prange`` are not a second public authoring surface.
"""

from functools import lru_cache
import os
import sys

from Infernux._jit_kernels import (
    JIT_AVAILABLE as _BACKEND_JIT_AVAILABLE,
    njit as _njit,
    warmup as _native_warmup,
)
from Infernux.jit_runtime import (
    CpuCompilationStatistics as Statistics,
    CpuPassTiming as PassTiming,
    CpuSpecializationStatistics as SpecializationStatistics,
)


_VECTOR_FIELDS = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
}


def _is_web_runtime() -> bool:
    return os.environ.get("INFERNUX_WEB_RUNTIME") == "1" or sys.platform == "emscripten"


def _jit_runtime_available() -> bool:
    """Return whether this process may execute the native CPU JIT."""
    # A host Python can have Numba installed while assembling a Web Player.
    # The target capability, not merely the host backend, controls execution.
    return _BACKEND_JIT_AVAILABLE and not _is_web_runtime()


# Describe the active Player. A host Python used while assembling Web content
# may have Numba installed, but the Web Player deliberately does not.
JIT_AVAILABLE = _jit_runtime_available()


@lru_cache(maxsize=1)
def _vector_layouts():
    """Native vector types and immutable layouts, independent of call data."""
    from Infernux.lib import Vector2, Vector3, vec4f
    import numpy as np
    return tuple(
        (vector_type, fields, np.dtype([(name, np.float32) for name in fields]))
        for vector_type, fields in (
            (Vector2, _VECTOR_FIELDS[2]),
            (Vector3, _VECTOR_FIELDS[3]),
            (vec4f, _VECTOR_FIELDS[4]),
        )
    )


def _vector_record(value):
    """Return a typed NumPy scalar for one public engine vector, if any."""
    import numpy as np
    for vector_type, fields, dtype in _vector_layouts():
        if isinstance(value, vector_type):
            record = np.array(tuple(float(getattr(value, name)) for name in fields), dtype=dtype)[()]
            return record, value, vector_type, fields
    return None


class _CompiledCpuFunction:
    """Thin public adapter from engine CPU buffers to native JIT storage."""

    def __init__(self, compiled):
        self._compiled = compiled
        self.__name__ = getattr(compiled, "__name__", type(compiled).__name__)
        self.__qualname__ = getattr(compiled, "__qualname__", self.__name__)
        self.__doc__ = getattr(compiled, "__doc__", None)
        self.__module__ = getattr(compiled, "__module__", __name__)

    @staticmethod
    def _arguments(args, kwargs):
        from Infernux.compute import Buffer
        import numpy as np

        owners = {}

        def unwrap(value):
            if value is None or isinstance(value, (bool, int, float, complex, np.number)):
                return value
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise TypeError("inx.jit.compile does not accept NumPy object arrays")
                return value
            if isinstance(value, Buffer):
                value._require_open()
                if value.device != "cpu":
                    raise RuntimeError("inx.jit.compile accepts CPU buffers; GPU buffers require inx.compute.launch")
                array = value.numpy(copy=False)
                owners[id(array)] = value
                return array
            identity = ("vector", id(value))
            if identity in owners:
                return owners[identity]
            vector = _vector_record(value)
            if vector is not None:
                record, original, vector_type, fields = vector
                owners.setdefault("vectors", []).append((record, original, vector_type, fields))
                owners[("vector_type", record.dtype.str)] = (vector_type, fields)
                owners[identity] = record
                return record
            raise TypeError(
                "inx.jit.compile arguments must be numeric scalars, NumPy arrays, "
                "CPU inx.buffer values, or Infernux vectors"
            )

        return tuple(unwrap(value) for value in args), {
            name: unwrap(value) for name, value in kwargs.items()
        }, owners

    @staticmethod
    def _restore(value, owners):
        import numpy as np

        owner = owners.get(id(value))
        if owner is not None:
            return owner
        if isinstance(value, np.void):
            vector = owners.get(("vector_type", value.dtype.str))
            if vector is not None:
                vector_type, fields = vector
                return vector_type(*(float(value[name]) for name in fields))
        if isinstance(value, tuple):
            return tuple(_CompiledCpuFunction._restore(item, owners) for item in value)
        if value is None or isinstance(value, (bool, int, float, complex, np.number, np.ndarray)):
            if isinstance(value, np.ndarray) and value.dtype.hasobject:
                raise TypeError("inx.jit.compile cannot return a NumPy object array")
            return value
        raise TypeError(
            "inx.jit.compile must return a numeric scalar, NumPy array, CPU inx.buffer, "
            "Infernux vector, tuple of supported values, or None"
        )

    @staticmethod
    def _commit_vectors(owners):
        for record, original, _vector_type, fields in owners.get("vectors", ()):
            for name in fields:
                setattr(original, name, float(record[name]))

    def __call__(self, *args, **kwargs):
        native_args, native_kwargs, owners = self._arguments(args, kwargs)
        try:
            result = self._compiled(*native_args, **native_kwargs)
        finally:
            # As with ndarray writes, an authored exception does not roll back
            # writes already performed by native code. Never replay the call.
            self._commit_vectors(owners)
        return self._restore(result, owners)

    def _infernux_warmup(self, *args, **kwargs):
        native_args, native_kwargs, _ = self._arguments(args, kwargs)
        prepare = getattr(self._compiled, "_infernux_warmup", None)
        if prepare is not None:
            return prepare(*native_args, **native_kwargs)
        from Infernux.jit_runtime import clone_call_arguments
        prepared_args, prepared_kwargs = clone_call_arguments(native_args, native_kwargs)
        self._compiled(*prepared_args, **prepared_kwargs)

    def __getattr__(self, name):
        return getattr(self._compiled, name)


def compile(fn=None, **options):
    """Compile a CPU function with Infernux's serial/parallel analysis.

    ``auto_parallel`` defaults to true. Callers may set it explicitly when a
    strictly serial native function is required. The returned dispatcher keeps
    the existing diagnostics, warmup, bounded specialization, and no-replay
    execution rules.

    Referenced globals and closures are compile-time values for a publication;
    publish a new function revision to change them. ``cache=True`` also keys
    disk entries by those dependencies, not just the source file timestamp.
    Editor artifacts live under ``Library/Artifacts/Compute/CPU``; Players use
    their writable application data root. Standalone compiler tools requesting
    disk caching must supply an explicit ``NUMBA_CACHE_DIR``.

    Each compiled implementation admits up to 64 input-type specializations.
    Array lengths do not create new specializations. Beyond capacity, new
    signatures are rejected before user code runs; existing ones remain valid.

    Automatic parallel selection also respects actual array storage. Shared
    parameters need a proven independent access pattern and identical layouts;
    offset aliases or overlapping elements select serial before execution.
    ``parallel_policy="required"`` rejects layouts without that proof. No
    partially executed call is replayed through another implementation.
    """
    if not _jit_runtime_available():
        raise RuntimeError(
            "inx.jit.compile requires the bundled Numba/llvmlite CPU JIT runtime"
        )
    options.setdefault("auto_parallel", True)
    if fn is None:
        def decorate(function):
            return _CompiledCpuFunction(_njit(**options)(function))
        return decorate
    if not callable(fn):
        raise TypeError("inx.jit.compile expects a callable")
    return _CompiledCpuFunction(_njit(fn, **options))


def warmup(fn, *args, **kwargs):
    """Prepare native CPU code without executing authored gameplay state.

    Targets without a CPU compiler remove public warmup calls while cooking
    project scripts. Reaching this runtime API therefore always requires the
    bundled native JIT instead of silently changing execution semantics.
    """
    if not _jit_runtime_available():
        raise RuntimeError(
            "inx.jit.warmup requires the bundled Numba/llvmlite CPU JIT runtime"
        )
    return _native_warmup(fn, *args, **kwargs)


def statistics(fn) -> Statistics:
    """Snapshot a CPU compilation without compiling or executing the function.

    Preparation time includes dependency compilation/cache loading, excluding
    waiting for the compiler lock; it is not steady-state execution time.
    Pass timings describe cold compilation only. The report owns no native
    code and remains readable after the function has retired.

    Memory counts require the Infernux llvmlite fork, otherwise they are None.
    They measure mapped code/data including allocator padding, not IR or RSS.
    Reachable memory includes shared linking dependencies and cannot be summed
    across function reports. This reports usage, not a memory-budget guarantee.
    """
    if not isinstance(fn, _CompiledCpuFunction):
        raise TypeError("jit.statistics expects a function returned by jit.compile")
    from Infernux._jit_backend import compilation_statistics

    compiled = fn._compiled
    if getattr(compiled, "auto_parallel", False):
        implementations = [("serial", compiled.serial)]
        if compiled.parallel is not compiled.serial:
            implementations.append(("parallel", compiled.parallel))
        mode = compiled.selected_mode
        diagnostic = compiled.last_diagnostic
        decisions = tuple((repr(key), value) for key, value in compiled.decisions.items())
    else:
        mode = "parallel" if compiled.targetoptions.get("parallel") else "serial"
        implementations = [(mode, compiled)]
        diagnostic = "explicit CPU implementation"
        decisions = ()
    return compilation_statistics(implementations, function_name=fn.__qualname__,
                                  selected_mode=mode, last_diagnostic=diagnostic, decisions=decisions)


__all__ = ["JIT_AVAILABLE", "compile", "warmup", "statistics", "Statistics", "PassTiming", "SpecializationStatistics"]
