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

import inspect
from functools import lru_cache
import os
import sys

from Infernux._jit_kernels import JIT_AVAILABLE, njit as _njit, warmup


_VECTOR_FIELDS = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
}


def _register_numeric_helpers(function) -> None:
    """Make referenced same-module Python helpers visible to nopython lowering."""
    from numba.extending import register_jitable

    visited = {id(function)}

    def register_dependencies(value) -> None:
        code = getattr(value, "__code__", None)
        globals_map = getattr(value, "__globals__", {})
        if code is None:
            return
        for name in code.co_names:
            dependency = globals_map.get(name)
            if (not inspect.isfunction(dependency)
                    or getattr(dependency, "__module__", None) != getattr(function, "__module__", None)
                    or id(dependency) in visited):
                continue
            visited.add(id(dependency))
            register_dependencies(dependency)
            register_jitable(dependency)

    register_dependencies(function)


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
    """
    if not JIT_AVAILABLE:
        if os.environ.get("INFERNUX_WEB_RUNTIME") == "1" or sys.platform == "emscripten":
            if fn is None:
                def decorate(function):
                    return function
                return decorate
            if not callable(fn):
                raise TypeError("inx.jit.compile expects a callable")
            return fn
        raise RuntimeError(
            "inx.jit.compile requires the bundled Numba/llvmlite CPU JIT runtime"
        )
    options.setdefault("auto_parallel", True)
    if fn is None:
        def decorate(function):
            _register_numeric_helpers(function)
            return _CompiledCpuFunction(_njit(**options)(function))
        return decorate
    if not callable(fn):
        raise TypeError("inx.jit.compile expects a callable")
    _register_numeric_helpers(fn)
    return _CompiledCpuFunction(_njit(fn, **options))


__all__ = ["JIT_AVAILABLE", "compile", "warmup"]
