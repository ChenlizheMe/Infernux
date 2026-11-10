"""Compiler-free CPU binding, task ownership and exact retirement.

JIT and cooked programs share this path and the original native JobSystem.
Only a prepared program supplies code; importing this module never loads it.
"""
from functools import lru_cache
from ._compiler.kernel_contract import ARRAY_DTYPES

_VECTOR_FIELDS = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
}


@lru_cache(maxsize=1)
def _vector_layouts():
    """Native vector types and immutable layouts, independent of call data."""
    from infernux.lib import Vector2, Vector3, vec4f
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


class _CpuArguments:
    @staticmethod
    def _arguments(args, kwargs):
        from infernux.compute import Buffer
        import numpy as np

        owners = {}

        def unwrap(value):
            if value is None or isinstance(value, (bool, int, float, complex, np.number)):
                return value
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    raise TypeError("CPU execution does not accept NumPy object arrays")
                return value
            if isinstance(value, Buffer):
                value._require_open()
                if value.device != "cpu":
                    raise RuntimeError("CPU execution accepts CPU buffers; GPU buffers require inx.compute.launch")
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
                "CPU execution arguments must be numeric scalars, NumPy arrays, "
                "CPU inx.buffer values, or Infernux vectors"
            )

        return tuple(unwrap(value) for value in args), {
            name: unwrap(value) for name, value in kwargs.items()
        }, owners


class CpuRangeProgram(_CpuArguments):
    """Runtime binding facts shared by JIT and target-native programs."""
    def __init__(self, parameter_names, parameter_kinds, domain_parameter, plan):
        self.parameter_names = tuple(parameter_names)
        self.parameter_kinds = tuple(parameter_kinds)
        self.array_parameters = frozenset(index for index, kind in enumerate(parameter_kinds) if kind in ARRAY_DTYPES)
        self.domain_parameter = domain_parameter
        self.continuation_plan = plan
        self._inner_variants = {}

    @property
    def native_blocks(self):
        return ()  # JIT allocation residency is not described by a native image.

    def _bind(self, params):
        import numpy as np

        if len(params) != len(self.parameter_names):
            raise TypeError(f"work item requires {len(self.parameter_names)} arguments, got {len(params)}")
        values, _, owners = self._arguments(params, {})
        values = list(values)
        arrays = {}
        for index, value in enumerate(values):
            name = self.parameter_names[index]
            if index in self.array_parameters:
                dtype = ARRAY_DTYPES[self.parameter_kinds[index]]
                if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype)
                        or value.ndim != 1 or not value.flags.c_contiguous or not value.flags.aligned):
                    raise TypeError(f"continuation buffer '{name}' requires aligned, contiguous one-dimensional {dtype} storage")
                if name in self.continuation_plan.outputs and not value.flags.writeable:
                    raise ValueError(f"continuation output '{name}' is read-only")
                arrays[index] = value
            elif self.parameter_kinds[index] == "int32":
                if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                        or not -0x80000000 <= int(value) <= 0x7fffffff):
                    raise TypeError(f"continuation control '{name}' requires an exact signed int32")
                values[index] = np.int32(value)
            else:
                if not isinstance(value, (float, np.floating)):
                    raise TypeError(f"continuation scalar '{name}' requires a float32/floating value")
                values[index] = np.float32(value)
        domain = arrays.get(self.domain_parameter)
        if domain is None:
            raise TypeError("continuation execution domain must be an array")
        count = len(domain)
        if count > 2**31 - 1:
            raise ValueError("continuation domain exceeds the i32 logical index range")
        control = self.continuation_plan.inner_range
        if control is not None:
            control.validate(values[control.begin_parameter], values[control.end_parameter])
        for extent in self.continuation_plan.resource_extents:
            extent.validate(count, len(arrays[extent.parameter]))
        for index, array in arrays.items():
            if self.parameter_names[index] not in self.continuation_plan.outputs:
                continue
            start = int(array.__array_interface__["data"][0])
            for other_index, other in arrays.items():
                if other_index == index:
                    continue
                other_start = int(other.__array_interface__["data"][0])
                if array.nbytes and other.nbytes and start < other_start + other.nbytes and other_start < start + array.nbytes:
                    raise ValueError(f"continuation output '{self.parameter_names[index]}' overlaps buffer '{self.parameter_names[other_index]}'")
        for index, array in arrays.items():
            if self.parameter_names[index] not in self.continuation_plan.outputs:
                array = array.view()
                array.flags.writeable = False
                values[index] = array
        return tuple(values), owners, count


    def _range_arguments(self, params, begin, end):
        import numpy as np

        values, owners, count = self._bind(params)
        for name, value in (("begin", begin), ("end", end)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"continuation {name} must be an integer, not bool")
        begin, end = int(begin), int(end)
        if not 0 <= begin <= end <= count:
            raise ValueError(f"continuation range [{begin},{end}) is outside [0,{count})")
        return values, owners, count, begin, end


class _CpuRangeTask:
    """Private completion ticket; it never publishes a partial output to a scene."""

    def __init__(self, engine, native, owners):
        import weakref

        self._engine = weakref.ref(engine)
        self._native = native
        self._owners = owners
        self._done = False
        self._cancelled = False
        self._result_code = None
        self._ran_on_worker = False
        self._execution_ns = None
        self._operation_execution_ns = ()

    def submit(self):
        """Admit this exact prepared ticket; pins never move to a second ledger."""
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
        from infernux.application import Application
        from infernux import lib

        assert_runtime_owner_thread()
        engine = self._engine()
        if engine is None or Application._current_engine() is not engine or not engine._cpu_range_accepting:
            raise RuntimeError("CPU range submission requires its active accepting Engine owner")
        native = self._native
        if self._done or self._cancelled or native is None or native.submitted:
            raise RuntimeError("CPU range ticket can only be submitted once")
        engine._cpu_range_tasks.add(self)  # Own pins before Schedule can accept.
        try:
            native.submit()
        except BaseException:
            if not native.submitted:
                self.cancel()
                self._poll()
            # An accepted failure remains in the one registry until retirement.
            raise
        if engine._mode != lib.RuntimeMode.Headless:
            engine.get_native_engine().request_full_speed_frame()
        return self

    def _poll(self):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread

        assert_runtime_owner_thread()
        if not self._done and self._native.is_complete:
            self._cancelled = self._cancelled or bool(self._native.cancelled)
            self._result_code = self._native.result_code
            self._ran_on_worker = bool(self._native.ran_on_worker)
            self._execution_ns = self._native.execution_ns
            self._operation_execution_ns = tuple(self._native.operation_execution_ns)
            self._done = True
            # JobHandle completion is authoritative, including a callback skipped
            # by cancellation. Only this owner point releases Python/code refs.
            self._native = None
            self._owners = None
            engine = self._engine()
            if engine is not None:
                engine._cpu_range_tasks.discard(self)
        return self._done

    @property
    def done(self):
        return self._poll()

    @property
    def cancelled(self):
        self._poll()
        return self._cancelled

    @property
    def result_code(self):
        self._poll()
        return self._result_code

    @property
    def ran_on_worker(self):
        self._poll()
        return self._ran_on_worker

    @property
    def execution_ns(self):
        """Completed native entry time; None if pending or cancelled before execution."""
        self._poll()
        return self._execution_ns

    @property
    def operation_execution_ns(self):
        self._poll()
        return self._operation_execution_ns

    @property
    def storage(self):
        return tuple(value for _, _, storage, _, _ in self._owners or () for value in storage)

    @property
    def scalar_storage(self):
        return tuple(value for compiled, _, storage, _, _ in self._owners or ()
                     for index, value in enumerate(storage) if index not in compiled.array_parameters)

    @property
    def native_blocks(self):
        return tuple(block for compiled, _, _, _, _ in self._owners or () for block in compiled.native_blocks)

    @property
    def scalar_storage_bytes(self):
        return sum(value.nbytes for value in self.scalar_storage)

    def cancel(self):
        from infernux.engine.runtime_dispatch import assert_runtime_owner_thread

        assert_runtime_owner_thread()
        if self._done or self._cancelled:
            return False
        # Withdraw publication even if native work completed just before Cancel.
        # Pins remain until _poll observes actual completion, never cancellation.
        self._cancelled = True
        self._native.cancel()
        return True


def _range_payload(compiled, params, begin, end):
    """One numerical/scalar binding for direct calls and queued entries."""
    import numpy as np

    values, borrowed, count, begin, end = compiled._range_arguments(params, begin, end)
    storage = tuple(value if index in compiled.array_parameters else np.array(
                        [value], dtype=np.int32 if compiled.parameter_kinds[index] == "int32" else np.float32)
                    for index, value in enumerate(values))
    slots = [(int(value.ctypes.data), int(value.nbytes), index not in compiled.array_parameters,
              compiled.parameter_names[index] in compiled.continuation_plan.outputs, int(value.dtype.itemsize))
             for index, value in enumerate(storage)]
    return storage, borrowed, slots, count, begin, end


def _prepare_cpu_range(compiled, *params, begin, end):
    """Prepare one retained ticket without admitting a native job.

    The caller must not mutate inputs/scalars, read unfinished outputs, resize
    storage, or overlap another writer until completion. Cancellation does not
    roll back writes. A nonzero result invalidates the whole request's output.
    """
    return _prepare_cpu_ranges(((compiled, params, begin, end),))


def _prepare_cpu_ranges(ranges):
    """Prepare a bounded ordered segment, pinned until one native job retires."""
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread
    from infernux.application import Application
    from infernux import lib

    assert_runtime_owner_thread()
    engine = Application._current_engine()
    if engine is None or not engine._cpu_range_accepting:
        raise RuntimeError("CPU range submission requires an active accepting Engine owner")
    native, owners = None, []
    for compiled, params, begin, end in ranges:
        if not isinstance(compiled, CpuRangeProgram):
            raise TypeError("CPU range submission requires a proven compiled work item")
        storage, borrowed, slots, count, begin, end = _range_payload(compiled, params, begin, end)
        entry = compiled._native_entry()  # Preparation never enters the worker.
        owners.append((compiled, entry, storage, borrowed, slots))
        if native is None:
            native = lib._Infernux._CpuRangeSlice(entry.address, slots, count, begin, end)
        else:
            native.append(entry.address, slots, count, begin, end)
    if native is None:
        raise ValueError('a CPU segment requires at least one range')
    return _CpuRangeTask(engine, native, tuple(owners))


def _submit_cpu_range(compiled, *params, begin, end):
    """Convenience admission through the same prepared ticket and native queue."""
    return _prepare_cpu_range(compiled, *params, begin=begin, end=end).submit()


def _poll_cpu_ranges(engine):
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread

    assert_runtime_owner_thread()
    for task in tuple(engine._cpu_range_tasks):
        task._poll()
    return bool(engine._cpu_range_tasks)


def _shutdown_cpu_ranges(engine):
    from infernux.engine.runtime_dispatch import assert_runtime_owner_thread

    assert_runtime_owner_thread()
    engine._cpu_range_accepting = False
    tasks = tuple(engine._cpu_range_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        if task._native is not None:
            # Binding releases GIL; inline JobSystems cannot admit these tasks.
            task._native.wait_for_completion()
        task._poll()
