"""
Batch read/write API for high-performance data transfer between
engine objects and numpy arrays.

Usage::

    import infernux as inx

    # Read world positions from a list of transforms → numpy (N, 3)
    positions = inx.batch_read(transforms, 'position')

    # Run a JIT kernel
    @inx.jit.compile
    def gravity(pos, dt):
        for i in range(len(pos)):
            pos[i, 1] -= 9.8 * dt
    gravity(positions, delta_time)

    # Write back
    inx.batch_write(transforms, positions, 'position')

Supported Transform properties:
    'position', 'local_position', 'local_scale',
    'euler_angles', 'local_euler_angles',
    'rotation', 'local_rotation'

For InxComponent subclass fields (int, float, Vector3, etc.):
    inx.batch_read(components, 'velocity')
    inx.batch_read(components, MyComponent.velocity)   # descriptor form
"""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from infernux.lib import TransformBatchHandle

# NumPy is a required engine runtime dependency. Import it on first batch API
# use so importing this module itself stays lightweight.
_np = None


def _get_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np

# Lazy import to avoid circular deps at module level.
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        from infernux import lib as _l
        _lib = _l
    return _lib


# ── Property name resolution ────────────────────────────────────────────

def _resolve_property_name(prop: Any) -> str:
    """Accept either a string or a descriptor and return the property name."""
    if isinstance(prop, str):
        return prop
    # SerializedFieldDescriptor (InxComponent fields)
    if hasattr(prop, 'metadata') and hasattr(prop.metadata, 'name'):
        return prop.metadata.name
    # pybind11 property descriptors don't have a nice .name — fall back
    raise TypeError(
        f"Cannot resolve property from {type(prop).__name__!r}. "
        f"Pass a string like 'position' or a SerializedFieldDescriptor."
    )


# ── Shape/dtype helpers ─────────────────────────────────────────────────

# Transform properties backed by C++ fast path.
_TRANSFORM_VEC3_PROPS = frozenset({
    'position', 'local_position', 'local_scale',
    'euler_angles', 'local_euler_angles',
})
_TRANSFORM_QUAT_PROPS = frozenset({
    'rotation', 'local_rotation',
})
_TRANSFORM_ALL_PROPS = _TRANSFORM_VEC3_PROPS | _TRANSFORM_QUAT_PROPS


def _is_transform_list(targets: Sequence) -> bool:
    """Check if the first element is a Transform (C++ pybind11 type)."""
    if not targets:
        return False
    lib = _get_lib()
    return isinstance(targets[0], lib.Transform)


def _validate_targets(targets: Sequence) -> None:
    """Reject ambiguous batches before touching any object's storage."""
    if not targets:
        raise ValueError("targets must be non-empty; use a TransformBatchHandle for an empty batch")
    cls = type(targets[0])
    if isinstance(targets[0], _get_lib().Transform):
        for target in targets:
            if type(target) is not cls:
                raise TypeError("batch targets must have the same concrete type")
        return
    from infernux.components.component import InxComponent

    if not isinstance(targets[0], InxComponent):
        raise TypeError("batch targets must be Transform or InxComponent instances")
    for target in targets:
        if type(target) is not cls:
            raise TypeError("batch targets must have the same concrete type")
        if target._is_destroyed:
            raise RuntimeError("batch contains a destroyed component")


def _component_metadata(targets: Sequence, prop_name: str):
    from infernux.components.fields import get_serialized_fields

    cls = type(targets[0])
    meta = get_serialized_fields(cls).get(prop_name)
    if meta is None:
        raise AttributeError(f"{cls.__name__} has no serialized field '{prop_name}'")
    return meta


def _cds_slots(targets: Sequence, class_id: int):
    """Pack validated CDS slot handles as an (N, 2) uint32 array."""
    packed: list[int] = []
    extend = packed.extend
    for target in targets:
        key = target._cds_key
        if key is None or key[0] != class_id:
            raise RuntimeError("batch contains an unavailable or mismatched CDS layout")
        extend(key[1])
    return _get_np().asarray(packed, dtype='uint32').reshape(-1, 2)


# ── ComponentDataStore fast path ────────────────────────────────────────

def _try_cds_gather(targets: Sequence, prop_name: str):
    """Attempt CDS C++ fast-path gather.  Returns ndarray or None."""
    from infernux.components._cds_bridge import get_class_info
    cls = type(targets[0])
    info = get_class_info(cls)
    if info is None:
        return None
    class_id, field_map = info
    entry = field_map.get(prop_name)
    if entry is None:
        return None
    field_id, type_code = entry

    # Collect generational slot handles as an (N, 2) uint32 array.
    np = _get_np()
    slots = _cds_slots(targets, class_id)
    lib = _get_lib()
    result = np.asarray(lib._cds_batch_gather(class_id, field_id, type_code, slots))
    # Native bool storage is byte-addressable; the public field type is bool.
    _, dtype = _field_shape_dtype(_component_metadata(targets, prop_name).field_type)
    return result.astype(dtype, copy=False)


def _try_cds_scatter(targets: Sequence, data: np.ndarray, prop_name: str) -> bool:
    """Attempt CDS C++ fast-path scatter.  Returns True if handled."""
    from infernux.components._cds_bridge import get_class_info
    cls = type(targets[0])
    info = get_class_info(cls)
    if info is None:
        return False
    class_id, field_map = info
    entry = field_map.get(prop_name)
    if entry is None:
        return False
    field_id, type_code = entry

    slots = _cds_slots(targets, class_id)
    lib = _get_lib()
    lib._cds_batch_scatter(class_id, field_id, type_code, slots, data)
    # Native scatter validates every generation before writing any value.
    # Publish the whole batch through one journal pass; per-component calls
    # dominated the write cost for large batches.
    from .components._component_lifecycle import notify_runtime_component_values

    notify_runtime_component_values(tuple(targets), prop_name)
    return True


class ComponentFieldBatch:
    """Validated component-field batch for repeated reads and writes.

    Creating a handle resolves the CDS layout, validates every target and packs
    the generational slot handles once. Repeated ``read``/``write`` calls then
    skip target validation and slot packing; the native store still validates
    generations, so a destroyed target keeps raising instead of writing to a
    reused slot.

    The handle holds strong references to its targets. Keep it for the lifetime
    of the batch (for example one training loop) rather than across a scene
    unload.
    """

    __slots__ = (
        "targets",
        "prop_name",
        "class_id",
        "field_id",
        "type_code",
        "slots",
        "dtype",
    )

    def __init__(self, targets, prop_name, class_id, field_id, type_code, slots, dtype):
        self.targets = targets
        self.prop_name = prop_name
        self.class_id = class_id
        self.field_id = field_id
        self.type_code = type_code
        self.slots = slots
        self.dtype = dtype

    def read(self):
        """Read the whole batch from the native store."""
        np = _get_np()
        result = np.asarray(
            _get_lib()._cds_batch_gather(
                self.class_id, self.field_id, self.type_code, self.slots
            )
        )
        return result.astype(self.dtype, copy=False)

    def write(self, data) -> None:
        """Scatter the whole batch; data is validated against the field contract."""
        prepared = _prepare_component_data(self.targets, data, self.prop_name)
        _get_lib()._cds_batch_scatter(
            self.class_id, self.field_id, self.type_code, self.slots, prepared
        )
        from .components._component_lifecycle import notify_runtime_component_values

        notify_runtime_component_values(self.targets, self.prop_name)


def create_component_batch_handle(targets: Sequence, prop: Any) -> ComponentFieldBatch:
    """Build a reusable handle for one CDS-backed component field."""
    from infernux.components._cds_bridge import get_class_info

    prop_name = _resolve_property_name(prop)
    _validate_targets(targets)
    cls = type(targets[0])
    info = get_class_info(cls)
    if info is None:
        raise TypeError(
            f"{cls.__name__} has no CDS-backed numeric fields; "
            "a component batch handle requires native field storage"
        )
    class_id, field_map = info
    entry = field_map.get(prop_name)
    if entry is None:
        raise TypeError(
            f"{cls.__name__}.{prop_name} is not a CDS-backed numeric field"
        )
    field_id, type_code = entry
    _elem_shape, dtype = _field_shape_dtype(
        _component_metadata(targets, prop_name).field_type
    )
    return ComponentFieldBatch(
        tuple(targets), prop_name, class_id, field_id, type_code,
        _cds_slots(targets, class_id), dtype,
    )


# ── FieldType → numpy dtype/shape mapping ───────────────────────────────

def _field_shape_dtype(field_type) -> tuple[tuple[int, ...], "np.dtype"]:
    """Return (per-element shape, dtype) for a FieldType enum."""
    from infernux.components.fields import FieldType
    np = _get_np()
    _map = {
        FieldType.INT:   ((), np.dtype('int64')),
        FieldType.FLOAT: ((), np.dtype('float64')),
        FieldType.BOOL:  ((), np.dtype('bool')),
        FieldType.VEC2:  ((2,), np.dtype('float32')),
        FieldType.VEC3:  ((3,), np.dtype('float32')),
        FieldType.VEC4:  ((4,), np.dtype('float32')),
    }
    result = _map.get(field_type)
    if result is None:
        raise TypeError(
            f"batch_read/batch_write does not support FieldType.{field_type.name}. "
            f"Only numeric types (INT, FLOAT, BOOL, VEC2, VEC3, VEC4) are supported."
        )
    return result


# ── Component (Python) slow path ────────────────────────────────────────

def _component_gather(targets: Sequence, prop_name: str) -> np.ndarray:
    """Gather a named attribute from a sequence of InxComponent instances."""
    from infernux.components.fields import FieldType

    # Look up metadata from the class
    meta = _component_metadata(targets, prop_name)

    elem_shape, dtype = _field_shape_dtype(meta.field_type)
    np = _get_np()
    n = len(targets)
    full_shape = (n, *elem_shape) if elem_shape else (n,)
    out = np.empty(full_shape, dtype=dtype)

    for i, obj in enumerate(targets):
        val = getattr(obj, prop_name)
        if elem_shape:
            # Vector type — extract components
            if meta.field_type == FieldType.VEC2:
                out[i, 0] = val.x
                out[i, 1] = val.y
            elif meta.field_type == FieldType.VEC3:
                out[i, 0] = val.x
                out[i, 1] = val.y
                out[i, 2] = val.z
            elif meta.field_type == FieldType.VEC4:
                out[i, 0] = val.x
                out[i, 1] = val.y
                out[i, 2] = val.z
                out[i, 3] = val.w
        else:
            out[i] = val

    return out


def _component_scatter(targets: Sequence, data: np.ndarray, prop_name: str) -> None:
    """Scatter a numpy array back into named attributes of InxComponent instances."""
    from infernux.components.fields import FieldType

    meta = _component_metadata(targets, prop_name)

    lib = _get_lib()

    for i, obj in enumerate(targets):
        if meta.field_type == FieldType.VEC2:
            setattr(obj, prop_name, lib.Vector2(float(data[i, 0]), float(data[i, 1])))
        elif meta.field_type == FieldType.VEC3:
            setattr(obj, prop_name, lib.Vector3(float(data[i, 0]), float(data[i, 1]), float(data[i, 2])))
        elif meta.field_type == FieldType.VEC4:
            setattr(obj, prop_name, lib.vec4f(float(data[i, 0]), float(data[i, 1]), float(data[i, 2]), float(data[i, 3])))
        else:
            setattr(obj, prop_name, data[i].item())


# ── Public API ──────────────────────────────────────────────────────────

def _prepare_component_data(targets: Sequence, data: NDArray, prop_name: str):
    """Validate and normalize the entire input before either storage path writes."""
    from infernux.components.fields import FieldType, normalize_runtime_field_value

    np = _get_np()
    meta = _component_metadata(targets, prop_name)
    elem_shape, dtype = _field_shape_dtype(meta.field_type)
    values = np.asarray(data)
    if (values.ndim != 1 + len(elem_shape)
            or values.shape[1:] != elem_shape or values.shape[0] < len(targets)):
        raise ValueError(
            f"batch data must have shape (N{''.join(', ' + str(d) for d in elem_shape)}), "
            f"with N >= {len(targets)}"
        )
    values = values[:len(targets)]
    if meta.range is not None and meta.field_type in (FieldType.INT, FieldType.FLOAT):
        # Use exactly the scalar range contract, including integer conversion,
        # without mutating the caller's array or exposing partially clamped data.
        return np.asarray([
            normalize_runtime_field_value(value.item(), meta) for value in values
        ], dtype=dtype)
    if meta.field_type == FieldType.BOOL:
        return np.ascontiguousarray(values, dtype=np.bool_)
    if not np.can_cast(values.dtype, dtype, casting='safe'):
        raise ValueError(f"batch data cannot be safely converted from {values.dtype} to {dtype}")
    return np.ascontiguousarray(values, dtype=dtype)


def batch_read(targets: Sequence, prop: Any = None) -> NDArray:
    """Read a property from all *targets* into a numpy array.

    Parameters
    ----------
    targets : list[Transform] | list[InxComponent] | TransformBatchHandle | ComponentFieldBatch
        Non-empty list of engine objects of the same concrete type, or a pre-built
        handle for repeated reads without re-validation.
    prop : str | descriptor
        Property name (``'position'``, ``'velocity'``) or a class-level
        descriptor (``MyComponent.velocity``).

    Returns
    -------
    numpy.ndarray
        Shape depends on the property type:
        - ``int`` / ``float`` / ``bool`` → ``(N,)``
        - ``Vector2`` → ``(N, 2)``
        - ``Vector3`` → ``(N, 3)``
        - ``quatf`` / ``vec4f`` → ``(N, 4)``
    """
    if isinstance(targets, ComponentFieldBatch):
        return targets.read()
    if prop is None:
        raise TypeError("batch_read requires a property for a target list")
    prop_name = _resolve_property_name(prop)

    # Handle-based fast path (cached Transform pointers).
    lib = _get_lib()
    if isinstance(targets, lib.TransformBatchHandle):
        if prop_name in _TRANSFORM_ALL_PROPS:
            return lib._transform_batch_read(targets, prop_name)
        raise ValueError(
            f"Unknown Transform property '{prop_name}'. "
            f"Supported: {sorted(_TRANSFORM_ALL_PROPS)}"
        )

    _validate_targets(targets)
    if _is_transform_list(targets):
        if prop_name in _TRANSFORM_ALL_PROPS:
            return lib._transform_batch_read(targets, prop_name)
        raise ValueError(
            f"Unknown Transform property '{prop_name}'. "
            f"Supported: {sorted(_TRANSFORM_ALL_PROPS)}"
        )

    # Try CDS C++ fast path for InxComponent numeric fields.
    result = _try_cds_gather(targets, prop_name)
    if result is not None:
        return result

    # Fallback: Python getattr loop.
    return _component_gather(targets, prop_name)


def batch_write(targets: Sequence, data: NDArray, prop: Any = None):
    """Write a numpy array back to a property on all *targets*.

    Parameters
    ----------
    targets : list[Transform] | list[InxComponent] | TransformBatchHandle | ComponentFieldBatch
        Same list used for the preceding ``batch_read``, or a pre-built handle.
    data : numpy.ndarray
        Array with ``data.shape[0] >= len(targets)`` and the field's element
        shape. Component numeric conversions must be safe; declared scalar
        ranges and boolean values follow normal field assignment semantics.
    prop : str | descriptor
        Same property specifier used for ``batch_read``.
    """
    if isinstance(targets, ComponentFieldBatch):
        targets.write(data)
        return
    if prop is None:
        raise TypeError("batch_write requires a property for a target list")
    prop_name = _resolve_property_name(prop)

    # Handle-based fast path.
    lib = _get_lib()
    if isinstance(targets, lib.TransformBatchHandle):
        if prop_name in _TRANSFORM_ALL_PROPS:
            return lib._transform_batch_write(targets, data, prop_name)
        raise ValueError(
            f"Unknown Transform property '{prop_name}'. "
            f"Supported: {sorted(_TRANSFORM_ALL_PROPS)}"
        )

    _validate_targets(targets)
    if _is_transform_list(targets):
        if prop_name in _TRANSFORM_ALL_PROPS:
            lib._transform_batch_write(targets, data, prop_name)
            return
        raise ValueError(
            f"Unknown Transform property '{prop_name}'. "
            f"Supported: {sorted(_TRANSFORM_ALL_PROPS)}"
        )

    # Try CDS C++ fast path for InxComponent numeric fields.
    data = _prepare_component_data(targets, data, prop_name)
    if _try_cds_scatter(targets, data, prop_name):
        return

    # Fallback: Python setattr loop.
    _component_scatter(targets, data, prop_name)


def create_batch_handle(targets: list, *, mode: str = "strict") -> "TransformBatchHandle":
    """Create a validated ``TransformBatchHandle`` for *targets*.

    Re-use the handle across ``batch_read`` / ``batch_write`` calls to avoid
    repeated pybind11 extraction overhead. Every operation validates the
    stored ECS generations and scene world IDs. ``mode="strict"`` rejects any
    stale transform. ``mode="compact"`` skips stale entries and makes reads
    return ``(values, valid_mask)`` while writes return ``valid_mask``.
    """
    lib = _get_lib()
    modes = {
        "strict": lib.TransformBatchMode.STRICT,
        "compact": lib.TransformBatchMode.COMPACT,
    }
    try:
        validation_mode = modes[mode]
    except KeyError as exc:
        raise ValueError("batch handle mode must be 'strict' or 'compact'") from exc
    return lib.TransformBatchHandle(targets, validation_mode)


def create_scene_batch_handle(scene, *, name_prefix: str = "", mode: str = "strict") -> "TransformBatchHandle":
    """Create a Transform batch from native Scene filtering.

    This avoids materializing every matching GameObject and Transform in Python,
    which is important for large authored scenes and Play Mode startup.
    """
    if scene is None:
        raise ValueError("scene is required")
    if not isinstance(name_prefix, str):
        raise TypeError("name_prefix must be a string")
    lib = _get_lib()
    modes = {
        "strict": lib.TransformBatchMode.STRICT,
        "compact": lib.TransformBatchMode.COMPACT,
    }
    try:
        validation_mode = modes[mode]
    except KeyError as exc:
        raise ValueError("batch handle mode must be 'strict' or 'compact'") from exc
    return lib._create_scene_transform_batch_handle(scene, name_prefix, validation_mode)
