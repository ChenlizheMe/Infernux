"""Declared scene fields sampled into the existing numerical request pipeline."""
from dataclasses import dataclass
import math
from time import perf_counter_ns

from infernux import compute
from infernux.compute_source import BufferSource
from ._tensor import Tensor, _empty


class Channel:
    """A batch source declaration; preparation never reads its numeric values."""
    @staticmethod
    def materials(materials):
        return _MaterialChannel(materials)

    @staticmethod
    def tensors(fields):
        return _TensorChannel(fields)

    @staticmethod
    def hinges(batch):
        return _HingeChannel(batch)

    @staticmethod
    def transforms(batch):
        return _TransformChannel(batch)


class _BatchChannel:
    def __init__(self, batch):
        self.batch = batch

    @property
    def rows(self):
        return len(self.batch)

    @property
    def signature(self):
        return self.kind, self.rows


class _HingeChannel(_BatchChannel):
    kind = "hinge-motion"

    def __init__(self, batch):
        from infernux.physics import HingeJointBatch
        if not isinstance(batch, HingeJointBatch):
            raise TypeError("hinge channels require a live HingeJointBatch")
        super().__init__(batch)

    def field_width(self, name):
        if name != "motion":
            raise ValueError("hinge channels expose 'motion' in radians and radians/second")
        return 2

    def snapshot(self, fields):
        return self.batch._native.read_motion(fields[0][3])


class _TransformChannel(_BatchChannel):
    kind = "transform"

    def __init__(self, batch):
        from infernux.lib import TransformBatchHandle
        if not isinstance(batch, TransformBatchHandle) or batch.is_compact:
            raise TypeError("transform channels require a strict TransformBatchHandle")
        super().__init__(batch)

    def field_width(self, name):
        return self.batch._field_width(name)

    def snapshot(self, fields):
        self.batch._snapshot_fields(fields)


class _MaterialChannel:
    def __init__(self, materials):
        from infernux.core.material import Material
        self._materials = tuple(materials)
        if not self._materials or any(not isinstance(value, Material) for value in self._materials):
            raise TypeError("material channels require a nonempty Material batch")

    @property
    def rows(self):
        return len(self._materials)


def _materials(value):
    from infernux.core.material import Material
    return value._materials if isinstance(value, _MaterialChannel) else (value,) if isinstance(value, Material) else None


class _TensorChannel:
    def __init__(self, fields):
        self.fields = dict(fields)
        if not self.fields or any(not isinstance(name, str) or not isinstance(value, Tensor)
                or len(value.shape) != 2 or value.requires_grad or value._run is not None or value._loss is not None
                for name, value in self.fields.items()):
            raise TypeError("tensor channels require named [rows,components] leaf/detached Tensor fields")
        self.rows = next(iter(self.fields.values())).shape[0]
        if any(value.shape[0] != self.rows for value in self.fields.values()):
            raise ValueError("channel fields must describe the same row batch")

    @property
    def signature(self):
        return tuple((name, value.shape, value.device, value.dtype.name) for name, value in sorted(self.fields.items()))


@dataclass(frozen=True)
class _Field:
    argument: str
    kind: str
    name: str
    default: float
    rows: int

    @property
    def shape(self):
        return self.rows, {"float": 1, "int": 1, "vector2": 2, "vector3": 3, "vector4": 4, "color": 4}[self.kind]


class _SourceProxy:
    def __init__(self, capture, argument, value):
        self.capture, self.argument, self.source = capture, argument, value
        self.rows = value.rows if isinstance(value, (_TensorChannel, _BatchChannel)) else len(_materials(value))

    def _field(self, kind, name, default=0.):
        from .program import _Value
        if not isinstance(name, str) or not name:
            raise TypeError("compiled source fields require names")
        field = _Field(self.argument, kind, name, float(default), self.rows)
        if isinstance(self.source, _BatchChannel):
            if kind != "vector" + str(self.source.field_width(name)):
                raise ValueError("scene field getter must match its native vector width")
        if isinstance(self.source, _TensorChannel):
            value = self.source.fields[name]
            if value.shape != field.shape:
                raise ValueError("channel field shape differs from its getter's components")
            expected = ("int32", "uint32", "int64") if kind == "int" else ("float32",)
            if value.dtype.name not in expected:
                raise TypeError("channel field dtype differs from its getter")
            sample = value.detach()
        else:
            if kind == "int":
                raise TypeError("integer getter fields require a Tensor channel")
            storage = _empty(field.shape, "cpu")
            sample = Tensor._from_buffer(storage, field.shape)
            storage.close()
        self.capture.samples.append(sample)
        return _Value(self.capture, sample, field)

    def get_float(self, name, default=0.):
        return self._field("float", name, default)

    def get_int(self, name, default=0):
        return self._field("int", name, default)

    def get_color(self, name):
        return self._field("color", name)

    def get_vector2(self, name):
        return self._field("vector2", name)

    def get_vector3(self, name):
        return self._field("vector3", name)

    def get_vector4(self, name):
        return self._field("vector4", name)


class _SourceInputs:
    """Private source allocations, then the same immutable Tensor/IO bindings."""
    def __init__(self, bindings, arguments, *, owner=None):
        self.buffers, self.values, self.groups = [], {}, {}
        self.done = False
        bindings = tuple(dict.fromkeys(binding for binding in bindings if binding is not None))
        sampled_fields = [binding for binding in bindings if isinstance(binding, _Field)
                           and not isinstance(arguments[binding.argument], _TensorChannel)]
        self.units = sum(binding.rows for binding in sampled_fields)
        self.nbytes = sum(math.prod(binding.shape) * 4 for binding in sampled_fields)
        self.service_key = tuple((arguments[field.argument].kind if isinstance(arguments[field.argument], _BatchChannel) else "material",
                                  field.kind, field.name, field.rows) for field in sampled_fields)
        for source in {id(arguments[field.argument]): arguments[field.argument] for field in sampled_fields}.values():
            if isinstance(source, _BatchChannel):
                source.batch.validate()
        if owner is not None:
            self.validate(owner)
        for binding in bindings:
            if binding is None or binding in self.values:
                continue
            if isinstance(binding, str):
                self.values[binding] = arguments[binding]
                continue
            source = arguments[binding.argument]
            if isinstance(source, _TensorChannel):
                value = source.fields[binding.name]
                if value.shape != binding.shape:
                    raise ValueError("channel field shape changed; prepare its current layout")
                self.values[binding] = value
                continue
            native_source = source if isinstance(source, _BatchChannel) else _materials(source)
            if native_source is None or (source.rows if isinstance(source, _BatchChannel) else len(native_source)) != binding.rows:
                raise ValueError("scene source layout changed; prepare the new batch shape")
            storage = _empty(binding.shape, "cpu")
            self.buffers.append(storage)
            self.values[binding] = Tensor._from_buffer(storage, binding.shape)
            self.groups.setdefault(binding.argument, (native_source, []))[1].append(
                (binding.kind, binding.name, binding.default, storage._array.reshape(binding.shape)))
        self.done = not self.groups

    def validate(self, owner):
        if (self.units > owner.budget.snapshot_rows or self.nbytes > owner.budget.snapshot_bytes
                or self.units * owner.budget.snapshot_ns_per_row >= owner._max_background_credit()):
            raise ValueError("scene source batch exceeds the atomic snapshot budget; use a smaller batch")

    def _snapshot(self):
        from infernux import lib
        step = None
        for source, fields in self.groups.values():
            if isinstance(source, _BatchChannel):
                sampled_step = source.snapshot(fields)
                if sampled_step is not None:
                    step = sampled_step
            else:
                lib._Infernux._snapshot_material_fields([material._native for material in source], fields)
        self.done = True
        self.groups.clear()
        return step

    def advance(self, owner, metrics):
        if self.units * owner.budget.snapshot_ns_per_row > owner._remaining_credit(background=True):
            return False
        started = perf_counter_ns()
        step = self._snapshot()
        metrics.update(source_snapshot_ns=perf_counter_ns() - started, source_snapshot_bytes=self.nbytes,
                       source_sample_opportunity=owner._last_opportunity)
        if step is not None:
            metrics["source_physics_step"] = step
        return True

    def offline(self):
        from ._tensor import Tensor
        for name, value in tuple(self.values.items()):
            if isinstance(value, BufferSource):
                storage = value.buffer()
                try:
                    self.values[name] = Tensor._from_buffer(storage, value.shape, shared_input=value.device == 'cpu')
                    self.buffers.append(storage)
                except BaseException:
                    storage._close(flush=False)
                    raise
        self._snapshot()

    def close(self):
        for buffer in self.buffers:
            buffer._close(flush=False)
        self.buffers.clear()
        self.values.clear()
        self.groups.clear()
