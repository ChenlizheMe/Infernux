"""Whole forward lowering to the common native compute-work recipe."""
import math
import struct
from types import SimpleNamespace

from infernux import compute
from infernux.lib import _Infernux as native
from . import _primitives as p
from ._tensor import Tensor, _empty


class GpuForward:
    def __init__(self, plan, inputs, *, run=None):
        self.plan = plan
        self.layout = self.input_layout(inputs)
        self.work = None
        self.parameters = tuple(sorted(plan.definition.used_parameters))
        prototypes = []
        try:
            # Keep SSA ports distinct even when the caller supplied one value
            # twice. Preserve the real allocation layout, including vector
            # buffers exposed to the model as scalar tensor views.
            def prototype(value, *, readonly=True):
                result = value.view(readonly=readonly)
                prototypes.append(result)
                return result
            values = tuple(prototype(value._data) for value in inputs)
            state = tuple(prototype(value) for value in (plan.root._state if run is None else run.state))
            if run is None:
                outputs = tuple(_empty(layer.output_shape, 'gpu', layer.dtype) for layer in plan.layers)
                prototypes.extend(outputs)
            else:
                outputs = tuple(prototype(value, readonly=False) for value in run.values[plan.input_count:])
            run = SimpleNamespace(values=(*values, *outputs), state=state,
                random_keys={index:(0, 0, 0, 0) for index, _ in plan.random_layers})
            operations = tuple(plan._forward_operation(run, index) for index in range(len(plan.operations)))
            used = {id(value) for _, params in operations for value in params if isinstance(value, compute.Buffer)}
            self.work = compute._prepare_work(operations,
                private_buffers=tuple(value for value in (*run.values, *state) if id(value) in used),
                inputs={**{f'x{index}':value for index, value in enumerate(values) if id(value) in used},
                        **{f'p{index}':state[index] for index in self.parameters}},
                outputs={f'y{index}':value for index, value in enumerate(run.values)
                         if index >= plan.input_count or index in plan.result_specs},
                argument_ports={index:f'rng{index}' for index, _ in plan.random_layers},
                logical_slots={value._generation:('value', index) for index, value in enumerate((*run.values, *state))})
            self.input_indices = tuple(index for index, value in enumerate(values) if id(value) in used)
            self.outputs = {int(name[1:]):(slot, offset, length)
                for name, slot, offset, length, is_input in self.work._ports if not is_input}
            self.uniforms = tuple((index, slot, template,
                tuple((int(row['offset']), '<i' if row['kind']=='int32' else '<f')
                      for row in plan._gpu_forward_executables[operations[index][0], False].artifact.argument_layout['parameters'][1:5]))
                for index, (slot, template) in self.work._argument_templates.items())
            self.storage = native._ComputeStorageGroup(tuple(slot[3] for slot in self.work._native.slots if slot[3] is not None))
        except BaseException:
            self.close()
            raise
        finally:
            p.retire(prototypes, flush=False)

    @staticmethod
    def input_layout(inputs):
        return tuple((value._data._native.element_count, value._data._native.lanes, value._data._byte_offset)
                     for value in inputs)

    def close(self):
        if self.work is not None:
            self.work.close()
            self.work = None

    def _bindings(self, values, state):
        return {**{f'x{index}':values[index]._native for index in self.input_indices},
                **{f'p{index}':state[index]._native for index in self.parameters}}

    def _execute(self, bindings, keys=None):
        compute._collect_launch_retirements()
        instance, updates = None, []
        try:
            instance = self.work._native.instantiate(bindings)
            if instance is None:
                raise MemoryError('native neural forward storage cannot be admitted')
            if keys is None:
                from ._random import _default
                keys = dict(zip((index for index, _ in self.plan.random_layers),
                                _default.reserve_many(tuple(count for _, count in self.plan.random_layers))))
            for index, slot, template, scalars in self.uniforms:
                arguments = bytearray(template)
                for (offset, kind), value in zip(scalars, keys[index], strict=True):
                    struct.pack_into(kind, arguments, offset, value)
                updates.append((instance.buffer(slot), bytes(arguments), 0))
            compute._execute_work_instance(instance, self.work._host, updates)
            return instance
        finally:
            bindings.clear()
            updates.clear()
            instance = None

    def forward_run(self, run):
        bindings = self._bindings(run.values, run.state)
        bindings.update((f'y{index}', run.values[index]._native) for index in self.outputs)
        instance = None
        try:
            instance = self._execute(bindings, run.random_keys)
            while run.cursor.current() is not None:
                run.cursor.complete()
        finally:
            bindings.clear()
            instance = None

    def forward(self, inputs):
        from ._plan import _map_result

        instance = None
        try:
            instance = self._execute(self._bindings(tuple(value._data for value in inputs), self.plan.root._state))
            def tensor(index):
                slot, offset, _ = self.outputs[index]
                result = self.plan.result_specs[index]
                storage = None
                try:
                    storage = compute.Buffer._from_native(instance.buffer(slot).freeze(),
                        shape=(math.prod(result.shape),), byte_offset=offset)
                    return Tensor._from_buffer(storage, result.shape, value_plan=self.plan, value_index=index)
                finally:
                    p.retire((storage,), flush=False)
            return _map_result(self.plan.result, tensor)
        finally:
            instance = None
