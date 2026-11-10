"""Apply authored Prefab updates while retaining resident objects and components."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from infernux.components.builtin_component import BuiltinComponent
from infernux.components.fields import get_serialized_fields
from infernux.components.missing_script import MissingScript
from infernux.components.value_codec import VALUE_CODECS
from infernux.engine._prefab_instance_hierarchy import (
    PrefabInstanceHierarchy,
    changed_component_slots,
    component_records,
    index_prefab_nodes,
    prepare_prefab_instance_hierarchy,
)
from infernux.engine.prefab_manager import _component_source_link


_OBJECT_VALUES = ("name", "tag", "layer", "is_static", "active")
_PREFAB_LINK_VALUES = {"prefab_guid": "", "prefab_root": False, "prefab_source_id": 0}
_OBJECT_PAYLOAD = frozenset((*_OBJECT_VALUES, *_PREFAB_LINK_VALUES,
                             "transform", "components", "children", "prefab_source"))
_TRANSFORM_PAYLOAD = frozenset(("position", "rotation", "scale", "enabled", "execution_order", "rect_layout"))


def _same_identity(before, after, payload):
    return ({key: value for key, value in before.items() if key not in payload}
            == {key: value for key, value in after.items() if key not in payload})


def _compatible_component_fields(before, after):
    return all(
        old["type_id"].startswith("native:") or old["data"].keys() == after[component_id]["data"].keys()
        for component_id, old in before.items()
        if component_id in after and old["type_id"] == after[component_id]["type_id"]
    )


def _compatible_node(before, after):
    return (
        _same_identity(before.document, after.document, _OBJECT_PAYLOAD)
        and _same_identity(before.document["transform"], after.document["transform"], _TRANSFORM_PAYLOAD)
        and _compatible_component_fields(component_records(before.document), component_records(after.document))
    )


@dataclass
class PrefabInstanceValues:
    edits: list[tuple[Callable, object, object]] = field(default_factory=list)
    validated_components: list = field(default_factory=list)
    hierarchy: PrefabInstanceHierarchy | None = None
    _applied: int = 0

    def commit(self):
        try:
            if self.hierarchy is not None:
                self.hierarchy.commit()
            for write, _old, new in self.edits:
                self._applied += 1
                if write(new) is False:
                    raise RuntimeError("Prefab instance rejected authored values")
            if self.hierarchy is not None:
                self.hierarchy.validate()
        except Exception:
            self.discard()
            raise

    def discard(self):
        try:
            while self._applied:
                self._applied -= 1
                write, old, _new = self.edits[self._applied]
                if write(old) is False:
                    raise RuntimeError("Prefab instance failed to restore authored values")
        finally:
            if self.hierarchy is not None:
                self.hierarchy.discard()

    def finish(self):
        if self.hierarchy is not None:
            self.hierarchy.finish()
        for component in self.validated_components:
            component._call_on_validate()


def prepare_prefab_instance_values(obj, before, after, asset_database):
    old_nodes, new_nodes = index_prefab_nodes(before), index_prefab_nodes(after)
    retained = [node_id for node_id in new_nodes if node_id in old_nodes]
    if before["id"] != after["id"] or any(
        not _compatible_node(old_nodes[node_id], new_nodes[node_id]) for node_id in retained
    ):
        return None
    component_slots = changed_component_slots(old_nodes, new_nodes)
    changed_owners = {slots.owner_id for slots in component_slots}

    plan = PrefabInstanceValues()
    lifecycle_edits = []

    def native_values(native, old, new):
        if old != new:
            native.validate_document(new)
            plan.edits.append((native.deserialize_document, old, new))

    def attribute(target, key, old, new, *, edits=None):
        if old != new:
            (plan.edits if edits is None else edits).append((partial(setattr, target, key), old, new))

    def visit(owner, old, new):
        if old == new:
            return
        for key in _OBJECT_VALUES:
            attribute(owner, key, old[key], new[key], edits=lifecycle_edits if key == "active" else None)
        transform = owner.transform
        old_transform = old["transform"]
        new_transform = new["transform"]
        native_values(transform, old_transform, {
            **new_transform, "enabled": old_transform["enabled"], "execution_order": old_transform["execution_order"],
        })
        attribute(transform, "enabled", old_transform["enabled"], new_transform["enabled"], edits=lifecycle_edits)
        attribute(transform, "execution_order", old_transform["execution_order"], new_transform["execution_order"])

        components = {component.component_id: component for component in owner.get_components()}
        old_records, new_records = component_records(old), component_records(new)
        if owner.id not in changed_owners and list(old_records) != list(new_records):
            plan.edits.append((owner.set_component_order, list(old_records), list(new_records)))
        for component_id, current in new_records.items():
            previous = old_records.get(component_id)
            if previous is None or previous["type_id"] != current["type_id"] or previous == current:
                continue
            component = components[component_id]
            is_native = current["type_id"].startswith("native:")
            old_data, new_data = previous["data"], current["data"]
            if old_data != new_data:
                if is_native:
                    native = component._require_cpp_component() if isinstance(component, BuiltinComponent) else component
                    header = {key: value for key, value in native.serialize_document().items() if key not in old_data}
                    native_values(native, {**header, **old_data}, {**header, **new_data})
                else:
                    if not isinstance(component, MissingScript):
                        fields = get_serialized_fields(type(component))
                        new_data = {key: value for key, value in new_data.items()
                                    if key in fields and value != old_data[key]}
                        old_data = {key: old_data[key] for key in new_data}
                        for key, value in new_data.items():
                            VALUE_CODECS.validate(value, fields[key], f"{type(component).__name__}.{key}")
                        if new_data:
                            plan.validated_components.append(component)
                    if new_data:
                        write = partial(component._deserialize_fields_document, _skip_on_after_deserialize=True)
                        plan.edits.append((write, old_data, new_data))
            old_source, new_source = _component_source_link(previous), _component_source_link(current)
            if old_source != new_source:
                native = (component._require_cpp_component() if isinstance(component, BuiltinComponent)
                          else component if is_native else component._cpp_component)
                attribute(native, "_prefab_source", old_source, new_source)
            attribute(component, "enabled", previous["enabled"], current["enabled"], edits=lifecycle_edits)
            attribute(component, "execution_order", previous["execution_order"], current["execution_order"])

        for key, default in _PREFAB_LINK_VALUES.items():
            attribute(owner, key, old.get(key, default), new.get(key, default))
        attribute(owner, "_prefab_source_document", old.get("prefab_source"), new.get("prefab_source"))

    scene = obj.scene
    for node_id in retained:
        visit(scene.find_by_id(node_id), old_nodes[node_id].document, new_nodes[node_id].document)
    plan.edits.extend(lifecycle_edits)
    plan.hierarchy = prepare_prefab_instance_hierarchy(
        scene, after, old_nodes, new_nodes, component_slots, asset_database,
    )
    return plan
