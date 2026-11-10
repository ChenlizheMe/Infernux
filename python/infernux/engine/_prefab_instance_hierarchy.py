"""Publish changes to a resident Prefab hierarchy by scene identity."""

from dataclasses import dataclass, field

from infernux.engine.component_restore import (
    PreparedPythonComponentGraph,
    _native_object_graph_document,
    _require_clean_pending_queue,
    attach_prepared_python_component,
    preflight_game_object_python_components,
    publish_prepared_scene_python_components,
)


@dataclass(frozen=True)
class PrefabNode:
    document: dict
    parent_id: int | None


def index_prefab_nodes(root):
    nodes = {}

    def visit(document, parent_id):
        node_id = document["id"]
        nodes[node_id] = PrefabNode(document, parent_id)
        for child in document["children"]:
            visit(child, node_id)

    visit(root, None)
    return nodes


def _child_orders(nodes):
    return {node_id: [child["id"] for child in node.document["children"]]
            for node_id, node in nodes.items()}


def component_records(document):
    return {record["component_id"]: record for record in document["components"]}


@dataclass(frozen=True)
class ComponentSlots:
    owner_id: int
    added: list[dict]
    removed: list[int]
    order: list[int]


def changed_component_slots(old_nodes, new_nodes):
    changes = []
    for node_id, node in new_nodes.items():
        if node_id not in old_nodes:
            continue
        before, after = component_records(old_nodes[node_id].document), component_records(node.document)
        retained = {key for key in before.keys() & after.keys()
                    if before[key]["type_id"] == after[key]["type_id"]}
        added = [record for key, record in after.items() if key not in retained]
        removed = [key for key in before if key not in retained]
        if added or removed:
            changes.append(ComponentSlots(node_id, added, removed, list(after)))
    return changes


@dataclass
class PrefabInstanceHierarchy:
    scene: object
    additions: list[tuple[int, dict]]
    moves: list[tuple[int, int, int]]
    removed_roots: list[int]
    old_orders: list[list[int]]
    new_orders: list[list[int]]
    prepared: PreparedPythonComponentGraph
    component_slots: list[ComponentSlots]
    prepared_slots: PreparedPythonComponentGraph
    _created: list = field(default_factory=list)
    _moved: list = field(default_factory=list)
    _attached: list = field(default_factory=list)
    _component_edits: list = field(default_factory=list)
    _started: bool = False

    def _reparent(self, object_id, parent_id):
        obj = self.scene.find_by_id(object_id)
        parent = self.scene.find_by_id(parent_id)
        obj._set_prefab_instance_parent(parent)
        if obj.get_parent() is not parent:
            raise RuntimeError(f"Failed to reparent Prefab object {object_id}")

    def _order(self, orders):
        for children in orders:
            for index, child_id in enumerate(children):
                child = self.scene.find_by_id(child_id)
                if child.transform.get_sibling_index() != index:
                    child.transform.set_sibling_index(index)

    def commit(self):
        _require_clean_pending_queue(self.scene)
        self._started = True
        # Only new branches enter the native document loader. They already
        # have their final scene IDs, including references to other branches.
        for parent_id, document in self.additions:
            root = self.scene.create_game_object(document["name"])
            self._created.append(root)
            root.set_parent(self.scene.find_by_id(parent_id), False)
            committed, _ids = root._commit_document(_native_object_graph_document(document), True)
            if not committed:
                raise RuntimeError(f"Failed to publish added Prefab object {document['name']!r}")
        for object_id, old_parent, new_parent in self.moves:
            self._moved.append((object_id, old_parent))
            self._reparent(object_id, new_parent)
        self._order(self.new_orders)
        for slots in self.component_slots:
            owner = self.scene.find_by_id(slots.owner_id)
            native = [record for record in slots.added if record["type_id"].startswith("native:")]
            self._component_edits.append(owner._stage_prefab_component_edit(native, slots.removed))
        self._attached = publish_prepared_scene_python_components(
            self.scene, self.prepared, clear_registries=False, defer_lifecycle=True,
        )
        for item in self.prepared_slots.components:
            owner = self.scene.find_by_id(item.game_object_id)
            attach_prepared_python_component(owner, item, len(owner.get_component_order()), self._attached)
        self.prepared_slots.consume()
        for slots in self.component_slots:
            owner = self.scene.find_by_id(slots.owner_id)
            if not owner.set_component_order(slots.order):
                raise RuntimeError(f"Failed to publish Prefab component order on object {slots.owner_id}")

    def validate(self):
        for edit in self._component_edits:
            edit.validate()
        for _owner, instance, _native in self._attached:
            instance._call_on_after_deserialize()

    def discard(self):
        self.prepared.discard()
        if not self._started:
            self.prepared_slots.discard()
            return
        self._started = False
        self.scene.take_pending_py_components()
        for owner, _instance, native in reversed(self._attached):
            owner._remove_prepared_py_component(native)
        self.prepared_slots.discard()
        for edit in reversed(self._component_edits):
            edit.rollback()
        for object_id, old_parent in reversed(self._moved):
            self._reparent(object_id, old_parent)
        for obj in reversed(self._created):
            self.scene._remove_game_object_immediately(obj)
        self._order(self.old_orders)

    def finish(self):
        for object_id in self.removed_roots:
            removed = self.scene.find_by_id(object_id)
            if removed is not None:
                self.scene._remove_game_object_immediately(removed)
        for edit in self._component_edits:
            edit.finish()
        for owner, _instance, native in self._attached:
            owner._activate_prepared_py_component(native)


def prepare_prefab_instance_hierarchy(scene, after, old_nodes, new_nodes, component_slots, asset_database):
    added = new_nodes.keys() - old_nodes.keys()
    removed = old_nodes.keys() - new_nodes.keys()
    moves = [(node_id, old_nodes[node_id].parent_id, node.parent_id)
             for node_id, node in new_nodes.items()
             if node_id in old_nodes and old_nodes[node_id].parent_id != node.parent_id]
    old_orders, new_orders = _child_orders(old_nodes), _child_orders(new_nodes)
    changed_parents = {node_id for node_id in old_orders.keys() | new_orders.keys()
                       if old_orders.get(node_id, []) != new_orders.get(node_id, [])}
    if not (added or removed or moves or changed_parents or component_slots):
        return None

    component_order = {}

    def added_branch(document):
        for component in document["components"]:
            if component["type_id"].startswith("python:"):
                component_order[component["component_id"]] = len(component_order)
        return {**document, "children": [added_branch(child) for child in document["children"] if child["id"] in added]}

    additions = [(node.parent_id, added_branch(node.document)) for node_id, node in new_nodes.items()
                 if node_id in added and node.parent_id not in added]
    removed_roots = [node_id for node_id, node in old_nodes.items() if node_id in removed and node.parent_id not in removed]
    direct_ids = {record["component_id"] for slots in component_slots for record in slots.added
                  if record["type_id"].startswith("python:")}
    prepared = preflight_game_object_python_components(
        after, asset_database, preserve_document_ids=True, prefer_loaded_types=True,
        reference_scene=scene, only_component_ids=set(component_order) | direct_ids,
    )
    added_slots = PreparedPythonComponentGraph([item for item in prepared.components if item.component_id in direct_ids])
    added_branches = PreparedPythonComponentGraph(sorted(
        (item for item in prepared.components if item.component_id not in direct_ids),
        key=lambda component: component_order[component.component_id],
    ))
    prepared.consume()
    return PrefabInstanceHierarchy(
        scene, additions, moves, removed_roots,
        [order for node_id, order in old_orders.items() if node_id in changed_parents],
        [order for node_id, order in new_orders.items() if node_id in changed_parents],
        added_branches, component_slots, added_slots,
    )
