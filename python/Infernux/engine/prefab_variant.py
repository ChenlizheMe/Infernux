"""Persistent Variant projections, using the ordinary Prefab identity merge.

This module owns authored data only. Publication, dependency ordering and Undo
belong to the existing asset transaction, not to a second Variant runtime.
"""

import copy

from Infernux.engine.prefab_manager import (
    PrefabDocumentError, _prefab_nodes, _prefab_next_local_id,
    _prefab_next_component_id, _validate_prefab_document,
)


def _parents(root):
    result = {root["local_id"]: None}
    for node in _prefab_nodes(root):
        result.update((child["local_id"], node["local_id"]) for child in node["children"])
    return result


def _validate_inherited_hierarchy(root, baseline, object_sources):
    """As in Unity, inherited objects cannot be deleted or reparented locally."""
    current = _parents(root)
    projection = {source: local for local, source in object_sources}
    for source, parent in _parents(baseline).items():
        local = projection[source]
        if local not in current:
            raise PrefabDocumentError("Deactivate inherited Variant objects instead of deleting them")
        if current[local] != (projection[parent] if parent is not None else None):
            raise PrefabDocumentError("Inherited Variant objects cannot be reparented")


def validate_variant_definition(definition):
    if not isinstance(definition, dict) or set(definition) != {"document", "base"}:
        raise PrefabDocumentError("Variant requires an authored document and a base projection")
    document, base = definition["document"], definition["base"]
    _validate_prefab_document(document)
    if not isinstance(base, dict) or set(base) != {
        "guid", "baseline", "object_sources", "component_sources", "property_overrides",
    }:
        raise PrefabDocumentError("Variant base projection has invalid fields")
    if not isinstance(base["guid"], str) or not base["guid"]:
        raise PrefabDocumentError("Variant requires a base asset GUID")
    _validate_prefab_document(base["baseline"])
    for key, source_ids, next_id in (
        ("object_sources", {n["local_id"] for n in _prefab_nodes(base["baseline"]["root_object"])},
         _prefab_next_local_id(document)),
        ("component_sources", {c["component_id"] for n in _prefab_nodes(base["baseline"]["root_object"])
                               for c in n["components"]}, _prefab_next_component_id(document)),
    ):
        pairs = base[key]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or any(type(value) is not int or value <= 0 for value in pair) for pair in pairs
        ):
            raise PrefabDocumentError(f"Variant {key} requires positive identity pairs")
        if len({pair[0] for pair in pairs}) != len(pairs) or len({pair[1] for pair in pairs}) != len(pairs):
            raise PrefabDocumentError(f"Variant {key} contains duplicate identities")
        if not source_ids.issubset(pair[1] for pair in pairs):
            raise PrefabDocumentError(f"Variant {key} does not cover its base")
        if any(pair[0] >= next_id for pair in pairs):
            raise PrefabDocumentError(f"Variant {key} exceeds the persistent identity watermark")
    _validate_inherited_hierarchy(document["root_object"], base["baseline"]["root_object"], base["object_sources"])
    if not isinstance(base["property_overrides"], list):
        raise PrefabDocumentError("Variant property overrides require an array")
    for item in base["property_overrides"]:
        if (not isinstance(item, dict) or set(item) != {"object", "component", "path", "value"}
                or type(item["object"]) is not int or item["object"] <= 0
                or type(item["component"]) is not int or item["component"] < 0
                or not isinstance(item["path"], list) or not item["path"]
                or any(not isinstance(key, str) or not key for key in item["path"])):
            raise PrefabDocumentError("Invalid Variant property override")


def _property_overrides(base, own):
    """Persist override intent even when a later base happens to match its value."""
    from Infernux.components.value_document import TYPE_KEY

    result = []
    original = {node["local_id"]: node for node in _prefab_nodes(base)}

    def fields(before, after, object_id, component_id=0, path=()):
        for key, value in after.items():
            if not path and key in {"local_id", "children", "components", "nested_prefab", "component_id", "type_id"}:
                continue
            previous = before.get(key)
            if previous == value:
                continue
            if isinstance(value, dict) and isinstance(previous, dict) and TYPE_KEY not in value:
                fields(previous, value, object_id, component_id, (*path, key))
            else:
                result.append(dict(object=object_id, component=component_id, path=[*path, key], value=copy.deepcopy(value)))

    for node in _prefab_nodes(own):
        before = original.get(node["local_id"])
        if before is None:
            continue
        fields(before, node, node["local_id"])
        components = {component["component_id"]: component for component in before["components"]}
        for component in node["components"]:
            previous = components.get(component["component_id"])
            if previous is not None:
                fields(previous, component, node["local_id"], component["component_id"])
    return result


def create_variant_definition(base_guid, base_document, authored_document=None):
    """Capture edits expressed in the base identity domain, never by node name."""
    _validate_prefab_document(base_document)
    document = copy.deepcopy(base_document if authored_document is None else authored_document)
    document["next_local_id"] = max(_prefab_next_local_id(base_document), _prefab_next_local_id(document))
    document["next_component_id"] = max(_prefab_next_component_id(base_document), _prefab_next_component_id(document))
    result = {"document": document, "base": {
        "guid": base_guid,
        "baseline": copy.deepcopy(base_document),
        "object_sources": [[node["local_id"], node["local_id"]]
                           for node in _prefab_nodes(base_document["root_object"])],
        "component_sources": [[component["component_id"], component["component_id"]]
                              for node in _prefab_nodes(base_document["root_object"])
                              for component in node["components"]],
        "property_overrides": _property_overrides(base_document["root_object"], document["root_object"]),
    }}
    validate_variant_definition(result)
    return result


def _project_root(root, object_sources, component_sources):
    from Infernux.engine.component_restore import _remap_local_reference_document

    objects = {source: local for local, source in object_sources}
    components = {source: local for local, source in component_sources}
    result = copy.deepcopy(root)
    for node in _prefab_nodes(result):
        node["local_id"] = objects[node["local_id"]]
        for component in node["components"]:
            component["component_id"] = components[component["component_id"]]
            for key, value in tuple(component.items()):
                component[key] = _remap_local_reference_document(value, objects, key, components)
        if "nested_prefab" in node:
            link = node["nested_prefab"]
            # Only the enclosing namespace changes. The inner source and its
            # baseline still belong to the nested asset's independent domain.
            link["object_sources"] = [[objects[local], source] for local, source in link["object_sources"]]
            link["component_sources"] = [[components[local], source] for local, source in link["component_sources"]]
    return result


def rebase_variant_definition(definition, base_document):
    """Return a new authored revision; callers publish it with its projections.

    Retired mappings remain reserved. A new base node cannot steal a Variant's
    private ID or reuse a deleted node's ID. Reading/merging never writes files.
    """
    from Infernux.engine.prefab_overrides import _merge_prefab_hierarchy

    validate_variant_definition(definition)
    _validate_prefab_document(base_document)
    result = copy.deepcopy(definition)
    document, base = result["document"], result["base"]
    for key, incoming, watermark in (
        ("object_sources", [node["local_id"] for node in _prefab_nodes(base_document["root_object"])], "next_local_id"),
        ("component_sources", [component["component_id"] for node in _prefab_nodes(base_document["root_object"])
                               for component in node["components"]], "next_component_id"),
    ):
        known = {source for _, source in base[key]}
        for source in incoming:
            if source not in known:
                base[key].append([document[watermark], source])
                document[watermark] += 1
                known.add(source)
    old_root = _project_root(base["baseline"]["root_object"], base["object_sources"], base["component_sources"])
    new_root = _project_root(base_document["root_object"], base["object_sources"], base["component_sources"])
    if old_root["local_id"] != new_root["local_id"]:
        raise PrefabDocumentError("Variant base root identity cannot be replaced")
    document["root_object"] = _merge_prefab_hierarchy(old_root, document["root_object"], new_root)
    nodes = {node["local_id"]: node for node in _prefab_nodes(document["root_object"])}
    for item in base["property_overrides"]:
        node = nodes.get(item["object"])
        if node is None:
            continue  # Source deletion can retire the target of an override.
        target = (node if item["component"] == 0 else
                  next((c for c in node["components"] if c["component_id"] == item["component"]), None))
        if target is None:
            continue
        for key in item["path"][:-1]:
            target = target.setdefault(key, {})
        target[item["path"][-1]] = copy.deepcopy(item["value"])
    base["baseline"] = copy.deepcopy(base_document)
    validate_variant_definition(result)
    return result


def rebase_variant_graph(definitions, load_base):
    """Stage a dependency-ordered set without partially publishing any asset.

    The caller supplies the frozen asset catalog and publishes the returned
    definitions as one authoring transaction. Shared ancestors are read once.
    """
    resolved, active, ordinary = {}, [], {}

    def resolve(guid):
        if guid in active:
            raise PrefabDocumentError("Variant source cycle: " + " -> ".join((*active, guid)))
        if guid in resolved:
            return resolved[guid]["document"]
        if guid not in definitions:
            if guid not in ordinary:
                ordinary[guid] = load_base(guid)
                _validate_prefab_document(ordinary[guid])
            return ordinary[guid]
        definition = definitions[guid]
        validate_variant_definition(definition)
        active.append(guid)
        try:
            base = resolve(definition["base"]["guid"])
            resolved[guid] = rebase_variant_definition(definition, base)
        finally:
            active.pop()
        return resolved[guid]["document"]

    for guid in definitions:
        resolve(guid)
    return resolved
