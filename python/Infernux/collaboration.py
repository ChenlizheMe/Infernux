"""Deterministic three-way merges for authored engine documents.

This file is also the Git merge driver. Run it directly: importing the engine
package is deliberately unnecessary for a merge or for driver installation.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # A direct script launch puts Infernux/ first on sys.path. Its math package
    # must never shadow the standard library used by this standalone driver.
    script_directory = os.path.dirname(__file__)
    sys.path[:] = [entry for entry in sys.path if entry != script_directory]

import argparse
import copy
import difflib
import heapq
import json
from pathlib import Path
import re
import runpy
from functools import cache
import shlex
import subprocess
from dataclasses import dataclass


_MISSING = object()
_PARENT = "__merge_parent__"
_GRAPH_SCHEMA = "infernux.graph_document"


def _collection_identity(record, field, path):
    """Use schema-owned collections, never names inside arbitrary script data."""
    if record.get("$schema") == _GRAPH_SCHEMA and field in {"nodes", "links"}:
        return "uid"
    if record.get("$schema") == "infernux.render_effect_group" and field == "entries":
        return "entry_id"
    if record.get("$schema") == "infernux.render_effect" and field == "dependencies":
        return "guid"
    if record.get("$schema") == "infernux.particle_graph" and field in {"emitters", "parameters", "event_types"}:
        return "stable_id"
    if "stable_id" in record and {"settings", "init", "update", "rendering"}.issubset(record):
        if field in {"attributes", "data_interfaces", "event_flows"}:
            return "stable_id"
    if {"default_state", "mode", "states", "parameters"}.issubset(record) and field in {"states", "parameters"}:
        return "stable_id"
    if "stable_id" in record and "clip_guid" in record and field == "transitions":
        return "stable_id"
    if "stable_id" in record and "target_state" in record and field == "conditions":
        return "stable_id"
    if {"duration", "apply_mode"}.issubset(record) and field == "keyframes":
        return "stable_id"
    if "authoring_texture_guid" in record and field == "frames":
        return "stable_id"
    if field == "components" and ({"local_id", "transform", "children"}.issubset(record)
                                  or re.fullmatch(r"/objects/[0-9]+", path)):
        return "component_id"
    if record.get("$schema") == "infernux.plugin_registry":
        return {"packages": "reference", "installed": "reference", "python_dependencies": "name"}.get(field)
    if "reference" in record and field == "files":
        return "guid"
    if {"name", "managed"}.issubset(record) and field == "owners":
        return "reference"
    if field in {"object_sources", "component_sources"} and "baseline" in record and "guid" in record:
        return field
    if field == "property_overrides" and "baseline" in record:
        return field
    if field == "value" and re.search(r"/metadata/sprite_frames$", path):
        return "stable_id"
    if field == "value" and re.search(r"/metadata/(model_textures|model_animations)$", path):
        return "key"
    return None


@dataclass(frozen=True)
class MergeConflict:
    path: str
    reason: str


@dataclass
class MergeResult:
    document: dict
    conflicts: list[MergeConflict]

    @property
    def clean(self):
        return not self.conflicts


class DocumentError(ValueError):
    pass


def _equal(left, right):
    if left is _MISSING or right is _MISSING:
        return left is right
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _clone(value):
    return _MISSING if value is _MISSING else copy.deepcopy(value)


def _path(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _conflict(conflicts, path, reason, ours):
    conflicts.append(MergeConflict(path or "/", reason))
    return _clone(ours)


def _key(value, identity):
    if identity == "property_overrides":
        return (value["object"], value["component"], tuple(value["path"]))
    if identity in {"object_sources", "component_sources"}:
        return value[0]
    key = value[identity]
    if identity == "reference" and isinstance(key, str):
        return key.casefold()
    if identity == "name" and isinstance(key, str):
        return re.sub(r"[-_.]+", "-", key).lower()
    return key


def _index(values, identity, path):
    result = {}
    for value in values:
        try:
            key = _key(value, identity)
            if type(key) not in (int, str, tuple) or key == "" or key in result:
                raise ValueError("missing or duplicate identity")
            result[key] = value
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DocumentError(f"{path}: invalid {identity} collection") from exc
    return result


def _order(base, ours, theirs, members, path, conflicts):
    """Combine moved-item and insertion constraints; unchanged order is passive."""
    sequences = [[key for key in values if key in members] for values in (base, ours, theirs)]
    b, o, t = sequences
    base_set = set(b)
    positions = [{key: index for index, key in enumerate(values)} for values in sequences]
    moved = set()
    for branch in (o, t):
        existing = [key for key in branch if key in base_set]
        existing_set = set(existing)
        original = [key for key in b if key in existing_set]
        matcher = difflib.SequenceMatcher(a=original, b=existing, autojunk=False)
        stable = {existing[index] for block in matcher.get_matching_blocks()
                  for index in range(block.b, block.b + block.size)}
        moved.update(set(existing) - stable)
    edges = {key: set() for key in members}
    incoming = {key: 0 for key in members}

    def edge(left, right):
        if left != right and right not in edges[left]:
            edges[left].add(right)
            incoming[right] += 1

    stable = [key for key in b if key not in moved]
    for left, right in zip(stable, stable[1:]):
        edge(left, right)
    handled = set()
    for key in moved:
        for other in b:
            pair = frozenset((key, other))
            if key == other or pair in handled:
                continue
            handled.add(pair)
            relations = [(p[key] < p[other]) if key in p and other in p else _MISSING for p in positions]
            before, own, other_side = relations
            relation = other_side if own is _MISSING else own
            if own is not _MISSING and other_side is not _MISSING:
                relation = other_side if own == before else own
            if relation is _MISSING:
                relation = before
            edge(key, other) if relation else edge(other, key)
    for branch in (o, t):
        for left, right in zip(branch, branch[1:]):
            if left not in base_set or right not in base_set:
                edge(left, right)
    priority = lambda key: (positions[0].get(key, len(b)), (0, key) if type(key) is int else (1, repr(key)))
    ready = [(priority(key), key) for key, degree in incoming.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        _, key = heapq.heappop(ready)
        ordered.append(key)
        for target in sorted(edges[key], key=priority):
            incoming[target] -= 1
            if incoming[target] == 0:
                heapq.heappush(ready, (priority(target), target))
    if len(ordered) != len(members):
        conflicts.append(MergeConflict(path, "incompatible sibling/component order"))
        return [*o, *sorted(set(members) - set(o), key=priority)]
    return ordered


def _merge_collection(base, ours, theirs, identity, path, conflicts):
    maps = [_index(values, identity, path) for values in (base, ours, theirs)]
    result = {}
    for key in sorted(set().union(*maps), key=repr):
        b, o, t = (values.get(key, _MISSING) for values in maps)
        if identity in {"uid", "stable_id", "entry_id"} and b is _MISSING and o is not _MISSING and t is not _MISSING and not _equal(o, t):
            value = _conflict(conflicts, _path(path, key), "different additions reused the same stable identity", o)
        else:
            value = _merge(b, o, t, _path(path, key), conflicts)
        if value is not _MISSING:
            result[key] = value
    order = _order(*(list(values) for values in maps), set(result), path, conflicts)
    return [result[key] for key in order]


def _merge_sequence(base, ours, theirs, path, conflicts):
    # Numeric arrays describe coordinates/channels, not identities or sets.
    if all(type(value) in (int, float) for array in (base, ours, theirs) for value in array):
        if len(base) == len(ours) == len(theirs):
            return [_merge(b, o, t, _path(path, index), conflicts)
                    for index, (b, o, t) in enumerate(zip(base, ours, theirs))]
        return _conflict(conflicts, path, "incompatible vector dimensions", ours)
    if path.rsplit("/", 1)[-1] in {"_shader_property_order", "dependencies"} and all(
        isinstance(value, str) for array in (base, ours, theirs) for value in array
    ):
        if all(len(values) == len(set(values)) for values in (base, ours, theirs)):
            members = (set(base) & set(ours) & set(theirs)) | (set(ours) - set(base)) | (set(theirs) - set(base))
            return _order(base, ours, theirs, members, path, conflicts)
    if len(base) == len(ours) == len(theirs):
        return [_merge(b, o, t, _path(path, index), conflicts)
                for index, (b, o, t) in enumerate(zip(base, ours, theirs))]
    return _conflict(conflicts, path, "array edits require stable identities", ours)


def _merge(base, ours, theirs, path, conflicts, collection_identity=None):
    if _equal(ours, theirs):
        return _clone(ours)
    if _equal(base, ours):
        return _clone(theirs)
    if _equal(base, theirs):
        return _clone(ours)
    if ours is _MISSING or theirs is _MISSING:
        return _conflict(conflicts, path, "deleted on one side and edited on the other", ours)
    if isinstance(ours, dict) and isinstance(theirs, dict) and (base is _MISSING or isinstance(base, dict)):
        original = {} if base is _MISSING else base
        for discriminator in ("$type", "$schema", "type_id", "type", "feature_type"):
            if discriminator in original and (
                not _equal(original.get(discriminator), ours.get(discriminator))
                or not _equal(original.get(discriminator), theirs.get(discriminator))
            ):
                return _conflict(conflicts, path, "type changed while the same record was edited", ours)
        result = {}
        for key in sorted(set(original) | set(ours) | set(theirs)):
            value = _merge(original.get(key, _MISSING), ours.get(key, _MISSING),
                           theirs.get(key, _MISSING), _path(path, key), conflicts,
                           _collection_identity(ours, key, path))
            if value is not _MISSING:
                result[key] = value
        return result
    if all(isinstance(value, list) for value in (base, ours, theirs)):
        if collection_identity:
            return _merge_collection(base, ours, theirs, collection_identity, path, conflicts)
        return _merge_sequence(base, ours, theirs, path, conflicts)
    return _conflict(conflicts, path, "both sides changed this value", ours)


def _hierarchy(document, prefab):
    field = "local_id" if prefab else "id"
    roots = [document["root_object"]] if prefab else document["objects"]
    if not isinstance(roots, list):
        raise DocumentError("/objects must be an array")
    nodes, orders, components = {}, {}, {}

    def visit(node, parent):
        if not isinstance(node, dict) or type(node.get(field)) is not int or node[field] <= 0:
            raise DocumentError(f"object {field} must be a positive integer")
        identity = node[field]
        if identity in nodes:
            raise DocumentError(f"duplicate object identity {identity}")
        if not isinstance(node.get("children"), list) or not isinstance(node.get("components"), list):
            raise DocumentError(f"object {identity}: children/components must be arrays")
        if not isinstance(node.get("transform"), dict):
            raise DocumentError(f"object {identity}: transform must be an object")
        value = {key: copy.deepcopy(value) for key, value in node.items() if key != "children"}
        value[_PARENT] = parent
        nodes[identity] = value
        orders.setdefault(parent, []).append(identity)
        records = list(node["components"])
        if not prefab:
            records.append(node["transform"])
        for component in records:
            component_id = component.get("component_id") if isinstance(component, dict) else None
            if type(component_id) is not int or component_id <= 0 or component_id in components:
                raise DocumentError(f"object {identity}: invalid/duplicate component identity {component_id}")
            components[component_id] = identity
        for child in node["children"]:
            visit(child, identity)

    for root in roots:
        visit(root, None)
    return nodes, orders, components


def _remap_references(value, objects, components, prefab, joint_data=False):
    if isinstance(value, list):
        return [_remap_references(item, objects, components, prefab) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    kind = value.get("$type")
    for key, item in value.items():
        if key in {"prefab_source", "baseline"}:
            result[key] = copy.deepcopy(item)
        elif (kind == "game_object_ref" and key == "object_id") or (
            kind == "component_ref" and key == "game_object_id"
        ):
            result[key] = objects.get(item, item)
        elif kind == "component_ref" and key == "component_id":
            result[key] = components.get(item, item)
        elif joint_data and key == "connected_body_component_id":
            result[key] = components.get(item, item)
        elif key in {"object_sources", "component_sources"}:
            mapping = objects if key == "object_sources" else components
            result[key] = [[mapping.get(pair[0], pair[0]), pair[1]] for pair in item]
        elif key == "property_overrides":
            result[key] = []
            for patch in item:
                rewritten = _remap_references(patch, objects, components, prefab)
                rewritten["object"] = objects.get(patch["object"], patch["object"])
                rewritten["component"] = components.get(patch["component"], patch["component"])
                result[key].append(rewritten)
        else:
            is_joint = key == "data" and value.get("type_id") in {
                "native:infernux.HingeJoint", "native:infernux.SliderJoint"
            }
            result[key] = _remap_references(item, objects, components, prefab, is_joint)
    return result


def _separate_additions(base, ours, theirs, prefab):
    structures = [_hierarchy(value, prefab) for value in (base, ours, theirs)]
    mappings = []
    for index in (0, 2):
        b, o, t = [set(value[index]) for value in structures]
        overlap = (o - b) & (t - b)
        next_id = max(b | o | t, default=0) + 1
        field = ("next_local_id" if index == 0 else "next_component_id") if prefab else (
            "nextObjectId" if index == 0 else "nextComponentId"
        )
        next_id = max(next_id, *(document.get(field, 1) for document in (base, ours, theirs)))
        mappings.append({old: next_id + offset for offset, old in enumerate(sorted(overlap))})
    objects, components = mappings
    if not objects and not components:
        return copy.deepcopy(theirs)
    result = _remap_references(theirs, objects, components, prefab)
    field = "local_id" if prefab else "id"

    def remap(node):
        node[field] = objects.get(node[field], node[field])
        if not prefab:
            transform = node["transform"]
            transform["component_id"] = components.get(transform["component_id"], transform["component_id"])
        for component in node["components"]:
            component["component_id"] = components.get(component["component_id"], component["component_id"])
        for child in node["children"]:
            remap(child)

    for root in ([result["root_object"]] if prefab else result["objects"]):
        remap(root)
    if "mainCameraComponentId" in result:
        result["mainCameraComponentId"] = components.get(result["mainCameraComponentId"], result["mainCameraComponentId"])
    return result


def _merge_hierarchy(base, ours, theirs, prefab, conflicts):
    for document in (base, ours, theirs):
        _validate_references(document, prefab)
    if _equal(ours, theirs) or _equal(base, theirs):
        return copy.deepcopy(ours)
    if _equal(base, ours):
        return copy.deepcopy(theirs)
    theirs = _separate_additions(base, ours, theirs, prefab)
    structures = [_hierarchy(value, prefab) for value in (base, ours, theirs)]
    maps = [value[0] for value in structures]
    merged = {}
    for identity in sorted(set().union(*maps)):
        value = _merge(*(values.get(identity, _MISSING) for values in maps),
                       _path("/objects", identity), conflicts)
        if value is not _MISSING:
            merged[identity] = value
    for identity, node in merged.items():
        seen, parent = {identity}, node[_PARENT]
        while parent is not None:
            if parent not in merged or parent in seen:
                raise DocumentError(f"/objects/{identity}: merged hierarchy has a missing parent or a cycle")
            seen.add(parent)
            parent = merged[parent][_PARENT]
    child_orders = {}
    for parent in {node[_PARENT] for node in merged.values()}:
        children = {identity for identity, node in merged.items() if node[_PARENT] == parent}
        child_orders[parent] = _order(*(value[1].get(parent, []) for value in structures), children,
                                     _path("/children", parent), conflicts)

    def rebuild(identity):
        node = {key: value for key, value in merged[identity].items() if key != _PARENT}
        node["children"] = [rebuild(child) for child in child_orders.get(identity, [])]
        return node

    ignored = {"root_object", "next_local_id", "next_component_id"} if prefab else {"objects", "nextObjectId", "nextComponentId"}
    envelopes = [{key: value for key, value in document.items() if key not in ignored}
                 for document in (base, ours, theirs)]
    result = _merge(*envelopes, "", conflicts)
    roots = [rebuild(identity) for identity in child_orders.get(None, [])]
    if prefab:
        if len(roots) != 1:
            raise DocumentError("Prefab requires exactly one root")
        result["root_object"] = roots[0]
        result["next_local_id"] = max(max(merged, default=0) + 1,
                                      *(document["next_local_id"] for document in (base, ours, theirs)))
        component_ids = _hierarchy(result, True)[2]
        result["next_component_id"] = max(max(component_ids, default=0) + 1,
                                          *(document["next_component_id"] for document in (base, ours, theirs)))
    else:
        result["objects"] = roots
        for field, identities in (("nextObjectId", merged), ("nextComponentId", _hierarchy(result, False)[2])):
            if any(field in document for document in (base, ours, theirs)):
                result[field] = max(max(identities, default=0) + 1,
                                    *(document.get(field, 1) for document in (base, ours, theirs)))
    _validate_references(result, prefab)
    return result


def _validate_references(document, prefab):
    nodes, _, components = _hierarchy(document, prefab)
    for field, identities in ((("next_local_id" if prefab else "nextObjectId"), nodes),
                              (("next_component_id" if prefab else "nextComponentId"), components)):
        if field in document and (type(document[field]) is not int or
                                  not max(identities, default=0) < document[field] < 2**64):
            raise DocumentError(f"/{field}: allocation watermark must exceed every surviving identity")

    def walk(value, path, joint_data=False):
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, _path(path, index))
        elif isinstance(value, dict):
            kind = value.get("$type")
            if kind in {"game_object_ref", "component_ref"}:
                field = "object_id" if kind == "game_object_ref" else "game_object_id"
                identity = value.get(field)
                if type(identity) is not int:
                    raise DocumentError(f"{path}: reference identity must be an integer")
                if identity and (not prefab or identity > 0):
                    if identity not in nodes:
                        raise DocumentError(f"{path}: reference points to a deleted object {identity}")
                    component_id = value.get("component_id", 0)
                    if type(component_id) is not int or (component_id and components.get(component_id) != identity):
                        raise DocumentError(f"{path}: component reference points to a deleted/mismatched component")
            for key, item in value.items():
                if key in {"prefab_source", "baseline"}:
                    continue
                if joint_data and key == "connected_body_component_id" and item and item not in components:
                    raise DocumentError(f"{path}: physics joint points to a deleted component")
                is_joint = key == "data" and value.get("type_id") in {
                    "native:infernux.HingeJoint", "native:infernux.SliderJoint"
                }
                walk(item, _path(path, key), is_joint)

    walk(document, "")
    camera = document.get("mainCameraComponentId", 0)
    if not prefab and "mainCameraComponentId" in document and (type(camera) is not int or camera <= 0):
        raise DocumentError("/mainCameraComponentId: expected a positive component identity")
    if not prefab and camera and camera not in components:
        raise DocumentError("/mainCameraComponentId: camera was deleted")
    if not prefab and camera:
        owner = nodes[components[camera]]
        if not any(component.get("component_id") == camera and component.get("type_id") == "native:infernux.Camera"
                   for component in owner["components"]):
            raise DocumentError("/mainCameraComponentId: target is not a Camera")


def _normalize_metadata(value):
    value = copy.deepcopy(value)
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            for name in ("content_hash", "last_modified"):
                metadata.pop(name, None)
            for name in ("model_textures", "model_animations"):
                entry = metadata.get(name)
                if entry and entry.get("type") == "string":
                    entry["value"] = _normalize_metadata(json.loads(entry["value"]))
        for key, item in value.items():
            value[key] = _normalize_metadata(item)
    elif isinstance(value, list):
        value = [_normalize_metadata(item) for item in value]
    return value


def _encode_metadata(value):
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            for name in ("model_textures", "model_animations"):
                entry = metadata.get(name)
                if entry and entry.get("type") == "string" and not isinstance(entry["value"], str):
                    entry["value"] = json.dumps(_encode_metadata(entry["value"]), sort_keys=True,
                                                ensure_ascii=False, separators=(",", ":"))
        for key, item in value.items():
            value[key] = _encode_metadata(item)
    elif isinstance(value, list):
        value = [_encode_metadata(item) for item in value]
    return value


def _validate_graphs(value, path=""):
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_graphs(item, _path(path, index))
    elif isinstance(value, dict):
        if value.get("$schema") == _GRAPH_SCHEMA:
            nodes = _index(value["nodes"], "uid", path + "/nodes")
            _index(value["links"], "uid", path + "/links")
            for link in value["links"]:
                if link["source_node"] not in nodes or link["target_node"] not in nodes:
                    raise DocumentError(f"{path}/links/{link['uid']}: link points to a deleted node")
        for key, item in value.items():
            _validate_graphs(item, _path(path, key))


def merge_documents(base, ours, theirs, asset_path):
    if not all(isinstance(value, dict) for value in (base, ours, theirs)):
        raise DocumentError("authored document roots must be objects")
    conflicts = []
    suffix = Path(asset_path).suffix.lower()
    try:
        if suffix in {".scene", ".prefab"}:
            result = _merge_hierarchy(base, ours, theirs, suffix == ".prefab", conflicts)
        elif suffix == ".meta":
            values = [_normalize_metadata(value) for value in (base, ours, theirs)]
            guids = [value["metadata"]["guid"]["value"] for value in values]
            if len(set(guids)) != 1:
                raise DocumentError("/metadata/guid: asset identity changed across branches")
            result = _encode_metadata(_merge(*values, "", conflicts))
        else:
            if suffix == ".mat":
                for side, other in ((ours, theirs), (theirs, ours)):
                    if not _equal(base.get("shaders"), side.get("shaders")) and not _equal(base.get("properties"), other.get("properties")):
                        raise DocumentError("/shaders: shader changed while the other side edited material properties")
            result = _merge(base, ours, theirs, "", conflicts)
            _validate_graphs(result)
    except DocumentError as exc:
        conflicts.append(MergeConflict("/", str(exc)))
        result = copy.deepcopy(ours)
    return MergeResult(result, conflicts)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DocumentError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def read_document(path):
    def invalid_constant(value):
        raise DocumentError(f"non-finite JSON value {value}")
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=_unique_pairs, parse_constant=invalid_constant)


@cache
def _path_operations():
    # Load the path owner as a standalone file. Importing its engine package
    # would initialize the native runtime inside Git/Hub's setup process.
    return runpy.run_path(str(Path(__file__).parent / "engine" / "path_utils.py"))


def configure_git_driver(project_root, python_executable=None):
    resolved_path = _path_operations()["resolved_path"]
    root = Path(resolved_path(project_root))
    if not any((parent / ".git").exists() for parent in (root, *root.parents)):
        return False
    interpreter = Path(resolved_path(python_executable or sys.executable)).as_posix()
    script = Path(resolved_path(__file__)).as_posix()
    command = f"{shlex.quote(interpreter)} {shlex.quote(script)} %O %A %B %P"
    for key, value in (("merge.infernux.name", "Infernux authored document merge"),
                       ("merge.infernux.driver", command), ("merge.infernux.recursive", "binary")):
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        subprocess.run(["git", "-C", str(root), "config", "--local", key, value], check=True,
                       capture_output=True, text=True, **options)
    attributes = root / ".gitattributes"
    text = attributes.read_text(encoding="utf-8") if attributes.exists() else ""
    patterns = {"*.scene", "*.prefab", "*.mat", "*.effect", "*.effectgroup", "*.particlegraph", "*.meta",
                "*.physicmaterial", "*.rendertexture", "*.inxdata", "*.animclip2d", "*.animclip3d",
                "*.animfsm", "*.timelinefsm", "*.animtimeline",
                "**/ProjectSettings/*.json", "**/InxPlugins.json"}
    lines, configured = [], set()
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in patterns:
            configured.add(fields[0])
            if "merge=binary" in fields[1:]:
                line = line.replace("merge=binary", "merge=infernux")
        lines.append(line)
    for pattern in sorted(patterns - configured):
        lines.append(f"{pattern} text eol=lf merge=infernux")
    updated = "\n".join(lines) + "\n"
    if updated != text:
        with attributes.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(updated)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", metavar="PROJECT")
    parser.add_argument("files", nargs="*")
    arguments = parser.parse_args(argv)
    if arguments.install:
        if arguments.files:
            parser.error("--install does not accept merge files")
        configure_git_driver(arguments.install)
        return 0
    if len(arguments.files) != 4:
        parser.error("expected BASE OURS THEIRS ASSET_PATH")
    base, ours, theirs, asset = arguments.files
    try:
        result = merge_documents(read_document(base), read_document(ours), read_document(theirs), asset)
        indent = 4 if Path(asset).suffix.lower() == ".meta" else 2
        payload = json.dumps(result.document, indent=indent, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        with open(ours, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        for conflict in result.conflicts:
            print(f"Infernux merge conflict: {asset}{conflict.path}: {conflict.reason}", file=sys.stderr)
        return 0 if result.clean else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Infernux merge failed: {asset}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
