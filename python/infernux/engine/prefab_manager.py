"""
Prefab system for infernux.

Handles saving GameObjects as .prefab files and instantiating them back into scenes.
Prefab files contain a typed GameObject document wrapped in a strict envelope.
"""

import json
import os
import copy

from infernux.debug import Debug
from infernux.engine.path_utils import path_key, resolved_path

PREFAB_EXTENSION = ".prefab"
_PREFAB_TEMPLATE_CACHE = {}


class PrefabDocumentError(ValueError):
    """Raised when a prefab is not the single active document schema."""


def _validate_game_object_document(
    document: dict,
    location: str = "root_object",
    local_ids: set[int] = None,
) -> None:
    if not isinstance(document, dict):
        raise PrefabDocumentError(f"{location} must be an object")
    if local_ids is None:
        local_ids = set()
    required = {
        "local_id", "name", "active", "is_static", "tag", "layer",
        "transform", "components", "children",
    }
    optional = {"nested_prefab", "model_source"}
    if not required.issubset(document) or set(document) - required - optional:
        missing = sorted(required - set(document))
        unknown = sorted(set(document) - required - optional)
        raise PrefabDocumentError(
            f"{location} fields do not match the current schema; missing={missing}, unknown={unknown}"
        )
    local_id = document["local_id"]
    if type(local_id) is not int or local_id <= 0 or local_id in local_ids:
        raise PrefabDocumentError(f"{location}.local_id must be a unique positive integer")
    local_ids.add(local_id)
    if not isinstance(document["name"], str) or type(document["active"]) is not bool:
        raise PrefabDocumentError(f"{location} has invalid name/active fields")
    if type(document["is_static"]) is not bool or not isinstance(document["tag"], str):
        raise PrefabDocumentError(f"{location} has invalid is_static/tag fields")
    if type(document["layer"]) is not int or not 0 <= document["layer"] < 32:
        raise PrefabDocumentError(f"{location}.layer must be an integer in [0, 31]")
    if not isinstance(document["transform"], dict):
        raise PrefabDocumentError(f"{location}.transform must be an object")
    for field in ("components", "children"):
        if not isinstance(document[field], list):
            raise PrefabDocumentError(f"{location}.{field} must be an array")
    if "model_source" in document:
        source = document["model_source"]
        if (
            not isinstance(source, dict) or set(source) != {"guid", "path"}
            or not isinstance(source["guid"], str) or not source["guid"]
            or not isinstance(source["path"], list)
            or any(not isinstance(part, str) or not part for part in source["path"])
        ):
            raise PrefabDocumentError(f"{location}.model_source requires a non-empty GUID and source path array")
    for index, child in enumerate(document["children"]):
        _validate_game_object_document(child, f"{location}.children[{index}]", local_ids)
    if "nested_prefab" in document:
        _validate_nested_prefab(document, location)


def _validate_nested_prefab(node, location):
    """Nested links belong to the author document, never the native ObjectGraph."""
    link = node["nested_prefab"]
    fields = {"guid", "baseline", "object_sources", "component_sources"}
    if not isinstance(link, dict) or set(link) != fields:
        raise PrefabDocumentError(f"{location}.nested_prefab has invalid fields")
    if not isinstance(link["guid"], str) or not link["guid"]:
        raise PrefabDocumentError(f"{location}.nested_prefab requires a source GUID")
    _validate_game_object_document(
        link["baseline"], location + ".nested_prefab.baseline", local_ids=set()
    )
    for key in ("object_sources", "component_sources"):
        pairs = link[key]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or any(type(value) is not int or value <= 0 for value in pair)
            for pair in pairs
        ):
            raise PrefabDocumentError(f"{location}.nested_prefab.{key} requires positive identity pairs")
        if len({pair[0] for pair in pairs}) != len(pairs) or len({pair[1] for pair in pairs}) != len(pairs):
            raise PrefabDocumentError(f"{location}.nested_prefab.{key} contains duplicate identities")
    if [node["local_id"], link["baseline"]["local_id"]] not in link["object_sources"]:
        raise PrefabDocumentError(f"{location}.nested_prefab does not identify its root")


def _validate_prefab_document(document: dict, file_path: str = "<memory>") -> None:
    if not isinstance(document, dict):
        raise PrefabDocumentError(f"Prefab '{file_path}' must contain an object")
    allowed = {"root_object", "source_canvas_name", "next_local_id", "next_component_id", "variant"}
    required = {"root_object", "next_local_id", "next_component_id"}
    if not required.issubset(document) or not set(document).issubset(allowed):
        raise PrefabDocumentError(f"Prefab '{file_path}' has missing or unknown envelope fields")
    if "source_canvas_name" in document and not isinstance(document["source_canvas_name"], str):
        raise PrefabDocumentError(f"Prefab '{file_path}' source_canvas_name must be a string")
    local_ids = set()
    _validate_game_object_document(document["root_object"], local_ids=local_ids)
    component_ids = [component["component_id"] for node in _prefab_nodes(document["root_object"])
                     for component in node["components"]]
    if any(type(value) is not int or value <= 0 for value in component_ids) or len(component_ids) != len(set(component_ids)):
        raise PrefabDocumentError(f"Prefab '{file_path}' component IDs must be unique positive integers")
    value = document["next_component_id"]
    if type(value) is not int or value <= max(component_ids, default=0):
        raise PrefabDocumentError(f"Prefab '{file_path}' next_component_id must exceed every component ID")
    next_id = document["next_local_id"]
    if type(next_id) is not int or next_id <= max(local_ids):
        raise PrefabDocumentError(f"Prefab '{file_path}' next_local_id must exceed every source node ID")
    if "variant" in document:
        from infernux.engine.prefab_variant import validate_variant_definition, variant_definition
        validate_variant_definition(variant_definition(document))


def _prefab_next_local_id(document):
    return document["next_local_id"]


def _prefab_nodes(root):
    yield root
    for child in root["children"]:
        yield from _prefab_nodes(child)


def _prefab_next_component_id(document):
    return document["next_component_id"]


_PREFAB_OCCURRENCE_FIELDS = ("outer_source_id", "outer_source_guid")


def _make_prefab_baseline(root, *, outer_source_id=0, previous=None):
    result = {"root_object": copy.deepcopy(root)}
    if previous:
        result.update((key, previous[key]) for key in _PREFAB_OCCURRENCE_FIELDS if key in previous)
    if outer_source_id:
        result["outer_source_id"] = outer_source_id
    return result


def _prefab_baseline_root(baseline):
    if set(baseline) - set(_PREFAB_OCCURRENCE_FIELDS) != {"root_object"}:
        raise PrefabDocumentError("Invalid prefab source baseline fields")
    if "outer_source_id" in baseline and (type(baseline["outer_source_id"]) is not int or baseline["outer_source_id"] <= 0):
        raise PrefabDocumentError("Nested Prefab outer source identity must be a positive integer")
    if "outer_source_guid" in baseline and (
        not isinstance(baseline["outer_source_guid"], str) or not baseline["outer_source_guid"]
        or "outer_source_id" not in baseline
    ):
        raise PrefabDocumentError("Nested Prefab outer source GUID requires an outer source identity")
    return baseline["root_object"]


def _validate_nested_source_ancestry(root, ancestry):
    for node in _prefab_nodes(root):
        link = node.get("nested_prefab")
        if link is None:
            continue
        guid = link["guid"]
        if guid in ancestry:
            raise PrefabDocumentError("Nested Prefab source cycle: " + " -> ".join((*ancestry, guid)))
        _validate_nested_source_ancestry(link["baseline"], (*ancestry, guid))


def _read_prefab_document(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    _validate_prefab_document(document, file_path)
    return document


def _invalidate_prefab_template_cache(file_path: str = None, guid: str = ""):
    del file_path  # File paths locate bytes; only the GUID can own a cache entry.
    if guid:
        _PREFAB_TEMPLATE_CACHE.pop(guid, None)


def _get_file_stamp(file_path: str):
    try:
        stat = os.stat(file_path)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        # Missing file is not an error in the cache-stamp probe path; the
        # caller (cached prefab template lookup) treats None as "no cache
        # entry" and re-reads the prefab from disk if it exists.
        return None


def _read_resolved_prefab_document(file_path, asset_database=None, *, path_for_guid=None, dependencies=None,
                                   resolve_base=None):
    """Resolve the current base chain without rewriting author files on read.

    Editor templates and cook share this projection. A stored Variant baseline
    records override intent, not authority to serve stale inherited values.
    """
    from infernux.engine.prefab_variant import rebase_variant_definition, variant_definition, variant_document

    resolved, active = {}, set()

    def read(path):
        key = path_key(path)
        if key in active:
            raise PrefabDocumentError(f"Variant source cycle at '{path}'")
        if key in resolved:
            return resolved[key]
        active.add(key)
        try:
            document = _read_prefab_document(path)
            if dependencies is not None:
                dependencies[path] = _get_file_stamp(path)
            if "variant" in document:
                guid = document["variant"]["guid"]
                if resolve_base is not None:
                    base = resolve_base(guid)
                else:
                    if path_for_guid is not None:
                        base_path = path_for_guid(guid)
                    else:
                        database = asset_database
                        if database is None:
                            from infernux.core.assets import AssetManager
                            database = AssetManager.require_asset_database()
                        base_path = database.get_path_from_guid(guid)
                    if not base_path:
                        raise PrefabDocumentError(f"Variant base asset is unavailable: {guid}")
                    base = read(base_path)
                base = {name: value for name, value in base.items() if name != "variant"}
                document = variant_document(rebase_variant_definition(variant_definition(document), base))
            resolved[key] = document
            return document
        finally:
            active.remove(key)

    return read(file_path)


def _published_prefab_sources_for(asset_database):
    from infernux.engine.scene_manager import SceneFileManager

    files = SceneFileManager.instance()
    if asset_database is not None and isinstance(files, SceneFileManager) and files._asset_database is asset_database:
        return files._published_prefab_sources
    return None


def _prefab_source_loader(asset_database, sources, published, *, refresh=()):
    """Stage a consistent set of authored documents for one runtime operation."""
    from infernux.engine.component_restore import validate_prefab_source_components

    loading = set()

    def load(guid):
        if guid in sources:
            return sources[guid]
        path = asset_database.get_path_from_guid(guid)
        if not path:
            raise PrefabDocumentError(f"Prefab source cannot be resolved: {guid}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if guid in published and guid not in refresh:
            sources[guid] = published[guid]
            return sources[guid]
        if guid in loading:
            raise PrefabDocumentError(f"Variant source cycle at '{guid}'")
        loading.add(guid)
        try:
            document = _read_resolved_prefab_document(path, asset_database, resolve_base=load)
            validate_prefab_source_components(document["root_object"])
            sources[guid] = document
            return document
        finally:
            loading.remove(guid)

    return load


def _load_prefab_template_payload(file_path: str, resolved_guid: str, asset_database=None, dependencies=None):
    try:
        prefab_data = _read_resolved_prefab_document(file_path, asset_database, dependencies=dependencies)
    except (OSError, json.JSONDecodeError, PrefabDocumentError) as exc:
        Debug.log_error(f"Failed to read prefab file: {exc}")
        return None

    def load_source(guid):
        if guid == resolved_guid:
            return prefab_data["root_object"]
        database = asset_database
        if database is None:
            from infernux.core.assets import AssetManager
            database = AssetManager.require_asset_database()
        path = database.get_path_from_guid(guid)
        if not path:
            raise PrefabDocumentError(f"Nested Prefab source cannot be resolved: {guid}")
        if dependencies is not None:
            dependencies[path] = _get_file_stamp(path)
        return _read_resolved_prefab_document(path, database, dependencies=dependencies)["root_object"]

    return _build_prefab_template_payload(prefab_data, resolved_guid, load_source)


def _build_prefab_template_payload(prefab_data, resolved_guid, load_source):
    root_obj_data = copy.deepcopy(prefab_data["root_object"])
    _strip_prefab_runtime_fields(root_obj_data)
    _stamp_prefab_guid(root_obj_data, resolved_guid)
    if resolved_guid:
        root_obj_data["prefab_source"] = _make_prefab_baseline(prefab_data["root_object"])

    if any(node.get("prefab_root") for child in root_obj_data["children"] for node in _prefab_nodes(child)):
        # A template uses a private document ID namespace. Fresh native IDs are
        # still allocated only once, when this prepared graph is instantiated.
        next_component = prefab_data["next_component_id"]
        for node in _prefab_nodes(root_obj_data):
            node["id"] = node.pop("local_id")
            node["transform"]["component_id"] = next_component
            next_component += 1

        from infernux.engine.prefab_overrides import resolve_scene_prefab_documents
        root_obj_data = resolve_scene_prefab_documents({"objects": [root_obj_data]}, load_source)["objects"][0]

    return root_obj_data


def _get_published_prefab_template(guid, asset_database, published):
    cached = _PREFAB_TEMPLATE_CACHE.get(guid)
    if cached and "sources" in cached and all(
        published.get(identity) is document and os.path.isfile(asset_database.get_path_from_guid(identity) or "")
        for identity, document in cached["sources"].items()
    ):
        return cached["document"]

    sources = {}
    load = _prefab_source_loader(asset_database, sources, published)
    try:
        prefab_data = load(guid)
        template = _build_prefab_template_payload(prefab_data, guid, lambda identity: load(identity)["root_object"])
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        Debug.log_error(f"Failed to prepare Prefab template '{guid}': {exc}")
        return None

    published.update(sources)
    _PREFAB_TEMPLATE_CACHE[guid] = {"sources": sources, "document": template}
    return template


def _get_cached_prefab_template(file_path: str, resolved_guid: str, asset_database=None):
    """Cache authored data, never live objects in a loaded gameplay world."""
    if resolved_guid:
        published = _published_prefab_sources_for(asset_database)
        if published is not None:
            return _get_published_prefab_template(resolved_guid, asset_database, published)

    stamp = _get_file_stamp(file_path)
    if stamp is None:
        Debug.log_warning(f"Prefab file not found: {file_path}")
        return None

    if not resolved_guid:
        # Explicit unregistered authoring inputs may still be loaded by low-level
        # tooling, but a filesystem path is never promoted to cache identity.
        return _load_prefab_template_payload(file_path, "", asset_database, {})

    cache_key = resolved_guid
    cached = _PREFAB_TEMPLATE_CACHE.get(cache_key)
    if cached and cached.get("stamp") == stamp and all(
        _get_file_stamp(path) == previous for path, previous in cached["dependencies"].items()
    ):
        return cached["document"]

    dependencies = {}
    template_payload = _load_prefab_template_payload(file_path, resolved_guid, asset_database, dependencies)
    if template_payload is None:
        return None

    _PREFAB_TEMPLATE_CACHE[cache_key] = {
        "stamp": stamp,
        "document": template_payload,
        "dependencies": dependencies,
    }
    return template_payload


def _belongs_to_prefab_source(root, node):
    guid = root.get("prefab_guid")
    return not guid or node.get("prefab_guid") == guid


def _component_source_link(component):
    return component.get("prefab_source_id", 0), component.get("prefab_source_guid", "")


def _source_component_id(root, node, component):
    source_id, explicit_guid = _component_source_link(component)
    guid = root.get("prefab_guid")
    return source_id if not guid or (explicit_guid or node.get("prefab_guid")) == guid else 0


def _strip_prefab_runtime_fields(obj_data: dict, *, next_local_id=1, instance_snapshot=False,
                                 object_id_map=None, next_component_id=1, component_id_map=None):
    """Localize a graph; negative IDs exist only in private instance merge snapshots.

    A negative scene ID identifies an added node or an external scene reference,
    never a source node. It is resolved back to a scene ID before preflight and
    is never written into a prefab asset.
    """
    if not isinstance(obj_data, dict):
        raise PrefabDocumentError("root_object must be an object")

    nodes = []

    def collect(node: dict, location: str) -> None:
        if not isinstance(node, dict):
            raise PrefabDocumentError(f"{location} must be an object")
        nodes.append((node, location))
        children = node.get("children")
        if not isinstance(children, list):
            raise PrefabDocumentError(f"{location}.children must be an array")
        for index, child in enumerate(children):
            collect(child, f"{location}.children[{index}]")

    collect(obj_data, "root_object")
    owned_objects = {node["id"] for node, _location in nodes
                     if "id" in node and _belongs_to_prefab_source(obj_data, node)}
    baseline = obj_data.get("prefab_source")
    has_runtime_ids = ["id" in node for node, _location in nodes]
    has_local_ids = ["local_id" in node for node, _location in nodes]
    if all(has_runtime_ids) and not any(has_local_ids):
        runtime_to_local = {}
        source_ids = (list(object_id_map.values()) if object_id_map is not None else
                      [node.get("prefab_source_id", 0) for node, _ in nodes if node["id"] in owned_objects])
        linked_ids = [value for value in source_ids if value > 0]
        if len(linked_ids) != len(set(linked_ids)):
            raise PrefabDocumentError("ObjectGraph contains duplicate prefab source identities")
        next_local_id = max(next_local_id, max(linked_ids, default=0) + 1)
        for node, location in nodes:
            runtime_id = node["id"]
            local_id = (object_id_map.get(runtime_id, 0) if object_id_map is not None else
                        node.get("prefab_source_id", 0) if runtime_id in owned_objects else 0)
            if local_id <= 0:
                if instance_snapshot:
                    local_id = -runtime_id
                else:
                    local_id = next_local_id
                    next_local_id += 1
            if type(runtime_id) is not int or runtime_id <= 0 or runtime_id in runtime_to_local:
                raise PrefabDocumentError(f"{location}.id must be a unique positive integer")
            runtime_to_local[runtime_id] = local_id
            node["local_id"] = local_id
    elif all(has_local_ids) and not any(has_runtime_ids):
        runtime_to_local = None
    else:
        raise PrefabDocumentError("ObjectGraph must contain either runtime id or local_id on every node")

    def rewrite_references(value, path: str):
        if isinstance(value, list):
            return [rewrite_references(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if not isinstance(value, dict):
            return value
        from infernux.components.value_document import TYPE_KEY, GAME_OBJECT_REF, COMPONENT_REF
        document_type = value.get(TYPE_KEY)
        if document_type == GAME_OBJECT_REF:
            target_id = value["object_id"]
            if target_id == 0 or runtime_to_local is None:
                return dict(value)
            if target_id not in runtime_to_local:
                if instance_snapshot:
                    return {**value, "object_id": -target_id}
                raise PrefabDocumentError(f"{path}: prefab cannot reference GameObject {target_id} outside its subtree")
            remapped = dict(value)
            remapped["object_id"] = runtime_to_local[target_id]
            return remapped
        if document_type == COMPONENT_REF:
            target_id = value["game_object_id"]
            if target_id == 0 or runtime_to_local is None:
                return copy.deepcopy(value)
            if target_id not in runtime_to_local:
                if instance_snapshot:
                    return {**value, "game_object_id": -target_id}
                raise PrefabDocumentError(f"{path}: prefab cannot reference component outside its subtree")
            remapped = dict(value)
            remapped["game_object_id"] = runtime_to_local[target_id]
            if "component_id" in value:
                identity = value["component_id"]
                if identity in transform_ids:
                    remapped["component_id"] = 0
                elif identity in component_ids:
                    remapped["component_id"] = component_ids[identity]
                elif identity and instance_snapshot:
                    remapped["component_id"] = -identity
                elif identity:
                    raise PrefabDocumentError(f"{path}: prefab references a missing component {identity}")
            return remapped
        return {key: rewrite_references(item, f"{path}.{key}") for key, item in value.items()}

    components = [component for node, _ in nodes for component in node["components"]]
    component_ids = {} if runtime_to_local is not None else None
    transform_ids = {node["transform"].get("component_id") for node, _ in nodes}
    if component_ids is not None:
        linked = (list(component_id_map.values()) if component_id_map is not None else
                  [_source_component_id(obj_data, node, component) for node, _ in nodes
                   for component in node["components"]])
        linked = [value for value in linked if value > 0]
        if len(linked) != len(set(linked)):
            raise PrefabDocumentError("ObjectGraph contains duplicate component source identities")
        next_component_id = max(next_component_id, max(linked, default=0) + 1)
    for node, location in nodes:
        node.pop("id", None)
        transform = node.get("transform")
        if isinstance(transform, dict):
            transform.pop("component_id", None)
        for index, component in enumerate(node.get("components", [])):
            if not isinstance(component, dict):
                continue
            runtime_id = component["component_id"]
            if component_ids is not None:
                source_id = (component_id_map.get(runtime_id, 0) if component_id_map is not None else
                             _source_component_id(obj_data, node, component))
                if source_id <= 0:
                    if instance_snapshot:
                        source_id = -runtime_id
                    else:
                        source_id = next_component_id
                        next_component_id += 1
                component_ids[runtime_id] = source_id
                component["component_id"] = source_id
            component.pop("instance_guid", None)
    # References can point forward to components on a later node.
    for node, location in nodes:
        for index, component in enumerate(node["components"]):
            if not isinstance(component.get("data"), dict):
                continue
            component["data"] = rewrite_references(
                component["data"],
                f"{location}.components[{index}].data",
            )
            if component_ids is not None:
                from infernux.engine.component_restore import rewrite_native_component_references

                def local_component(target):
                    if not target:
                        return 0
                    if target in component_ids:
                        return component_ids[target]
                    if instance_snapshot:
                        return -target
                    raise PrefabDocumentError(
                        f"{location}.components[{index}]: prefab references a component outside its subtree"
                    )

                component["data"] = rewrite_native_component_references(
                    component["type_id"], component["data"], local_component,
                )
    if runtime_to_local is not None and baseline:
        # Native nodes carry their outer asset identity. Recover nested ownership
        # from that asset's immutable author baseline when creating a merge view.
        nested = {node["local_id"]: node["nested_prefab"]
                  for node in _prefab_nodes(_prefab_baseline_root(baseline)) if "nested_prefab" in node}
        for node, _location in nodes:
            if node["local_id"] in nested:
                node["nested_prefab"] = copy.deepcopy(nested[node["local_id"]])
    return runtime_to_local, next_local_id, component_ids, next_component_id


def read_prefab_source_canvas(file_path: str = None, guid: str = None,
                              asset_database=None) -> str:
    """Return the ``source_canvas_name`` stored in a prefab, or ``""``."""
    if asset_database is not None:
        resolved_guid = str(guid or "").strip()
        if not resolved_guid and file_path:
            try:
                resolved_guid = str(asset_database.get_guid_from_path(file_path) or "").strip()
            except Exception:
                resolved_guid = ""
        if not resolved_guid:
            return ""
        try:
            file_path = str(asset_database.get_path_from_guid(resolved_guid) or "").strip()
        except Exception:
            return ""
        published = _published_prefab_sources_for(asset_database)
        if file_path and os.path.isfile(file_path) and published is not None:
            if resolved_guid not in published and _get_published_prefab_template(resolved_guid, asset_database, published) is None:
                return ""
            return published[resolved_guid].get("source_canvas_name", "")
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        data = _read_prefab_document(file_path)
        return data.get("source_canvas_name", "")
    except (OSError, json.JSONDecodeError, PrefabDocumentError):
        return ""


def serialize_prefab_document(
    game_object,
    *,
    source_canvas_name: str = "",
    root_document_template: dict = None,
    next_local_id: int = 1,
    next_component_id: int = 1,
    preserve_root_properties: bool = True,
) -> dict:
    """Capture the exact strict prefab document owned by a GameObject tree."""
    if game_object is None:
        raise ValueError("cannot serialize prefab without a GameObject")

    from infernux.engine.component_restore import (
        serialize_game_object_document_authoritatively,
    )

    go_data = serialize_game_object_document_authoritatively(game_object)
    baseline = game_object._prefab_source_document
    if baseline and "prefab_source" not in go_data:
        go_data["prefab_source"] = baseline
    return _serialize_prefab_snapshot(
        go_data, source_canvas_name=source_canvas_name,
        root_document_template=root_document_template, next_local_id=next_local_id,
        next_component_id=next_component_id,
        preserve_root_properties=preserve_root_properties,
    )[0]


def _serialize_prefab_snapshot(go_data, *, source_canvas_name="", root_document_template=None,
                               next_local_id=1, next_component_id=1, preserve_root_properties=True):
    """Capture asset content and its scene-to-source projection from one snapshot."""
    go_data = copy.deepcopy(go_data)
    if not isinstance(go_data, dict):
        raise TypeError("GameObject.serialize_document() did not return a dict")
    if preserve_root_properties and isinstance(root_document_template, dict):
        for key in ("name", "active", "is_static", "tag", "layer"):
            if key in root_document_template:
                go_data[key] = copy.deepcopy(root_document_template[key])
        root_transform = go_data.get("transform")
        template_transform = root_document_template.get("transform")
        if isinstance(root_transform, dict) and isinstance(template_transform, dict):
            for key in ("position", "rotation"):
                if key in template_transform:
                    root_transform[key] = copy.deepcopy(template_transform[key])

    # Strip linkage and convert runtime IDs/references to prefab-local IDs.
    object_ids, next_local_id, component_ids, next_component_id = _localize_prefab_snapshot(
        go_data, source=root_document_template, next_local_id=next_local_id, next_component_id=next_component_id,
    )

    prefab_data = {
        "root_object": go_data,
        "next_local_id": next_local_id,
        "next_component_id": next_component_id,
    }
    if source_canvas_name:
        prefab_data["source_canvas_name"] = source_canvas_name
    _validate_prefab_document(prefab_data)
    return prefab_data, object_ids, component_ids


def _localize_prefab_snapshot(root, *, source=None, instance_snapshot=False,
                              object_id_map=None, component_id_map=None,
                              next_local_id=1, next_component_id=1):
    from infernux.engine.prefab_overrides import _instance_component_ids
    nested = any(node.get("prefab_root") for child in root["children"] for node in _prefab_nodes(child))
    if nested and object_id_map is None:
        object_id_map, component_id_map = _nested_source_projection(root, source)
    elif component_id_map is None and source is not None:
        component_id_map = _instance_component_ids(root, source)
    instances = _capture_nested_instances(root)
    objects, next_local_id, components, next_component_id = _strip_prefab_runtime_fields(
        root, instance_snapshot=instance_snapshot, object_id_map=object_id_map,
        component_id_map=component_id_map, next_local_id=next_local_id, next_component_id=next_component_id,
    )
    for node, link, inner_objects, inner_components in instances:
        node["nested_prefab"] = {
            **link,
            "object_sources": [[objects[identity], source] for identity, source in inner_objects.items() if source > 0],
            "component_sources": [[components[identity], source] for identity, source in inner_components.items() if source > 0],
        }
    _strip_prefab_fields(root)
    return objects, next_local_id, components, next_component_id


def _nested_source_projection(root, source=None):
    """Map innermost runtime ownership into this enclosing asset's namespace."""
    baseline = root.get("prefab_source")
    source = _prefab_baseline_root(baseline) if baseline else source
    links = {node["local_id"]: node["nested_prefab"] for node in _prefab_nodes(source)
             if "nested_prefab" in node} if source else {}
    objects, components = {}, {}
    guid = root.get("prefab_guid")

    def collect_outer_components(node):
        for component in node["components"]:
            if source_id := _source_component_id(root, node, component):
                components[component["component_id"]] = source_id

    def collect_outer_members(node):
        if node.get("prefab_root"):
            if node.get("prefab_source", {}).get("outer_source_guid") == guid:
                project_instance(node)
                return
        elif _belongs_to_prefab_source(root, node):
            if source_id := node.get("prefab_source_id"):
                objects[node["id"]] = source_id
        collect_outer_components(node)
        for child in node["children"]:
            collect_outer_members(child)

    def project_instance(node):
        inner_objects, inner_components = _nested_source_projection(node)
        baseline = node.get("prefab_source", {})
        owner = baseline.get("outer_source_guid", guid)
        link = links.get(baseline.get("outer_source_id")) if owner == guid else None
        if link and link["guid"] != node["prefab_guid"]:
            link = None
        object_sources = {inner: outer for outer, inner in link["object_sources"]} if link else {}
        component_sources = {inner: outer for outer, inner in link["component_sources"]} if link else {}
        objects.update((identity, object_sources.get(inner, -identity)) for identity, inner in inner_objects.items())
        components.update((identity, component_sources.get(inner, -identity)) for identity, inner in inner_components.items())
        if guid:
            collect_outer_components(node)
            for child in node["children"]:
                collect_outer_members(child)

    def visit(node, is_root=False):
        if not is_root and node.get("prefab_root") and node.get("prefab_guid"):
            project_instance(node)
            return
        owned = _belongs_to_prefab_source(root, node)
        objects[node["id"]] = (node.get("prefab_source_id", 0) if owned else 0) or -node["id"]
        for component in node["components"]:
            identity = component["component_id"]
            components[identity] = _source_component_id(root, node, component) or -identity
        for child in node["children"]:
            visit(child)

    visit(root, True)
    return objects, components


def _capture_nested_instances(root):
    """Move embedded instances into independent source namespaces before save.

    The outer asset owns flat local IDs for references; each nested root records
    a separate projection to its original asset. Repeated instances of one source
    therefore cannot collide, and inner baselines retain deeper nested links.
    """
    captured = []
    guid = root.get("prefab_guid")

    def visit(node, inside_instance=False):
        if node.get("prefab_root") and node.get("prefab_guid"):
            baseline = node.get("prefab_source")
            if not baseline:
                raise PrefabDocumentError("Nested Prefab requires a synchronized source baseline before saving")
            if not inside_instance or (guid and baseline.get("outer_source_guid") == guid):
                objects, components = _nested_source_projection(node)
                captured.append((node, {"guid": node["prefab_guid"],
                                        "baseline": copy.deepcopy(_prefab_baseline_root(baseline))}, objects, components))
            inside_instance = True
        for child in node["children"]:
            visit(child, inside_instance)

    for child in root["children"]:
        visit(child)
    return captured


def save_prefab_document(prefab_data: dict, file_path: str, asset_database=None,
                         *, expected_file_state=None) -> bool:
    """Durably publish an already captured strict prefab document."""
    if not file_path.lower().endswith(PREFAB_EXTENSION):
        file_path += PREFAB_EXTENSION

    try:
        _validate_prefab_document(prefab_data, file_path)
    except PrefabDocumentError as exc:
        Debug.log_error(f"Refusing to save invalid prefab: {exc}")
        return False

    try:
        os.makedirs(os.path.dirname(resolved_path(file_path)), exist_ok=True)
        from infernux.core.document_store import write_document_text
        payload = {**prefab_data, "next_local_id": _prefab_next_local_id(prefab_data),
                   "next_component_id": _prefab_next_component_id(prefab_data)}
        content = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        write_document_text(file_path, content, expected_file_state=expected_file_state)
    except (OSError, RuntimeError) as exc:
        Debug.log_error(f"Failed to write prefab file: {exc}")
        return False

    if asset_database:
        try:
            from infernux.core.assets import AssetManager
            mutation = AssetManager.import_asset(file_path, database=asset_database)
            if not mutation:
                raise RuntimeError(mutation.error)
            guid = mutation.guid
            _invalidate_prefab_template_cache(file_path, guid)
        except Exception as exc:
            _invalidate_prefab_template_cache(file_path, "")
            Debug.log_error(f"Failed to register prefab in AssetDatabase: {exc}")
            return False
    else:
        _invalidate_prefab_template_cache(file_path, "")

    return True


def save_prefab(game_object, file_path: str, asset_database=None,
               source_canvas_name: str = "", root_document_template: dict = None,
               *, expected_file_state=None) -> bool:
    """Serialize and durably publish a GameObject hierarchy as a prefab."""
    if not file_path.lower().endswith(PREFAB_EXTENSION):
        file_path += PREFAB_EXTENSION
    try:
        existing_save = expected_file_state is None or expected_file_state.exists
        previous = _read_prefab_document(file_path) if existing_save and os.path.isfile(file_path) else None
        prefab_data = serialize_prefab_document(
            game_object,
            source_canvas_name=source_canvas_name,
            root_document_template=root_document_template,
            next_local_id=previous["next_local_id"] if previous else 1,
            next_component_id=previous["next_component_id"] if previous else 1,
        )
        if previous and "variant" in previous:
            from infernux.engine.prefab_variant import edit_variant_document
            prefab_data = edit_variant_document(previous, prefab_data)
    except Exception as exc:
        Debug.log_error(f"Failed to serialize GameObject for prefab: {exc}")
        return False
    return save_prefab_document(
        prefab_data, file_path, asset_database, expected_file_state=expected_file_state,
    )


def instantiate_prefab(file_path: str = None, guid: str = None,
                       scene=None, parent=None, asset_database=None,
                       *, instantiate_in_world_space: bool = False,
                       configure_created=None):
    """Instantiate a prefab into the active scene.

    Supply either *file_path* or *guid* (GUID is resolved via asset_database).
    Returns the root GameObject, or None on failure.
    """
    resolved_guid = str(guid or "").strip()
    if asset_database is not None:
        # ``file_path`` is accepted only as an editor authoring boundary. Once
        # selected, immediately collapse it to the registered GUID and resolve
        # the current location back from that GUID. A stale supplied path never
        # participates in runtime identity or cache selection.
        if not resolved_guid and file_path:
            try:
                resolved_guid = str(asset_database.get_guid_from_path(file_path) or "").strip()
            except Exception:
                resolved_guid = ""
        if not resolved_guid:
            return None
        try:
            file_path = str(asset_database.get_path_from_guid(resolved_guid) or "").strip()
        except Exception:
            file_path = ""

    if not file_path or not os.path.isfile(file_path):
        Debug.log_warning(f"Prefab file not found: {file_path}")
        return None

    if scene is None:
        from infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
    if scene is None:
        Debug.log_warning("No active scene — cannot instantiate prefab.")
        return None

    template = _get_cached_prefab_template(file_path, resolved_guid, asset_database)
    if template is None:
        return None

    from infernux.engine.component_restore import instantiate_game_object_document_transactionally

    def configure_instance(created):
        created.name = f"{created.name} (Clone)"
        if parent is not None and instantiate_in_world_space:
            created.set_parent(parent, True)
        if configure_created is not None:
            configure_created(created)

    try:
        new_obj = instantiate_game_object_document_transactionally(
            scene,
            template,
            None if instantiate_in_world_space else parent,
            asset_database,
            configure_created=configure_instance,
        )
    except RuntimeError as exc:
        Debug.log_error(f"Failed to instantiate prefab document: {exc}")
        return None
    if new_obj is None:
        Debug.log_error("Failed to instantiate prefab from cached template.")
        return None

    return new_obj


def _stamp_prefab_guid(obj_data: dict, guid: str, is_root: bool = True, *, source_ids=None, component_source_ids=None):
    """Recursively stamp prefab_guid (and prefab_root on root) into JSON data."""
    if "nested_prefab" in obj_data:
        link = obj_data.pop("nested_prefab")
        outer_id = obj_data["local_id"]
        # Stamp inner ownership using inner IDs, while retaining the outer IDs
        # in the ObjectGraph so ordinary reference remapping remains unchanged.
        object_sources = dict(link["object_sources"])
        component_sources = dict(link["component_sources"])
        baseline_nodes = {node["local_id"]: node for node in _prefab_nodes(link["baseline"])}
        saved = []
        outer_nodes = []
        foreign_instances = {}
        object_namespace, component_namespace = {}, {}
        for node in _prefab_nodes(obj_data):
            local_id = node["local_id"]
            source_id = object_sources.get(local_id, -abs(local_id))
            if source_id < 0 < local_id and (source_ids is None or local_id in source_ids):
                outer_nodes.append(node)
            if source_id < 0 and "nested_prefab" in node:
                foreign_instances[local_id] = node
            saved.append((node, "local_id", local_id))
            node["local_id"] = source_id
            object_namespace[local_id] = source_id
            original = baseline_nodes.get(source_id)
            if original and "nested_prefab" in original:
                node["nested_prefab"] = copy.deepcopy(original["nested_prefab"])
            for component in node["components"]:
                identity = component["component_id"]
                saved.append((component, "component_id", identity))
                component["component_id"] = component_sources.get(identity, -abs(identity))
                component_namespace[identity] = component["component_id"]
        for node in foreign_instances.values():
            nested = node["nested_prefab"]
            for key, namespace in (("object_sources", object_namespace), ("component_sources", component_namespace)):
                nested[key] = [[namespace[outer], inner] for outer, inner in nested[key] if outer in namespace]
        _stamp_prefab_guid(obj_data, link["guid"], is_root=True,
                           source_ids=set(object_sources.values()), component_source_ids=set(component_sources.values()))
        for record, key, value in saved:
            record[key] = value
        for node in outer_nodes:
            if node.get("prefab_root"):
                # Its placement is owned by the enclosing asset even though
                # another nested instance now contains it in the scene.
                if node["local_id"] in foreign_instances:
                    node["prefab_source"]["outer_source_id"] = node["local_id"]
                    node["prefab_source"]["outer_source_guid"] = guid
                continue
            if node.get("prefab_source_id"):
                continue
            node["prefab_guid"] = guid
            node["prefab_source_id"] = node["local_id"]
        for node in _prefab_nodes(obj_data):
            for component in node["components"]:
                identity = component["component_id"]
                if (not component.get("prefab_source_id") and identity > 0
                        and (component_source_ids is None or identity in component_source_ids)):
                    component["prefab_source_id"] = identity
                    if node["prefab_guid"] != guid:
                        component["prefab_source_guid"] = guid
        obj_data["prefab_source"] = _make_prefab_baseline(link["baseline"], outer_source_id=outer_id if outer_id > 0 else 0)
        return
    obj_data["prefab_guid"] = guid
    local_id = obj_data["local_id"]
    if source_ids is None or local_id in source_ids:
        obj_data["prefab_source_id"] = local_id
    else:
        obj_data.pop("prefab_source_id", None)
    if is_root:
        obj_data["prefab_root"] = True
    for component in obj_data["components"]:
        component.pop("prefab_source_guid", None)
        source_id = component["component_id"]
        if source_id > 0 and (component_source_ids is None or source_id in component_source_ids):
            component["prefab_source_id"] = source_id
        else:
            component.pop("prefab_source_id", None)
    for child in obj_data.get("children", []):
        _stamp_prefab_guid(child, guid, is_root=False, source_ids=source_ids, component_source_ids=component_source_ids)


def _link_prefab_components(obj, component_links):
    """Assign links on existing native handles without restoring author state."""
    from infernux.components import InxComponent
    for component in obj.get_components():
        native = component._cpp_component if isinstance(component, InxComponent) else component
        if native.component_id in component_links:
            native._prefab_source = component_links[native.component_id]


def _link_created_prefab_source(game_object, file_path: str, asset_database) -> bool:
    """Turn the saved source hierarchy into an instance of its new Prefab."""
    if game_object is None or asset_database is None:
        return False
    try:
        guid = str(asset_database.get_guid_from_path(file_path) or "")
    except Exception as exc:
        Debug.log_warning(f"Failed to resolve created prefab GUID: {exc}")
        return False
    if not guid:
        Debug.log_warning(f"Created prefab has no AssetDatabase GUID: {file_path}")
        return False

    document = _read_prefab_document(file_path)["root_object"]
    try:
        _link_prefab_hierarchy(game_object, document, guid)
    except Exception as exc:
        Debug.log_warning(f"Failed to link created prefab source: {exc}")
        return False
    return True


def _link_prefab_hierarchy(game_object, document, guid):
    """Publish a saved asset's layered ownership onto the existing handles."""
    linked_document = copy.deepcopy(document)
    _stamp_prefab_guid(linked_document, guid)

    def _link(obj, node, is_root: bool) -> None:
        obj.prefab_guid = node["prefab_guid"]
        obj.prefab_root = node.get("prefab_root", False)
        obj.prefab_source_id = node.get("prefab_source_id", 0)
        obj._prefab_source_document = node.get("prefab_source")
        records = obj.serialize_document()["components"]
        _link_prefab_components(obj, {current["component_id"]: _component_source_link(source)
                                     for current, source in zip(records, node["components"], strict=True)})
        for child, source in zip(obj.get_children(), node["children"], strict=True):
            _link(child, source, False)

    _link(game_object, linked_document, True)
    game_object._prefab_source_document = _make_prefab_baseline(document)


def _strip_prefab_fields(obj_data: dict):
    """Recursively remove prefab_guid/prefab_root so the template is clean."""
    obj_data.pop("prefab_guid", None)
    obj_data.pop("prefab_root", None)
    obj_data.pop("prefab_source_id", None)
    obj_data.pop("prefab_source", None)
    for component in obj_data.get("components", []):
        component.pop("prefab_source_id", None)
        component.pop("prefab_source_guid", None)
    for child in obj_data.get("children", []):
        _strip_prefab_fields(child)
