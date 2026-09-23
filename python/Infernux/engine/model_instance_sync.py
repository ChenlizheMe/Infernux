"""Synchronize live model instances after an external source reimport.

Imported models are references, not flattened scene content.  A scene keeps
the instance hierarchy and its authored transforms while the source asset owns
the node list and geometry.  This module reconciles the two at the owner
safe-point after a model publication.
"""

from __future__ import annotations

import json
import math
from typing import Any


_MODEL_RENDERER_TYPE_IDS = frozenset(
    {
        "native:infernux.MeshRenderer",
        "native:infernux.SkinnedMeshRenderer",
    }
)
_MODEL_GEOMETRY_TYPE_IDS = _MODEL_RENDERER_TYPE_IDS | frozenset(
    {"native:infernux.MeshCollider"}
)


def _model_mesh_identity_map(asset_database: Any, guid: str) -> dict[str, tuple[str, ...]] | None:
    """Read the current imported mesh identity map for one model asset.

    ``None`` means the asset has no model mesh manifest and must be left to
    normal native validation.  Duplicate stable identities are an importer
    contract violation and deliberately fail instead of falling back to a
    display path.
    """
    if asset_database is None:
        return None
    meta = asset_database.get_meta_by_guid(str(guid or ""))
    if meta is None or not meta.has_key("model_meshes"):
        return None
    manifest = json.loads(meta.get_string("model_meshes"))
    if not isinstance(manifest, list):
        raise ValueError("model_meshes metadata must be a list")
    result: dict[str, tuple[str, ...]] = {}
    seen_paths: set[tuple[str, ...]] = set()
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("model_meshes metadata entries must be objects")
        identifier = str(entry.get("subresource_id") or "")
        path = entry.get("path")
        if not identifier or not isinstance(path, list) or not path:
            raise ValueError("model_meshes metadata entries require identity and path")
        current_path = tuple(str(part) for part in path)
        if any(not part for part in current_path):
            raise ValueError("model_meshes metadata paths cannot contain empty names")
        if identifier in result:
            raise ValueError(
                f"model mesh stable identity is duplicated: {identifier} (guid={guid})"
            )
        if current_path in seen_paths:
            raise ValueError(
                f"model mesh source path is duplicated: {'/'.join(current_path)} (guid={guid})"
            )
        seen_paths.add(current_path)
        result[identifier] = current_path
    return result


def _model_generates_colliders(asset_database: Any, guid: str) -> bool:
    meta = asset_database.get_meta_by_guid(str(guid or ""))
    if meta is None:
        raise ValueError(f"model source metadata is missing: {guid}")
    document = meta.serialize_document()
    metadata = document.get("metadata") if isinstance(document, dict) else None
    setting = metadata.get("generate_colliders") if isinstance(metadata, dict) else None
    return bool(setting.get("value")) if isinstance(setting, dict) else False


def _document_has_authored_content(node: dict[str, Any], source_guid: str) -> bool:
    for component in node.get("components") or ():
        if not isinstance(component, dict):
            return True
        if str(component.get("type_id") or "") not in _MODEL_GEOMETRY_TYPE_IDS:
            return True
    for child in node.get("children") or ():
        if not isinstance(child, dict):
            return True
        source = child.get("model_source")
        child_guid = str(source.get("guid") or "") if isinstance(source, dict) else ""
        if child_guid != source_guid or _document_has_authored_content(child, source_guid):
            return True
    return False


def _document_has_direct_authored_content(node: dict[str, Any]) -> bool:
    """Whether the source host itself, rather than a descendant, is authored."""
    for component in node.get("components") or ():
        if not isinstance(component, dict):
            return True
        if _component_type(component) not in _MODEL_GEOMETRY_TYPE_IDS:
            return True
    return False


def reconcile_scene_document_model_instances(
    document: Any, asset_database: Any
) -> int:
    """Reconcile imported model identities before native Scene commit.

    The native component loader remains strict: a stable identity that no
    longer exists is still invalid.  This document-wide step can, however,
    retire the complete generated scene node atomically before component
    deserialization.  A same-name node recreated in Blender is therefore a
    new object; the source-graph phase materializes it with a new object ID.

    Returns the number of document mutations.  Native Play snapshots and
    documents without an AssetDatabase are intentionally untouched.
    """
    if not isinstance(document, dict) or asset_database is None:
        return 0
    roots = document.get("objects")
    if not isinstance(roots, list):
        return 0

    manifests: dict[str, dict[str, tuple[str, ...]] | None] = {}
    changed = 0

    def manifest_for(guid: str) -> dict[str, tuple[str, ...]] | None:
        if guid not in manifests:
            manifests[guid] = _model_mesh_identity_map(asset_database, guid)
        return manifests[guid]

    def visit(nodes: list[Any]) -> None:
        nonlocal changed
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if not isinstance(node, dict):
                index += 1
                continue
            children = node.get("children")
            if isinstance(children, list):
                visit(children)

            components = node.get("components")
            if not isinstance(components, list):
                index += 1
                continue

            stale_guids: set[str] = set()
            for component in components:
                if not isinstance(component, dict):
                    continue
                if str(component.get("type_id") or "") not in _MODEL_RENDERER_TYPE_IDS:
                    continue
                data = component.get("data")
                if not isinstance(data, dict):
                    continue
                guid = str(data.get("meshAssetGuid") or "")
                identifier = str(data.get("modelSubresourceId") or "")
                if not guid or not identifier:
                    continue
                identity_map = manifest_for(guid)
                if identity_map is None:
                    continue
                current_path = identity_map.get(identifier)
                if current_path is None:
                    stale_guids.add(guid)
                    continue
                serialized_path = tuple(str(part) for part in (data.get("modelNodePath") or ()))
                if serialized_path != current_path:
                    data["modelNodePath"] = list(current_path)
                    changed += 1

            if not stale_guids:
                index += 1
                continue

            source = node.get("model_source")
            source_guid = str(source.get("guid") or "") if isinstance(source, dict) else ""
            stale_source_guid = source_guid if source_guid in stale_guids else sorted(stale_guids)[0]
            if not _document_has_authored_content(node, stale_source_guid):
                # A source mesh can be removed while one of its children is
                # moved elsewhere in the DCC hierarchy.  Children were
                # reconciled first; splice the surviving identities into the
                # old parent list so the source-graph phase can attach them to
                # their current source parent without losing authored local
                # TRS or renderer overrides.
                surviving_children = list(node.get("children") or ())
                nodes[index:index + 1] = surviving_children
                changed += 1
                continue

            # The GameObject has author-owned state.  Keep that state, but
            # retire imported geometry and its source binding.  The current
            # source node (including a same-name recreation) is materialized
            # independently by the source-graph phase before native commit.
            retained = [
                component
                for component in components
                if not (
                    isinstance(component, dict)
                    and str(component.get("type_id") or "") in _MODEL_GEOMETRY_TYPE_IDS
                )
            ]
            changed += len(components) - len(retained)
            node["components"] = retained
            if "model_source" in node:
                node.pop("model_source", None)
                changed += 1
            index += 1

    visit(roots)
    return changed


def _load_model_source_nodes(guid: str) -> tuple[dict[str, Any], ...]:
    """Return the already-preloaded authoritative model hierarchy."""
    from Infernux.core.mesh import Mesh

    mesh = Mesh.load_guid(guid)
    if mesh is None:
        raise RuntimeError(f"model source is not resident after dependency preflight: {guid}")
    nodes = tuple(dict(node) for node in mesh.model_nodes)
    if not nodes:
        raise ValueError(f"model source has no hierarchy: {guid}")
    return nodes


def _source_graph(nodes: tuple[dict[str, Any], ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Validate and index parent-before-child model node records."""
    paths: list[tuple[str, ...]] = []
    graph: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, raw in enumerate(nodes):
        name = str(raw.get("name") or "")
        parent_index = int(raw.get("parent_index", -1))
        if not name:
            raise ValueError("model source node name cannot be empty")
        if parent_index < -1 or parent_index >= index:
            raise ValueError("model source hierarchy must be parent-before-child")
        parent_path = paths[parent_index] if parent_index >= 0 else ()
        path = parent_path + (name,)
        if path in graph:
            raise ValueError(f"model source path is duplicated: {'/'.join(path)}")
        record = dict(raw)
        record["path"] = path
        record["parent_path"] = parent_path
        graph[path] = record
        paths.append(path)
    return graph


def _component_type(component: Any) -> str:
    return str(component.get("type_id") or "") if isinstance(component, dict) else ""


def _renderer_record(node: dict[str, Any]) -> dict[str, Any] | None:
    for component in node.get("components") or ():
        if _component_type(component) in _MODEL_RENDERER_TYPE_IDS:
            return component
    return None


def _source_binding(node: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    source = node.get("model_source")
    if not isinstance(source, dict):
        return "", ()
    return str(source.get("guid") or ""), tuple(str(part) for part in source.get("path") or ())


def _iter_document_nodes(nodes: list[Any]):
    for node in tuple(nodes):
        if not isinstance(node, dict):
            continue
        yield node
        children = node.get("children")
        if isinstance(children, list):
            yield from _iter_document_nodes(children)


class _DocumentIdAllocator:
    def __init__(self, document: dict[str, Any]):
        highest = 0
        for node in _iter_document_nodes(document.get("objects") or []):
            highest = max(highest, int(node.get("id") or 0))
            transform = node.get("transform")
            if isinstance(transform, dict):
                highest = max(highest, int(transform.get("component_id") or 0))
            for component in node.get("components") or ():
                if isinstance(component, dict):
                    highest = max(highest, int(component.get("component_id") or 0))
        self._next = highest + 1

    def take(self) -> int:
        value = self._next
        self._next += 1
        return value


def _matrix_trs(matrix: Any) -> tuple[list[float], list[float], list[float]]:
    """Decompose a source row-major affine matrix into current Transform TRS."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 4:
        raise ValueError("model source local_matrix must contain four rows")
    rows = [[float(value) for value in row] for row in matrix]
    if any(len(row) != 4 or any(not math.isfinite(value) for value in row) for row in rows):
        raise ValueError("model source local_matrix must be a finite 4x4 matrix")
    if any(abs(rows[3][column] - (1.0 if column == 3 else 0.0)) > 1.0e-5 for column in range(4)):
        raise ValueError("model source local_matrix must be affine")

    position = [rows[row][3] for row in range(3)]
    columns = [[rows[row][column] for row in range(3)] for column in range(3)]
    scale = [math.sqrt(sum(value * value for value in column)) for column in columns]
    if any(value <= 1.0e-8 for value in scale):
        raise ValueError("model source local_matrix has a singular scale")
    rotation = [[rows[row][column] / scale[column] for column in range(3)] for row in range(3)]
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if determinant < 0.0:
        scale[0] = -scale[0]
        for row in range(3):
            rotation[row][0] = -rotation[row][0]

    # Match glm::quat_cast followed by Transform's public XYZ Euler surface.
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * root
        qx = (rotation[2][1] - rotation[1][2]) / root
        qy = (rotation[0][2] - rotation[2][0]) / root
        qz = (rotation[1][0] - rotation[0][1]) / root
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        root = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / root
        qx = 0.25 * root
        qy = (rotation[0][1] + rotation[1][0]) / root
        qz = (rotation[0][2] + rotation[2][0]) / root
    elif rotation[1][1] > rotation[2][2]:
        root = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / root
        qx = (rotation[0][1] + rotation[1][0]) / root
        qy = 0.25 * root
        qz = (rotation[1][2] + rotation[2][1]) / root
    else:
        root = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / root
        qx = (rotation[0][2] + rotation[2][0]) / root
        qy = (rotation[1][2] + rotation[2][1]) / root
        qz = 0.25 * root
    from Infernux.lib import quatf

    euler = quatf(qx, qy, qz, qw).euler_angles
    return position, [float(euler.x), float(euler.y), float(euler.z)], scale


def _new_source_node_document(
    guid: str,
    path: tuple[str, ...],
    source: dict[str, Any],
    identity_by_path: dict[tuple[str, ...], str],
    allocator: _DocumentIdAllocator,
    generate_colliders: bool,
) -> dict[str, Any]:
    position, rotation, scale = _matrix_trs(source.get("local_matrix"))
    object_id = allocator.take()
    components: list[dict[str, Any]] = []
    node_group = int(source.get("node_group", -1))
    if node_group >= 0:
        identifier = identity_by_path.get(path)
        if not identifier:
            raise ValueError(f"model geometry node has no stable identity: {'/'.join(path)}")
        components.append(
            {
                "component_id": allocator.take(),
                "type_id": "native:infernux.MeshRenderer",
                "enabled": bool(source.get("visible", True)),
                "execution_order": 0,
                "data": {
                    "meshId": 0,
                    "materials": [],
                    "castShadows": True,
                    "receivesShadows": True,
                    "boundsMin": [0.0, 0.0, 0.0],
                    "boundsMax": [0.0, 0.0, 0.0],
                    "useInlineMesh": False,
                    "meshAssetGuid": guid,
                    "nodeGroup": node_group,
                    "modelNodePath": list(path),
                    "modelSubresourceId": identifier,
                },
            }
        )
        if generate_colliders:
            components.append(
                {
                    "component_id": allocator.take(),
                    "type_id": "native:infernux.MeshCollider",
                    "enabled": True,
                    "execution_order": 0,
                    "data": {
                        "is_trigger": False,
                        "center": [0.0, 0.0, 0.0],
                        "physic_material_guid": "",
                        "convex": False,
                    },
                }
            )
    return {
        "id": object_id,
        "name": path[-1],
        "active": True,
        "is_static": False,
        "tag": "Untagged",
        "layer": 0,
        "transform": {
            "component_id": allocator.take(),
            "type": "Transform",
            "enabled": True,
            "execution_order": 0,
            "position": position,
            "rotation": rotation,
            "scale": scale,
        },
        "components": components,
        "children": [],
        "model_source": {"guid": guid, "path": list(path)},
    }


def _detach_node(nodes: list[Any], target: dict[str, Any]) -> bool:
    for index, node in enumerate(nodes):
        if node is target:
            nodes.pop(index)
            return True
        if isinstance(node, dict) and isinstance(node.get("children"), list):
            if _detach_node(node["children"], target):
                return True
    return False


def _parent_map(root: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    def visit(parent: dict[str, Any]) -> None:
        for child in parent.get("children") or ():
            if isinstance(child, dict):
                result[id(child)] = parent
                visit(child)

    visit(root)
    return result


def _reconcile_document_instance_graph(
    root: dict[str, Any],
    guid: str,
    source_graph: dict[tuple[str, ...], dict[str, Any]],
    identity_map: dict[str, tuple[str, ...]],
    allocator: _DocumentIdAllocator,
    generate_colliders: bool,
) -> int:
    changed = 0
    parent_before = _parent_map(root)
    source_nodes = [
        node
        for node in _iter_document_nodes([root])
        if node is not root and _source_binding(node)[0] == guid
    ]
    old_path_by_node = {id(node): _source_binding(node)[1] for node in source_nodes}
    by_identity: dict[str, dict[str, Any]] = {}
    path_migrations: dict[tuple[str, ...], tuple[str, ...]] = {}
    for node in source_nodes:
        renderer = _renderer_record(node)
        data = renderer.get("data") if isinstance(renderer, dict) else None
        identifier = str(data.get("modelSubresourceId") or "") if isinstance(data, dict) else ""
        if not identifier:
            continue
        if identifier in by_identity:
            raise ValueError(f"scene model instance duplicates stable identity: {identifier}")
        by_identity[identifier] = node
        new_path = identity_map.get(identifier)
        if new_path is None:
            continue
        old_path = old_path_by_node[id(node)]
        if old_path != new_path:
            path_migrations[old_path] = new_path
            node["model_source"] = {"guid": guid, "path": list(new_path)}
            if old_path and str(node.get("name") or "") == old_path[-1]:
                node["name"] = new_path[-1]
            changed += 1
        if tuple(str(part) for part in data.get("modelNodePath") or ()) != new_path:
            data["modelNodePath"] = list(new_path)
            changed += 1

    # Geometry identity changes provide unambiguous pivot rename mappings.
    prefix_map: dict[tuple[str, ...], tuple[str, ...]] = {}
    ambiguous: set[tuple[str, ...]] = set()
    original_paths = frozenset(old_path_by_node.values())
    for old_path, new_path in path_migrations.items():
        if len(old_path) != len(new_path):
            continue
        for length in range(1, len(old_path)):
            old_prefix, new_prefix = old_path[:length], new_path[:length]
            # A cross-parent move between two existing pivots is not a pivot
            # rename. Only migrate a vanished prefix into a genuinely new one.
            if old_prefix in source_graph or new_prefix in original_paths:
                continue
            previous = prefix_map.get(old_prefix)
            if previous is not None and previous != new_prefix:
                ambiguous.add(old_prefix)
            else:
                prefix_map[old_prefix] = new_prefix
    for path in ambiguous:
        prefix_map.pop(path, None)
    for node in source_nodes:
        if _renderer_record(node) is not None:
            continue
        old_path = old_path_by_node[id(node)]
        new_path = prefix_map.get(old_path)
        if new_path is None or new_path == old_path or new_path not in source_graph:
            continue
        node["model_source"] = {"guid": guid, "path": list(new_path)}
        if old_path and str(node.get("name") or "") == old_path[-1]:
            node["name"] = new_path[-1]
        changed += 1

    current_by_path: dict[tuple[str, ...], dict[str, Any]] = {(): root}
    duplicates: set[tuple[str, ...]] = set()
    for node in source_nodes:
        current_path = _source_binding(node)[1]
        if current_path in current_by_path:
            duplicates.add(current_path)
        else:
            current_by_path[current_path] = node
    if duplicates:
        raise ValueError(
            "scene model instance duplicates source path: "
            + ", ".join("/".join(path) for path in sorted(duplicates))
        )

    # Retire deleted source nodes before additions. Authored hosts survive but
    # lose imported renderer/source ownership; generated hosts are removed and
    # their children are hoisted for the following authoritative reparent pass.
    stale = [
        node for path, node in current_by_path.items()
        if path and path not in source_graph
    ]
    for node in sorted(stale, key=lambda item: len(_source_binding(item)[1]), reverse=True):
        path = _source_binding(node)[1]
        parent = parent_before.get(id(node), root)
        parent_children = parent.get("children")
        if not isinstance(parent_children, list):
            parent_children = []
            parent["children"] = parent_children
        authored = _document_has_direct_authored_content(node)
        if authored:
            components = node.get("components") or []
            node["components"] = [
                component for component in components
                if _component_type(component) not in _MODEL_RENDERER_TYPE_IDS
            ]
            node.pop("model_source", None)
            changed += 1
        else:
            children = list(node.get("children") or ())
            try:
                position = parent_children.index(node)
            except ValueError:
                continue
            parent_children[position:position + 1] = children
            changed += 1
        current_by_path.pop(path, None)

    # Materialize every genuinely new source path directly in the candidate
    # document. No live Scene mutation is required after native commit.
    for path, source in sorted(source_graph.items(), key=lambda item: (len(item[0]), item[0])):
        if path in current_by_path:
            continue
        node = _new_source_node_document(
            guid,
            path,
            source,
            {v: k for k, v in identity_map.items()},
            allocator,
            generate_colliders,
        )
        current_by_path[path] = node
        changed += 1

    # Rebuild source-owned edges. An author-reparented node is recognized from
    # its old actual parent not matching its old source parent and stays put.
    actual_parent_by_id = _parent_map(root)
    for path, node in sorted(current_by_path.items(), key=lambda item: (len(item[0]), item[0])):
        if not path:
            continue
        old_path = old_path_by_node.get(id(node))
        if old_path is not None:
            old_parent = parent_before.get(id(node))
            old_parent_path = old_path_by_node.get(id(old_parent), ()) if old_parent is not None else ()
            if old_parent_path != old_path[:-1]:
                continue
        target_parent = current_by_path.get(path[:-1])
        if target_parent is None:
            raise ValueError(f"model source parent is missing: {'/'.join(path[:-1])}")
        actual_parent = actual_parent_by_id.get(id(node))
        if actual_parent is target_parent:
            continue
        if actual_parent is not None:
            actual_parent["children"] = [
                child for child in actual_parent["children"] if child is not node
            ]
        target_parent.setdefault("children", []).append(node)
        actual_parent_by_id[id(node)] = target_parent
        changed += 1
    return changed


def reconcile_scene_document_model_source_graphs(document: Any, asset_database: Any) -> int:
    """Atomically project current model source graphs into a scene document.

    Call this only after resource preflight: it reads the resident model's
    hierarchy and mutates the candidate document before native commit.
    """
    if not isinstance(document, dict) or asset_database is None:
        return 0
    roots = document.get("objects")
    if not isinstance(roots, list):
        return 0
    instances: list[tuple[dict[str, Any], str]] = []
    for node in _iter_document_nodes(roots):
        guid, path = _source_binding(node)
        if guid and not path:
            instances.append((node, guid))
    if not instances:
        return 0
    allocator = _DocumentIdAllocator(document)
    cached: dict[
        str,
        tuple[dict[tuple[str, ...], dict[str, Any]], dict[str, tuple[str, ...]], bool],
    ] = {}
    changed = 0
    for root, guid in instances:
        if guid not in cached:
            identity_map = _model_mesh_identity_map(asset_database, guid)
            if identity_map is None:
                raise ValueError(f"model source has no stable mesh identity manifest: {guid}")
            cached[guid] = (
                _source_graph(_load_model_source_nodes(guid)),
                identity_map,
                _model_generates_colliders(asset_database, guid),
            )
        source_graph, identity_map, generate_colliders = cached[guid]
        changed += _reconcile_document_instance_graph(
            root, guid, source_graph, identity_map, allocator, generate_colliders
        )
    return changed


def _object_path_map(root: Any) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {(): root}

    def visit(parent: Any, prefix: tuple[str, ...]) -> None:
        for child in tuple(parent.get_children() or ()):
            path = prefix + (str(child.name),)
            result[path] = child
            visit(child, path)

    visit(root, ())
    return result


def _descendants(root: Any) -> tuple[Any, ...]:
    result: list[Any] = []

    def visit(parent: Any) -> None:
        for child in tuple(parent.get_children() or ()):
            result.append(child)
            visit(child)

    visit(root)
    return tuple(result)


def _source_node_has_authored_content(obj: Any, guid: str) -> bool:
    """Return whether a removed source node carries author-owned state.

    Model creation contributes only the renderer/collider pair (and the
    optional skinned renderer).  Any other native or Python component, or an
    explicitly authored child, means the GameObject is part of the scene
    authoring rather than disposable imported geometry.
    """
    generated = {"Transform", "MeshRenderer", "SkinnedMeshRenderer", "MeshCollider"}
    for component in tuple(obj.get_components() or ()):
        if type(component).__name__ not in generated:
            return True
    if tuple(obj.get_py_components() or ()):
        return True
    for child in _descendants(obj):
        if str(getattr(child, "_model_source_guid", "") or "") != guid:
            return True
    return False


def _retire_source_geometry(obj: Any) -> None:
    """Detach imported geometry while retaining an author-owned GameObject."""
    # MeshCollider derives its geometry from the sibling renderer.  Retaining
    # it after the imported mesh identity disappears creates a permanently
    # invalid component that cannot cook a shape.  Imported geometry therefore
    # retires as one renderer/collider unit while unrelated authored state and
    # the GameObject's Transform remain intact.
    for component_type in ("MeshCollider", "MeshRenderer", "SkinnedMeshRenderer"):
        component = obj.get_component(component_type)
        if component is not None:
            obj.remove_component(component)
    obj._set_model_source("", [])


def _model_instance_roots(scene: Any, guid: str) -> tuple[Any, ...]:
    """Find current model containers from persisted GUID source identities."""
    return tuple(
        obj
        for obj in tuple(scene.get_all_objects() or ())
        if str(getattr(obj, "_model_source_guid", "") or "") == guid
        and not tuple(getattr(obj, "_model_source_path", ()) or ())
    )


def _source_paths(mesh: Any) -> dict[tuple[str, ...], dict]:
    nodes = tuple(mesh.model_nodes or ())
    # Keep the node path construction identical to the native CreateModelObject
    # contract: paths are source names, never scene object IDs or transforms.
    result: dict[tuple[str, ...], dict] = {}
    computed: list[tuple[str, ...]] = []
    for node in nodes:
        parent_index = int(node.get("parent_index", -1))
        prefix = computed[parent_index] if parent_index >= 0 else ()
        path = prefix + (str(node["name"]),)
        computed.append(path)
        result[path] = node
    return result


def _source_subresource_ids(guid: str) -> dict[tuple[str, ...], str]:
    """Return unique imported mesh identities for the current model source.

    The importer owns this manifest. Missing or duplicate identities are
    rejected rather than guessed from mutable scene paths.
    """
    from Infernux.core.assets import AssetManager

    database = AssetManager.require_asset_database()
    meta = database.get_meta_by_guid(guid)
    if meta is None or not meta.has_key("model_meshes"):
        return {}
    manifest = json.loads(meta.get_string("model_meshes"))
    if not isinstance(manifest, list):
        raise ValueError("model_meshes metadata must be a list")
    entries: list[tuple[tuple[str, ...], str]] = []
    seen: set[str] = set()
    seen_paths: set[tuple[str, ...]] = set()
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("model_meshes metadata entries must be objects")
        path = entry.get("path")
        identifier = str(entry.get("subresource_id") or "")
        if not isinstance(path, list) or not path or not identifier:
            raise ValueError("model_meshes metadata entries require identity and path")
        key = tuple(str(part) for part in path)
        if any(not part for part in key):
            raise ValueError("model_meshes metadata paths cannot contain empty names")
        if identifier in seen:
            raise ValueError(
                f"model mesh stable identity is duplicated: {identifier} (guid={guid})"
            )
        if key in seen_paths:
            raise ValueError(
                f"model mesh source path is duplicated: {'/'.join(key)} (guid={guid})"
            )
        seen.add(identifier)
        seen_paths.add(key)
        entries.append((key, identifier))
    return dict(entries)


def _migrate_source_paths_by_identity(scene: Any, root: Any, guid: str, source_ids: dict[tuple[str, ...], str]) -> bool:
    """Move a live model hierarchy's source bindings across unambiguous renames.

    The object itself is retained, so authored TRS, materials and user-added
    components remain intact.  A user-renamed GameObject keeps its display
    name; only its hidden source binding follows the DCC path.
    """
    if not source_ids:
        return False
    candidates = (root, *_descendants(root))
    by_id: dict[str, Any] = {}
    for obj in candidates:
        if str(getattr(obj, "_model_source_guid", "") or "") != guid:
            continue
        renderer = obj.get_component("MeshRenderer")
        if renderer is None:
            renderer = obj.get_component("SkinnedMeshRenderer")
        identifier = str(getattr(renderer, "model_subresource_id", "") or "") if renderer else ""
        if identifier and identifier in source_ids.values() and identifier not in by_id:
            by_id[identifier] = obj

    # A path-depth change is a real source reparent, not a rename.  Only
    # same-depth identity pairs may project their prefixes onto imported empty
    # pivots; otherwise an old pivot could collide with the moved mesh path.
    direct_maps: dict[tuple[str, ...], tuple[str, ...]] = {}
    same_depth_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for new_path, identifier in source_ids.items():
        obj = by_id.get(identifier)
        if obj is None:
            continue
        old_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        if not old_path or old_path == new_path:
            continue
        direct_maps[old_path] = new_path
        if len(old_path) != len(new_path):
            continue
        same_depth_pairs.append((old_path, new_path))

    prefix_maps: dict[tuple[str, ...], tuple[str, ...]] = {}
    for old_path, new_path in same_depth_pairs:
        for length in range(1, len(old_path) + 1):
            old_prefix = old_path[:length]
            new_prefix = new_path[:length]
            previous = prefix_maps.get(old_prefix)
            if previous is not None and previous != new_prefix:
                prefix_maps.clear()
                break
            prefix_maps[old_prefix] = new_prefix
        if not prefix_maps:
            break
    if not prefix_maps and not direct_maps:
        return False

    # Snapshot source paths and parent edges before changing either.  The
    # source path is the only identity we can use for imported pivots (they do
    # not have a mesh subresource id), while the live parent edge tells us
    # whether an author deliberately reparented the object in the scene.
    old_paths_by_id = {
        int(obj.id): tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        for obj in candidates
        if str(getattr(obj, "_model_source_guid", "") or "") == guid
    }
    old_parents_by_id = {int(obj.id): obj.get_parent() for obj in candidates}
    changed = False
    for obj in candidates:
        if str(getattr(obj, "_model_source_guid", "") or "") != guid:
            continue
        old_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        new_path = prefix_maps.get(old_path)
        if new_path is None:
            new_path = direct_maps.get(old_path)
        if new_path is None or new_path == old_path:
            continue
        if old_path and str(obj.name) == old_path[-1]:
            obj.name = new_path[-1]
        obj._set_model_source(guid, list(new_path))
        for renderer_type in ("MeshRenderer", "SkinnedMeshRenderer"):
            renderer = obj.get_component(renderer_type)
            if renderer is not None and renderer.model_node_path == list(old_path):
                renderer.set_model_mesh(guid, list(new_path))
        changed = True

    # Rebuild only edges that still match the old source edge.  If a scene
    # author moved an imported child under another GameObject, the live parent
    # no longer carries the old source path and therefore remains authoritative.
    # This lets a DCC cross-parent move update imported structure without
    # trampling authored hierarchy edits.
    source_by_new_path: dict[tuple[str, ...], Any] = {}
    duplicate_new_paths: set[tuple[str, ...]] = set()
    for obj in candidates:
        if str(getattr(obj, "_model_source_guid", "") or "") != guid:
            continue
        new_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        if new_path in source_by_new_path:
            duplicate_new_paths.add(new_path)
        else:
            source_by_new_path[new_path] = obj
    for path in duplicate_new_paths:
        source_by_new_path.pop(path, None)
    for obj in candidates:
        object_id = int(obj.id)
        old_path = old_paths_by_id.get(object_id)
        new_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        if old_path is None or old_path == new_path or not old_path:
            continue
        previous_parent = old_parents_by_id.get(object_id)
        if previous_parent is None:
            continue
        previous_parent_path = old_paths_by_id.get(int(previous_parent.id))
        if previous_parent_path != old_path[:-1]:
            continue
        # The destination parent is defined by the migrated source path, so a
        # cross-parent move naturally targets ``new_path[:-1]``.  For a pure
        # rename this is equivalent to the migrated old parent path.
        target_parent_path = new_path[:-1]
        target_parent = source_by_new_path.get(target_parent_path)
        if target_parent is None or target_parent is previous_parent:
            continue
        obj.set_parent(target_parent, world_position_stays=False)
        changed = True

    return changed


def _destroy_stale_geometry(
    scene: Any,
    root: Any,
    guid: str,
    source: dict[tuple[str, ...], dict],
    source_ids: dict[tuple[str, ...], str],
) -> bool:
    paths = _object_path_map(root)
    source_objects: dict[tuple[str, ...], Any] = {}
    for candidate in (root, *_descendants(root)):
        if str(getattr(candidate, "_model_source_guid", "") or "") != guid:
            continue
        source_path = tuple(str(part) for part in (getattr(candidate, "_model_source_path", ()) or ()))
        source_objects[source_path] = candidate
    stale: dict[tuple[str, ...], Any] = {}
    if source_objects:
        current_ids = frozenset(source_ids.values())
        for path, obj in tuple(source_objects.items()):
            if not path:
                continue
            renderer = obj.get_component("MeshRenderer")
            if renderer is None:
                renderer = obj.get_component("SkinnedMeshRenderer")
            identifier = str(getattr(renderer, "model_subresource_id", "") or "") if renderer else ""
            # Identity owns model geometry.  A DCC node deleted and recreated
            # with the same display path is a new object, not a continuation
            # of the old scene instance.
            if identifier and identifier not in current_ids:
                stale[path] = obj
                continue
            if path not in source:
                stale[path] = obj
                continue
            source_node = source[path]
            has_renderer = obj.get_component("MeshRenderer") is not None or obj.get_component(
                "SkinnedMeshRenderer"
            ) is not None
            if int(source_node.get("node_group", -1)) < 0 and has_renderer:
                stale[path] = obj
        if not stale:
            return False
        # Destroy only the highest stale source node. Its descendants belong
        # to the same removed source subtree and are retired with it.
        selected = []
        stale_paths = set(stale)
        for path, obj in sorted(stale.items(), key=lambda item: (len(item[0]), item[0])):
            if any(path[:index] in stale_paths for index in range(1, len(path))):
                continue
            selected.append(obj)
        if selected:
            from Infernux.debug import Debug

            for obj in selected:
                source_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
                if _source_node_has_authored_content(obj, guid):
                    Debug.log_warning(
                        "Model source node was removed; retaining author-owned instance "
                        f"'{obj.name}' and retiring imported geometry at source path "
                        f"'{('/'.join(source_path))}' (guid={guid})"
                    )
                    _retire_source_geometry(obj)
                else:
                    Debug.log_warning(
                        "Model source node was removed from the imported asset; "
                        f"retiring instance '{obj.name}' at source path "
                        f"'{('/'.join(source_path))}' (guid={guid})"
                    )
        for obj in selected:
            if obj is not root and not _source_node_has_authored_content(obj, guid):
                scene.destroy_game_object(obj)
        scene.process_pending_destroys()
        return True
    for path, obj in tuple(paths.items()):
        if not path:
            continue
        renderer = obj.get_component("MeshRenderer")
        if renderer is None:
            renderer = obj.get_component("SkinnedMeshRenderer")
        if renderer is None:
            continue
        if not str(renderer.mesh_asset_guid or ""):
            continue
        if str(renderer.mesh_asset_guid) != guid:
            # This branch is only a defensive guard for user-authored children;
            # the caller filters instances by source GUID.
            continue
        source_node = source.get(path)
        if source_node is None or int(source_node.get("node_group", -1)) < 0:
            stale[path] = obj
            continue
        # A removed source pivot must take its old geometry with it.  Destroy
        # the first missing ancestor, not each descendant independently.
        prefix: tuple[str, ...] = ()
        for part in path:
            prefix += (part,)
            if prefix not in source:
                stale[prefix] = paths.get(prefix, obj)
                break
    if not stale:
        return False
    from Infernux.debug import Debug

    for path in sorted(stale):
        Debug.log_warning(
            "Model source node was removed from the imported asset; "
            f"retiring instance source path '{('/'.join(path))}' (guid={guid})"
        )
    for obj in stale.values():
        if obj is not root:
            scene.destroy_game_object(obj)
    scene.process_pending_destroys()
    return True


def _sync_instance(scene: Any, root: Any, guid: str, mesh: Any, scale_ratio: float = 1.0) -> bool:
    source = _source_paths(mesh)
    source_ids = _source_subresource_ids(guid)
    changed = _migrate_source_paths_by_identity(scene, root, guid, source_ids)
    changed = _destroy_stale_geometry(scene, root, guid, source, source_ids) or changed
    source_bound = {
        tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ())): obj
        for obj in (root, *_descendants(root))
        if str(getattr(obj, "_model_source_guid", "") or "") == guid
    }
    existing = source_bound or _object_path_map(root)
    missing = [path for path in source if path not in existing]
    if abs(float(scale_ratio) - 1.0) > 1.0e-7:
        from Infernux.lib import Vector3

        # Scale Factor is an importer-space operation. Geometry was already
        # republished at the new size, so apply the same ratio to every
        # existing source-node offset. The scene-authored container remains
        # the pivot and is deliberately not moved.
        for path, obj in existing.items():
            if not path or path in missing or str(getattr(obj, "_model_source_guid", "") or "") != guid:
                continue
            position = obj.transform.local_position
            obj.transform.local_position = Vector3(
                float(position.x) * scale_ratio,
                float(position.y) * scale_ratio,
                float(position.z) * scale_ratio,
            )
            changed = True
    if not missing:
        return changed

    # Native model creation already performs TRS decomposition and creates the
    # correct renderer/material bindings.  Use it as the source-node factory,
    # then move only missing nodes into the existing instance.  Existing nodes
    # and all authored transforms/components remain untouched.
    temporary = scene.create_from_model(guid, "__InfernuxModelRefresh__")
    if temporary is None:
        raise RuntimeError(f"cannot stage refreshed model instance for {guid}")
    staged = _object_path_map(temporary)
    try:
        for path in sorted(missing, key=lambda item: (len(item), item)):
            source_object = staged.get(path)
            if source_object is None:
                continue
            parent_path = path[:-1]
            parent = root if not parent_path else existing.get(parent_path)
            if parent is None:
                raise RuntimeError(f"model source parent is missing: {parent_path}")
            source_object.set_parent(parent, world_position_stays=False)
            existing[path] = source_object
            changed = True
    finally:
        # Existing source nodes stayed under the temporary container and are
        # discarded. Newly moved nodes are now owned by the real instance.
        scene.destroy_game_object(temporary)
        scene.process_pending_destroys()
    return changed


def synchronize_model_instances(scene: Any, guid: str, *, scale_ratio: float = 1.0) -> int:
    """Reconcile every live instance of *guid* in one loaded scene.

    Returns the number of instance roots changed.  The function is intentionally
    owner-thread only and is called after AssetDatabase publication.
    """
    guid = str(guid or "")
    if not guid or scene is None:
        return 0
    from Infernux.core.mesh import Mesh

    mesh = Mesh.load_guid(guid)
    if mesh is None or not mesh.model_nodes:
        return 0
    changed = 0
    for root in _model_instance_roots(scene, guid):
        if _sync_instance(scene, root, guid, mesh, scale_ratio):
            changed += 1
    return changed


def _mark_scene_dirty(scene: Any) -> None:
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.engine.scene_manager import SceneFileManager

    manager = SceneFileManager.instance()
    if manager is None or manager._is_play_mode():
        return
    document_id = manager.document_id_for_scene(scene)
    if document_id:
        DocumentRegistry.instance().mark_changed(document_id)


def synchronize_loaded_model_instances(guid: str, *, scale_ratio: float = 1.0) -> int:
    """Synchronize all currently resident editor scenes for one model GUID."""
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.lib import SceneManager

    editor_manager = SceneFileManager.instance()
    if editor_manager is not None and editor_manager._is_play_mode():
        return 0
    manager = SceneManager.instance()
    total = 0
    for index in range(int(manager.scene_count)):
        scene = manager.get_scene_at(index)
        changed = synchronize_model_instances(scene, guid, scale_ratio=scale_ratio)
        if changed:
            _mark_scene_dirty(scene)
        total += changed
    return total


def synchronize_scene_model_instances(scene: Any, *, mark_dirty: bool = True) -> int:
    """Reconcile every compound model source referenced by one loaded scene."""
    if scene is None:
        return 0
    from Infernux.engine.scene_manager import SceneFileManager

    editor_manager = SceneFileManager.instance()
    if editor_manager is not None and editor_manager._is_play_mode():
        return 0
    guids: set[str] = set()
    for obj in tuple(scene.get_all_objects() or ()):
        source_guid = str(getattr(obj, "_model_source_guid", "") or "")
        source_path = tuple(getattr(obj, "_model_source_path", ()) or ())
        if source_guid and not source_path:
            guids.add(source_guid)
        for renderer_type in ("MeshRenderer", "SkinnedMeshRenderer"):
            renderer = obj.get_component(renderer_type)
            if renderer is not None and renderer.model_node_path and renderer.mesh_asset_guid:
                guids.add(str(renderer.mesh_asset_guid))
    total = 0
    for guid in sorted(guids):
        changed = synchronize_model_instances(scene, guid)
        if changed and mark_dirty:
            _mark_scene_dirty(scene)
        total += changed
    return total
