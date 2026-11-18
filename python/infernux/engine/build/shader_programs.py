"""Cook the material programs selected by the resolved Player scenes."""

import json
from pathlib import Path

from infernux._compiler.shader_library import using_cooked_shader_sources
from infernux._source_kinds import source_extension
from infernux.core.asset_document import read_asset_document
from infernux.engine.runtime_artifact_catalog import _asset_refs, logical_asset_type, logical_type_for_path
from infernux.lib import _Infernux


_IMPLICIT_FRAGMENT = {
    "MeshRenderer": "infernux/lit", "SkinnedMeshRenderer": "infernux/lit",
    "SpriteRenderer": "infernux/unlit", "LineRenderer": "infernux/unlit", "TrailRenderer": "infernux/unlit",
}


def stage_scene_shader_programs(data_root, catalog_path, source_entries, scenes, resource_root, build_cache, web):
    data_root, resource_root = Path(data_root), Path(resource_root)
    records = {r["guid"]: r for r in json.loads(Path(catalog_path).read_bytes())["entries"]}
    paths, shader_paths, shader_ids = {}, {}, {}
    for guid, record in records.items():
        primary = next(a for a in record["runtime_artifacts"]
                       if a["runtime_artifact_id"] == record["primary_runtime_artifact_id"])
        path = primary["runtime_path"]
        paths[guid] = (data_root / path if primary["package"] == "Content.inxpkg"
                       else resource_root / path.removeprefix("infernux/resources/"))
        metadata = record["metadata"].get("metadata", {})
        name = metadata.get("shader_id", {}).get("value")
        if name:
            key = name, source_extension(path)
            # A GUID remains usable when two authors have the same symbolic ID.
            shader_ids[key] = guid if key not in shader_ids else None
            shader_paths[key] = str(paths[guid]) if shader_ids[key] else None

    selections = set()

    def shader_guid(ref, suffix):
        if ref.get("guid"):
            return ref["guid"]
        name = ref.get("shader_id")
        if not name:
            return ""
        guid = shader_ids.get((name, suffix))
        if not guid:
            raise ValueError(f"Player material shader is missing or ambiguous: {name} ({suffix})")
        return guid

    def select(fragment, vertex):
        frag = shader_guid(fragment, ".frag.py")
        vert = shader_guid(vertex, ".vert.py")
        selections.add((frag, ""))
        if vert:
            selections.add((vert, ""))
            if vert != shader_ids.get(("infernux/standard", ".vert.py")):
                selections.add((frag, vert))

    def material(document):
        stages = document["shaders"]
        select(stages["fragment"], stages["vertex"])

    def select_builtin(fragment, vertex="infernux/standard"):
        select({"shader_id": fragment}, {"shader_id": vertex})

    def visit_object(obj):
        for component in obj.get("components", ()):
            kind = component["type_id"].removeprefix("native:infernux.")
            if kind in _IMPLICIT_FRAGMENT:
                for slot in component["data"].get("materials", ()) or (None,):
                    if slot is None:
                        select_builtin(_IMPLICIT_FRAGMENT[kind])
                    elif isinstance(slot, dict):
                        material(slot["material"])
        for child in obj.get("children", ()):
            visit_object(child)

    pending = list(scenes)
    seen = set()
    while pending:
        guid = pending.pop()
        if guid in seen or guid not in records:
            continue
        seen.add(guid)
        entry = source_entries[guid]
        pending.extend(entry.get("dependencies", ()))
        kind = logical_type_for_path(records[guid]["runtime_path"])
        if kind not in {"material", "scene", "prefab"}:
            kind = logical_asset_type(entry)
        if kind == "particlegraph":
            particle = json.loads(paths[guid].read_bytes())
            for emitter in particle["hir"]["emitters"]:
                for output in emitter["render_plan"]:
                    if output["shader"]:
                        select({"shader_id": output["shader"]}, {})
            continue
        if kind not in {"material", "scene", "prefab"}:
            continue
        document = read_asset_document(str(paths[guid]))
        if kind == "material":
            material(document)
            continue
        pending.extend(ref for ref, _ in _asset_refs(document))
        if kind == "scene":
            sky = document.get("environment", {}).get("skyboxMaterialGuid")
            if sky:
                pending.append(sky)
            else:
                select_builtin("infernux/skybox/procedural", "infernux/skybox/vertex")
            for obj in document["objects"]:
                visit_object(obj)
        else:
            visit_object(document["root_object"])

    if selections:
        with using_cooked_shader_sources(resource_root, shader_paths):
            _Infernux._cook_shader_author_programs(str(data_root), str(resource_root), str(build_cache),
                                                  str(catalog_path), web, sorted(selections))
