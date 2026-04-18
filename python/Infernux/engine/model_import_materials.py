"""Create lit ``.mat`` assets from model ``.meta`` ``import_materials`` (Assimp) and assign to MeshRenderers."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from Infernux.debug import Debug

_LIT_MAT_TEMPLATE: Dict[str, Any] = {
    "name": "ImportedLit",
    "builtin": False,
    "shaders": {"vertex": "standard", "fragment": "lit"},
    "renderState": {
        "cullMode": 2,
        "frontFace": 1,
        "polygonMode": 0,
        "depthTestEnable": True,
        "depthWriteEnable": True,
        "depthCompareOp": 1,
        "blendEnable": False,
        "srcColorBlendFactor": 6,
        "dstColorBlendFactor": 7,
        "colorBlendOp": 0,
        "renderQueue": 2000,
    },
    "properties": {
        "baseColor": {"type": 7, "value": [1.0, 1.0, 1.0, 1.0]},
        "metallic": {"type": 0, "value": 0.0},
        "smoothness": {"type": 0, "value": 0.5},
        "ambientOcclusion": {"type": 0, "value": 1.0},
        "emissionColor": {"type": 7, "value": [0.0, 0.0, 0.0, 0.0]},
        "normalScale": {"type": 0, "value": 1.0},
        "specularHighlights": {"type": 0, "value": 1.0},
        "texSampler": {"type": 6, "guid": ""},
        "metallicMap": {"type": 6, "guid": ""},
        "smoothnessMap": {"type": 6, "guid": ""},
        "aoMap": {"type": 6, "guid": ""},
        "normalMap": {"type": 6, "guid": ""},
    },
}


def _parse_import_materials(meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not meta:
        return []
    raw = meta.get("import_materials")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _sanitize_filename(s: str) -> str:
    s = (s or "Material").strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:64] if s else "Material"


def _resolve_texture_guid(mesh_dir: str, rel: str, adb) -> str:
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or rel.startswith("*"):
        return ""
    # Strip Assimp/FBX prefix paths like "../textures/foo.png"
    clean = rel.lstrip("./")
    if os.path.isabs(clean):
        abs_tex = os.path.normpath(clean)
    else:
        abs_tex = os.path.normpath(os.path.join(mesh_dir, clean))
    if not os.path.isfile(abs_tex):
        return ""
    try:
        g = adb.get_guid_from_path(abs_tex) or ""
        if g:
            return g
        return adb.import_asset(abs_tex) or ""
    except Exception as _exc:
        Debug.log(f"[Suppressed] texture import {abs_tex}: {type(_exc).__name__}: {_exc}")
        return ""


def _lit_mat_dict(spec: Dict[str, Any], mesh_dir: str, adb, albedo_guid: str) -> Dict[str, Any]:
    root = copy.deepcopy(_LIT_MAT_TEMPLATE)
    name = str(spec.get("name") or "ImportedLit")
    root["name"] = name
    bc = spec.get("baseColor") or [1, 1, 1, 1]
    if isinstance(bc, (list, tuple)) and len(bc) >= 3:
        a = float(bc[3]) if len(bc) > 3 else 1.0
        root["properties"]["baseColor"]["value"] = [
            float(bc[0]), float(bc[1]), float(bc[2]), a,
        ]
    root["properties"]["metallic"]["value"] = float(spec.get("metallic", 0.0))
    root["properties"]["smoothness"]["value"] = float(spec.get("smoothness", 0.5))
    root["properties"]["ambientOcclusion"]["value"] = float(spec.get("ambientOcclusion", 1.0))

    ec = spec.get("emissionColor") or [0.0, 0.0, 0.0, 0.0]
    if isinstance(ec, (list, tuple)) and len(ec) >= 3:
        ea = float(ec[3]) if len(ec) > 3 else 1.0
        root["properties"]["emissionColor"]["value"] = [
            float(ec[0]), float(ec[1]), float(ec[2]), ea,
        ]

    mr_path = str(spec.get("metallicRoughnessTexturePath", "") or "")
    mr_g = _resolve_texture_guid(mesh_dir, mr_path, adb) if mr_path else ""
    met_g = _resolve_texture_guid(mesh_dir, str(spec.get("metallicTexturePath", "") or ""), adb)
    rough_g = _resolve_texture_guid(mesh_dir, str(spec.get("roughnessTexturePath", "") or ""), adb)
    if not met_g and mr_g:
        met_g = mr_g
    if not rough_g and mr_g:
        rough_g = mr_g

    norm_g = _resolve_texture_guid(mesh_dir, str(spec.get("normalTexturePath", "") or ""), adb)
    ao_g = _resolve_texture_guid(mesh_dir, str(spec.get("occlusionTexturePath", "") or ""), adb)

    root["properties"]["texSampler"]["guid"] = albedo_guid or ""
    root["properties"]["metallicMap"]["guid"] = met_g or ""
    root["properties"]["smoothnessMap"]["guid"] = rough_g or ""
    root["properties"]["normalMap"]["guid"] = norm_g or ""
    root["properties"]["aoMap"]["guid"] = ao_g or ""
    return root


def ensure_imported_material_files(mesh_abs_path: str, adb) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Write or update ``{stem}_ImportedMaterials/*.mat`` and return (guids, specs) in mesh slot order."""
    from Infernux.core.asset_types import read_meta_file
    from Infernux.core.assets import AssetManager

    specs = _parse_import_materials(read_meta_file(mesh_abs_path))
    if not specs or not mesh_abs_path:
        return [], specs

    stem = os.path.splitext(os.path.basename(mesh_abs_path))[0]
    mesh_dir = os.path.dirname(os.path.abspath(mesh_abs_path))
    out_dir = os.path.join(mesh_dir, f"{stem}_ImportedMaterials")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as _exc:
        Debug.log_warning(f"ImportedMaterials folder: {_exc}")
        return [], specs

    guids: List[str] = []
    for i, spec in enumerate(specs):
        safe = _sanitize_filename(str(spec.get("name", f"slot_{i}")))
        mat_path = os.path.join(out_dir, f"{i:02d}_{safe}.mat")
        tex_guid = _resolve_texture_guid(mesh_dir, str(spec.get("albedoTexturePath", "")), adb)
        data = _lit_mat_dict(spec, mesh_dir, adb, tex_guid)
        try:
            with open(mat_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        except OSError as _exc:
            Debug.log_warning(f"Write material {mat_path}: {_exc}")
            guids.append("")
            continue
        AssetManager.reimport_asset(mat_path)
        try:
            guids.append(adb.get_guid_from_path(mat_path) or "")
        except Exception:
            guids.append("")
    return guids, specs


def _mesh_slots_for_renderer(mesh, node_group: int) -> List[int]:
    """Map MeshRenderer material slot index -> mesh material slot index (matches C++ node-group logic)."""
    try:
        n = int(mesh.submesh_count)
    except Exception:
        n = 0
    if node_group < 0:
        try:
            c = int(mesh.material_slot_count)
        except Exception:
            c = 0
        return list(range(max(c, 1)))

    slots: List[int] = []
    seen = set()
    for i in range(n):
        try:
            info = mesh.get_submesh_info(i)
        except Exception:
            continue
        if int(info.get("node_group", -1)) != int(node_group):
            continue
        s = int(info.get("material_slot", 0))
        if s not in seen:
            seen.add(s)
            slots.append(s)
    # C++ uses std::set<uint32_t> iteration → sorted order
    slots_sorted = sorted(slots)
    return slots_sorted if slots_sorted else [0]


def _iter_game_object_tree(root):
    yield root
    try:
        children = list(root.get_children())
    except Exception:
        return
    for ch in children:
        yield from _iter_game_object_tree(ch)


def apply_imported_materials_to_model_instance(root_go, mesh_guid: str) -> None:
    """Assign per-slot ``.mat`` GUIDs from ``import_materials`` meta for a hierarchy from ``create_from_model``."""
    if root_go is None or not mesh_guid:
        return
    try:
        from Infernux.lib import AssetRegistry
        registry = AssetRegistry.instance()
        adb = registry.get_asset_database() if registry else None
    except Exception:
        adb = None
    if adb is None:
        return
    try:
        mesh_path = adb.get_path_from_guid(mesh_guid) or ""
    except Exception:
        mesh_path = ""
    if not mesh_path or not os.path.isfile(mesh_path):
        return

    slot_guids, specs = ensure_imported_material_files(mesh_path, adb)
    if not slot_guids or not specs:
        return

    for obj in _iter_game_object_tree(root_go):
        try:
            cpp = obj.get_cpp_component("MeshRenderer")
        except Exception:
            cpp = None
        if cpp is None or not getattr(cpp, "has_mesh_asset", False):
            continue
        try:
            if cpp.mesh_asset_guid != mesh_guid:
                continue
        except Exception:
            continue
        mesh = cpp.get_mesh_asset()
        if mesh is None:
            continue
        try:
            ng = int(getattr(cpp, "node_group", -1))
        except Exception:
            ng = -1
        order = _mesh_slots_for_renderer(mesh, ng)
        guids: List[str] = []
        for mesh_slot in order:
            if 0 <= mesh_slot < len(slot_guids):
                g = slot_guids[mesh_slot]
                if g:
                    guids.append(g)
        try:
            need = int(cpp.material_count)
        except Exception:
            need = len(guids)
        if need <= 0 or len(guids) != need or not all(guids):
            continue
        try:
            cpp.set_materials(guids)
        except Exception as _exc:
            Debug.log_warning(f"set_materials failed: {_exc}")
