"""
Prefab override diff system.

Compares a live prefab instance hierarchy against its source .prefab asset
to compute property-level overrides. Supports apply (write overrides back
to the .prefab file) and revert (reset instance to match the prefab).

Identification strategy:
  Nodes are matched by *name-path* (e.g. "Root/Child/GrandChild") since
  instance GameObjects get fresh IDs on instantiation. Name-path is stable
  as long as the user does not rename nodes — an acceptable trade-off for
  this iteration of the override system.
"""

import json
import os
from typing import Dict, List, Optional

from Infernux.debug import Debug


# ─── Public data types ────────────────────────────────────────────────────

class Override:
    """A single property-level override on one node."""
    __slots__ = ("node_path", "key", "prefab_value", "instance_value")

    def __init__(self, node_path: str, key: str, prefab_value, instance_value):
        self.node_path = node_path
        self.key = key
        self.prefab_value = prefab_value
        self.instance_value = instance_value

    def __repr__(self):
        return f"Override({self.node_path!r}, {self.key!r})"


# ─── Core diff ────────────────────────────────────────────────────────────

_SKIP_KEYS = frozenset({
    "id", "schema_version", "children", "components", "py_components",
    "transform", "prefab_guid", "prefab_root",
})

_TRANSFORM_KEYS = ("position", "rotation", "scale")


def compute_overrides(instance_obj, prefab_path: str,
                      asset_database=None) -> List[Override]:
    """Compare *instance_obj* (live GameObject) against the .prefab file.

    Returns a list of Override objects describing every property difference.
    """
    prefab_data = _load_prefab_root(prefab_path)
    if prefab_data is None:
        return []

    instance_data = _serialize_obj(instance_obj)
    if instance_data is None:
        return []

    overrides: List[Override] = []
    _diff_node(instance_data, prefab_data, "", overrides)
    return overrides


def apply_overrides_to_prefab(instance_obj, prefab_path: str,
                               asset_database=None) -> bool:
    """Write the current instance state back to the .prefab file.

    Resets the instance to non-overridden state (the prefab file now
    matches the instance).
    """
    instance_data = _serialize_obj(instance_obj)
    if instance_data is None:
        Debug.log_error("Failed to serialize instance for apply.")
        return False

    # Strip prefab linkage fields from what we write to disk
    from Infernux.engine.prefab_manager import _strip_prefab_fields
    _strip_prefab_fields(instance_data)

    try:
        with open(prefab_path, "r", encoding="utf-8") as f:
            prefab_file = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        Debug.log_error(f"Failed to read prefab for apply: {exc}")
        return False

    prefab_file["root_object"] = instance_data

    try:
        with open(prefab_path, "w", encoding="utf-8") as f:
            json.dump(prefab_file, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        Debug.log_error(f"Failed to write prefab: {exc}")
        return False

    Debug.log_internal(f"Applied overrides to prefab: {os.path.basename(prefab_path)}")
    return True


def revert_overrides(instance_obj, prefab_path: str,
                     asset_database=None) -> bool:
    """Reset the instance hierarchy to match the source .prefab file.

    Preserves the instance's transform (position in scene) and its
    prefab linkage fields.
    """
    prefab_data = _load_prefab_root(prefab_path)
    if prefab_data is None:
        Debug.log_error("Failed to load prefab for revert.")
        return False

    # Keep current instance transform (user may have moved the object)
    try:
        current_json = json.loads(instance_obj.serialize())
        current_transform = current_json.get("transform")
    except Exception:
        current_transform = None

    # Keep prefab linkage
    prefab_guid = getattr(instance_obj, 'prefab_guid', '')

    # Stamp prefab linkage into the template
    from Infernux.engine.prefab_manager import _stamp_prefab_guid
    if prefab_guid:
        _stamp_prefab_guid(prefab_data, prefab_guid, is_root=True)

    # Restore transform
    if current_transform:
        prefab_data["transform"] = current_transform

    try:
        instance_obj.deserialize(json.dumps(prefab_data))
    except Exception as exc:
        Debug.log_error(f"Failed to deserialize during revert: {exc}")
        return False

    Debug.log_internal("Reverted prefab instance to source.")
    return True


# ─── Internal helpers ─────────────────────────────────────────────────────

def _load_prefab_root(prefab_path: str) -> Optional[dict]:
    """Load and return the root_object dict from a .prefab file."""
    if not prefab_path or not os.path.isfile(prefab_path):
        return None
    try:
        with open(prefab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("root_object")
    except (OSError, json.JSONDecodeError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None


def _serialize_obj(obj) -> Optional[dict]:
    """Serialize a live GameObject to a dict."""
    try:
        return json.loads(obj.serialize())
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None


def _diff_node(instance: dict, prefab: dict, path: str,
               out: List[Override]):
    """Recursively diff one node."""
    node_name = instance.get("name", "")
    current_path = f"{path}/{node_name}" if path else node_name

    # Compare top-level scalar properties
    for key in set(instance.keys()) | set(prefab.keys()):
        if key in _SKIP_KEYS:
            continue
        iv = instance.get(key)
        pv = prefab.get(key)
        if iv != pv:
            out.append(Override(current_path, key, pv, iv))

    # Compare transform sub-keys
    i_transform = instance.get("transform", {})
    p_transform = prefab.get("transform", {})
    for tk in _TRANSFORM_KEYS:
        iv = i_transform.get(tk)
        pv = p_transform.get(tk)
        if iv != pv:
            out.append(Override(current_path, f"transform.{tk}", pv, iv))

    # Compare components by type name matching
    _diff_components(
        instance.get("components", []),
        prefab.get("components", []),
        current_path, "components", out,
    )
    _diff_components(
        instance.get("py_components", []),
        prefab.get("py_components", []),
        current_path, "py_components", out,
    )

    # Recurse children (match by index → name)
    i_children = instance.get("children", [])
    p_children = prefab.get("children", [])
    p_by_name = {c.get("name"): c for c in p_children}

    for i_child in i_children:
        child_name = i_child.get("name", "")
        p_child = p_by_name.get(child_name)
        if p_child is None:
            out.append(Override(current_path, f"added_child:{child_name}", None, child_name))
        else:
            _diff_node(i_child, p_child, current_path, out)

    for p_child in p_children:
        child_name = p_child.get("name", "")
        i_names = {c.get("name") for c in i_children}
        if child_name not in i_names:
            out.append(Override(current_path, f"removed_child:{child_name}", child_name, None))


def _diff_components(instance_comps: list, prefab_comps: list,
                     node_path: str, section: str,
                     out: List[Override]):
    """Diff component lists by type_name matching."""
    type_key = "type" if section == "components" else "py_type_name"

    p_by_type: Dict[str, dict] = {}
    for c in prefab_comps:
        tn = c.get(type_key, "")
        if tn:
            p_by_type[tn] = c

    for ic in instance_comps:
        tn = ic.get(type_key, "")
        if not tn:
            continue
        pc = p_by_type.get(tn)
        if pc is None:
            out.append(Override(node_path, f"added_{section}:{tn}", None, tn))
            continue
        # Compare fields within this component
        skip = {"type", "py_type_name", "component_id", "script_guid", "enabled"}
        for key in set(ic.keys()) | set(pc.keys()):
            if key in skip:
                continue
            if ic.get(key) != pc.get(key):
                out.append(Override(node_path, f"{section}:{tn}.{key}",
                                   pc.get(key), ic.get(key)))

    i_types = {c.get(type_key, "") for c in instance_comps}
    for pc in prefab_comps:
        tn = pc.get(type_key, "")
        if tn and tn not in i_types:
            out.append(Override(node_path, f"removed_{section}:{tn}", tn, None))
