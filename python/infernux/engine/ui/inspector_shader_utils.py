"""Inspector material fields and references using the shared typed source catalog."""

import os
from infernux.engine.path_utils import portable_path
from infernux.core._shader_catalog import (
    _is_builtin_shader_path, _matches_stage, _read_compiled_shader_metadata,
    bump_shader_property_generation, get_shader_property_generation,
    get_shader_candidates, get_shader_file_path, is_shader_hidden,
    parse_shader_id, shader_menu_folders, shader_picker_name,
)


# Only published material fields are cached here; discovery is core-owned.
_shader_properties_cache_generation = -1
_shader_properties_cache: dict[tuple[str, ...], tuple[str, int, list | None]] = {}


def _get_shader_property_record(reference, ext: str):
    """Resolve the imported GUID; only name-only references use the catalog.

    None means an unavailable stage, distinct from a valid empty schema.
    Native publication versions also invalidate fields when a scene worker
    accepts a program without passing through the Python asset facade.
    """
    from infernux.engine.project_context import get_project_root
    from infernux.lib import AssetRegistry
    registry = AssetRegistry.instance()
    guid = str(reference.get("guid", "") or "").strip() if isinstance(reference, dict) else ""
    shader_id = shader_ref_id(reference)
    if not shader_id and isinstance(reference, dict):
        shader_id = str(reference.get("builtin", "") or "").strip()
    global _shader_properties_cache_generation
    generation = get_shader_property_generation()
    if generation != _shader_properties_cache_generation:
        _shader_properties_cache.clear()
        _shader_properties_cache_generation = generation
    cache_key = (get_project_root() or "", "guid" if guid else "name", guid or shader_id, ext)
    cached = _shader_properties_cache.get(cache_key)
    if cached is not None and registry.get_asset_version(cached[0]) == cached[1]:
        return cached
    database = registry.get_asset_database()
    if guid:
        shader_path = database.get_path_from_guid(guid) if database else None
    else:
        shader_path = get_shader_file_path(shader_id, ext)
    source_guid = guid or (database.get_guid_from_path(shader_path) if database and shader_path else "")
    props = (parse_shader_properties(shader_path)
             if shader_path and _matches_stage(shader_path, ext) else None)
    record = (source_guid, registry.get_asset_version(source_guid), props)
    _shader_properties_cache[cache_key] = record
    return record


def _get_shader_properties_cached(reference, ext: str) -> list | None:
    return _get_shader_property_record(reference, ext)[2]


def get_shader_property_revision(reference, ext: str):
    return _get_shader_property_record(reference, ext)[:2] if reference else ("", 0)


def parse_shader_properties(filepath: str) -> list:
    """Read the accepted native schema, never an uncompiled source draft."""
    from infernux.core.shader import Shader
    from infernux._compiler.shader_parameters import material_properties
    schema = Shader.get_schema(filepath)
    return material_properties(schema["parameters"]) if schema["domain"] in {"mesh", "vertex", "particle", "screen_ui", "world_ui"} else []


def shader_ref_id(value) -> str:
    """Return the compiler shader ID from a catalog value or structured reference."""
    if isinstance(value, dict):
        shader_id = value.get("shader_id", "")
        return shader_id.strip() if isinstance(shader_id, str) else ""
    return value.strip() if isinstance(value, str) else ""


def make_shader_reference(value, ext: str) -> dict[str, str]:
    """Build a canonical material shader reference.

    ``value`` may be a catalog shader ID, an asset path, or an existing
    reference. GUID and path are enriched whenever imported metadata is
    available; built-in ID-only references remain valid.
    """
    structured = isinstance(value, dict)
    existing = value if structured else {}
    guid = existing.get("guid", "") if isinstance(existing.get("guid", ""), str) else ""
    shader_id = shader_ref_id(value)
    if not shader_id and structured and isinstance(existing.get("builtin", ""), str):
        shader_id = existing.get("builtin", "").strip()
    # A structured value already crossed the editor assignment boundary. Its
    # old path_hint is ignored and current display data is resolved by GUID.
    path_hint = ""

    if isinstance(value, str):
        candidate = value.strip()
        if _matches_stage(candidate, ext) or os.path.isfile(candidate):
            path_hint = candidate
            # A picker/drop payload is a source location, never a shader ID.
            # Resolve ShaderInfo.Name below so material pipeline keys remain
            # stable and never contain project-specific absolute paths.
            shader_id = ""
        else:
            shader_id = candidate

    from infernux.lib import AssetRegistry
    database = AssetRegistry.instance().get_asset_database()

    resolved_path = ""
    if guid:
        # A persisted GUID owns identity. A missing asset stays missing even
        # when another project or builtin stage has the same display name.
        if database:
            resolved_path = database.get_path_from_guid(guid) or ""
    else:
        if not structured and path_hint and os.path.isfile(path_hint):
            resolved_path = path_hint
        if not resolved_path:
            # Name-only references are builtin IDs or legacy authoring input.
            resolved_path = get_shader_file_path(shader_id, ext) or ""

    if resolved_path:
        resolved_path = portable_path(resolved_path)
        metadata = _read_compiled_shader_metadata(resolved_path)
        if metadata is not None:
            imported_id = metadata.get("shader_id", "")
            imported_guid = metadata.get("guid", "")
            if isinstance(imported_id, str) and imported_id.strip():
                shader_id = imported_id.strip()
            if isinstance(imported_guid, str) and imported_guid.strip():
                guid = imported_guid.strip()
        if not shader_id:
            shader_id = parse_shader_id(resolved_path) or ""
        if not guid and database:
            guid = database.get_guid_from_path(resolved_path) or ""
        path_hint = resolved_path

    if _is_builtin_shader_path(path_hint):
        # Engine shaders are addressed by ShaderInfo.Name. Persisting a
        # generated Library or package path makes materials machine-specific
        # and leaks the implementation path into pipeline IDs.
        guid = ""
        path_hint = ""

    return {
        "guid": guid,
        "shader_id": shader_id,
        "path_hint": path_hint,
    }


def shader_display_from_value(value, items):
    """Map a shader value to its display string for UI."""
    shader_id = shader_ref_id(value)
    for display, v in items:
        if v == shader_id:
            return display
    return shader_id


_SHADER_TYPE_MAP = {
    'Float': 0,
    'Float2': 1,
    'Float3': 2,
    'Float4': 3,
    'Color': 7,
    'Int': 4,
    'Mat4': 5,
    'Texture2D': 6,
    'Bool': 10,
    'UInt': 11,
    'Mat3': 12,
}


_PROPERTY_VALUE_DEFAULTS = {
    "Float": 0.0,
    "Float2": [0.0, 0.0],
    "Float3": [0.0, 0.0, 0.0],
    "Float4": [0.0, 0.0, 0.0, 0.0],
    "Color": [1.0, 1.0, 1.0, 1.0],
    "Int": 0,
    "UInt": 0,
    "Bool": False,
    "Mat3": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "Mat4": [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ],
}


def _coerce_property_value(property_type: str, value):
    """Return a canonical material value, or ``None`` when incompatible."""
    try:
        if property_type == "Float":
            if isinstance(value, (list, tuple, dict, bool)):
                return None
            return float(value)
        if property_type == "Bool":
            return value if isinstance(value, bool) else None
        if property_type == "UInt":
            return value if type(value) is int and 0 <= value <= 0xFFFFFFFF else None
        if property_type == "Int":
            if isinstance(value, (list, tuple, dict, bool)):
                return None
            return int(value)

        lengths = {
            "Float2": 2,
            "Float3": 3,
            "Float4": 4,
            "Color": 4,
            "Mat4": 16,
            "Mat3": 9,
        }
        expected = lengths.get(property_type)
        if expected is None or not isinstance(value, (list, tuple)) or len(value) != expected:
            return None
        return [float(component) for component in value]
    except (TypeError, ValueError, OverflowError):
        return None


def _apply_shader_props_to_mat(mat_data: dict, all_props: list[dict],
                                remove_unknown: bool = False):
    """Apply a merged list of imported shader property dicts to mat_data.

    Shared implementation for `sync_properties_from_shader` and
    `sync_all_shader_properties`.
    """
    props = mat_data.setdefault("properties", {})

    seen_names: set[str] = set()
    ordered_names: list[str] = []
    for sp in all_props:
        name = sp.get('name', '')
        if name and name not in seen_names:
            ordered_names.append(name)
            seen_names.add(name)
    mat_data["_shader_property_order"] = ordered_names

    shader_prop_names: set[str] = set()
    for sp in all_props:
        name = sp.get('name', '')
        ptype_str = sp.get('type', 'Float')
        default = sp.get('default')
        hdr = sp.get('hdr', False)
        internal = bool(sp.get('internal', False))
        authored_range = sp.get('range')

        if not name:
            continue

        shader_prop_names.add(name)
        ptype = _SHADER_TYPE_MAP.get(ptype_str, 0)

        existing = props.get(name)
        if not isinstance(existing, dict):
            existing = {}
            props[name] = existing

        existing['type'] = ptype
        existing['hdr'] = hdr
        if sp.get('label'):
            existing['label'] = sp['label']
        else:
            existing.pop('label', None)
        if internal:
            existing['internal'] = True
        else:
            existing.pop('internal', None)
        if ptype == 6:
            guid = existing.get('guid', '')
            existing['guid'] = guid if isinstance(guid, str) else ''
            existing.pop('value', None)
        else:
            fallback = _coerce_property_value(ptype_str, default)
            if fallback is None:
                fallback = _coerce_property_value(
                    ptype_str, _PROPERTY_VALUE_DEFAULTS.get(ptype_str)
                )
            current = _coerce_property_value(ptype_str, existing.get('value'))
            existing['value'] = current if current is not None else fallback
            existing.pop('guid', None)

        if (
            ptype in (0, 4)
            and isinstance(authored_range, (list, tuple))
            and len(authored_range) == 2
        ):
            if ptype == 4:
                props[name]['range'] = [int(authored_range[0]), int(authored_range[1])]
            else:
                props[name]['range'] = [float(authored_range[0]), float(authored_range[1])]
        else:
            props[name].pop('range', None)

    if remove_unknown:
        for k in [k for k in props if k not in shader_prop_names]:
            del props[k]


def sync_properties_from_shader(mat_data: dict, shader_id: str, ext: str,
                                remove_unknown: bool = False):
    """Sync material properties from ShaderInfo declarations.
    Adds new properties from shader, keeps existing values if property exists.
    If *remove_unknown* is True, removes properties not defined in shader.
    """
    shader_props = _get_shader_properties_cached(shader_id, ext)
    if shader_props is None:
        # An unavailable stage must preserve the last published schema.
        return
    _apply_shader_props_to_mat(mat_data, shader_props, remove_unknown=remove_unknown)


def _merged_shader_properties(vertex, fragment) -> list[dict] | None:
    all_props = []
    has_stage = False
    for reference, ext in ((vertex, ".vert"), (fragment, ".frag")):
        if not reference or (isinstance(reference, dict) and not any(
                reference.get(key) for key in ("guid", "shader_id", "builtin"))):
            continue
        has_stage = True
        properties = _get_shader_properties_cached(reference, ext)
        if properties is None:
            return None
        all_props.extend(properties)
    return all_props if has_stage else None


def sync_all_shader_properties(mat_data: dict, vert_shader_id, frag_shader_id,
                               remove_unknown: bool = False):
    """Sync material properties from both vertex and fragment shader annotations.

    Merges properties from both shader stages. Vertex properties appear
    first in the display order, followed by fragment properties.
    If *remove_unknown* is True, removes properties not defined in either shader.
    """
    all_props = _merged_shader_properties(vert_shader_id, frag_shader_id)
    if all_props is not None:
        _apply_shader_props_to_mat(mat_data, all_props, remove_unknown=remove_unknown)


def get_all_shader_property_names(vert_shader_id, frag_shader_id) -> list[str]:
    """Return all declared material property names from the active vertex and fragment shaders."""
    ordered_names: list[str] = []
    seen_names: set[str] = set()

    for sp in _merged_shader_properties(vert_shader_id, frag_shader_id) or ():
        name = sp.get("name", "")
        if name and name not in seen_names:
            ordered_names.append(name)
            seen_names.add(name)

    return ordered_names


def get_material_property_display_order(mat_data: dict) -> list[str]:
    """Return material properties in shader declaration order only.

    Properties not declared in the shader (phantom / stale) are excluded.
    """
    props = mat_data.get("properties", {})
    if not props:
        return []

    shader_order = mat_data.get("_shader_property_order", [])
    shader_set = set(shader_order) if shader_order else None

    ordered = []
    seen = set()
    for name in shader_order:
        if (
            name in props
            and name not in seen
            and not bool(props[name].get("internal", False))
        ):
            ordered.append(name)
            seen.add(name)

    # Only include extras if there is no shader metadata (e.g. unloaded shader)
    if shader_set is None:
        for name in sorted(props.keys()):
            if name not in seen:
                ordered.append(name)

    return ordered
