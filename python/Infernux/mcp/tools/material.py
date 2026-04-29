"""Material MCP tools."""

from __future__ import annotations

from typing import Any

from Infernux.mcp.tools.common import main_thread, notify_asset_changed, register_tool_metadata, resolve_project_path, serialize_value


def register_material_tools(mcp, project_path: str) -> None:
    _register_metadata()

    @mcp.tool(name="material.create")
    def material_create(path: str, template: str = "lit", overwrite: bool = False, properties: dict[str, Any] | None = None) -> dict:
        """Create a material asset and optionally set properties."""

        def _create():
            import os
            from Infernux.core.material import Material
            file_path = resolve_project_path(project_path, path)
            if os.path.exists(file_path) and not overwrite:
                raise FileExistsError(f"Material already exists: {path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            name = os.path.splitext(os.path.basename(file_path))[0]
            mat = Material.create_unlit(name) if str(template).lower() == "unlit" else Material.create_lit(name)
            _set_properties(mat, properties or {})
            mat.save(file_path)
            notify_asset_changed(file_path, "created")
            return {"path": os.path.relpath(file_path, project_path).replace("\\", "/"), "properties": _properties(mat)}

        return main_thread("material.create", _create)

    @mcp.tool(name="material.get_properties")
    def material_get_properties(path: str) -> dict:
        """Read material properties."""

        def _get():
            import os
            mat = _load_material(project_path, path)
            return {"path": os.path.relpath(resolve_project_path(project_path, path), project_path).replace("\\", "/"), "properties": _properties(mat)}

        return main_thread("material.get_properties", _get)

    @mcp.tool(name="material.set_property")
    def material_set_property(path: str, name: str, value: Any, value_type: str = "auto") -> dict:
        """Set one material property."""

        def _set():
            file_path = resolve_project_path(project_path, path)
            mat = _load_material(project_path, path)
            _set_one(mat, name, value, value_type)
            mat.flush()
            mat.save(file_path)
            notify_asset_changed(file_path, "modified")
            return {"path": path, "name": name, "value": serialize_value(mat.get_property(name))}

        return main_thread("material.set_property", _set)


def _load_material(project_path: str, path: str):
    from Infernux.core.material import Material
    file_path = resolve_project_path(project_path, path)
    mat = Material.load(file_path)
    if mat is None:
        raise FileNotFoundError(f"Material not found or failed to load: {path}")
    return mat


def _set_properties(mat, properties: dict[str, Any]) -> None:
    for name, value in properties.items():
        _set_one(mat, name, value, "auto")


def _set_one(mat, name: str, value: Any, value_type: str) -> None:
    kind = str(value_type or "auto").lower()
    if kind == "float" or (kind == "auto" and isinstance(value, float)):
        mat.set_float(name, float(value))
    elif kind == "int" or (kind == "auto" and isinstance(value, int) and not isinstance(value, bool)):
        mat.set_int(name, int(value))
    elif kind == "color" or (kind == "auto" and isinstance(value, (list, tuple)) and len(value) == 4):
        mat.set_color(name, float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    elif kind == "vector2" or (kind == "auto" and isinstance(value, (list, tuple)) and len(value) == 2):
        mat.set_vector2(name, float(value[0]), float(value[1]))
    elif kind == "vector3" or (kind == "auto" and isinstance(value, (list, tuple)) and len(value) == 3):
        mat.set_vector3(name, float(value[0]), float(value[1]), float(value[2]))
    elif kind == "texture":
        mat.set_texture(name, value)
    else:
        mat.set_param(name, value)


def _properties(mat) -> dict[str, Any]:
    try:
        return serialize_value(mat.get_all_properties())
    except Exception:
        return {}


def _register_metadata() -> None:
    for name, summary in {
        "material.create": "Create a material asset.",
        "material.get_properties": "Read material properties.",
        "material.set_property": "Set a material shader property.",
    }.items():
        register_tool_metadata(name, summary=summary)
