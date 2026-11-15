"""GPU asset discovery and creation shared by local automation transports."""

from infernux._source_kinds import SHADER_SOURCE_KINDS
from infernux.core.shader import Shader
from infernux.lib import AssetRegistry

from .asset_operations import _create_asset
from .operation_support import asset_database, on_editor, operation
from .operations import Operation, OperationKind


def build_shader_operations(project_path: str) -> tuple[Operation, ...]:
    stages = tuple(kind[0] for kind in SHADER_SOURCE_KINDS.values())
    return (
        operation(
            "infernux.shader.list", OperationKind.QUERY,
            "List imported GPU assets by registered shader ID, GUID, stage, domain and visibility. Reads the native catalog without compiling sources or creating pipelines.",
            _list, capability="asset.read",
            input_properties={"query": {"type": "string", "default": ""},
                              "stage": {"type": "string", "enum": ["", *stages], "default": ""},
                              "include_hidden": {"type": "boolean", "default": False},
                              "offset": {"type": "integer", "minimum": 0, "default": 0},
                              "limit": {"type": "integer", "minimum": 1, "default": 50}},
            tags=("shader", "catalog", "guid", "着色器"),
        ),
        operation(
            "infernux.shader.library.search", OperationKind.QUERY,
            "Find typed ShaderLib exports in the current engine and project. Returns source signatures, documentation and per-function fragment requirements from the engine frontend; does not execute Python author files.",
            _library_search, capability="asset.read",
            input_properties={"query": {"type": "string", "default": ""},
                              "asset": {"type": "string", "default": ""},
                              "offset": {"type": "integer", "minimum": 0, "default": 0},
                              "limit": {"type": "integer", "minimum": 1, "default": 50}},
            tags=("shader", "shaderlib", "function", "library", "函数库"),
        ),
        operation(
            "infernux.shader.create", OperationKind.COMMAND,
            "Create and import a static GPU asset with the Project creation service and shared Undo. Vertex, fragment/surface, shading-model and library templates use Python GPU types and portable filenames.",
            lambda directory, name, stage="fragment": _create_shader(project_path, directory, name, stage),
            capability="asset.write",
            input_properties={"directory": {"type": "string"}, "name": {"type": "string"},
                              "stage": {"type": "string", "enum": ["vertex", "fragment", "shadingmodel", "library"], "default": "fragment"}},
            required=("directory", "name"), reversible=True,
            side_effects=("Creates and imports a GPU author source through asset history.",),
            tags=("shader", "shaderlib", "create", "authoring", "着色器"),
        ),
    )


def _catalog():
    database = asset_database()
    records = []
    for guid in database.get_all_guids():
        meta = database.get_meta_by_guid(guid)
        if meta is None or not meta.has_key("shader_source_kind"):
            continue
        records.append({"guid": guid, "shader_id": meta.get_string("shader_id"),
                        "path": database.get_path_from_guid(guid), "stage": meta.get_string("shader_source_kind"),
                        "domain": meta.get_string("shader_domain"), "hidden": meta.get_bool("shader_hidden")})
    return sorted(records, key=lambda record: (record["shader_id"], record["stage"], record["path"]))


def _list(query="", stage="", include_hidden=False, offset=0, limit=50):
    def read():
        terms = query.casefold().split()
        matches = [record for record in _catalog()
                   if (not stage or record["stage"] == stage) and (include_hidden or not record["hidden"])
                   and all(term in (record["shader_id"] + " " + record["path"]).casefold() for term in terms)]
        return {"assets": matches[offset:offset + limit], "total": len(matches), "offset": offset}
    return on_editor("infernux.shader.list", read)


def _library_search(query="", asset="", offset=0, limit=50):
    def read():
        if asset:
            database, path, guid = Shader._author_source(asset)
            schema = Shader.get_schema(asset)
            record = {"guid": guid, "shader_id": schema["info"]["shader_id"], "path": path}
            records = ((record, schema),)
        else:
            records = ((record, None) for record in _catalog() if record["stage"] == "library")
        terms = query.casefold().split()
        registry = AssetRegistry.instance()
        matches = []
        for record, schema in records:
            if schema is None:
                schema = Shader.get_schema(record["guid"])
            if schema["domain"] != "library":
                raise ValueError("Shader library search requires a .shaderlib.py asset: " + record["path"])
            for function in schema["functions"]:
                if function["name"].startswith("_"):
                    continue
                text = " ".join((record["shader_id"], function["signature"], function["documentation"])).casefold()
                if all(term in text for term in terms):
                    matches.append(dict(function, asset_guid=record["guid"], shader_id=record["shader_id"],
                                        path=record["path"], asset_version=registry.get_asset_version(record["guid"])))
        return {"functions": matches[offset:offset + limit], "total": len(matches), "offset": offset}
    return on_editor("infernux.shader.library.search", read)


def _create_shader(project_path, directory, name, stage):
    suffix = next(suffix for suffix, kind in SHADER_SOURCE_KINDS.items() if kind[0] == stage)
    variant = suffix.removeprefix(".").removesuffix(".py")
    return _create_asset(project_path, "shader", directory, name, variant)
