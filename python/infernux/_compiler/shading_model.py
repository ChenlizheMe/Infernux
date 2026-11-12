"""Typed surface-lighting responses compiled without executing host Python."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

from .shader_library import ShaderEmitter, declarations
from .slang_ast import qualified
from .shader_info import INFO_DEFAULTS, info_field, read_shader_info


def _builtin_root():
    from infernux.resources import resources_path

    return Path(resources_path) / "shaders"
_LIGHTING = {
    "lighting.pbr": ("inx_shade_pbr<tiled_lighting>", "float3", ("InxSurfaceData", "InxShadingContext")),
    "lighting.pbr_indirect": ("inx_pbr_indirect", "float3", ("InxSurfaceData", "InxShadingContext")),
    "lighting.main_light": ("inx_main_light<tiled_lighting>", "InxLight", ("InxShadingContext", "float3")),
    "lighting.view_direction": ("inx_view_direction", "float3", ("InxShadingContext",)),
    "lighting.direct": ("inx_author_direct<tiled_lighting>", "float3", ("InxSurfaceData", "InxShadingContext")),
    "lighting.ambient": ("inx_ambient_irradiance", "float3", ("InxShadingContext", "float3")),
}


_DIRECT_RESPONSE = """
public struct InxAuthorLightResponse<let tiled_lighting : bool> : IInxLightResponse
{
    InxSurfaceData surface;
    InxShadingContext context;
    public float3 evaluate(InxLight light)
    {
        return inx_shading_model::shade_light<tiled_lighting>(surface, context, light);
    }
};
float3 inx_author_direct<let tiled_lighting : bool>(InxSurfaceData surface, InxShadingContext context)
{
    InxAuthorLightResponse<tiled_lighting> response;
    response.surface = surface;
    response.context = context;
    return inx_direct_lighting<tiled_lighting, InxAuthorLightResponse<tiled_lighting>>(
        context, normalize(surface.normal_ws), response);
}
"""


class _ModelEmitter(ShaderEmitter):
    def __init__(self, *args, contract, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = contract

    def infer(self, node):
        if isinstance(node, ast.Call) and self.symbol(qualified(node.func)) in _LIGHTING:
            return _LIGHTING[self.symbol(qualified(node.func))][1]
        return super().infer(node)

    def visit_Call(self, node):
        operation = self.symbol(qualified(node.func))
        if operation not in _LIGHTING:
            return super().visit_Call(node)
        if self.contract == "unlit":
            self.error(node, "An unlit shading model does not request lighting resources", code="author.capability")
        symbol, _, arguments = _LIGHTING[operation]
        if node.keywords or len(node.args) != len(arguments):
            self.error(node, "Lighting arguments must match the typed signature", code="author.type")
        for value, expected in zip(node.args, arguments):
            if self.infer(value) != expected:
                self.error(value, "Lighting argument requires " + expected, code="author.type")
        return symbol + "(" + ", ".join(self.visit(value) for value in node.args) + ")"


@dataclass(frozen=True)
class ShadingModel:
    name: str
    path: str
    contract: str
    source: str
    dependencies: tuple[str, ...]
    source_ranges: tuple
    deferred_model: str | None
    info: dict


def read_shading_model(path, graph, owner, location, *, source=None):
    path = Path(path).resolve()
    if source is None:
        try:
            source = graph.reader(str(path))
        except FileNotFoundError:
            source = None
        if source is None:
            owner.error(location, "Shading model source does not exist: " + str(path), code="author.reference")
    source = source.removeprefix("\ufeff")
    alias, imports, body = declarations(source, str(path))
    author = ShaderEmitter(source, str(path), aliases=(alias,))
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(value=None, lineno=1, col_offset=0), "Declare one shader.shadingmodel class")
    declaration = body[0]
    if declaration.decorator_list or len(declaration.bases) != 1 or qualified(declaration.bases[0]) != alias + ".shadingmodel":
        author.error(declaration, "Declare a shader.shadingmodel class")
    contract, methods = None, []
    for statement in declaration.body:
        if info_field(statement) in INFO_DEFAULTS:
            continue
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and qualified(statement.targets[0]) == "lighting_contract" and contract is None):
            if not isinstance(statement.value, ast.Constant) or statement.value.value not in {
                "standard_pbr", "unlit", "custom_response",
            }:
                author.error(statement, "lighting_contract is standard_pbr, unlit or custom_response", code="author.capability")
            contract = statement.value.value
        elif isinstance(statement, ast.FunctionDef):
            methods.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "Shading models contain a lighting_contract and typed methods")
    if contract is None:
        author.error(declaration, "Shading models declare their lighting_contract", code="author.capability")
    functions, dependencies = graph.imports(imports, author)
    for method in methods:
        name = "self." + method.name
        if name in functions:
            author.error(method, "Duplicate shading model method: " + method.name)
        functions[name] = replace(author.signature(method, "inx_shading_model::" + method.name, method=True),
                                  lighting_specialization=True)
    entry = functions.get("self.shade")
    if entry is None or entry.arguments != ("InxSurfaceData", "InxShadingContext") or entry.result != "float3":
        author.error(declaration, "shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) returns shader.vec3",
                     code="author.type")
    direct_calls = [node for method in methods for node in ast.walk(method)
                    if isinstance(node, ast.Call) and qualified(node.func) == alias + ".lighting.direct"]
    if direct_calls:
        response = functions.get("self.shade_light")
        if (response is None or response.arguments != ("InxSurfaceData", "InxShadingContext", "InxLight")
                or response.result != "float3"):
            author.error(direct_calls[0], "lighting.direct requires shade_light(self, surface: shader.SurfaceData, "
                         "ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3", code="author.type")
    code = ["import inx_geometry;", *("import " + name + ";" for name in dependencies)]
    if contract != "unlit":
        code.append("import inx_lighting;")
    code.append("namespace inx_shading_model {")
    ranges = []
    for method in methods:
        emitter = _ModelEmitter(source, str(path), aliases=(alias,), functions=functions, contract=contract)
        code.extend(emitter.emit_function(method, functions["self." + method.name], method=True, public=True))
        ranges.extend(emitter.source_ranges)
    code.append("}")
    if direct_calls:
        code.append(_DIRECT_RESPONSE)
    builtin_models = {_builtin_root().joinpath(name + ".shadingmodel.py").resolve(): model
                      for name, model in (("pbr", "PBR"), ("unlit", "Unlit"), ("toon", "Toon"))}
    return ShadingModel(declaration.name, str(path), contract, "\n".join(code), dependencies, tuple(ranges),
                        builtin_models.get(path), read_shader_info(source, str(path)))


def default_shading_model():
    return _builtin_root() / "pbr.shadingmodel.py"
