"""Static author declarations lowered to the shared native Slang compiler.

The first vertical slice is a fullscreen fragment with typed parameters and
pure expressions. General shader domains and control-flow analysis build on
this source model; unsupported author syntax is diagnosed at its source line.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .shader_library import LibraryGraph, ShaderEmitter, declarations


@dataclass(frozen=True)
class ShaderParameter:
    name: str
    kind: str
    default: object
    label: str
    range: tuple[float, float] | None = None
    hdr: bool = False


@dataclass(frozen=True)
class ShaderDefinition:
    name: str
    source_path: str
    module_name: str
    parameters: tuple[ShaderParameter, ...]
    generated_source: str
    libraries: tuple[tuple[str, str, str], ...] = ()

    def compile(self, targets=("spirv", "wgsl")):
        from infernux.lib import _Infernux
        return _Infernux._compile_shader_program(
            self.generated_source, self.source_path, self.module_name, self.libraries, targets,
        )


_PARAMETER_TYPES = {"float": "float", "int": "int", "bool": "uint", "color": "float4"}
_DEFAULTS = {"float": 0.0, "int": 0, "bool": False, "color": (1.0, 1.0, 1.0, 1.0)}


class _Author(ShaderEmitter):
    def __init__(self, source: str, path: str):
        super().__init__(source, path)
        self.source = source
        self.alias = ""
        self.parameters = {}

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self.qualified(node.value) + "." + node.attr
        self.error(node, "Expected a static shader declaration name")

    def literal(self, node):
        if isinstance(node, ast.Call) and self.qualified(node.func) in {
            self.alias + ".vec2", self.alias + ".vec3", self.alias + ".vec4",
        }:
            return tuple(self.literal(value) for value in node.args)
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            self.error(node, "Shader parameter defaults and presentation metadata must be literal values")

    def parameter(self, statement):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            self.error(statement, "Declare one named shader parameter per field")
        call = statement.value
        if not isinstance(call, ast.Call):
            self.error(statement, "Expected shader.parameter declaration")
        declaration = self.qualified(call.func)
        prefix = self.alias + ".parameter."
        if not declaration.startswith(prefix) or declaration[len(prefix):] not in _PARAMETER_TYPES:
            self.error(call, "Unsupported shader parameter declaration")
        kind = declaration[len(prefix):]
        values = {keyword.arg: self.literal(keyword.value) for keyword in call.keywords}
        if call.args:
            self.error(call, "Declare the parameter's default with default=...")
        unknown = values.keys() - {"default", "label", "range", "hdr"}
        if unknown:
            self.error(call, "Unknown shader parameter metadata: " + ", ".join(sorted(unknown)))
        name = statement.targets[0].id
        value = ShaderParameter(name, kind, values.get("default", _DEFAULTS[kind]), values.get("label", name),
                                values.get("range"), values.get("hdr", False))
        self.parameters[name] = value
        self.types["self." + name] = "bool" if kind == "bool" else _PARAMETER_TYPES[kind]

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr not in self.parameters:
                self.error(node, "Unknown shader parameter: " + node.attr)
            name = "inx_parameters." + node.attr
            return "(" + name + " != 0u)" if self.parameters[node.attr].kind == "bool" else name
        return super().visit_Attribute(node)

    def function(self, main):
        if main.decorator_list or main.args.vararg or main.args.kwarg or main.args.kwonlyargs or main.args.defaults:
            self.error(main, "Shader entry parameters have a fixed typed signature")
        arguments = main.args.args
        if not arguments or arguments[0].arg != "self":
            self.error(main, "Shader main is an instance method")
        if len(arguments) > 2:
            self.error(main, "Fullscreen main accepts self and an optional vec2 UV")
        signature = ""
        if len(arguments) == 2:
            uv = arguments[1]
            if uv.annotation is None or self.qualified(uv.annotation) != self.alias + ".vec2":
                self.error(uv, "Fullscreen UV must be annotated as shader.vec2")
            signature = "float2 " + uv.arg + " : TEXCOORD0"
            self.types[uv.arg] = "float2"
        if main.returns is None or self.qualified(main.returns) != self.alias + ".vec4":
            self.error(main, "Fullscreen main returns shader.vec4")
        self.aliases = (self.alias,)
        self.declare_locals(main.body)
        self.local_declarations()
        self.block(main.body)
        return ['[shader("fragment")]', "float4 fragment_main(" + signature + ") : SV_Target0", "{", *self.lines, "}"]


_FULLSCREEN_VERTEX = """
struct FullscreenVertex { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[shader("vertex")]
FullscreenVertex vertex_main(uint id : SV_VertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    FullscreenVertex result;
    result.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    result.uv = uv;
    return result;
}
"""


def parse_shader(source: str, source_path: str, module_name: str) -> ShaderDefinition:
    author = _Author(source, source_path)
    author.alias, imports, body = declarations(source, source_path)
    graph = LibraryGraph()
    author.functions, dependencies = graph.imports(imports, author)
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(lineno=1, col_offset=0), "Declare one shader class")
    declaration = body[0]
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) != author.alias + ".fullscreen":
        author.error(declaration, "The first authoring slice supports shader.fullscreen")
    main, helpers = None, []
    for statement in declaration.body:
        if isinstance(statement, ast.Assign):
            author.parameter(statement)
        elif isinstance(statement, ast.FunctionDef) and statement.name == "main" and main is None:
            main = statement
        elif isinstance(statement, ast.FunctionDef) and statement.name != "main":
            author.functions["self." + statement.name] = author.signature(statement, statement.name, method=True)
            helpers.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "Expected a shader parameter or typed method")
    if main is None:
        author.error(declaration, "Shader class needs a main entry")
    code = [*("import " + name + ";" for name in dependencies), _FULLSCREEN_VERTEX]
    if author.parameters:
        code.append("struct ShaderParameters {")
        code.extend("    " + _PARAMETER_TYPES[field.kind] + " " + field.name + ";" for field in author.parameters.values())
        code.extend(["};", "ConstantBuffer<ShaderParameters> inx_parameters;"])
    for helper in helpers:
        emitter = _Author(source, source_path)
        emitter.alias = author.alias
        emitter.aliases = (author.alias,)
        emitter.parameters = author.parameters
        emitter.types = author.types.copy()
        emitter.functions = author.functions
        code.extend(emitter.emit_function(helper, author.functions["self." + helper.name], method=True))
    code.extend(author.function(main))
    return ShaderDefinition(declaration.name, source_path, module_name, tuple(author.parameters.values()),
                            "\n".join(code), graph.sources())


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)
