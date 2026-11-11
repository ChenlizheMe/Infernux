"""Static GPU author fields, typed inputs and functions lowered to native Slang."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .shader_library import LibraryGraph, ShaderEmitter, declarations
from .shader_types import RESOURCE_MODULE, RESOURCE_TYPES


@dataclass(frozen=True)
class ShaderParameter:
    name: str
    kind: str
    default: object
    label: str
    range: tuple[float, float] | None = None
    hdr: bool = False


@dataclass(frozen=True)
class ShaderInput:
    name: str
    semantic: str
    kind: str
    domain: str | None = None


@dataclass(frozen=True)
class ShaderDefinition:
    name: str
    source_path: str
    module_name: str
    parameters: tuple[ShaderParameter, ...]
    generated_source: str
    libraries: tuple[tuple[str, str, str], ...] = ()
    inputs: tuple[ShaderInput, ...] = ()
    source_ranges: tuple = ()

    def compile(self, targets=("spirv", "wgsl")):
        from infernux.lib import _Infernux
        return _Infernux._compile_shader_program(
            self.generated_source, self.source_path, self.module_name, self.libraries, targets,
            source_ranges=self.source_ranges,
        )


_PARAMETER_TYPES = {"float": "float", "int": "int", "uint": "uint", "bool": "uint", "color": "float4",
                    "vec2": "float2", "vec3": "float3", "vec4": "float4", "mat3": "float3x3", "mat4": "float4x4"}
_DEFAULTS = {"float": 0.0, "int": 0, "uint": 0, "bool": False, "color": (1.0, 1.0, 1.0, 1.0),
             "vec2": (0.0, 0.0), "vec3": (0.0, 0.0, 0.0), "vec4": (0.0, 0.0, 0.0, 0.0),
             "mat3": tuple(tuple(float(row == column) for column in range(3)) for row in range(3)),
             "mat4": tuple(tuple(float(row == column) for column in range(4)) for row in range(4))}


class _Author(ShaderEmitter):
    def __init__(self, source: str, path: str):
        super().__init__(source, path)
        self.source = source
        self.alias = ""
        self.fields = {}
        self.field_nodes = {}

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
            dimension = int(self.qualified(node.func)[-1])
            values = (self.literal(value) for value in node.args)
            components = tuple(component for value in values for component in (value if isinstance(value, tuple) else (value,)))
            if len(components) == 1:
                components *= dimension
            if len(components) != dimension:
                self.error(node, "Vector defaults must fill every component")
            return components
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            self.error(node, "Shader parameter defaults and presentation metadata must be literal values")

    def parameter(self, statement):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            self.error(statement, "Declare one named shader parameter per field")
        call = statement.value
        if not isinstance(call, ast.Call):
            self.error(statement, "Expected shader.parameter declaration")
        declaration = self.qualified(call.func)
        name = targets[0].id
        self.field_nodes[name] = statement
        if declaration == self.alias + ".input":
            if len(call.args) != 2:
                self.error(call, "shader.input names a semantic and a typed GPU resource")
            kind = self.qualified(call.args[1]).removeprefix(self.alias + ".")
            if kind not in RESOURCE_TYPES:
                self.error(call.args[1], "Unknown GPU input resource type")
            metadata = {keyword.arg: self.literal(keyword.value) for keyword in call.keywords}
            if metadata.keys() - {"domain"}:
                self.error(call, "GPU image inputs declare a color domain")
            self.fields[name] = ShaderInput(name, self.literal(call.args[0]), kind, metadata.get("domain"))
            self.types["self." + name] = RESOURCE_TYPES[kind]
            return
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
        value = ShaderParameter(name, kind, values.get("default", _DEFAULTS[kind]), values.get("label", name),
                                values.get("range"), values.get("hdr", False))
        self.validate_default(value, call)
        self.fields[name] = value
        self.types["self." + name] = "bool" if kind == "bool" else _PARAMETER_TYPES[kind]

    def validate_default(self, field, node):
        value, kind = field.default, field.kind
        numeric = lambda item: type(item) in (int, float)
        valid = True
        if kind == "float":
            valid = numeric(value)
        elif kind in {"int", "uint"}:
            minimum, maximum = (-2**31, 2**31 - 1) if kind == "int" else (0, 2**32 - 1)
            valid = type(value) is int and minimum <= value <= maximum
        elif kind == "bool":
            valid = type(value) is bool
        elif kind == "color" or kind.startswith("vec"):
            count = 4 if kind == "color" else int(kind[-1])
            valid = isinstance(value, (tuple, list)) and len(value) == count and all(map(numeric, value))
        elif kind.startswith("mat"):
            count = int(kind[-1])
            valid = (isinstance(value, (tuple, list)) and len(value) == count
                     and all(isinstance(row, (tuple, list)) and len(row) == count and all(map(numeric, row)) for row in value))
        if not valid:
            self.error(node, f"Default of {field.name!r} must match its declared {kind} type and shape", code="author.parameter")
        if field.range is not None:
            valid = (kind in {"float", "int", "uint"} and isinstance(field.range, (tuple, list))
                     and len(field.range) == 2 and all(map(numeric, field.range)) and field.range[0] <= field.range[1])
            if not valid:
                self.error(node, "A slider range is an ordered pair for a scalar numeric parameter", code="author.parameter")

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            field = self.fields.get(node.attr)
            if isinstance(field, ShaderInput):
                return "inx_" + node.attr
            if field is None:
                self.error(node, "Unknown shader parameter: " + node.attr)
            name = "inx_parameters." + node.attr
            return "(" + name + " != 0u)" if field.kind == "bool" else name
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
            if uv.annotation is None or self.annotation(uv.annotation) != "float2":
                self.error(uv, "Fullscreen UV must be annotated as shader.vec2")
            signature = "float2 " + uv.arg + " : TEXCOORD0"
            self.types[uv.arg] = "float2"
        if main.returns is None or self.annotation(main.returns) != "float4":
            self.error(main, "Fullscreen main returns shader.vec4")
        self.aliases = (self.alias,)
        self.declare_locals(main.body)
        self.local_declarations()
        self.block(main.body)
        return ['[shader("fragment")]', self.source_directive(main),
                "float4 fragment_main(" + signature + ") : SV_Target0", "{", *self.lines, "}"]


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


def parse_shader(source: str, source_path: str, module_name: str, *, reader=None) -> ShaderDefinition:
    author = _Author(source, source_path)
    author.alias, imports, body = declarations(source, source_path)
    author.aliases = (author.alias,)
    graph = LibraryGraph(reader)
    author.functions, dependencies = graph.imports(imports, author)
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(lineno=1, col_offset=0), "Declare one shader class")
    declaration = body[0]
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) != author.alias + ".fullscreen":
        author.error(declaration, "The first authoring slice supports shader.fullscreen")
    main, helpers = None, []
    for statement in declaration.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
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
    parameters = tuple(field for field in author.fields.values() if isinstance(field, ShaderParameter))
    inputs = tuple(field for field in author.fields.values() if isinstance(field, ShaderInput))
    code = ["import " + RESOURCE_MODULE[0] + ";", *("import " + name + ";" for name in dependencies), _FULLSCREEN_VERTEX]
    for field in inputs:
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     RESOURCE_TYPES[field.kind] + " inx_" + field.name + ";"])
    if parameters:
        code.append("struct ShaderParameters {")
        for field in parameters:
            code.extend([author.source_directive(author.field_nodes[field.name]),
                         "    " + _PARAMETER_TYPES[field.kind] + " " + field.name + ";"])
        code.extend(["};", "ConstantBuffer<ShaderParameters> inx_parameters;"])
    for helper in helpers:
        emitter = _Author(source, source_path)
        emitter.alias = author.alias
        emitter.aliases = (author.alias,)
        emitter.fields = author.fields
        emitter.types = author.types.copy()
        emitter.functions = author.functions
        code.extend(emitter.emit_function(helper, author.functions["self." + helper.name], method=True))
        author.source_ranges.extend(emitter.source_ranges)
    code.extend(author.function(main))
    return ShaderDefinition(declaration.name, source_path, module_name, parameters,
                            "\n".join(code), graph.sources(), inputs,
                            tuple(dict.fromkeys((*graph.ranges(), *author.source_ranges))))


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)


def _prepare_source(source, source_path, module_name, reader):
    """Return a plain native preparation snapshot; no Python object enters GPU work."""
    from infernux._source_kinds import source_extension

    if source_extension(source_path) == ".shaderlib.py":
        graph = LibraryGraph(reader)
        owner = ShaderEmitter(source, source_path)
        module = graph.load(Path(source_path), owner, ast.Constant(lineno=1, col_offset=0), source=source)
        schema = {"name": Path(source_path).name[:-len(".shaderlib.py")], "domain": "library",
                  "parameters": [], "inputs": [],
                  "functions": [{"name": name, "return_type": function.result, "arguments": function.arguments}
                                for name, function in module.functions.items()]}
        libraries = tuple(item for item in graph.sources() if item[0] != module.name)
        return module.source, libraries, json.dumps(schema, ensure_ascii=False), graph.ranges()
    definition = parse_shader(source.removeprefix("\ufeff"), source_path, module_name, reader=reader)
    schema = {"name": definition.name, "domain": "fullscreen",
              "parameters": [asdict(field) for field in definition.parameters],
              "inputs": [asdict(field) for field in definition.inputs]}
    return definition.generated_source, definition.libraries, json.dumps(schema, ensure_ascii=False), definition.source_ranges
