"""Static author declarations lowered to the shared native Slang compiler.

The first vertical slice is a fullscreen fragment with typed parameters and
pure expressions. General shader domains and control-flow analysis build on
this source model; unsupported author syntax is diagnosed at its source line.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path

from infernux.shader._language import INTRINSICS, TYPES


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

    def compile(self, targets=("spirv", "wgsl")):
        from infernux.lib import _Infernux
        return _Infernux._compile_shader_program(
            self.generated_source, self.source_path, self.module_name, targets=targets,
        )


_PARAMETER_TYPES = {"float": "float", "int": "int", "bool": "uint", "color": "float4"}
_DEFAULTS = {"float": 0.0, "int": 0, "bool": False, "color": (1.0, 1.0, 1.0, 1.0)}
_BINARY = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_UNARY = {ast.UAdd: "+", ast.USub: "-", ast.Not: "!", ast.Invert: "~"}
_COMPARE = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


class _Author:
    def __init__(self, source: str, path: str):
        self.source = source
        self.path = path
        self.lines = source.splitlines()
        self.alias = ""
        self.parameters = {}
        self.locals = set()

    def error(self, node, message):
        raise SyntaxError(message, (self.path, node.lineno, node.col_offset + 1, self.lines[node.lineno - 1]))

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

    def expression(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int, bool)):
            return json.dumps(node.value)
        if isinstance(node, ast.Name) and node.id in self.locals:
            return node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                if node.attr not in self.parameters:
                    self.error(node, "Unknown shader parameter: " + node.attr)
                name = "inx_parameters." + node.attr
                return "(" + name + " != 0u)" if self.parameters[node.attr].kind == "bool" else name
            return self.expression(node.value) + "." + node.attr
        if isinstance(node, ast.Call):
            name = self.qualified(node.func)
            prefix = self.alias + "."
            symbol = name[len(prefix):] if name.startswith(prefix) else name
            functions = TYPES | INTRINSICS
            if symbol not in functions or (not name.startswith(prefix) and name not in {"float", "int", "bool", "abs", "min", "max"}):
                self.error(node, "GPU calls must name a shader constructor or intrinsic")
            if node.keywords:
                self.error(node, "GPU intrinsic arguments are positional")
            return functions[symbol] + "(" + ", ".join(self.expression(arg) for arg in node.args) + ")"
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return "(" + self.expression(node.left) + " " + _BINARY[type(node.op)] + " " + self.expression(node.right) + ")"
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return "(" + _UNARY[type(node.op)] + self.expression(node.operand) + ")"
        if isinstance(node, ast.IfExp):
            return "(" + self.expression(node.test) + " ? " + self.expression(node.body) + " : " + self.expression(node.orelse) + ")"
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            parts = ["(" + self.expression(left) + " " + _COMPARE[type(op)] + " " + self.expression(right) + ")"
                     for left, op, right in zip(operands, node.ops, operands[1:]) if type(op) in _COMPARE]
            if len(parts) != len(node.ops):
                self.error(node, "Unsupported GPU comparison")
            return "(" + " && ".join(parts) + ")"
        if isinstance(node, ast.BoolOp):
            return "(" + (" && " if isinstance(node.op, ast.And) else " || ").join(self.expression(value) for value in node.values) + ")"
        if isinstance(node, ast.Subscript):
            return self.expression(node.value) + "[" + self.expression(node.slice) + "]"
        self.error(node, "GPU expression is not supported by the current source contract: " + type(node).__name__)

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
            self.locals.add(uv.arg)
        if main.returns is None or self.qualified(main.returns) != self.alias + ".vec4":
            self.error(main, "Fullscreen main returns shader.vec4")
        lines = ["[shader(\"fragment\")]", "float4 fragment_main(" + signature + ") : SV_Target0", "{"]
        for statement in main.body:
            lines.append("#line " + str(statement.lineno) + " " + json.dumps(self.path, ensure_ascii=False))
            if isinstance(statement, ast.Return):
                lines.append("    return " + self.expression(statement.value) + ";")
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                value = self.expression(statement.value)
                lines.append("    " + ("" if name in self.locals else "var ") + name + " = " + value + ";")
                self.locals.add(name)
            else:
                self.error(statement, "Fullscreen entries currently support local values and a return expression")
        lines.append("}")
        return lines


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
    tree = ast.parse(source, filename=source_path)
    classes = []
    for statement in tree.body:
        if isinstance(statement, ast.Import) and len(statement.names) == 1 and statement.names[0].name == "infernux.shader":
            author.alias = statement.names[0].asname or "infernux.shader"
        elif isinstance(statement, ast.ClassDef):
            classes.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "GPU author files contain shader imports and declarations; host statements are not executed")
    if not author.alias or len(classes) != 1:
        raise SyntaxError("Declare one shader class and import infernux.shader", (source_path, 1, 1, author.lines[0] if author.lines else ""))
    declaration = classes[0]
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) != author.alias + ".fullscreen":
        author.error(declaration, "The first authoring slice supports shader.fullscreen")
    main = None
    for statement in declaration.body:
        if isinstance(statement, ast.Assign):
            author.parameter(statement)
        elif isinstance(statement, ast.FunctionDef) and statement.name == "main" and main is None:
            main = statement
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "Expected a parameter declaration or the main shader entry")
    if main is None:
        author.error(declaration, "Shader class needs a main entry")
    code = [_FULLSCREEN_VERTEX]
    if author.parameters:
        code.append("struct ShaderParameters {")
        code.extend("    " + _PARAMETER_TYPES[field.kind] + " " + field.name + ";" for field in author.parameters.values())
        code.extend(["};", "ConstantBuffer<ShaderParameters> inx_parameters;"])
    code.extend(author.function(main))
    return ShaderDefinition(declaration.name, source_path, module_name, tuple(author.parameters.values()), "\n".join(code))


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)
