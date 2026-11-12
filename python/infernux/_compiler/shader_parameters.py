"""Logical GPU fields and their shared static author frontend."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .shader_library import ShaderEmitter
from .shader_types import RESOURCE_TYPES

@dataclass(frozen=True)
class ShaderParameter:
    name: str
    kind: str
    default: object
    label: str
    range: tuple[float, float] | None = None
    hdr: bool = False
    internal: bool = False
    stages: tuple[str, ...] = ("fragment",)


@dataclass(frozen=True)
class ShaderInput:
    name: str
    semantic: str
    kind: str
    domain: str | None = None


_PARAMETER_TYPES = {"float": "float", "int": "int", "uint": "uint", "bool": "uint", "color": "float4",
                    "vec2": "float2", "vec3": "float3", "vec4": "float4", "mat3": "float3x3", "mat4": "float4x4"}
_DEFAULTS = {"float": 0.0, "int": 0, "uint": 0, "bool": False, "color": (1.0, 1.0, 1.0, 1.0),
             "texture_2d": "white",
             "vec2": (0.0, 0.0), "vec3": (0.0, 0.0, 0.0), "vec4": (0.0, 0.0, 0.0, 0.0),
             "mat3": tuple(tuple(float(row == column) for column in range(3)) for row in range(3)),
             "mat4": tuple(tuple(float(row == column) for column in range(4)) for row in range(4))}


class ParameterAuthor(ShaderEmitter):
    def __init__(self, source: str, path: str):
        super().__init__(source, path)
        self.source = source
        self.alias = ""
        self.fields = {}
        self.field_nodes = {}
        self.field_sources = {}
        self.domain = "fullscreen"
        self.is_surface = False

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
        self.field_sources[name] = (self, statement)
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
        prefix = self.alias + (".material." if self.domain == "mesh" else ".parameter.")
        if not declaration.startswith(prefix) or declaration[len(prefix):] not in {*_PARAMETER_TYPES, "texture_2d"}:
            self.error(call, "Expected " + prefix + " followed by a supported value type")
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
        self.types["self." + name] = (RESOURCE_TYPES["Color2D"] if kind == "texture_2d" else
                                     "bool" if kind == "bool" else _PARAMETER_TYPES[kind])

    def validate_default(self, field, node):
        value, kind = field.default, field.kind
        numeric = lambda item: type(item) in (int, float)
        valid = True
        if kind == "texture_2d":
            if self.domain != "mesh":
                self.error(node, "Texture asset properties belong to mesh materials; fullscreen images use shader.input",
                           code="author.parameter")
            valid = value in ("white", "black", "normal")
        elif kind == "float":
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
            if isinstance(field, ShaderInput) or (isinstance(field, ShaderParameter) and field.kind == "texture_2d"):
                return "inx_" + node.attr
            if field is None:
                self.error(node, "Unknown shader parameter: " + node.attr)
            name = "inx_parameters." + node.attr
            return "(" + name + " != 0u)" if field.kind == "bool" else name
        return super().visit_Attribute(node)

    def parameter_declarations(self, parameters):
        code = []
        for index, field in enumerate(field for field in parameters if field.kind == "texture_2d"):
            owner, node = self.field_sources[field.name]
            code.extend([owner.source_directive(node),
                         f"[[vk::binding({1 + 2 * index}, 0)]] InxColor2D inx_{field.name};"])
        uniforms = tuple(field for field in parameters if field.kind != "texture_2d")
        if uniforms:
            code.append("struct ShaderParameters {")
            for field in uniforms:
                owner, node = self.field_sources[field.name]
                code.extend([owner.source_directive(node), "    " + _PARAMETER_TYPES[field.kind] + " " + field.name + ";"])
            binding = "[[vk::binding(0, 0)]] " if self.domain == "mesh" else ""
            code.extend(["};", binding + "ConstantBuffer<ShaderParameters> inx_parameters;"])
        return code
