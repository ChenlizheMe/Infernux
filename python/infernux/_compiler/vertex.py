"""Static local-space deformation; the engine owns projection and pass variants."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

from .shader_info import INFO_DEFAULTS, info_field, read_shader_info
from .shader_library import declarations
from .shader_parameters import ParameterAuthor, ShaderParameter


@dataclass(frozen=True)
class VertexDefinition:
    name: str
    path: str
    source: str
    author: ParameterAuthor
    methods: tuple
    dependencies: tuple[str, ...]
    info: dict

    def emit(self):
        code, ranges = ["namespace inx_vertex_author {"], []
        for method in self.methods:
            emitter = ParameterAuthor(self.source, self.path)
            emitter.alias = self.author.alias
            emitter.aliases = (emitter.alias,)
            emitter.fields = self.author.fields
            emitter.types = self.author.types.copy()
            emitter.functions = self.author.functions
            code.extend(emitter.emit_function(method, emitter.functions["self." + method.name], method=True))
            ranges.extend(emitter.source_ranges)
        code.append("}")
        return "\n".join(code), tuple(ranges)


def read_vertex(path, graph, owner, location, *, source=None):
    path = Path(path).resolve()
    if source is None:
        try:
            source = graph.reader(str(path))
        except FileNotFoundError:
            source = None
        if source is None:
            owner.error(location, "Vertex source does not exist: " + str(path), code="author.reference")
    source = source.removeprefix("\ufeff")
    alias, imports, body = declarations(source, str(path))
    author = ParameterAuthor(source, str(path))
    author.alias, author.aliases, author.domain = alias, (alias,), "mesh"
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(lineno=1, col_offset=0), "Declare one shader.vert class")
    declaration = body[0]
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) != alias + ".vert":
        author.error(declaration, "Declare a shader.vert class")
    methods = []
    for statement in declaration.body:
        if info_field(statement) in INFO_DEFAULTS:
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            author.parameter(statement)
        elif isinstance(statement, ast.FunctionDef):
            methods.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "Vertex assets declare material fields and typed methods")
    author.functions, dependencies = graph.imports(imports, author)
    for method in methods:
        name = "self." + method.name
        if name in author.functions:
            author.error(method, "Duplicate vertex method: " + method.name)
        author.functions[name] = author.signature(method, "inx_vertex_author::" + method.name, method=True)
    entry = author.functions.get("self.deform")
    if entry is None or entry.arguments != ("InxVertexData", "InxVertexContext") or entry.result != "InxVertexData":
        author.error(declaration, "deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) returns shader.VertexData",
                     code="author.type")
    for name, parameter in author.fields.items():
        if not isinstance(parameter, ShaderParameter) or parameter.kind in {"bool", "uint", "mat3"}:
            author.error(author.field_nodes[name], "This value type is not supported by the mesh material owner", code="author.parameter")
        author.fields[name] = replace(parameter, stages=("vertex",))
    return VertexDefinition(declaration.name, str(path), source, author, tuple(methods), dependencies,
                            read_shader_info(source, str(path)))
