"""Static ShaderLib imports and typed functions shared by GPU author domains."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .slang_ast import SlangEmitter, qualified
from .shader_types import RESOURCE_MODULE, RESOURCE_TYPES


@dataclass(frozen=True)
class Function:
    symbol: str
    result: str
    arguments: tuple[str, ...]


@dataclass
class Module:
    name: str
    path: str
    source: str
    functions: dict[str, Function]


class ShaderEmitter(SlangEmitter):
    def __init__(self, source, path, *, aliases=("shader",), functions=None, types=None):
        super().__init__(source, path, aliases=aliases, types=types)
        self.functions = dict(functions or {})

    def infer(self, node):
        if (isinstance(node, ast.Call) and self.symbol(qualified(node.func)) != qualified(node.func)
                and self.symbol(qualified(node.func)) in {"sample", "sample_level", "sample_grad", "load"}):
            return "float4"
        if isinstance(node, ast.Call) and (function := self.functions.get(qualified(node.func))):
            return function.result
        return super().infer(node)

    def visit_Call(self, node):
        operation = self.symbol(qualified(node.func))
        image_operations = {"sample": (2, "Sample"), "sample_level": (3, "SampleLevel"),
                            "sample_grad": (4, "SampleGrad"), "load": (3, "Load")}
        if operation in image_operations and operation != qualified(node.func):
            arity, intrinsic = image_operations[operation]
            if node.keywords or len(node.args) != arity:
                self.error(node, f"shader.{operation} takes {arity} typed positional arguments")
            if self.infer(node.args[0]) not in RESOURCE_TYPES.values():
                self.error(node, "Texture operations receive a declared GPU image")
            image, *arguments = (self.visit(argument) for argument in node.args)
            if operation == "load":
                return f"{image}.texture.Load(int3({arguments[0]}, {arguments[1]}))"
            return image + ".texture." + intrinsic + "(" + ", ".join([image + ".sampler", *arguments]) + ")"
        function = self.functions.get(qualified(node.func))
        if function is None:
            return super().visit_Call(node)
        if node.keywords or len(node.args) != len(function.arguments):
            self.error(node, "Shader helper arguments must match its typed positional signature")
        return function.symbol + "(" + ", ".join(self.visit(arg) for arg in node.args) + ")"

    def annotation(self, node):
        name = qualified(node).rsplit(".", 1)[-1]
        return RESOURCE_TYPES[name] if name in RESOURCE_TYPES else super().annotation(node)

    def signature(self, definition, symbol, *, method=False):
        arguments = definition.args
        if (definition.decorator_list or arguments.vararg or arguments.kwarg
                or arguments.kwonlyargs or arguments.defaults or arguments.posonlyargs):
            self.error(definition, "GPU helpers use a fixed typed positional signature")
        fields = arguments.args
        if method:
            if not fields or fields[0].arg != "self":
                self.error(definition, "Shader helpers are instance methods")
            fields = fields[1:]
        for field in fields:
            if field.annotation is None:
                self.error(field, "GPU helper parameters need type annotations")
        if definition.returns is None:
            self.error(definition, "GPU helpers need a return type annotation")
        return Function(symbol, self.annotation(definition.returns),
                        tuple(self.annotation(field.annotation) for field in fields))

    def emit_function(self, definition, function, *, method=False, public=False):
        fields = definition.args.args[1:] if method else definition.args.args
        self.types.update((field.arg, kind) for field, kind in zip(fields, function.arguments))
        self.declare_locals(definition.body)
        self.local_declarations()
        self.block(definition.body)
        parameters = ", ".join(kind + " " + field.arg for field, kind in zip(fields, function.arguments))
        return [f'{"public " if public else ""}{function.result} {definition.name}({parameters})',
                "{", *self.lines, "}"]


def declarations(source, path):
    """Separate GPU declarations without importing or executing their Python module."""
    emitter = ShaderEmitter(source, path)
    imports, body, aliases = [], [], []
    for statement in ast.parse(source, filename=path).body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name != "infernux.shader":
                    emitter.error(statement, "GPU files import infernux.shader and static ShaderLib declarations")
                aliases.append(alias.asname or alias.name)
        elif isinstance(statement, ast.ImportFrom):
            imports.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        elif isinstance(statement, (ast.ClassDef, ast.FunctionDef)):
            body.append(statement)
        else:
            emitter.error(statement, "GPU files contain static declarations; host statements are not executed")
    if len(aliases) != 1:
        emitter.error(ast.Constant(lineno=1, col_offset=0), "Import infernux.shader once in each GPU source")
    return aliases[0], imports, body


class LibraryGraph:
    def __init__(self, reader=None):
        self.modules: dict[Path, Module] = {}
        self.ordered: list[Module] = []
        self.loading: set[Path] = set()
        self.reader = reader or (lambda path: Path(path).read_text(encoding="utf-8-sig"))

    def imports(self, imports, owner: ShaderEmitter):
        functions, dependencies = {}, []
        for statement in imports:
            if any(item.name == "*" for item in statement.names):
                owner.error(statement, "ShaderLib imports name explicit exports")
            if statement.level:
                base = Path(owner.path).parent
                for _ in range(statement.level - 1):
                    base = base.parent
                base = base.joinpath(*(statement.module or "").split("."))
            elif statement.module and statement.module.startswith("infernux.shader.lib."):
                base = Path(__file__).parents[1] / "resources" / "shaders" / "lib"
                base = base.joinpath(*statement.module.split(".")[3:])
            else:
                owner.error(statement, "ShaderLib imports use a relative path or infernux.shader.lib")
            if statement.module:
                module = self.load(base.with_name(base.name + ".shaderlib.py"), owner, statement)
                dependencies.append(module.name)
                for alias in statement.names:
                    if alias.name not in module.functions:
                        owner.error(statement, f"ShaderLib {base.name!r} has no export {alias.name!r}")
                    name = alias.asname or alias.name
                    if name in functions:
                        owner.error(statement, "Duplicate imported ShaderLib name: " + name)
                    functions[name] = module.functions[alias.name]
            else:
                for alias in statement.names:
                    module = self.load(base / (alias.name + ".shaderlib.py"), owner, statement)
                    dependencies.append(module.name)
                    for name, function in module.functions.items():
                        functions[(alias.asname or alias.name) + "." + name] = function
        return functions, tuple(dict.fromkeys(dependencies))

    def load(self, path, owner, location, *, source=None):
        path = path.resolve()
        if path in self.loading:
            owner.error(location, "ShaderLib import cycle reaches " + str(path))
        if path in self.modules:
            return self.modules[path]
        if source is None:
            try:
                source = self.reader(str(path))
            except FileNotFoundError:
                owner.error(location, "ShaderLib source does not exist: " + str(path))
        if source is None:
            owner.error(location, "ShaderLib source does not exist: " + str(path))
        source = source.removeprefix("\ufeff")
        alias, imports, body = declarations(source, str(path))
        emitter = ShaderEmitter(source, str(path), aliases=(alias,))
        module = Module("inx_library_" + str(len(self.modules)), str(path), "", {})
        self.modules[path] = module
        self.loading.add(path)
        functions, dependencies = self.imports(imports, emitter)
        for definition in body:
            if not isinstance(definition, ast.FunctionDef):
                emitter.error(definition, "ShaderLib exports typed functions")
            if definition.name in functions:
                emitter.error(definition, "Duplicate ShaderLib function: " + definition.name)
            function = emitter.signature(definition, module.name + "::" + definition.name)
            module.functions[definition.name] = functions[definition.name] = function
        code = ["import " + RESOURCE_MODULE[0] + ";", *("import " + name + ";" for name in dependencies),
                "namespace " + module.name + " {"]
        for definition in body:
            function_emitter = ShaderEmitter(source, str(path), aliases=(alias,), functions=functions)
            code.extend(function_emitter.emit_function(definition, module.functions[definition.name], public=True))
        code.append("}")
        module.source = "\n".join(code)
        self.loading.remove(path)
        self.ordered.append(module)
        return module

    def sources(self):
        return (RESOURCE_MODULE, *((module.name, module.path, module.source) for module in self.ordered))
