"""Shared structured Python arithmetic lowering for Shader and numerical programs.

Source frontends own resources and entry signatures. This emitter owns typed
locals, expressions and structured control flow; Slang owns GPU IR and targets.
"""

from __future__ import annotations

import ast
import json
from functools import reduce

from infernux.shader._language import INTRINSICS, TYPES
from .diagnostics import ShaderSourceError


_BINARY = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Mod: "%",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
}
_UNARY = {ast.UAdd: "+", ast.USub: "-", ast.Not: "!", ast.Invert: "~"}
_COMPARE = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_FUNCTIONS = TYPES | INTRINSICS


def qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified(node.value) + "." + node.attr
    return ""


def common_type(left, right):
    if left == right:
        return left
    if left is None or right is None:
        return left or right
    # Unsuffixed GPU literals have 32-bit types. Vector/scalar operations
    # retain the vector shape; floating operands promote integer operands.
    shape = next((kind for kind in (left, right) if kind and kind[-1:].isdigit()), "")
    scalar = "float" if any(kind.startswith("float") for kind in (left, right)) else "uint" if any(kind.startswith("uint") for kind in (left, right)) else "int"
    if not shape and scalar != "float" and "int64_t" in (left, right):
        return "int64_t"
    prefix = next((index for index, character in enumerate(shape) if character.isdigit()), len(shape))
    return scalar + shape[prefix:] if shape else scalar


def floating_type(kind):
    return shaped_type(kind, "float")


def shaped_type(kind, scalar):
    if kind and kind[-1:].isdigit():
        prefix = next(index for index, character in enumerate(kind) if character.isdigit())
        return scalar + kind[prefix:]
    return scalar


def product_type(left, right):
    """Slang mul / Python @ contracts, including rectangular matrices."""
    def dimensions(kind):
        prefix = next((i for i, char in enumerate(kind) if char.isdigit()), len(kind))
        return kind[:prefix], tuple(map(int, kind[prefix:].split("x"))) if prefix < len(kind) else ()
    if left is None or right is None:
        return left or right
    scalar, a = dimensions(left)
    _, b = dimensions(right)
    if len(a) == 2 and len(b) == 1:
        return scalar + str(a[0])
    if len(a) == 1 and len(b) == 2:
        return scalar + str(b[1])
    if len(a) == 2 and len(b) == 2:
        return scalar + str(a[0]) + "x" + str(b[1])
    if len(a) == len(b) == 1:
        return common_type(left, right).rstrip("0123456789")
    return common_type(left, right)


class SlangEmitter(ast.NodeVisitor):
    def __init__(self, source: str, path: str, *, aliases=("shader",), types=None):
        self.path = path
        self.source_lines = source.splitlines()
        self.aliases = aliases
        self.types = dict(types or {})
        self.locals = {}
        self.lines = []
        self.indent = 1
        self.line_offset = 0
        self.loop_flags = []
        self.loop_sequence = 0
        self.current_statement = None
        self.source_ranges = []

    def source_range(self, node):
        line = getattr(node, "lineno", 1)
        text = self.source_lines[line - 1] if 0 < line <= len(self.source_lines) else ""
        end_line = getattr(node, "end_lineno", None) or line
        end_text = self.source_lines[end_line - 1] if 0 < end_line <= len(self.source_lines) else ""
        # AST columns count UTF-8 bytes; editor/SyntaxError columns count characters.
        column = len(text.encode("utf-8")[:getattr(node, "col_offset", 0)].decode("utf-8")) + 1
        end_byte = getattr(node, "end_col_offset", None)
        end_column = len(end_text.encode("utf-8")[:end_byte].decode("utf-8")) + 1 if end_byte is not None else column
        return self.path, line + self.line_offset, column, end_line + self.line_offset, end_column

    def error(self, node, message, *, code="author.syntax"):
        path, line, column, end_line, end_column = self.source_range(node)
        source_line = line - self.line_offset
        text = self.source_lines[source_line - 1] if 0 < source_line <= len(self.source_lines) else ""
        raise ShaderSourceError(message, (path, line, column, text, end_line, end_column), code=code)

    def source_directive(self, node):
        location = self.source_range(node)
        self.source_ranges.append(location)
        return f"#line {location[1]} {json.dumps(self.path, ensure_ascii=False)}"

    def line(self, text):
        if self.current_statement is not None:
            self.lines.append(self.source_directive(self.current_statement))
        self.lines.append("    " * self.indent + text)

    def block(self, statements):
        previous = self.current_statement
        for statement in statements:
            self.current_statement = statement
            self.visit(statement)
        self.current_statement = previous

    def declare_locals(self, statements):
        """Python locals have function scope, including names assigned in branches."""
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                kind = self.annotation(statement.annotation) if isinstance(statement, ast.AnnAssign) else self.infer(statement.value)
                for target in targets:
                    if isinstance(target, ast.Name):
                        previous = self.locals.get(target.id)
                        self.locals[target.id] = common_type(previous, kind) if previous else kind
                        self.types[target.id] = self.locals[target.id]
            elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                name = statement.target.id
                value = ast.BinOp(left=statement.target, op=statement.op, right=statement.value)
                self.locals[name] = self.types[name] = self.infer(value)
            elif isinstance(statement, ast.For):
                if not isinstance(statement.target, ast.Name):
                    self.error(statement.target, "GPU ranges use one integer index")
                self.locals[statement.target.id] = self.types[statement.target.id] = "int"
            if isinstance(statement, (ast.If, ast.For, ast.While)):
                self.declare_locals(statement.body)
                self.declare_locals(statement.orelse)

    def local_declarations(self):
        for name, kind in self.locals.items():
            if kind is None:
                raise TypeError(f"{self.path}: GPU local {name!r} needs an explicit type annotation")
            self.line(kind + " " + name + ";")

    def annotation(self, node):
        name = qualified(node).rsplit(".", 1)[-1]
        if name not in TYPES:
            self.error(node, "Unknown GPU value type: " + name)
        return TYPES[name]

    def symbol(self, name):
        for alias in self.aliases:
            prefix = alias + "."
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def infer(self, node):
        if isinstance(node, ast.Constant):
            return "bool" if isinstance(node.value, bool) else "int" if isinstance(node.value, int) else "float"
        if isinstance(node, ast.Name):
            return self.types.get(node.id)
        if isinstance(node, ast.Attribute):
            key = qualified(node)
            if key in self.types:
                return self.types[key]
            base = self.infer(node.value)
            if base and set(node.attr) <= set("xyzwrgba"):
                scalar = base.rstrip("0123456789")
                return scalar + str(len(node.attr)) if len(node.attr) > 1 else scalar
        if isinstance(node, ast.UnaryOp):
            kind = self.infer(node.operand)
            return shaped_type(kind, "bool") if isinstance(node.op, ast.Not) else kind
        if isinstance(node, ast.BoolOp):
            return shaped_type(reduce(common_type, map(self.infer, node.values)), "bool")
        if isinstance(node, ast.Compare):
            return shaped_type(reduce(common_type, map(self.infer, (node.left, *node.comparators))), "bool")
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.MatMult):
                return product_type(self.infer(node.left), self.infer(node.right))
            return common_type(self.infer(node.left), self.infer(node.right))
        if isinstance(node, ast.IfExp):
            return common_type(self.infer(node.body), self.infer(node.orelse))
        if isinstance(node, ast.Call):
            name = self.symbol(qualified(node.func))
            if name in TYPES:
                return TYPES[name]
            if name in INTRINSICS and not node.args:
                self.error(node, "GPU intrinsics need explicit positional arguments", code="author.call")
            if name in {"dot", "length", "distance", "determinant"}:
                return "float"
            if name in {"all", "any"}:
                return "bool"
            if name in {"isnan", "isinf", "isfinite"}:
                return shaped_type(self.infer(node.args[0]), "bool")
            if name in {"as_float", "f16_to_f32"}:
                return shaped_type(self.infer(node.args[0]), "float")
            if name == "as_int":
                return shaped_type(self.infer(node.args[0]), "int")
            if name in {"as_uint", "f32_to_f16", "count_bits", "first_bit_high", "first_bit_low"}:
                return shaped_type(self.infer(node.args[0]), "uint")
            if name == "mul" and len(node.args) == 2:
                return product_type(*(self.infer(argument) for argument in node.args))
            if name in INTRINSICS:
                if INTRINSICS[name] in {"min", "max", "clamp", "step", "smoothstep", "pow", "atan2", "lerp", "mad"}:
                    return reduce(common_type, map(self.infer, node.args))
                return self.infer(node.args[0])
        if isinstance(node, ast.Subscript):
            base = self.infer(node.value)
            if base and "x" in base and base[-1:].isdigit():
                prefix = next(index for index, char in enumerate(base) if char.isdigit())
                return base[:prefix] + base.split("x")[1]
            return base.rstrip("0123456789") if base else None
        return None

    def visit_Constant(self, node):
        if not isinstance(node.value, (float, int, bool)):
            self.error(node, "GPU expressions use numeric literals")
        return json.dumps(node.value)

    def visit_Name(self, node):
        if node.id not in self.types:
            self.error(node, "Unknown GPU value: " + node.id)
        return node.id

    def visit_Attribute(self, node):
        return self.visit(node.value) + "." + node.attr

    def visit_Call(self, node):
        name = qualified(node.func)
        symbol = self.symbol(name)
        if symbol not in _FUNCTIONS or (symbol == name and symbol not in {"float", "int", "bool", "abs", "min", "max"}):
            self.error(node, "GPU calls must name a shader constructor, intrinsic or declared helper")
        if node.keywords:
            self.error(node, "GPU intrinsic arguments are positional")
        return _FUNCTIONS[symbol] + "(" + ", ".join(self.visit(arg) for arg in node.args) + ")"

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return "pow(" + self.visit(node.left) + ", " + self.visit(node.right) + ")"
        if isinstance(node.op, ast.MatMult):
            return "mul(" + self.visit(node.left) + ", " + self.visit(node.right) + ")"
        if type(node.op) not in _BINARY:
            self.error(node, "GPU operator is not supported: " + type(node.op).__name__)
        return "(" + self.visit(node.left) + " " + _BINARY[type(node.op)] + " " + self.visit(node.right) + ")"

    def visit_UnaryOp(self, node):
        return "(" + _UNARY[type(node.op)] + self.visit(node.operand) + ")"

    def visit_IfExp(self, node):
        return "(" + self.visit(node.test) + " ? " + self.visit(node.body) + " : " + self.visit(node.orelse) + ")"

    def visit_Compare(self, node):
        operands = [node.left, *node.comparators]
        parts = []
        for left, op, right in zip(operands, node.ops, operands[1:]):
            if type(op) not in _COMPARE:
                self.error(node, "GPU comparisons operate on values")
            parts.append("(" + self.visit(left) + " " + _COMPARE[type(op)] + " " + self.visit(right) + ")")
        return "(" + " && ".join(parts) + ")"

    def visit_BoolOp(self, node):
        return "(" + (" && " if isinstance(node.op, ast.And) else " || ").join(self.visit(value) for value in node.values) + ")"

    def visit_Subscript(self, node):
        return self.visit(node.value) + "[" + self.visit(node.slice) + "]"

    def assign(self, target, value):
        self.line(self.visit(target) + " = " + value + ";")

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            self.error(node, "GPU assignments name one destination")
        self.assign(node.targets[0], self.visit(node.value))

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.assign(node.target, self.visit(node.value))

    def visit_AugAssign(self, node):
        value = ast.BinOp(left=node.target, op=node.op, right=node.value)
        ast.copy_location(value, node)
        self.assign(node.target, self.visit(value))

    def visit_Return(self, node):
        self.line("return" + (" " + self.visit(node.value) if node.value is not None else "") + ";")

    def visit_If(self, node):
        self.line("if (" + self.visit(node.test) + ") {")
        self.indent += 1
        self.block(node.body)
        self.indent -= 1
        if node.orelse:
            self.line("} else {")
            self.indent += 1
            self.block(node.orelse)
            self.indent -= 1
        self.line("}")

    def visit_For(self, node):
        call = node.iter
        if not isinstance(call, ast.Call) or qualified(call.func) != "range" or call.keywords or not 1 <= len(call.args) <= 3:
            self.error(node, "GPU for loops use range(start, stop, step)")
        flag = self.begin_loop(node)
        arguments = [self.visit(arg) for arg in call.args]
        begin, end, step = ("0", arguments[0], "1") if len(arguments) == 1 else (*arguments, "1") if len(arguments) == 2 else arguments
        index = node.target.id
        scope = self.loop_sequence
        self.line(f"let inx_start_{scope} = {begin};")
        self.line(f"let inx_stop_{scope} = {end};")
        self.line(f"let inx_step_{scope} = {step};")
        self.line(f"for ({index} = inx_start_{scope}; inx_step_{scope} > 0 ? {index} < inx_stop_{scope} : {index} > inx_stop_{scope}; {index} += inx_step_{scope}) {{")
        self.indent += 1
        self.block(node.body)
        self.indent -= 1
        self.line("}")
        self.end_loop(node, flag)

    def visit_While(self, node):
        flag = self.begin_loop(node)
        self.line("while (" + self.visit(node.test) + ") {")
        self.indent += 1
        self.block(node.body)
        self.indent -= 1
        self.line("}")
        self.end_loop(node, flag)

    def begin_loop(self, node):
        flag = "inx_loop_broken_" + str(self.loop_sequence) if node.orelse else None
        self.loop_sequence += 1
        if flag:
            self.line("bool " + flag + " = false;")
        self.loop_flags.append(flag)
        return flag

    def end_loop(self, node, flag):
        self.loop_flags.pop()
        if flag:
            self.line("if (!" + flag + ") {")
            self.indent += 1
            self.block(node.orelse)
            self.indent -= 1
            self.line("}")

    def visit_Break(self, node):
        if not self.loop_flags:
            self.error(node, "break belongs to a GPU loop")
        if self.loop_flags[-1]:
            self.line(self.loop_flags[-1] + " = true;")
        self.line("break;")

    def visit_Continue(self, node):
        if not self.loop_flags:
            self.error(node, "continue belongs to a GPU loop")
        self.line("continue;")

    def visit_Pass(self, node):
        pass

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            self.line(self.visit(node.value) + ";")

    def generic_visit(self, node):
        self.error(node, "Unsupported GPU syntax: " + type(node).__name__)
