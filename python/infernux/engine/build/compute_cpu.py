"""Build-time lowering of ``inx.compute`` kernels for CPU-only targets."""

from __future__ import annotations

import ast
import copy

from infernux._compiler.source_metadata import compute_decorator_names


def _attribute_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _compute_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    return (compute_decorator_names(tree, kinds=("kernel",)),
            compute_decorator_names(tree, kinds=("atomic_add",)))


class _CpuComputeLowering(ast.NodeTransformer):
    def __init__(self, kernels: set[str], atomics: set[str]) -> None:
        self.kernels = kernels
        self.atomics = atomics
        self.rewrote = False

    def _is_kernel(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        return any(
            _attribute_name(item.func if isinstance(item, ast.Call) else item)
            in self.kernels
            for item in node.decorator_list
        )

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            call = node.value
            if _attribute_name(call.func) in self.atomics:
                if len(call.args) != 2 or call.keywords:
                    raise ValueError(
                        f"compute.atomic_add at line {node.lineno} requires target and value"
                    )
                target = copy.deepcopy(call.args[0])
                if not isinstance(target, ast.Subscript):
                    raise ValueError(
                        f"compute.atomic_add at line {node.lineno} requires a buffer element target"
                    )
                target.ctx = ast.Store()
                replacement = ast.AugAssign(
                    target=target,
                    op=ast.Add(),
                    value=self.visit(call.args[1]),
                )
                self.rewrote = True
                return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if _attribute_name(node.func) in self.atomics:
            raise ValueError(
                f"compute.atomic_add at line {node.lineno} must be a standalone statement"
            )
        return self.generic_visit(node)


def _is_call(node: ast.AST, names: set[str]) -> bool:
    return isinstance(node, ast.Call) and _attribute_name(node.func) in names


def _buffer_access(node: ast.expr):
    """Return ``(buffer, index, optional_lane)`` for authored buffer access."""
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.value, ast.Subscript):
        inner = node.value
        return inner.value, inner.slice, node.slice
    if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
        return node.value, node.slice.elts[0], node.slice.elts[1]
    return node.value, node.slice, None


class _VectorKernelLowering(ast.NodeTransformer):
    """Lift independent SIMT work items into NumPy array operations."""

    def __init__(self, index_names: set[str], atomic_names: set[str]) -> None:
        self.index_names = index_names
        self.atomic_names = atomic_names
        self.work_index = ""
        self.active_masks: list[ast.expr] = []
        self.defined: set[str] = set()
        self.mask_serial = 0
        self.scalar_loop_depth = 0

    @property
    def active_mask(self):
        return copy.deepcopy(self.active_masks[-1]) if self.active_masks else None

    def _masked_value(self, new: ast.expr, old: ast.expr) -> ast.expr:
        return ast.Call(
            func=ast.Attribute(ast.Name("__inx_np", ast.Load()), "where", ast.Load()),
            args=[self.active_mask, new, old],
            keywords=[],
        )

    def _masked_target(self, target: ast.Subscript) -> ast.Subscript:
        access = _buffer_access(target)
        if access is None:
            raise ValueError("CPU vector cook requires a buffer assignment target")
        base, index, lane = access
        if not isinstance(index, ast.Name) or index.id != self.work_index:
            raise ValueError("CPU vector cook cannot mask an indirect buffer assignment")
        items = [self.active_mask]
        if lane is not None:
            items.append(self.visit(lane))
        sliced = items[0] if len(items) == 1 else ast.Tuple(items, ast.Load())
        return ast.Subscript(self.visit(base), sliced, target.ctx)

    def visit_Assign(self, node: ast.Assign):
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_call(node.value, self.index_names)
        ):
            domain = self.visit(node.value.args[0])
            self.work_index = node.targets[0].id
            self.defined.add(self.work_index)
            value = ast.Call(
                func=ast.Attribute(ast.Name("__inx_np", ast.Load()), "arange", ast.Load()),
                args=[ast.Call(ast.Name("len", ast.Load()), [domain], [])],
                keywords=[ast.keyword("dtype", ast.Attribute(ast.Name("__inx_np", ast.Load()), "int64", ast.Load()))],
            )
            return ast.copy_location(ast.Assign(node.targets, value), node)
        value = self.visit(node.value)
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                if self.active_masks and target.id in self.defined:
                    value = self._masked_value(value, ast.Name(target.id, ast.Load()))
                self.defined.add(target.id)
                targets.append(target)
            elif isinstance(target, ast.Subscript):
                transformed = self.visit(target)
                targets.append(
                    self._masked_target(target) if self.active_masks else transformed
                )
                if self.active_masks:
                    value = ast.Subscript(value, self.active_mask, ast.Load())
            else:
                targets.append(self.visit(target))
        return ast.copy_location(ast.Assign(targets, value), node)

    def visit_AugAssign(self, node: ast.AugAssign):
        value = self.visit(node.value)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            self.defined.add(name)
            if self.active_masks:
                combined = ast.BinOp(ast.Name(name, ast.Load()), node.op, value)
                return ast.copy_location(
                    ast.Assign([ast.Name(name, ast.Store())], self._masked_value(combined, ast.Name(name, ast.Load()))),
                    node,
                )
            return ast.copy_location(ast.AugAssign(node.target, node.op, value), node)
        target = self.visit(node.target)
        if self.active_masks:
            target = self._masked_target(node.target)
            value = ast.Subscript(value, self.active_mask, ast.Load())
        return ast.copy_location(ast.AugAssign(target, node.op, value), node)

    def visit_If(self, node: ast.If):
        condition = self.visit(node.test)
        outer = self.active_mask
        if outer is not None:
            condition = ast.Call(
                ast.Attribute(ast.Name("__inx_np", ast.Load()), "logical_and", ast.Load()),
                [outer, condition], [],
            )
        self.mask_serial += 1
        mask_name = f"__inx_mask_{self.mask_serial}"
        assignment = ast.Assign([ast.Name(mask_name, ast.Store())], condition)
        self.active_masks.append(ast.Name(mask_name, ast.Load()))
        body = [item for statement in node.body for item in _statement_list(self.visit(statement))]
        self.active_masks.pop()
        output = [ast.copy_location(assignment, node)]
        if self.scalar_loop_depth:
            any_active = ast.Call(
                ast.Attribute(ast.Name("__inx_np", ast.Load()), "any", ast.Load()),
                [ast.Name(mask_name, ast.Load())],
                [],
            )
            output.append(ast.copy_location(ast.If(any_active, body, []), node))
        else:
            output.extend(body)
        if node.orelse:
            inverted = ast.UnaryOp(ast.Invert(), ast.Name(mask_name, ast.Load()))
            if outer is not None:
                inverted = ast.Call(
                    ast.Attribute(ast.Name("__inx_np", ast.Load()), "logical_and", ast.Load()),
                    [outer, inverted], [],
                )
            self.mask_serial += 1
            other_name = f"__inx_mask_{self.mask_serial}"
            output.append(ast.Assign([ast.Name(other_name, ast.Store())], inverted))
            self.active_masks.append(ast.Name(other_name, ast.Load()))
            other_body = [
                item
                for statement in node.orelse
                for item in _statement_list(self.visit(statement))
            ]
            self.active_masks.pop()
            if self.scalar_loop_depth:
                any_other = ast.Call(
                    ast.Attribute(ast.Name("__inx_np", ast.Load()), "any", ast.Load()),
                    [ast.Name(other_name, ast.Load())],
                    [],
                )
                output.append(ast.If(any_other, other_body, []))
            else:
                output.extend(other_body)
        return output

    def visit_For(self, node: ast.For):
        target = self.visit(node.target)
        iterator = self.visit(node.iter)
        self.scalar_loop_depth += 1
        try:
            body = [
                item
                for statement in node.body
                for item in _statement_list(self.visit(statement))
            ]
        finally:
            self.scalar_loop_depth -= 1
        return ast.copy_location(
            ast.For(target=target, iter=iterator, body=body, orelse=[], type_comment=node.type_comment),
            node,
        )

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and _attribute_name(node.value.func) in self.atomic_names:
            call = node.value
            if len(call.args) != 2 or call.keywords:
                raise ValueError("CPU vector atomic_add requires target and value")
            access = _buffer_access(call.args[0])
            if access is None:
                raise ValueError("CPU vector atomic_add requires a buffer target")
            base, index, lane = access
            target_slice = ast.Slice()
            if lane is not None:
                target_slice = ast.Tuple([target_slice, self.visit(lane)], ast.Load())
            target = ast.Subscript(self.visit(base), target_slice, ast.Load())
            indices = self.visit(index)
            value = self.visit(call.args[1])
            helper = copy.deepcopy(call.func)
            if not isinstance(helper, ast.Attribute):
                raise ValueError(
                    "CPU vector atomic_add requires a qualified compute decorator"
                )
            helper.attr = "_cpu_atomic_add"
            mask = self.active_mask
            if mask is None:
                mask = ast.Call(
                    ast.Attribute(ast.Name("__inx_np", ast.Load()), "ones_like", ast.Load()),
                    [ast.Name(self.work_index, ast.Load())],
                    [ast.keyword("dtype", ast.Attribute(ast.Name("__inx_np", ast.Load()), "bool_", ast.Load()))],
                )
            result = ast.Call(
                helper,
                [
                    target,
                    indices,
                    value,
                    mask,
                ],
                [],
            )
            return ast.copy_location(ast.Expr(result), node)
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        access = _buffer_access(node)
        if access is None:
            return self.generic_visit(node)
        base, index, lane = access
        index = self.visit(index)
        if isinstance(index, ast.Name) and index.id == self.work_index:
            index = ast.Slice()
        items = [index]
        if lane is not None:
            items.append(self.visit(lane))
        sliced = items[0] if len(items) == 1 else ast.Tuple(items, ast.Load())
        return ast.copy_location(ast.Subscript(self.visit(base), sliced, node.ctx), node)

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if _attribute_name(node.func) == "math.sqrt":
            node.func = ast.Attribute(ast.Name("__inx_np", ast.Load()), "sqrt", ast.Load())
        return node

    def visit_IfExp(self, node: ast.IfExp):
        return ast.copy_location(ast.Call(
            ast.Attribute(ast.Name("__inx_np", ast.Load()), "where", ast.Load()),
            [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)], [],
        ), node)

    def visit_BoolOp(self, node: ast.BoolOp):
        function = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        values = [self.visit(value) for value in node.values]
        result = values[0]
        for value in values[1:]:
            result = ast.Call(
                ast.Attribute(ast.Name("__inx_np", ast.Load()), function, ast.Load()),
                [result, value], [],
            )
        return ast.copy_location(result, node)


def _statement_list(value):
    if value is None:
        return ()
    return value if isinstance(value, list) else (value,)


def _supported_vector_loop(node: ast.For) -> bool:
    """A scalar range loop may surround an array-vectorized work-item body."""
    if not isinstance(node.target, ast.Name) or node.orelse:
        return False
    if not _is_call(node.iter, {"range"}) or not 1 <= len(node.iter.args) <= 3:
        return False
    return not any(
        isinstance(item, (ast.Break, ast.Continue, ast.AsyncFor))
        for statement in node.body
        for item in ast.walk(statement)
    )


def _can_vectorize(node: ast.FunctionDef | ast.AsyncFunctionDef, atomic_names: set[str]) -> bool:
    if isinstance(node, ast.AsyncFunctionDef) or any(isinstance(item, ast.While) for item in ast.walk(node)):
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.For) and not _supported_vector_loop(item):
            return False
        if isinstance(item, (ast.Assign, ast.AugAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else (item.target,)
            for target in targets:
                access = _buffer_access(target) if isinstance(target, ast.Subscript) else None
                if access is not None and isinstance(access[1], ast.Constant):
                    return False
        if isinstance(item, ast.Call) and _attribute_name(item.func) in atomic_names:
            access = _buffer_access(item.args[0]) if item.args else None
            if access is None:
                return False
    return True


class _ComputeFunctionCook(ast.NodeTransformer):
    def __init__(self, kernels: set[str], atomics: set[str], indices: set[str]) -> None:
        self.kernels = kernels
        self.atomics = atomics
        self.indices = indices
        self.vectorized = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_kernel = any(
            _attribute_name(item.func if isinstance(item, ast.Call) else item) in self.kernels
            for item in node.decorator_list
        )
        if not is_kernel or not _can_vectorize(node, self.atomics):
            return self.generic_visit(node)
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _attribute_name(target) not in self.kernels:
                continue
            if isinstance(target, ast.Attribute):
                target.attr = "_cpu_kernel"
            elif isinstance(target, ast.Name):
                # Direct imported decorators have no stable owner to retarget.
                return self.generic_visit(node)
        lowering = _VectorKernelLowering(self.indices, self.atomics)
        body = [item for statement in node.body for item in _statement_list(lowering.visit(statement))]
        if not lowering.work_index:
            return self.generic_visit(node)
        index_assignment = next((
            index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == lowering.work_index
            and _is_call(statement.value, {"__inx_np.arange"})
        ), None)
        if index_assignment is not None and not any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id == lowering.work_index
            for index, statement in enumerate(body)
            if index != index_assignment
            for item in ast.walk(statement)
        ):
            body.pop(index_assignment)
        node.body = body
        self.vectorized = True
        return node

def build_cpu_compute_source(source: str) -> str:
    """Cook authored GPU kernels into the deterministic Web CPU backend.

    Independent work items become NumPy array operations. Kernels whose order
    is part of their behavior remain sequential, and atomics use deterministic
    scatter semantics. Public ``device='gpu'`` declarations stay unchanged.
    """
    tree = ast.parse(source)
    kernels, atomics = _compute_names(tree)
    indices = {name.rsplit(".", 1)[0] + ".index" for name in kernels}
    vector = _ComputeFunctionCook(kernels, atomics, indices)
    tree = vector.visit(tree)
    lowering = _CpuComputeLowering(kernels, atomics)
    tree = lowering.visit(tree)
    if not lowering.rewrote and not vector.vectorized:
        return source
    insertion = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
        tree.body[0].value, ast.Constant
    ) and isinstance(tree.body[0].value.value, str):
        insertion = 1
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom) \
            and tree.body[insertion].module == "__future__":
        insertion += 1
    tree.body.insert(insertion, ast.Import(names=[ast.alias("numpy", "__inx_np")]))
    ast.fix_missing_locations(tree)
    cooked = ast.unparse(tree) + "\n"
    compile(cooked, "<web-cpu-compute>", "exec")
    return cooked


__all__ = ["build_cpu_compute_source"]
