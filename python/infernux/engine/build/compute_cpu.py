"""Build-time lowering of ``inx.compute`` kernels for CPU-only targets."""

from __future__ import annotations

import ast
import copy

from infernux._compiler.source_metadata import compute_decorator_names
from infernux._compiler.kernel_contract import execution_domain, attribute_name as _attribute_name
from .compute_vector import analyze_kernel, VectorLowering


def _compute_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    return (compute_decorator_names(tree, kinds=("kernel",)),
            compute_decorator_names(tree, kinds=("atomic_add",)))


class _CpuComputeLowering(ast.NodeTransformer):
    def __init__(self, atomics: set[str]) -> None:
        self.atomics = atomics
        self.rewrote = False

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


class _ComputeFunctionCook(ast.NodeTransformer):
    def __init__(self, kernels: set[str], atomics: set[str], indices: set[str], symbols) -> None:
        self.kernels = kernels
        self.atomics = atomics
        self.indices = indices
        self.symbols = symbols
        self.vectorized = False
        self.rewrote = False
        self.direct_decorator = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_kernel = any(
            _attribute_name(item.func if isinstance(item, ast.Call) else item) in self.kernels
            for item in node.decorator_list
        )
        if not is_kernel:
            return self.generic_visit(node)
        domain_parameter = execution_domain(node, self.indices)[3]
        plan = analyze_kernel(node, self.indices, self.atomics)
        vectorized = plan.ordered_reason is None
        for position, decorator in enumerate(node.decorator_list):
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if _attribute_name(target) not in self.kernels:
                continue
            if isinstance(target, ast.Attribute):
                target.attr = "_cpu_kernel"
            elif isinstance(target, ast.Name):
                target = ast.Attribute(ast.Name(self.symbols[2], ast.Load()), "_cpu_kernel", ast.Load())
                self.direct_decorator = True
            node.decorator_list[position] = ast.Call(target, [], [
                ast.keyword("domain_parameter", ast.Constant(domain_parameter)),
                ast.keyword("vectorized", ast.Constant(vectorized)),
            ])
        self.rewrote = True
        if not vectorized:
            return self.generic_visit(node)
        node = VectorLowering(node, plan, self.indices, self.atomics, self.symbols).lower(node)
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
    indices = compute_decorator_names(tree, kinds=("index",))
    occupied = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            occupied.add(node.id)
        elif isinstance(node, ast.arg):
            occupied.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            occupied.add(node.name)
        elif isinstance(node, ast.alias):
            occupied.add(node.asname or node.name.partition('.')[0])
    symbols = []
    for stem in ('_inx_cpu_np', '_inx_cpu_lanes', '_inx_cpu_compute'):
        candidate = stem
        while candidate in occupied:
            candidate += '_'
        occupied.add(candidate)
        symbols.append(candidate)
    vector = _ComputeFunctionCook(kernels, atomics, indices, symbols)
    tree = vector.visit(tree)
    lowering = _CpuComputeLowering(atomics)
    tree = lowering.visit(tree)
    if not lowering.rewrote and not vector.rewrote:
        return source
    insertion = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(
        tree.body[0].value, ast.Constant
    ) and isinstance(tree.body[0].value.value, str):
        insertion = 1
    while insertion < len(tree.body) and isinstance(tree.body[insertion], ast.ImportFrom) \
            and tree.body[insertion].module == "__future__":
        insertion += 1
    tree.body.insert(insertion, ast.Import(names=[ast.alias("numpy", symbols[0])]))
    if vector.direct_decorator or vector.vectorized:
        tree.body.insert(insertion, ast.Import(names=[ast.alias("infernux.compute", symbols[2])]))
    if vector.vectorized:
        tree.body.insert(insertion, ast.Import(names=[ast.alias("infernux._compiler.cpu_lanes", symbols[1])]))
    ast.fix_missing_locations(tree)
    cooked = ast.unparse(tree) + "\n"
    compile(cooked, "<web-cpu-compute>", "exec")
    return cooked


__all__ = ["build_cpu_compute_source"]
