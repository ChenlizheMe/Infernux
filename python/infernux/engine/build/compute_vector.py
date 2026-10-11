"""Compile explicit work-item shapes and control dependencies into NumPy.

Uniform expressions stay scalar. Varying locals have one leading work-item
axis; each block evaluates only its active indices. In particular, array
gathers and short-circuit operands are never evaluated in inactive lanes.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from infernux._compiler.kernel_contract import attribute_name, execution_domain


_BINARY = {ast.Add: 'add', ast.Sub: 'subtract', ast.Mult: 'multiply', ast.Div: 'true_divide',
           ast.FloorDiv: 'floor_divide', ast.Mod: 'remainder', ast.Pow: 'power',
           ast.BitAnd: 'bitwise_and', ast.BitOr: 'bitwise_or', ast.BitXor: 'bitwise_xor',
           ast.LShift: 'left_shift', ast.RShift: 'right_shift'}
_COMPARE = {ast.Eq: 'equal', ast.NotEq: 'not_equal', ast.Lt: 'less', ast.LtE: 'less_equal',
            ast.Gt: 'greater', ast.GtE: 'greater_equal'}
_MATH = {'sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'floor', 'ceil', 'fabs'}
_PURE = {'abs', 'min', 'max', 'len', *(f'math.{name}' for name in _MATH)}


def buffer_access(node):
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.value, ast.Subscript):
        return node.value.value, node.value.slice, node.slice
    if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
        return node.value, node.slice.elts[0], node.slice.elts[1]
    return node.value, node.slice, None


@dataclass(frozen=True)
class VectorPlan:
    work_index: str
    domain: str
    varying: frozenset[str]
    ordered_reason: str | None

    def varies(self, node):
        return any(isinstance(item, ast.Name) and item.id in self.varying for item in ast.walk(node))


def analyze_kernel(definition, indices, atomics):
    _, work_index, domain, _ = execution_domain(definition, indices)
    varying = {work_index}

    def varies(node):
        return any(isinstance(item, ast.Name) and item.id in varying for item in ast.walk(node))

    def propagate(statements, divergent=False):
        for node in statements:
            if isinstance(node, ast.If):
                controlled = divergent or varies(node.test)
                propagate(node.body, controlled)
                propagate(node.orelse, controlled)
            elif isinstance(node, (ast.For, ast.While)):
                propagate(node.body, divergent)
                propagate(node.orelse, divergent)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if node.value is not None and (divergent or varies(node.value)):
                    varying.update(target.id for target in targets if isinstance(target, ast.Name))

    while True:
        before = set(varying)
        propagate(definition.body)
        if varying == before:
            break

    reason = None

    def ordered(message):
        nonlocal reason
        if reason is None:
            reason = message

    def expression(node):
        if isinstance(node, (ast.Name, ast.Constant)):
            return
        if isinstance(node, ast.Attribute):
            expression(node.value)
        elif isinstance(node, ast.Subscript):
            expression(node.value)
            expression(node.slice)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                expression(item)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            expression(node.left)
            expression(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.Invert, ast.UAdd, ast.USub)):
            expression(node.operand)
        elif isinstance(node, ast.BoolOp):
            for item in node.values:
                expression(item)
        elif isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            for item in (node.left, *node.comparators):
                expression(item)
        elif isinstance(node, ast.IfExp):
            expression(node.test)
            expression(node.body)
            expression(node.orelse)
        elif isinstance(node, ast.Call) and attribute_name(node.func) in indices | _PURE:
            for item in node.args:
                expression(item)
            if node.keywords:
                ordered('call keywords require the scalar work-item contract')
            function = attribute_name(node.func)
            if function in {'min', 'max'} and len(node.args) != 2:
                ordered('an item-local reduction requires scalar work-item execution')
            if function == 'len' and any(varies(item) for item in node.args):
                ordered('an item-local length requires scalar work-item execution')
        else:
            ordered('expression has no batch operation in the work-item contract')

    def target(node):
        if isinstance(node, ast.Name):
            return
        access = buffer_access(node)
        if access is None:
            ordered('write has no explicit scalar or buffer address')
            return
        base, index, lane = access
        if isinstance(base, ast.Name) and base.id in varying:
            # A lane of an item-local vector, not an indirect shared write.
            expression(index)
        elif not isinstance(index, ast.Name) or index.id != work_index:
            ordered('shared write does not have a proven unique work-item address')
        if lane is not None:
            expression(lane)

    def statements(nodes):
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                if node.value is None:
                    continue
                expression(node.value)
                for item in node.targets if isinstance(node, ast.Assign) else (node.target,):
                    target(item)
            elif isinstance(node, ast.If):
                expression(node.test)
                statements(node.body)
                statements(node.orelse)
            elif isinstance(node, ast.For):
                if (not isinstance(node.target, ast.Name) or node.target.id in varying
                    or not isinstance(node.iter, ast.Call) or attribute_name(node.iter.func) != 'range'
                    or not 1 <= len(node.iter.args) <= 3 or node.iter.keywords
                    or any(varies(item) for item in node.iter.args) or node.orelse):
                    ordered('loop bounds or control vary across work items')
                statements(node.body)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and attribute_name(node.value.func) in atomics:
                call = node.value
                if len(call.args) != 2 or call.keywords or buffer_access(call.args[0]) is None:
                    ordered('atomic write requires an explicit buffer address and value')
                else:
                    expression(call.args[0])
                    expression(call.args[1])
            elif isinstance(node, ast.Pass) or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            else:
                ordered('control or effect requires ordered work-item execution')

    statements(definition.body)
    # Distinguish independent item writes from visible cross-item effects.
    # A same-storage gather after/before a write cannot be lifted into a
    # whole-array statement without changing the ordered Web contract.
    written = set()
    atomic_targets = set()
    for node in ast.walk(definition):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        elif isinstance(node, ast.Call) and attribute_name(node.func) in atomics and node.args:
            targets = [node.args[0]]
            part = node.args[0]
            while isinstance(part, ast.Subscript):
                atomic_targets.add(id(part))
                part = part.value
        for item in targets:
            access = buffer_access(item)
            if access is not None and isinstance(access[0], ast.Name) and access[0].id not in varying:
                written.add(access[0].id)
    for node in ast.walk(definition):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and id(node) not in atomic_targets:
            base, index, _ = buffer_access(node)
            if isinstance(base, ast.Name) and base.id in written:
                if not isinstance(index, ast.Name) or index.id != work_index:
                    ordered('a gather observes storage written by other work items')
    parameters = {argument.arg for argument in (*definition.args.posonlyargs, *definition.args.args)}
    if varying.intersection(parameters):
        ordered('mutable parameters require the scalar work-item binding')
    stores_to_index = sum(isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == work_index
                          for node in ast.walk(definition))
    if stores_to_index != 1:
        ordered('the work-item address is reassigned')
    return VectorPlan(work_index, domain, frozenset(varying), reason)


@dataclass
class Value:
    prefix: list
    node: ast.expr
    varying: bool


def name(value, ctx=None):
    return ast.Name(value, ctx or ast.Load())


def call(owner, member, *args, **kwargs):
    return ast.Call(ast.Attribute(name(owner), member, ast.Load()), list(args),
                    [ast.keyword(key, value) for key, value in kwargs.items()])


def assign(target, value):
    return ast.Assign([name(target, ast.Store())], value)


class VectorLowering:
    def __init__(self, definition, plan, indices, atomics, symbols):
        self.plan, self.indices, self.atomics = plan, indices, atomics
        self.numpy, self.lanes, self.compute = symbols
        self.reserved = {node.id for node in ast.walk(definition) if isinstance(node, ast.Name)}
        self.reserved.update(node.arg for node in ast.walk(definition) if isinstance(node, ast.arg))
        self.serial = 0
        self.active = self.fresh()
        self.width = self.fresh()

    def fresh(self):
        while True:
            self.serial += 1
            candidate = f'__inx_lane_{self.serial}'
            if candidate not in self.reserved:
                self.reserved.add(candidate)
                return candidate

    def binary(self, operation, left, right):
        return Value([*left.prefix, *right.prefix], call(self.lanes, 'binary',
            ast.Attribute(name(self.numpy), operation, ast.Load()), left.node, right.node,
            ast.Constant(left.varying), ast.Constant(right.varying)), left.varying or right.varying)

    def expr(self, node, active):
        varying = self.plan.varies(node)
        if isinstance(node, ast.Name):
            if node.id == self.plan.work_index:
                return Value([], name(active), True)
            value = copy.deepcopy(node)
            if varying and active != self.active:
                value = ast.Subscript(value, name(active), ast.Load())
            return Value([], value, varying)
        if isinstance(node, ast.Constant):
            return Value([], copy.deepcopy(node), False)
        if isinstance(node, ast.BinOp):
            return self.binary(_BINARY[type(node.op)], self.expr(node.left, active), self.expr(node.right, active))
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, active)
            if isinstance(node.op, ast.Not) and varying:
                result = call(self.numpy, 'logical_not', value.node)
            else:
                result = ast.UnaryOp(copy.deepcopy(node.op), value.node)
            return Value(value.prefix, result, varying)
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1:
                return self.binary(_COMPARE[type(node.ops[0])], self.expr(node.left, active), self.expr(node.comparators[0], active))
            operands = [node.left, *node.comparators]
            return self.expr(ast.BoolOp(ast.And(), [ast.Compare(operands[i], [op], [operands[i + 1]])
                                                   for i, op in enumerate(node.ops)]), active)
        if isinstance(node, ast.BoolOp):
            if not varying:
                return Value([], copy.deepcopy(node), False)
            left = self.expr(node.values[0], active)
            remainder = node.values[1] if len(node.values) == 2 else ast.BoolOp(node.op, node.values[1:])
            return self.choice(left, remainder if isinstance(node.op, ast.And) else None,
                               None if isinstance(node.op, ast.And) else remainder, active)
        if isinstance(node, ast.IfExp):
            return self.choice(self.expr(node.test, active), node.body, node.orelse, active)
        if isinstance(node, ast.Subscript):
            base, index, lane = buffer_access(node)
            if isinstance(base, ast.Name) and base.id in self.plan.varying:
                # v[0] addresses a component of every active item-local vector.
                item = self.expr(index, active)
                address = ast.Tuple([name(active), item.node], ast.Load())
                return Value(item.prefix, ast.Subscript(copy.deepcopy(base), address, ast.Load()), True)
            base_value, index_value = self.expr(base, active), self.expr(index, active)
            address = index_value.node
            if (active == self.active and isinstance(index, ast.Name) and index.id == self.plan.work_index
                and (lane is None or not self.plan.varies(lane))):
                address = ast.Slice(upper=name(self.width))
            prefix = [*base_value.prefix, *index_value.prefix]
            if lane is not None:
                lane_value = self.expr(lane, active)
                prefix.extend(lane_value.prefix)
                address = ast.Tuple([address, lane_value.node], ast.Load())
            return Value(prefix, ast.Subscript(base_value.node, address, ast.Load()), varying)
        if isinstance(node, ast.Attribute):
            base = self.expr(node.value, active)
            return Value(base.prefix, ast.Attribute(base.node, node.attr, ast.Load()), varying)
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.expr(item, active) for item in node.elts]
            prefix = [statement for value in values for statement in value.prefix]
            if varying:
                result = call(self.lanes, 'pack', ast.List([value.node for value in values], ast.Load()),
                              ast.List([ast.Constant(value.varying) for value in values], ast.Load()),
                              ast.Call(name('len'), [name(active)], []))
            else:
                result = type(node)([value.node for value in values], ast.Load())
            return Value(prefix, result, varying)
        if isinstance(node, ast.Call):
            function = attribute_name(node.func)
            values = [self.expr(item, active) for item in node.args]
            prefix = [statement for value in values for statement in value.prefix]
            if function in {'min', 'max'} and len(values) == 2:
                return self.binary('minimum' if function == 'min' else 'maximum', *values)
            function_node = copy.deepcopy(node.func)
            if function.startswith('math.') and function.partition('.')[2] in _MATH:
                function_node = ast.Attribute(name(self.numpy), function.partition('.')[2], ast.Load())
            return Value(prefix, ast.Call(function_node, [value.node for value in values], []), varying)
        raise AssertionError(f'unplanned vector expression: {ast.dump(node)}')

    def choice(self, test, yes, no, active):
        if not test.varying and not any(branch is not None and self.plan.varies(branch) for branch in (yes, no)):
            captured, result = self.fresh(), self.fresh()
            branches = []
            for branch in (yes, no):
                value = self.expr(branch, active) if branch is not None else Value([], name(captured), False)
                branches.append([*value.prefix, assign(result, value.node)])
            return Value([*test.prefix, assign(captured, test.node), ast.If(name(captured), *branches)], name(result), False)
        captured, true_positions, false_positions, result = (self.fresh() for _ in range(4))
        prefix = [*test.prefix, assign(captured, test.node), assign(result, ast.Constant(None)),
                  ast.Assign([ast.Tuple([name(true_positions, ast.Store()), name(false_positions, ast.Store())], ast.Store())],
                             call(self.lanes, 'partition', ast.Call(name('len'), [name(active)], []), name(captured)))]
        for positions, branch in ((true_positions, yes), (false_positions, no)):
            branch_active = self.fresh()
            if branch is None:
                value = Value([], ast.Subscript(name(captured), name(positions), ast.Load()) if test.varying else name(captured), test.varying)
            else:
                value = self.expr(branch, branch_active)
            branch_indices = name(positions) if active == self.active else ast.Subscript(name(active), name(positions), ast.Load())
            body = [assign(branch_active, branch_indices), *value.prefix,
                    assign(result, call(self.lanes, 'store', name(result), name(positions), value.node,
                        ast.Constant(value.varying), ast.Call(name('len'), [name(active)], []), promote=ast.Constant(True)))]
            prefix.append(ast.If(ast.Call(name('len'), [name(positions)], []), body, []))
        return Value(prefix, name(result), True)

    def write(self, target, value, active, operation=None):
        if isinstance(target, ast.Name):
            if operation is not None:
                value = self.binary(operation, self.expr(name(target.id), active), value)
            if target.id in self.plan.varying:
                indices = ast.Call(name('slice'), [name(self.width)], []) if active == self.active else name(active)
                return [*value.prefix, assign(target.id, call(self.lanes, 'store', name(target.id), indices,
                            value.node, ast.Constant(value.varying), name(self.width)))]
            return [*value.prefix, assign(target.id, value.node)]
        base, index, lane = buffer_access(target)
        if isinstance(base, ast.Name) and base.id in self.plan.varying:
            lane, index = index, name(active)
            index_value = Value([], index, True)
            contiguous = active == self.active
        else:
            index_value = self.expr(index, active)
            contiguous = active == self.active and isinstance(index, ast.Name) and index.id == self.plan.work_index
        lane_value = self.expr(lane, active) if lane is not None else Value([], ast.Constant(None), False)
        if contiguous and not lane_value.varying:
            index_value = Value([], ast.Call(name('slice'), [name(self.width)], []), True)
        op = ast.Attribute(name(self.numpy), operation, ast.Load()) if operation else ast.Constant(None)
        return [*value.prefix, *index_value.prefix, *lane_value.prefix,
                ast.Expr(call(self.lanes, 'write', copy.deepcopy(base), index_value.node, lane_value.node,
                              value.node, ast.Constant(value.varying), op))]

    def statements(self, nodes, active):
        output = []
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                if node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if (len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == self.plan.work_index
                    and isinstance(node.value, ast.Call) and attribute_name(node.value.func) in self.indices):
                    continue
                value = self.expr(node.value, active)
                if len(targets) > 1:
                    captured = self.fresh()
                    output.extend([*value.prefix, assign(captured, value.node)])
                    value = Value([], name(captured), value.varying)
                for target in targets:
                    output.extend(self.write(target, value, active, _BINARY[type(node.op)] if isinstance(node, ast.AugAssign) else None))
            elif isinstance(node, ast.If):
                condition = self.expr(node.test, active)
                if not condition.varying:
                    output.extend([*condition.prefix, ast.If(condition.node, self.statements(node.body, active), self.statements(node.orelse, active))])
                    continue
                captured, truth, falsehood = (self.fresh() for _ in range(3))
                output.extend([*condition.prefix, assign(captured, condition.node)])
                if node.orelse:
                    output.append(ast.Assign([ast.Tuple([name(truth, ast.Store()), name(falsehood, ast.Store())], ast.Store())],
                        call(self.lanes, 'partition', ast.Call(name('len'), [name(active)], []), name(captured))))
                else:
                    output.append(assign(truth, call(self.lanes, 'selected', ast.Call(name('len'), [name(active)], []), name(captured))))
                for positions, body in ((truth, node.body), (falsehood, node.orelse)):
                    if body:
                        branch = self.fresh()
                        branch_indices = name(positions) if active == self.active else ast.Subscript(name(active), name(positions), ast.Load())
                        output.append(ast.If(ast.Call(name('len'), [name(positions)], []),
                            [assign(branch, branch_indices), *self.statements(body, branch)], []))
            elif isinstance(node, ast.For):
                iterator = self.expr(node.iter, active)
                output.extend([*iterator.prefix, ast.For(copy.deepcopy(node.target), iterator.node, self.statements(node.body, active), [])])
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                source = node.value
                base, index, lane = buffer_access(source.args[0])
                indices, value = self.expr(index, active), self.expr(source.args[1], active)
                target = copy.deepcopy(base)
                if lane is not None:
                    target = ast.Subscript(target, ast.Tuple([ast.Slice(), copy.deepcopy(lane)], ast.Load()), ast.Load())
                output.extend([*indices.prefix, *value.prefix,
                    ast.Expr(call(self.compute, '_cpu_atomic_add', target, indices.node, value.node,
                        call(self.numpy, 'ones', ast.Call(name('len'), [name(active)], []),
                             dtype=ast.Attribute(name(self.numpy), 'bool_', ast.Load()))))])
            else:
                output.append(copy.deepcopy(node))
        return output

    def lower(self, definition):
        locals_init = [assign(variable, ast.Constant(None)) for variable in sorted(self.plan.varying - {self.plan.work_index})]
        body = self.statements(definition.body, self.active)
        reads = {item.id for statement in body for item in ast.walk(statement)
                 if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
        prefix = [assign(self.width, ast.Call(name('len'), [name(self.plan.domain)], []))]
        if self.active in reads or self.plan.work_index in reads:
            prefix.append(assign(self.active, call(self.numpy, 'arange', name(self.width),
                                                   dtype=ast.Attribute(name(self.numpy), 'intp', ast.Load()))))
        if self.plan.work_index in reads:
            prefix.append(assign(self.plan.work_index, name(self.active)))
        definition.body = [*prefix, *locals_init, *body]
        return definition
