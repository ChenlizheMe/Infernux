"""Static import-time contract for user script candidates.

The policy pass only reads an AST.  It never imports or executes user code.
Ordinary imports, factories, decorators, and class metadata are Python code,
not an engine-maintained capability allowlist. This pass diagnoses explicit
import-time filesystem, process, or live-module mutations; it is not a sandbox
and cannot prove arbitrary library code free of side effects.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScriptCandidatePolicyIssue:
    code: str
    operation: str
    message: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None


@dataclass(frozen=True, slots=True)
class ScriptCandidatePolicyReport:
    blocked: tuple[ScriptCandidatePolicyIssue, ...] = ()
    runtime_guard_required: tuple[ScriptCandidatePolicyIssue, ...] = ()

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked)

    @property
    def requires_runtime_guard(self) -> bool:
        return bool(self.runtime_guard_required)

    @property
    def is_rejected(self) -> bool:
        """Whether candidate loading must stop before any import/execute step."""
        return self.is_blocked or self.requires_runtime_guard


_BLOCKED_MODULE_WRITE = "NX-R1-STATIC-MODULE-WRITE"
_BLOCKED_ENVIRONMENT_WRITE = "NX-R1-STATIC-ENVIRONMENT-WRITE"
_BLOCKED_FILE_WRITE = "NX-R1-STATIC-FILE-WRITE"
_BLOCKED_PROCESS = "NX-R1-STATIC-PROCESS"
_BLOCKED_DYNAMIC_CODE = "NX-R1-STATIC-DYNAMIC-CODE"

_PROCESS_DIRECT_MEMBERS = frozenset(
    {
        ("subprocess", "run"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("subprocess", "Popen"),
        ("os", "system"),
        ("os", "popen"),
        ("multiprocessing", "Process"),
        ("multiprocessing", "Pool"),
        ("multiprocessing", "Manager"),
        ("multiprocessing", "Pipe"),
        ("threading", "Thread"),
        ("threading", "Timer"),
        ("_thread", "start_new_thread"),
        ("socket", "socket"),
        ("socket", "socketpair"),
        ("socket", "create_connection"),
        ("socket", "create_server"),
        ("atexit", "register"),
        ("atexit", "unregister"),
        ("asyncio", "run"),
        ("asyncio", "create_task"),
        ("asyncio", "ensure_future"),
        ("asyncio", "run_coroutine_threadsafe"),
        ("concurrent.futures", "ThreadPoolExecutor"),
        ("concurrent.futures", "ProcessPoolExecutor"),
    }
)
_PATH_MUTATORS = frozenset({"unlink", "rename", "replace", "touch", "mkdir", "rmdir"})
_FILE_MODULE_MEMBERS = frozenset(
    {
        ("os", "remove"),
        ("os", "unlink"),
        ("os", "rename"),
        ("os", "replace"),
        ("os", "mkdir"),
        ("os", "makedirs"),
        ("shutil", "move"),
        ("shutil", "copy"),
        ("shutil", "copy2"),
        ("shutil", "rmtree"),
        ("shutil", "unlink"),
        ("numpy", "save"),
        ("numpy", "savez"),
        ("numpy", "savez_compressed"),
        ("numpy", "savetxt"),
    }
)
_DYNAMIC_CODE_NAMES = frozenset({"exec", "eval", "compile"})


def _decode_source(source: bytes | bytearray | memoryview | str) -> str:
    if isinstance(source, str):
        return source
    payload = bytes(source)
    encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
    return payload.decode(encoding)


def _chain(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _chain(node.value)
        return (*parent, node.attr) if parent else None
    if isinstance(node, ast.Subscript):
        return _chain(node.value)
    return None


def _root_name(node: ast.AST | None) -> str | None:
    chain = _chain(node)
    return chain[0] if chain else None


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _issue(node: ast.AST, code: str, operation: str, message: str) -> ScriptCandidatePolicyIssue:
    return ScriptCandidatePolicyIssue(
        code=code,
        operation=operation,
        message=message,
        line=int(getattr(node, "lineno", 1)),
        column=int(getattr(node, "col_offset", 0)),
        end_line=getattr(node, "end_lineno", None),
        end_column=getattr(node, "end_col_offset", None),
    )


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imported_modules: dict[str, str] = {}
        self.imported_members: dict[str, tuple[str, str]] = {}
        self.path_bindings: set[str] = set()
        self.blocked: list[ScriptCandidatePolicyIssue] = []
        self.runtime_guard_required: list[ScriptCandidatePolicyIssue] = []

    def _blocked(self, node: ast.AST, code: str, operation: str, message: str) -> None:
        self.blocked.append(_issue(node, code, operation, message))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_modules[alias.asname or alias.name.split(".", 1)[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            self.imported_members[bound] = (module, alias.name)
            if module == "pathlib" and alias.name == "Path":
                self.path_bindings.add(bound)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_header(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition_header(node)

    def _visit_definition_header(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Defaults, annotations, and decorators execute at import time.  The
        # function body does not and is intentionally skipped.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_Lambda(self, _node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class bodies execute at import time, but ordinary class declarations
        # and serialized-field calls remain allowed by the normal visitors.
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)

    def _target_is_imported(self, target: ast.AST) -> bool:
        root = _root_name(target)
        return root in self.imported_modules or root in self.imported_members

    def _check_target(self, node: ast.AST, target: ast.AST) -> None:
        if not self._target_is_imported(target):
            return
        chain = _chain(target)
        operation = ".".join(chain) if chain else "imported binding"
        root = _root_name(target) or ""
        module = self.imported_modules.get(root)
        member_module = self.imported_members.get(root, ("", ""))[0]
        code = (
            _BLOCKED_ENVIRONMENT_WRITE
            if (
                (module == "os" and len(chain or ()) > 1 and (chain or ())[1] == "environ")
                or (member_module == "os" and self.imported_members[root][1] == "environ")
            )
            else _BLOCKED_MODULE_WRITE
        )
        self._blocked(
            node,
            code,
            operation,
            f"top-level assignment mutates imported module/member '{operation}'",
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_target(node, target)
        if isinstance(node.value, ast.Call):
            path = self._imported_path(node.value.func)
            if path and path[-1] == "Path":
                self.path_bindings.update(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_target(node, node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_target(node, node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self._check_target(node, target)
        self.generic_visit(node)

    def _imported_path(self, node: ast.AST | None) -> tuple[str, ...] | None:
        chain = _chain(node)
        if not chain:
            return None
        module = self.imported_modules.get(chain[0])
        if module is not None:
            return (module, *chain[1:])
        member = self.imported_members.get(chain[0])
        if member is not None:
            return (member[0], member[1], *chain[1:])
        return None

    def _is_open(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            return True
        path = self._imported_path(node.func)
        return bool(path and path[0] in {"io", "builtins"} and path[-1] == "open")

    def _open_mode(self, node: ast.Call) -> str | None:
        mode_node = node.args[1] if len(node.args) > 1 else None
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_node = keyword.value
        if mode_node is None:
            return "r"
        return _literal_string(mode_node)

    def _is_path_mutator(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr not in _PATH_MUTATORS and not node.func.attr.startswith("write"):
            return False
        receiver = node.func.value
        if isinstance(receiver, ast.Call):
            path = self._imported_path(receiver.func)
            return bool(path and path[-1] == "Path")
        if isinstance(receiver, ast.Name) and receiver.id in self.path_bindings:
            return True
        path = self._imported_path(receiver)
        return bool(path and path[-1] == "Path")

    def visit_Call(self, node: ast.Call) -> None:
        path = self._imported_path(node.func)
        direct_name = node.func.id if isinstance(node.func, ast.Name) else None

        if direct_name in _DYNAMIC_CODE_NAMES or (
            path and path[0] == "builtins" and path[-1] in _DYNAMIC_CODE_NAMES
        ):
            self._blocked(
                node,
                _BLOCKED_DYNAMIC_CODE,
                direct_name or ".".join(path or ()),
                "top-level dynamic code execution is not allowed for a reload candidate",
            )
        elif path and path[0].lstrip(".") == "importlib" and path[-1] == "reload":
            self._blocked(
                node,
                _BLOCKED_DYNAMIC_CODE,
                ".".join(path),
                "top-level importlib.reload is not allowed for a reload candidate",
            )
        elif self._is_open(node):
            mode = self._open_mode(node)
            if mode is None:
                self._blocked(node, _BLOCKED_FILE_WRITE, "open", "top-level open mode must be an explicit read-only mode")
            elif any(flag in mode for flag in "wax+"):
                self._blocked(
                    node,
                    _BLOCKED_FILE_WRITE,
                    "open",
                    "top-level open with a write-capable mode is not allowed",
                )
        elif self._is_path_mutator(node):
            self._blocked(
                node,
                _BLOCKED_FILE_WRITE,
                ".".join(_chain(node.func) or ("Path", node.func.attr)),
                "top-level filesystem mutation is not allowed for a reload candidate",
            )
        elif path:
            module = path[0].lstrip(".")
            if module == "os" and len(path) > 1 and path[1] == "environ":
                self._blocked(
                    node,
                    _BLOCKED_ENVIRONMENT_WRITE,
                    ".".join(path),
                    "top-level environment mutation is not allowed for a reload candidate",
                )
            elif module == "numpy" and path[-1] in {"seterr", "seterrcall", "set_printoptions"}:
                self._blocked(node, _BLOCKED_ENVIRONMENT_WRITE, ".".join(path),
                              "top-level process-wide NumPy configuration mutation is not allowed")
            elif module == "sys" and len(path) > 1 and path[1] in {"path", "modules", "meta_path"}:
                self._blocked(
                    node,
                    _BLOCKED_MODULE_WRITE,
                    ".".join(path),
                    "top-level interpreter import-state mutation is not allowed",
                )
            elif (module, path[-1]) in _FILE_MODULE_MEMBERS:
                self._blocked(
                    node,
                    _BLOCKED_FILE_WRITE,
                    ".".join(path),
                    "top-level filesystem mutation is not allowed for a reload candidate",
                )
            elif (
                (module, path[-1]) in _PROCESS_DIRECT_MEMBERS
                or (module == "os" and path[-1].startswith("spawn"))
            ):
                self._blocked(
                    node,
                    _BLOCKED_PROCESS,
                    ".".join(path),
                    "top-level process/thread/timer/socket/atexit operation is not allowed",
                )

        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def finish(self) -> ScriptCandidatePolicyReport:
        key = lambda item: (item.line, item.column, item.code, item.operation)
        return ScriptCandidatePolicyReport(
            blocked=tuple(sorted(self.blocked, key=key)),
            runtime_guard_required=tuple(sorted(self.runtime_guard_required, key=key)),
        )


def analyze_script_candidate_tree(tree: ast.AST) -> ScriptCandidatePolicyReport:
    visitor = _PolicyVisitor()
    visitor.visit(tree)
    return visitor.finish()


def analyze_script_candidate(
    source: bytes | bytearray | memoryview | str,
    *,
    filename: str = "<script>",
) -> ScriptCandidatePolicyReport:
    tree = ast.parse(_decode_source(source), filename=filename, mode="exec")
    return analyze_script_candidate_tree(tree)


__all__ = [
    "ScriptCandidatePolicyIssue",
    "ScriptCandidatePolicyReport",
    "analyze_script_candidate",
    "analyze_script_candidate_tree",
]
