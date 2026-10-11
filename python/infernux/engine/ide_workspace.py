"""Keep VS Code's Python analysis aligned with the running editor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from infernux.core.document_store import read_document_text_snapshot, write_document_text
from .jsonc_document import JsoncDocument
from .path_utils import (
    is_path_within,
    portable_path,
    relative_path,
    resolve_project_path,
    resolved_path,
    same_path,
)


def synchronize_vscode_workspace(project_root: str) -> None:
    """Update engine-owned runtime settings while retaining user preferences."""
    root = Path(resolved_path(project_root))
    spec = importlib.util.find_spec("infernux")
    if spec is None or spec.origin is None or spec.submodule_search_locations is None:
        raise RuntimeError("The running editor environment has no infernux package.")
    package_root = Path(resolved_path(spec.origin)).parent
    module_root = package_root.parent
    try:
        module_path = relative_path(module_root, root, allow_root=True)
    except ValueError:
        module_path = portable_path(module_root)
    interpreter = resolved_path(sys.executable)
    try:
        interpreter_path = "${workspaceFolder}/" + relative_path(interpreter, root)
    except ValueError:
        interpreter_path = portable_path(interpreter)

    def read_document(path: Path):
        text, state = read_document_text_snapshot(str(path))
        try:
            document = JsoncDocument(text if text is not None else "{}\n")
        except ValueError as exc:
            raise ValueError(f"Invalid IDE configuration {path}: {exc}") from exc
        return document, state

    def search_paths(previous: list[str]) -> list[str]:
        if not isinstance(previous, list) or any(not isinstance(value, str) for value in previous):
            raise ValueError("IDE extraPaths must be an array of strings")
        paths = [module_path]
        for value in previous:
            resolved = resolve_project_path(value, root)
            if is_path_within(resolved, root):
                relative = relative_path(resolved, root, allow_root=True)
                if relative.split("/", 1)[0] in (".venv", ".runtime"):
                    continue
            if (
                not same_path(resolved, module_root)
                and not same_path(resolved, package_root)
                and value not in paths
            ):
                paths.append(value)
        return paths

    settings_path = root / ".vscode" / "settings.json"
    pyright_path = root / "pyrightconfig.json"
    settings_source, settings_state = read_document(settings_path)
    pyright_source, pyright_state = read_document(pyright_path)
    settings = dict(settings_source.value)
    pyright = dict(pyright_source.value)
    settings["python.defaultInterpreterPath"] = interpreter_path
    settings["python.analysis.extraPaths"] = search_paths(
        settings.get("python.analysis.extraPaths", [])
    )
    settings.setdefault("python.analysis.typeCheckingMode", "basic")
    settings.setdefault("python.analysis.autoImportCompletions", True)
    pyright.pop("venv", None)
    pyright.pop("venvPath", None)
    pyright.pop("pythonPath", None)
    pyright["pythonVersion"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    pyright["extraPaths"] = search_paths(pyright.get("extraPaths", []))
    pyright.setdefault("include", ["Assets"])
    changes = (
        (settings_path, settings_source, settings_state, settings_source.render(settings)),
        (pyright_path, pyright_source, pyright_state, pyright_source.render(pyright)),
    )
    for path, original, state, content in changes:
        if not state.exists or original.source != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_document_text(str(path), content, expected_file_state=state)
