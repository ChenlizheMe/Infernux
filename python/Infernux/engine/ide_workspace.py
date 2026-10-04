"""Keep VS Code's Python analysis aligned with the running editor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def synchronize_vscode_workspace(project_root: str) -> None:
    """Update engine-owned runtime settings while retaining user preferences."""
    root = Path(project_root).resolve()
    spec = importlib.util.find_spec("infernux")
    if spec is None or spec.origin is None:
        raise RuntimeError("The running editor environment has no infernux module.")
    module_root = Path(spec.origin).resolve().parent
    try:
        module_path = module_root.relative_to(root).as_posix()
    except ValueError:
        module_path = module_root.as_posix()
    interpreter = Path(sys.executable).resolve()
    try:
        interpreter_path = "${workspaceFolder}/" + interpreter.relative_to(root).as_posix()
    except ValueError:
        interpreter_path = interpreter.as_posix()

    def read_document(path: Path) -> dict:
        if not path.exists():
            return {}
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError(f"IDE configuration must be a JSON object: {path}")
        return document

    def search_paths(previous: list[str]) -> list[str]:
        paths = [module_path]
        for value in previous:
            resolved = (root / value).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                relative = None
            if relative and relative.parts[0] in (".venv", ".runtime"):
                continue
            if resolved != module_root and value not in paths:
                paths.append(value)
        return paths

    settings_path = root / ".vscode" / "settings.json"
    pyright_path = root / "pyrightconfig.json"
    settings = read_document(settings_path)
    pyright = read_document(pyright_path)
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
    for path, document in ((settings_path, settings), (pyright_path, pyright)):
        content = json.dumps(document, ensure_ascii=False, indent=4) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8-sig") != content:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
