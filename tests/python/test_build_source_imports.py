"""Private source analysis cannot publish or reuse a live project module."""
import sys
import types
from infernux.engine.build.source_imports import BuildSourceImports
from infernux.engine.project_context import get_project_root, set_project_root


def test_private_dataclasses_import_cycles_and_current_snapshots(tmp_path):
    previous = get_project_root()
    assets = tmp_path / "Assets"
    assets.mkdir()
    set_project_root(str(tmp_path))
    live = types.ModuleType("BuildOnly")
    live.value = -1
    prior = sys.modules.get("BuildOnly")
    sys.modules["BuildOnly"] = live
    imports = BuildSourceImports()
    try:
        imports.register("BuildOnly", str(assets / "BuildOnly.py"), source=(
            "from dataclasses import dataclass\nimport BuildHelper\n"
            "@dataclass\nclass Data:\n    value: int = 8\nvalue = BuildHelper.value\n"))
        imports.register("BuildHelper", str(assets / "BuildHelper.py"), source="import BuildOnly\nvalue = 12\n")
        current = imports.load("BuildOnly")
        assert current.value == 12 and current.Data().value == 8
        assert sys.modules["BuildOnly"] is live and live.value == -1
        assert "BuildHelper" not in sys.modules
    finally:
        imports.close()
        if prior is None:
            sys.modules.pop("BuildOnly", None)
        else:
            sys.modules["BuildOnly"] = prior
        set_project_root(previous)
