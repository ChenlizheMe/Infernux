from Infernux.engine.nuitka_builder import NuitkaBuilder
import subprocess
import sys
import os
from pathlib import Path

import Infernux


def test_player_retains_author_gizmo_import_but_not_editor_collection():
    from Infernux.engine.player_package_audit import (
        PLAYER_FORBIDDEN_RUNTIME_MODULES, PLAYER_FORBIDDEN_RUNTIME_PREFIXES,
    )
    excluded = NuitkaBuilder._is_player_runtime_excluded_source
    assert not excluded("gizmos/__init__.py")
    assert not excluded("gizmos/gizmos.py")
    assert excluded("gizmos/collector.py")
    for path in ("Infernux/gizmos/__init__.pyc", "Infernux/gizmos/gizmos.pyc"):
        assert path not in PLAYER_FORBIDDEN_RUNTIME_MODULES
        assert not any(path.startswith(prefix) for prefix in PLAYER_FORBIDDEN_RUNTIME_PREFIXES)
    assert "Infernux/gizmos/collector.pyc" in PLAYER_FORBIDDEN_RUNTIME_MODULES


def test_author_drawing_api_import_does_not_load_collector():
    # pytest's pythonpath does not propagate to a fresh interpreter. Test the
    # same package, not an unrelated release wheel installed in this environment.
    env = os.environ.copy()
    package_parent = str(Path(Infernux.__file__).parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (package_parent, env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; from Infernux.gizmos import Gizmos; "
            "assert callable(Gizmos.draw_wire_sphere); "
            "assert 'Infernux.gizmos.collector' not in sys.modules"
        )], capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
