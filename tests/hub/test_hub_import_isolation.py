"""Source Hub helpers must preserve the operator's engine import environment."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("preload_lock", [False, True])
def test_source_hub_preserves_engine_search_path(tmp_path, preload_lock):
    installed = tmp_path / "installed"
    package = installed / "infernux"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("origin = 'installed'\n", encoding="utf-8")
    shutil.copy2(ROOT / "python/infernux_project_lock.py", installed)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", """
from pathlib import Path
import sys
root, installed = map(Path, sys.argv[1:3])
sys.path.insert(0, str(installed))
if sys.argv[3] == 'True':
    import infernux_project_lock
    original_lock = infernux_project_lock
sys.path.insert(0, str(root / 'packaging'))
before = list(sys.path)
import hub_utils
assert sys.path == before, 'Hub changed the engine module search path'
assert 'infernux' not in sys.modules, 'Hub imported the engine'
import infernux
assert Path(infernux.__file__).resolve() == installed / 'infernux/__init__.py'
assert infernux.origin == 'installed'
import infernux_project_lock
assert hub_utils.project_lock is infernux_project_lock
if sys.argv[3] == 'True':
    assert infernux_project_lock is original_lock
    assert Path(infernux_project_lock.__file__).resolve() == installed / 'infernux_project_lock.py'
else:
    assert Path(infernux_project_lock.__file__).resolve() == root / 'python/infernux_project_lock.py'
print('Hub preserves installed engine and shared lock identity')
""", str(ROOT), str(installed), str(preload_lock)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
