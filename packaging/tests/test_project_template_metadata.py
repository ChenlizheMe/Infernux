from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGING_DIR / "model"
for directory in (PACKAGING_DIR, MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from project_model import _create_default_project_content, configure_project_collaboration


def test_default_render_assets_seed_identity_without_derived_fingerprints(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "ProjectSettings").mkdir(parents=True)

    _create_default_project_content(str(staging), "Project")

    rendering = staging / "Assets" / "Rendering"
    for asset in (
        rendering / "Bloom.effect",
        rendering / "ACES Tone Mapping.effect",
        rendering / "Default Post Processing.effectgroup",
    ):
        metadata = json.loads(asset.with_name(asset.name + ".meta").read_text(encoding="utf-8"))
        assert "content_hash" not in metadata["metadata"]
        assert len(metadata["metadata"]["guid"]["value"]) == 32
        assert metadata["metadata"]["resource_type"]["value"] == "RenderEffect"


def test_hub_installs_the_target_runtime_driver_without_importing_native_engine(tmp_path, monkeypatch):
    repo_root = PACKAGING_DIR.parent
    package = tmp_path / "shadow" / "Infernux"
    (package / "engine").mkdir(parents=True)
    for initializer in (package / "__init__.py", package / "engine/__init__.py"):
        initializer.write_text("raise AssertionError('Engine must not be imported during Git setup')\n", encoding="utf-8")
    shutil.copyfile(repo_root / "python/Infernux/collaboration.py", package / "collaboration.py")
    shutil.copyfile(repo_root / "python/Infernux/engine/path_utils.py", package / "engine/path_utils.py")
    project = tmp_path / "Project"
    project.mkdir()
    import subprocess
    subprocess.run(["git", "init", str(project)], capture_output=True, check=True)
    monkeypatch.setenv("PYTHONPATH", str(package.parent))
    configure_project_collaboration(str(project), sys.executable)
    driver = subprocess.run(["git", "-C", str(project), "config", "--get", "merge.infernux.driver"], capture_output=True, text=True)
    assert driver.returncode == 0
    assert (package / "collaboration.py").as_posix() in driver.stdout
