from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGING_DIR / "model"
for directory in (PACKAGING_DIR, MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import project_model
from project_model import ProjectModel, _create_default_project_content


def test_default_render_assets_seed_identity_without_derived_fingerprints(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "ProjectSettings").mkdir(parents=True)

    _create_default_project_content(str(staging), "Project")

    rendering = staging / "Assets" / "Settings"
    for asset in (
        rendering / "Bloom.effect",
        rendering / "ACES Tone Mapping.effect",
        rendering / "Default Post Processing.effectgroup",
    ):
        metadata = json.loads(asset.with_name(asset.name + ".meta").read_text(encoding="utf-8"))
        assert "content_hash" not in metadata["metadata"]
        assert len(metadata["metadata"]["guid"]["value"]) == 32
        assert metadata["metadata"]["resource_type"]["value"] == "RenderEffect"


def test_installed_hub_uses_the_selected_wheels_template(tmp_path, monkeypatch):
    wheel = tmp_path / "selected.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("infernux/resources/project_templates/default_scene.json", b'{"name":"selected"}')
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    model = ProjectModel(None, SimpleNamespace(get_wheel_path=lambda version: str(wheel)), object())
    assert model._read_bundled_support_file("default_scene.json", "0.4.1") == b'{"name":"selected"}'


def test_installed_hub_rejects_missing_wheel_template_even_when_source_exists(tmp_path, monkeypatch):
    wheel = tmp_path / "incomplete.whl"
    with zipfile.ZipFile(wheel, "w"):
        pass
    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    model = ProjectModel(None, SimpleNamespace(get_wheel_path=lambda version: str(wheel)), object())
    with pytest.raises(RuntimeError, match="found 0"):
        model._read_bundled_support_file("default_scene.json", "0.4.1")
