from __future__ import annotations

import json
from pathlib import Path

from infernux.engine.library_sync import sync_resources


def test_library_sync_does_not_mirror_wheel_mandatory_packages(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "builtin.inxpkg").write_bytes(b"mandatory")
    (resources / "shaders").mkdir()
    (resources / "shaders" / "surface.glsl").write_bytes(b"resource")
    nested = resources / "official_packages"
    nested.mkdir()
    (nested / "catalog-artifact.inxpkg").write_bytes(b"catalog")
    project = tmp_path / "project"
    for name in ("schemas", "licenses", "project_templates"):
        (resources / name).mkdir()
        (resources / name / "non-asset.txt").write_text("package metadata", encoding="utf-8")
        stale = project / "Library" / "Resources" / name
        stale.mkdir(parents=True)
        (stale / "non-asset.txt").write_text("old mirror", encoding="utf-8")
    monkeypatch.setattr(
        "infernux.resources.get_package_resources_path", lambda: str(resources)
    )

    destination = Path(sync_resources(str(project)))

    assert not (destination / "builtin.inxpkg").exists()
    assert (destination / "shaders" / "surface.glsl").read_bytes() == b"resource"
    assert not (destination / "official_packages").exists()
    for name in ("schemas", "licenses", "project_templates"):
        assert not (destination / name).exists()
    manifest = json.loads(
        (project / "Library" / ".InfernuxResources.json").read_text(encoding="utf-8")
    )
    assert set(manifest) == {"entries"}
    assert manifest["entries"]["shaders/surface.glsl"]["size"] == len(b"resource")
