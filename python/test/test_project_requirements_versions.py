from __future__ import annotations

import importlib.metadata

from Infernux.engine import project_requirements as requirements


def test_importable_wrong_version_does_not_satisfy_project_pin(monkeypatch):
    class Distribution:
        version = "1.0"

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(requirements, "_has_module", lambda name: True)
    assert not requirements._has_requirement("demo==2.0", "demo")
    assert requirements._has_requirement("demo>=1,<2", "demo")


def test_project_requirements_respect_platform_markers(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text('demo==2; python_version >= "3.0"\nskipped==1; python_version < "2.0"\n', encoding="utf-8")
    assert requirements._parse_requirements(str(path)) == [('demo==2; python_version >= "3.0"', "demo")]
