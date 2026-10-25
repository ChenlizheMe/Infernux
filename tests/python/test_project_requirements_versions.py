from __future__ import annotations

import importlib.metadata
from types import SimpleNamespace

import pytest

from infernux.engine import project_requirements as requirements


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


def test_missing_authored_requirements_never_creates_a_shared_file(tmp_path):
    assert not requirements.ensure_project_requirements(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("spec", ["Infernux==0.4.1"])
def test_project_requirements_cannot_install_or_replace_the_pinned_engine(tmp_path, spec):
    path = tmp_path / "requirements.txt"
    path.write_text(spec + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact engine version"):
        requirements._parse_requirements(str(path))


def test_editor_stops_before_native_initialization_when_dependencies_are_missing(tmp_path, monkeypatch):
    from infernux.engine.bootstrap import EditorBootstrap

    monkeypatch.setattr(requirements, "ensure_project_requirements", lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match="Project Python requirements could not be satisfied"):
        EditorBootstrap._ensure_project_requirements(SimpleNamespace(project_path=str(tmp_path)))


def test_dependency_check_does_not_replace_a_live_environment(tmp_path, monkeypatch):
    path = tmp_path / "ProjectSettings" / "requirements.txt"
    path.parent.mkdir()
    path.write_text("demo==2.0\n", encoding="utf-8")
    monkeypatch.setenv("_INFERNUX_PROJECT_REQS_CHECKED", "1")
    monkeypatch.setattr(requirements, "_has_requirement", lambda *_args: False)
    before = path.read_bytes()
    assert not requirements.ensure_project_requirements(str(tmp_path))
    assert path.read_bytes() == before
