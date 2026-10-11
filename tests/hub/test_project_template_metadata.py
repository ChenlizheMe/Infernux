from __future__ import annotations

import json
import importlib.metadata
import sys
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


PACKAGING_DIR = (Path(__file__).resolve().parents[2] / "packaging")
MODEL_DIR = PACKAGING_DIR / "model"
for directory in (PACKAGING_DIR, MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import project_model
from project_model import ProjectModel, _create_default_project_content
from engine_wheel import editor_launch_script, read_project_template, runtime_package_script


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
        archive.writestr("infernux/templates/project/default_scene.json", b'{"name":"selected"}')
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


@pytest.mark.parametrize("directory", [
    "infernux/templates/project", "Infernux/resources/project_templates",
    "infernux/relocated/project_defaults",
])
def test_template_lookup_follows_unique_filename_not_a_hub_layout(tmp_path, directory):
    wheel = tmp_path / "engine.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{directory}/project.gitignore.txt", b"selected engine template")
        archive.writestr("somewhere/not-project.gitignore.txt", b"not a filename match")
    assert read_project_template(str(wheel), "project.gitignore.txt") == b"selected engine template"


@pytest.mark.parametrize("duplicate", [False, True])
def test_ambiguous_templates_are_not_silently_selected(tmp_path, duplicate):
    wheel = tmp_path / "ambiguous.whl"
    first = "infernux/templates/project/requirements.txt"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(first, b"first")
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(first, b"second")
        else:
            archive.writestr("Infernux/resources/project_templates/requirements.txt", b"second")
    with pytest.raises(RuntimeError, match="found 2"):
        read_project_template(str(wheel), "requirements.txt")


def test_wheel_without_authored_scenes_leaves_bootstrap_to_its_editor(tmp_path):
    (tmp_path / "ProjectSettings").mkdir()
    _create_default_project_content(str(tmp_path), "OlderEngine", read_template=lambda _: None)
    assert not list(tmp_path.rglob("*.scene"))
    assert not (tmp_path / "ProjectSettings/BuildSettings.json").exists()


def test_partial_scene_templates_fail_instead_of_using_new_hub_defaults(tmp_path):
    (tmp_path / "ProjectSettings").mkdir()
    with pytest.raises(RuntimeError, match="Incomplete engine scene templates"):
        _create_default_project_content(
            str(tmp_path), "Incomplete",
            read_template=lambda name: b"{}" if name == "default_scene.json" else None,
        )
    assert not list(tmp_path.rglob("*.scene"))


@pytest.mark.parametrize("package", ["Infernux", "infernux"])
def test_editor_entry_uses_installed_distribution_spelling(package, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: SimpleNamespace(
        files=[PurePosixPath(f"{package}/__init__.py")],
    ))
    calls = []
    level = object()
    monkeypatch.setitem(sys.modules, package + ".engine", SimpleNamespace(
        release_engine=lambda **kwargs: calls.append(kwargs),
    ))
    monkeypatch.setitem(sys.modules, package + ".lib", SimpleNamespace(
        LogLevel=SimpleNamespace(Info=level),
    ))
    monkeypatch.setattr(sys, "argv", ["-c", "project path"])
    exec(editor_launch_script(installed=True), {})
    assert calls == [{"engine_log_level": level, "project_path": "project path"}]


@pytest.mark.parametrize("files", [[], ["Infernux/__init__.py", "infernux/__init__.py"]])
def test_installed_engine_entry_rejects_missing_or_ambiguous_package(files, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: SimpleNamespace(
        files=[PurePosixPath(name) for name in files],
    ))
    with pytest.raises(RuntimeError, match="no unique engine package"):
        exec(runtime_package_script(installed=True), {})


def test_source_engine_entry_does_not_consult_installed_wheel_metadata(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: pytest.fail("source entry queried a wheel"))
    scope = {}
    exec(runtime_package_script(installed=False), scope)
    assert scope["_engine_package"] == "infernux"
