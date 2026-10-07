import subprocess
import zipfile
from types import SimpleNamespace

import pytest

import model.project_model as project_model


@pytest.mark.parametrize("distribution_name", ["Infernux", "infernux"])
def test_lowercase_wheel_upgrade_removes_retired_package_and_typing_entrances(
    tmp_path, distribution_name,
):
    site_packages = tmp_path / "site-packages"
    legacy_package = site_packages / "Infernux"
    legacy_package.mkdir(parents=True)
    (legacy_package / "__init__.py").write_text("OLD", encoding="utf-8")
    for name in ("infernux.py", "infernux.pyi"):
        (site_packages / name).write_text("OLD", encoding="utf-8")
    legacy_stubs = site_packages / "infernux-stubs"
    legacy_stubs.mkdir()
    (legacy_stubs / "__init__.pyi").write_text("OLD", encoding="utf-8")
    legacy_metadata = site_packages / "infernux-0.4.0.dist-info"
    legacy_metadata.mkdir()
    (legacy_metadata / "METADATA").write_text("Name: Infernux", encoding="utf-8")
    unrelated = site_packages / "numpy"
    unrelated.mkdir()
    (unrelated / "__init__.py").write_text("KEEP", encoding="utf-8")
    wheel = tmp_path / "infernux-0.4.1-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("infernux/__init__.py", "NEW")
        archive.writestr("infernux/__init__.pyi", "NEW_TYPES")
        archive.writestr("infernux/py.typed", "")
        archive.writestr("infernux/renderstack/__init__.py", "SUBMODULE")
        archive.writestr("infernux-0.4.1.dist-info/METADATA", "Name: infernux")

    project_model._install_wheel_direct(str(wheel), str(site_packages), distribution_name)

    assert (site_packages / "infernux" / "__init__.py").read_text(encoding="utf-8") == "NEW"
    assert (site_packages / "infernux" / "__init__.pyi").read_text(encoding="utf-8") == "NEW_TYPES"
    assert (site_packages / "infernux" / "renderstack" / "__init__.py").is_file()
    assert (site_packages / "infernux" / "py.typed").is_file()
    assert not legacy_metadata.exists()
    assert not legacy_stubs.exists()
    assert not (site_packages / "infernux.py").exists()
    assert not (site_packages / "infernux.pyi").exists()
    assert "Infernux" not in {path.name for path in site_packages.iterdir()}
    assert (unrelated / "__init__.py").read_text(encoding="utf-8") == "KEEP"


@pytest.mark.parametrize("marker_state", ["missing", "stale", "current"])
def test_project_uses_wheel_identity_not_only_distribution_version(tmp_path, monkeypatch, marker_state):
    project = tmp_path / "Project"
    site_packages = project / ".runtime" / "site-packages"
    package = site_packages / "infernux"
    package.mkdir(parents=True)
    installed_source = package / "__init__.py"
    installed_source.write_text("OLD", encoding="utf-8")
    python = project / ".runtime" / "python.exe"
    python.touch()
    wheel = tmp_path / "infernux-0.4.0-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("infernux/__init__.py", "NEW")

    monkeypatch.setattr(project_model, "is_frozen", lambda: True)
    monkeypatch.setattr(project_model.ProjectModel, "_get_project_python", staticmethod(lambda path: str(python)))
    monkeypatch.setattr(project_model.ProjectModel, "_get_site_packages", staticmethod(lambda path: str(site_packages)))
    monkeypatch.setattr(project_model, "_project_python_version", lambda path: "3.13")
    monkeypatch.setattr(project_model.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "0.4.0\n", ""))
    validations = []
    monkeypatch.setattr(project_model.ProjectModel, "validate_python_runtime", staticmethod(validations.append))
    marker = project / ".runtime" / ".infernux-wheel"
    if marker_state != "missing":
        marker.write_text(
            project_model._wheel_install_fingerprint(str(wheel)) if marker_state == "current" else "old wheel\n",
            encoding="utf-8",
        )
    manager = SimpleNamespace(get_wheel_path=lambda *args: str(wheel))
    model = project_model.ProjectModel(None, manager, runtime_manager=object())
    model._install_infernux_in_runtime(str(project), "0.4.0", validate_current=False)
    assert installed_source.read_text(encoding="utf-8") == ("OLD" if marker_state == "current" else "NEW")
    assert len(validations) == (0 if marker_state == "current" else 1)
    assert marker.read_text(encoding="utf-8") == project_model._wheel_install_fingerprint(str(wheel))
    model._install_infernux_in_runtime(str(project), "0.4.0", validate_current=False)
    assert len(validations) == (0 if marker_state == "current" else 1)
