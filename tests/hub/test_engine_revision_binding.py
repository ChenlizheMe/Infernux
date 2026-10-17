"""A Git project pin must never resolve to a different wheel revision."""

import zipfile
from types import SimpleNamespace

import pytest

import version_manager as vm


def wheel(root, build, *, version="0.4.1"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"infernux-{version}-{build}-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("infernux/__init__.py", "")
        archive.writestr(
            f"infernux-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: infernux\nVersion: {version}\n"
        )
        archive.writestr(
            f"infernux-{version}.dist-info/WHEEL", f"Wheel-Version: 1.0\nBuild: {build}\nTag: cp313-cp313-win_amd64\n"
        )
    return path


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_VERSIONS_DIR", tmp_path)
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({"win_amd64"}))
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform="win32"))
    return vm.VersionManager()


def test_bare_pin_does_not_select_newer_revision(manager, tmp_path):
    wheel(tmp_path / "0.4.1", 2)
    assert not manager.is_installed("0.4.1")
    original = wheel(tmp_path / "0.4.1", 1)
    assert manager.get_wheel_path("0.4.1") == str(original)


def test_exact_revisions_coexist_in_existing_cache(manager, tmp_path):
    first = wheel(tmp_path / "0.4.1", 2)
    second = wheel(tmp_path / "0.4.1", 10)
    assert manager.get_wheel_path("0.4.1-v2") == str(first)
    assert manager.get_wheel_path("0.4.1-v10") == str(second)
    assert manager.installed_versions() == ["0.4.1-v10", "0.4.1-v2"]
    assert not manager.is_installed("0.4.1-v3")


def test_local_import_preserves_build_identity(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_require_installed_python", lambda *args, **kwargs: None)
    path = wheel(tmp_path / "incoming", 3)
    assert manager.install_local_wheel(str(path)) == "0.4.1-v3"
    assert manager.get_wheel_path("0.4.1-v3")
    assert not manager.is_installed("0.4.1")


def test_removing_one_revision_preserves_other_projects_engine(manager, tmp_path):
    removed = wheel(tmp_path / "0.4.1", 2)
    retained = wheel(tmp_path / "0.4.1", 3)
    assert manager.remove_version("0.4.1-v2")
    assert not removed.exists()
    assert retained.is_file()
    assert manager.is_installed("0.4.1-v3")
    assert not manager.remove_version("0.4.1-v2")


def test_git_clone_preserves_exact_release_without_runtime(tmp_path, manager):
    import subprocess

    source = tmp_path / "author"
    source.mkdir()
    vm.VersionManager.write_project_version(str(source), "0.4.1-v2")
    (source / ".gitignore").write_text(".runtime/\n", encoding="utf-8")
    (source / ".runtime").mkdir()
    (source / ".runtime/private.txt").write_text("machine-local", encoding="utf-8")
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=Repair", "-c", "user.email=repair@example.invalid", "commit", "-qm", "Project pin"],
    ):
        subprocess.run(["git", "-C", str(source), *command], check=True, capture_output=True)
    target = tmp_path / "collaborator"
    subprocess.run(["git", "clone", "-q", str(source), str(target)], check=True, capture_output=True)
    assert not (target / ".runtime").exists()
    pin = manager.read_project_version(str(target))
    assert pin == "0.4.1-v2"
    wheel(tmp_path / "0.4.1", 3)
    assert not manager.is_installed(pin)
    matching = wheel(tmp_path / "0.4.1", 2)
    assert manager.get_wheel_path(pin) == str(matching)
    assert (target / ".infernux-version").read_bytes() == (source / ".infernux-version").read_bytes()
