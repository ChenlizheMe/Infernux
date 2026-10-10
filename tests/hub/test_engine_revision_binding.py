"""A Git project pin must never resolve to a different wheel revision."""
import zipfile
from types import SimpleNamespace
from pathlib import Path

import pytest

import version_manager as vm


def wheel(root, build, *, version="0.4.1"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"infernux-{version}-{build}-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("infernux/__init__.py", "")
        archive.writestr(f"infernux-{version}.dist-info/METADATA",
                         f"Metadata-Version: 2.1\nName: infernux\nVersion: {version}\n")
        archive.writestr(f"infernux-{version}.dist-info/WHEEL",
                         f"Wheel-Version: 1.0\nBuild: {build}\nTag: cp313-cp313-win_amd64\n")
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


def test_cache_folder_cannot_relabel_another_package_version(manager, tmp_path):
    wheel(tmp_path / "0.4.1", 1, version="0.4.0")
    assert not manager.is_installed("0.4.1")


def test_github_and_pypi_revisions_are_separate_entries(manager, monkeypatch):
    names = [f"infernux-0.4.1-{build}-cp313-cp313-win_amd64.whl" for build in (2, 10)]
    releases = vm._merge_release_catalogs(
        [{"tag_name": "v0.4.1-v2", "assets": [{"name": names[0], "browser_download_url": "https://example.invalid/old"}]}],
        {"releases": {"0.4.1": [{"packagetype": "bdist_wheel", "filename": name,
                                   "url": "https://example.invalid/" + name} for name in names]}},
    )
    monkeypatch.setattr(manager, "_fetch_releases", lambda: releases)
    versions = manager.list_versions()
    assert [entry.version for entry in versions] == ["0.4.1-v10", "0.4.1-v2"]
    assert [Path(entry.wheel_url).name for entry in versions] == list(reversed(names))
    assert len(versions[1].wheel_options) == 2


def test_catalog_cannot_relabel_a_wheel_revision(manager, monkeypatch):
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [{"tag_name": "v0.4.1-v2", "assets": [{
        "name": "infernux-0.4.1-3-cp313-cp313-win_amd64.whl", "browser_download_url": "https://example.invalid/wrong",
    }]}])
    versions = manager.list_versions()
    assert not versions or not versions[0].wheel_options


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


@pytest.mark.parametrize("value", ["../other", "..\\other", "E:/other", "0.4.1-v0", "0.4.1-v3/other"])
def test_invalid_release_cannot_address_another_directory(manager, value):
    assert manager.get_wheel_path(value) is None
    assert not manager.is_installed(value)
    with pytest.raises(ValueError):
        manager.remove_version(value)


def test_missing_revision_disables_real_hub_card(tmp_path, manager, monkeypatch):
    from PySide6.QtWidgets import QApplication, QLabel
    from types import SimpleNamespace
    import ui_project_list
    from i18n import tr

    app = QApplication.instance() or QApplication([])
    project = tmp_path / "Project"
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "ProjectSettings/PythonRuntime.json").write_text('{"pythonVersion":"3.13"}', encoding="utf-8")
    vm.VersionManager.write_project_version(str(project), "0.4.1-v3")
    wheel(tmp_path / "0.4.1", 2)
    monkeypatch.setattr(ui_project_list, "is_frozen", lambda: True)
    record = SimpleNamespace(project_id="exact", name="Exact revision", created_at="", path=str(project))
    pane = ui_project_list.ProjectListPane(SimpleNamespace(all_projects=lambda: [record]), manager)
    try:
        card = pane.project_cards["exact"]
        # Unavailable rows cannot launch, but their action menu stays usable.
        assert not card.launchable and not card.can_select
        assert card._actions_button.isEnabled()
        assert any(label.text().startswith(tr("Install required version")) for label in card.findChildren(QLabel))
        pane.select_project("exact")
        assert pane.get_selected_project_id() is None
        wheel(tmp_path / "0.4.1", 3)
        pane.refresh()
        app.processEvents()
        assert pane.project_cards["exact"].launchable
        pane.select_project("exact")
        assert pane.get_selected_project_id() == "exact"
    finally:
        pane.close()


def test_git_clone_preserves_exact_release_without_runtime(tmp_path, manager):
    import subprocess

    source = tmp_path / "author"
    source.mkdir()
    vm.VersionManager.write_project_version(str(source), "0.4.1-v2")
    (source / ".gitignore").write_text(".runtime/\n", encoding="utf-8")
    (source / ".runtime").mkdir()
    (source / ".runtime/private.txt").write_text("machine-local", encoding="utf-8")
    for command in (["init", "-q"], ["add", "."], ["-c", "user.name=Repair", "-c", "user.email=repair@example.invalid", "commit", "-qm", "Project pin"]):
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


def test_setup_emits_the_runtime_revision_in_the_wheel_build_tag(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from hub_release import project_build_number

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "wheel-source"
    package = source / "python/infernux"
    (package / "lib").mkdir(parents=True)
    shutil.copy2(root / "setup.py", source / "setup.py")
    shutil.copy2(root / "python/infernux_project_lock.py", source / "python/infernux_project_lock.py")
    shutil.copy2(root / "python/infernux/version.py", package / "version.py")
    (package / "__init__.py").write_text("", encoding="utf-8")
    # Only the packaging identity contract is under test; no native code runs.
    (package / "lib/_Infernux.pyd").write_bytes(b"packaging fixture")
    (package / "lib/PlayerNativeContract.json").write_text(
        '{"contract":"infernux.player-native","runtime_linkage":"static"}', encoding="utf-8")
    (source / "pyproject.toml").write_text(
        '[project]\nname = "infernux"\nversion = "0.4.1"\n'
        '[tool.setuptools]\npy-modules = ["infernux_project_lock"]\n'
        '[tool.setuptools.packages.find]\nwhere = ["python"]\n', encoding="utf-8")
    subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=source,
                   env={**os.environ, "INFERNUX_STAGED_WHEEL_BUILD": "1"},
                   check=True, capture_output=True)
    [artifact] = (source / "dist").glob("*.whl")
    assert vm.wheel_build(str(artifact)) == (project_build_number(root), "")
    with zipfile.ZipFile(artifact) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        assert f"Build: {project_build_number(root)}\n" in archive.read(metadata).decode()
        assert archive.read("infernux/version.py") == (root / "python/infernux/version.py").read_bytes()
        assert archive.read("infernux_project_lock.py") == (root / "python/infernux_project_lock.py").read_bytes()
