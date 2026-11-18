"""Native packages; installed MCP workflows live in tests/isolated/test_editor_automation_workflow.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from infernux.engine import player_package_native
from infernux.plugins import InxPackage, PluginManager


@pytest.fixture(autouse=True)
def _isolated_package_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "INFERNUX_PACKAGE_CACHE_ROOT",
        str(tmp_path / "hub-package-cache"),
    )


def _native_available() -> bool:
    try:
        player_package_native._backend()
        return not player_package_native.using_test_backend()
    except Exception:
        return False


@pytest.mark.skipif(not _native_available(), reason="native InxPack backend unavailable")
def test_export_uses_writer_manifest_without_reading_its_own_package(tmp_path):
    import infernux.plugins.package as package_module

    source = tmp_path / "author"
    source.mkdir()
    (source / "message.txt").write_text("exported text", encoding="utf-8")
    with (
        patch.object(package_module, "read_manifest", wraps=package_module.read_manifest) as manifests,
        patch.object(package_module, "read_entry", wraps=package_module.read_entry) as entries,
    ):
        preview = InxPackage.export(
            str(source), [str(source)], str(tmp_path / "Output.inxpkg")
        )
        assert (manifests.call_count, entries.call_count) == (0, 0)
    assert preview == InxPackage.inspect(preview.package_path)


@pytest.mark.skipif(not _native_available(), reason="native InxPack backend unavailable")
@pytest.mark.parametrize("worker", [False, True])
def test_package_temporary_workspace_belongs_to_output(tmp_path, monkeypatch, worker):
    import tempfile

    source = tmp_path / "message.txt"
    source.write_text("owned temporary data", encoding="utf-8")
    destination = tmp_path / "output" / "probe.inxpkg"
    workspaces = []
    create = tempfile.TemporaryDirectory

    def temporary_directory(*args, **kwargs):
        workspace = create(*args, **kwargs)
        workspaces.append(Path(workspace.name))
        return workspace

    monkeypatch.setattr(tempfile, "TemporaryDirectory", temporary_directory)
    if worker:
        player_package_native.write_pack_isolated([("message.txt", source)], destination)
    else:
        InxPackage.export(str(tmp_path), [str(source)], str(destination))
    assert workspaces
    assert all(path.parent == destination.parent for path in workspaces)
    assert all(not path.exists() for path in workspaces)


def _source(path: Path, reference: str) -> Path:
    path.mkdir(parents=True)
    (path / "inx_package.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "name": reference,
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.skipif(not _native_available(), reason="native InxPack backend unavailable")
def test_repository_package_scripts_are_standalone_deterministic_and_native_compatible(
    tmp_path,
):
    repository = Path(__file__).parents[2]
    plugin_roots = tuple(
        repository / "external" / "plugins" / name
        for name in (
            "infernux_android",
            "infernux_linux",
            "infernux_web",
            "infernux_windows",
        )
    )
    scripts = [(root / "package.py").read_bytes() for root in plugin_roots]
    for script in scripts:
        assert b"from infernux" not in script
        assert b"import infernux" not in script

    outputs = (tmp_path / "first.inxpkg", tmp_path / "second.inxpkg")
    for destination in outputs:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(plugin_roots[-1] / "package.py"),
                str(destination),
            ],
            cwd=tmp_path,
            env={"PATH": os.environ.get("PATH", "")},
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert destination.is_file()

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    preview = InxPackage.inspect(str(outputs[0]))
    assert preview.metadata["reference"] == "infernux/platform-windows"
    assert preview.logical_entries
    assert all(not path.startswith("README") for path in preview.logical_entries)
    assert "package.py" not in preview.logical_entries


@pytest.mark.skipif(not _native_available(), reason="native InxPack backend unavailable")
def test_native_inxpackage_installs_only_explicit_nested_requirement(tmp_path):
    child = _source(tmp_path / "child", "native/child")
    (child / "child.txt").write_text("child payload", encoding="utf-8")
    child_package = tmp_path / "Child.inxpkg"
    InxPackage.export_source(str(child), str(child_package))

    parent = _source(tmp_path / "parent", "native/parent")
    (parent / "vendor").mkdir()
    shutil.copy2(child_package, parent / "vendor" / "Child.inxpkg")
    (parent / "parent.txt").write_text("parent payload", encoding="utf-8")
    package_without_requirement = tmp_path / "ParentOptional.inxpkg"
    InxPackage.export_source(str(parent), str(package_without_requirement))

    project = tmp_path / "project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    manager = PluginManager(str(project))
    manager.install_package(str(package_without_requirement), install_dependencies=True)
    assert {item["reference"] for item in manager.registry.installed()} == {
        "native/parent"
    }
    assert (project / "Assets/Plugins/vendor/Child.inxpkg").is_file()
    manager.uninstall("native/parent")

    (parent / "requirements.txt").write_text(
        "vendor/Child.inxpkg\n", encoding="utf-8"
    )
    manifest = parent / "inx_package.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["version"] = "2.0.0"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    required_package = tmp_path / "ParentRequired.inxpkg"
    InxPackage.export_source(str(parent), str(required_package))
    manager.install_package(str(required_package), install_dependencies=True)
    assert {item["reference"] for item in manager.registry.installed()} == {
        "native/child",
        "native/parent",
    }
    assert (
        project / "Assets/Plugins/child.txt"
    ).read_text(encoding="utf-8") == "child payload"
    with pytest.raises(RuntimeError, match="required by"):
        manager.uninstall("native/child")
    manager.uninstall("native/parent")
    manager.uninstall("native/child")
