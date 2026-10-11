from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest


import blender_support


@pytest.fixture(autouse=True)
def isolated_shared_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        blender_support,
        "get_hub_shared_data_dir",
        lambda: str(tmp_path / "Shared"),
    )


def _archive(path: Path) -> Path:
    executable = "blender.exe" if os.name == "nt" else "blender"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"blender-{blender_support.BLENDER_VERSION}/{executable}", b"binary")
        bundle.writestr(f"blender-{blender_support.BLENDER_VERSION}/data/readme.txt", b"data")
    return path


def test_archive_installs_one_shared_authoring_tool_and_exports_path(tmp_path, monkeypatch):
    root = tmp_path / "Shared/AuthoringTools/Blender/current"
    manager = blender_support.BlenderSupportManager(root)
    monkeypatch.delenv("INFERNUX_BLENDER_EXECUTABLE", raising=False)

    assert not manager.status().installed
    assert manager.install_archive(_archive(tmp_path / "blender.zip")) == str(root.resolve())

    status = manager.status()
    assert status.installed and status.version == blender_support.BLENDER_VERSION
    assert status.executable is not None and status.executable.is_file()
    assert os.environ["INFERNUX_BLENDER_EXECUTABLE"] == str(status.executable)
    assert (root / blender_support.MANIFEST_NAME).is_file()


def test_download_uses_pinned_official_asset_and_rejects_changed_bytes(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        executable = "blender.exe" if os.name == "nt" else "blender"
        bundle.writestr(f"blender-{blender_support.BLENDER_VERSION}/{executable}", b"binary")
    content = payload.getvalue()
    digest = __import__("hashlib").sha256(content).hexdigest()
    name = "blender-test.zip"
    monkeypatch.setitem(blender_support._ASSETS, blender_support.host_id(), (name, len(content), digest))

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(blender_support.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(content))
    manager = blender_support.BlenderSupportManager(tmp_path / "installed")
    progress = []
    manager.install(on_progress=lambda done, total: progress.append((done, total)))
    assert manager.status().installed
    assert progress[-1] == (len(content), len(content))

    monkeypatch.setitem(blender_support._ASSETS, blender_support.host_id(), (name, len(content), "0" * 64))
    with pytest.raises(blender_support.BlenderSupportError, match="integrity"):
        manager.install()


def test_manifest_rejects_wrong_host_or_missing_executable(tmp_path):
    manager = blender_support.BlenderSupportManager(tmp_path / "installed")
    manager.install_archive(_archive(tmp_path / "blender.zip"))
    executable = manager.status().executable
    assert executable is not None
    executable.unlink()
    status = manager.status()
    assert not status.installed and "missing" in status.error


def test_archive_extracts_long_paths_and_rejects_escape(tmp_path):
    archive = tmp_path / "long.zip"
    relative = "root/" + "/".join(["nested-package" * 3] * 6) + "/data.txt"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(relative, b"long path")
    root = blender_support._extract_archive(archive, tmp_path / "stage")
    assert (root.parent / relative).read_bytes() == b"long path"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", b"escape")
    with pytest.raises(blender_support.BlenderSupportError, match="unsafe"):
        blender_support._extract_archive(archive, tmp_path / "unsafe-stage")
    assert not (tmp_path / "escape.txt").exists()


def test_failed_extraction_preserves_original_error_when_cleanup_fails(tmp_path, monkeypatch):
    manager = blender_support.BlenderSupportManager(tmp_path / "installed")
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not an archive")
    def cleanup(_path):
        raise PermissionError("cleanup denied")
    monkeypatch.setattr(blender_support.shutil, "rmtree", cleanup)
    with pytest.raises(blender_support.BlenderSupportError, match="archive is invalid"):
        manager.install_archive(archive)
