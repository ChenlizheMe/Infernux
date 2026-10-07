"""One portable manifest path contract at release and update-staging boundaries."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

import pytest

import hub_release
import hub_updater


def manifest(paths, version="1.1.0", platform="windows-x64"):
    return {"$schema": hub_release.MANIFEST_SCHEMA, "product": hub_release.PRODUCT_NAME,
            "version": version, "platform": platform,
            "files": [{"path": path} for path in paths]}


@pytest.mark.parametrize("path", [
    "C:/outside.txt", "C:outside.txt", "nested/C:/outside.txt", "nested/file:stream",
    "//server/share/file", "\\\\server\\share\\file", "/absolute", "\\absolute",
    "../outside", "nested/../outside", "nested/./file", "nested//file", "./file",
    "nested\\file", "", ".", None, 123, "nul.txt", "nested/file.", "nested/file ",
])
def test_manifest_rejects_nonportable_file_identity(path):
    with pytest.raises(ValueError, match="Unsafe update path"):
        hub_release.validate_manifest(manifest([path]))


@pytest.mark.parametrize("platform", ["windows-x64", "linux-x64"])
def test_manifest_rejects_shared_data_aliases(platform):
    with pytest.raises(ValueError, match="shared resources"):
        hub_release.validate_manifest(manifest(["InfernuxHubData/Shared/user.txt"], platform=platform))


def test_windows_manifest_rejects_case_aliases():
    with pytest.raises(ValueError, match="Duplicate"):
        hub_release.validate_manifest(manifest(["runtime/Core.dll", "RUNTIME/core.dll"]))


@pytest.mark.parametrize("path", ["运行时/字体.ttf", "runtime/Library.dll", "Infernux Hub.exe"])
def test_manifest_preserves_valid_relative_paths(path):
    document = manifest([path])
    assert hub_release.validate_manifest(document) is document
    assert document["files"] == [{"path": path}]


@pytest.mark.parametrize("unsafe", [False, True])
def test_real_zip_stage_rejects_drive_member_without_writing_outside(
    tmp_path, monkeypatch, unsafe,
):
    monkeypatch.chdir(tmp_path)
    # Different drives reproduce the original Windows drive-relative join. On
    # other hosts the same manifest identity is invalid before any extraction.
    repository = Path(__file__).resolve().parents[2]
    (repository / "out").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hub-stage-", dir=repository / "out") as workspace:
        owned = Path(workspace)
        shared, installed = owned / "shared", owned / "installed"
        installed.mkdir()
        (installed / "old.txt").write_bytes(b"old app")
        hub_release.write_manifest(manifest(["old.txt"], "1.0.0"),
                                   installed / hub_release.manifest_asset_name("windows-x64"))
        member = (tmp_path.drive + "/escaped.txt") if unsafe and os.name == "nt" else (
            "C:/escaped.txt" if unsafe else "nested/new.txt")
        stage = shared / "Updates/1.1.0/stage"
        would_write = stage.joinpath(*PurePosixPath(member).parts).resolve()
        if os.name == "nt" and unsafe:
            assert would_write.is_relative_to(tmp_path) or would_write.is_relative_to(owned)
        incoming = owned / "incoming.zip"
        with zipfile.ZipFile(incoming, "w") as stream:
            stream.writestr("first.txt", b"first")
            stream.writestr(member, b"new payload")
        update = hub_updater.HubUpdate("1.0.0", "1.1.0", "", "full.zip", "", "",
                                      incoming.stat().st_size, "", "", "windows-x64")
        monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(shared))
        monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(installed))
        monkeypatch.setattr(hub_updater, "_download", lambda _u, dest, _p: shutil.copy2(incoming, dest))
        monkeypatch.setattr(hub_updater, "_request_bytes", lambda *a:
                            json.dumps(manifest(["first.txt", member])).encode())
        if unsafe:
            with pytest.raises(ValueError, match="Unsafe update path"):
                hub_updater.stage_update(update)
            assert not would_write.exists()
            assert not (stage / "first.txt").exists(), "Validate the complete manifest before extracting"
        else:
            result = hub_updater.stage_update(update)
            assert (result / "stage/nested/new.txt").read_bytes() == b"new payload"
        assert (installed / "old.txt").read_bytes() == b"old app"


def test_stage_rejects_resolved_destination_outside_root(tmp_path, monkeypatch):
    installed, outside = tmp_path / "app", tmp_path / "outside"
    installed.mkdir()
    outside.mkdir()
    (installed / "old.txt").write_text("old")
    hub_release.write_manifest(manifest(["old.txt"], "1.0.0"),
                               installed / hub_release.manifest_asset_name("windows-x64"))
    shared = tmp_path / "Shared"
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(shared))
    monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(installed))
    monkeypatch.setattr(hub_updater, "_request_bytes", lambda *a: json.dumps(manifest(["nested/file.txt"])).encode())
    def download(_update, destination, _progress):
        # Only the test's own temporary stage is linked, before extraction.
        link = destination.parent / "stage/nested"
        if os.name == "nt":
            subprocess.run([
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "New-Item -ItemType Junction -Path $env:INFERNUX_TEST_LINK "
                "-Target $env:INFERNUX_TEST_TARGET -ErrorAction Stop | Out-Null",
            ], env=dict(os.environ, INFERNUX_TEST_LINK=str(link), INFERNUX_TEST_TARGET=str(outside)),
                capture_output=True, text=True, check=True, timeout=15, creationflags=0x08000000)
        else:
            link.symlink_to(outside, target_is_directory=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("nested/file.txt", b"not allowed")
    monkeypatch.setattr(hub_updater, "_download", download)
    update = hub_updater.HubUpdate("1.0.0", "1.1.0", "", "full.zip", "", "", 0, "", "", "windows-x64")
    with pytest.raises(ValueError, match="outside.*staging"):
        hub_updater.stage_update(update)
    assert not (outside / "file.txt").exists()
