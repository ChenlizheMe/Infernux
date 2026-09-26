import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import hub_updater
import python_runtime
from hub_release import create_manifest, manifest_asset_name, write_manifest
from hub_updater import HubUpdate, HubUpdateStatus, check_for_update, stage_update


def _platform_release(target: str, platform: str, size: int = 1) -> dict:
    full = f"InfernuxHub-{target}-{platform}-full.zip"
    installer = (
        f"InfernuxHubInstaller-{target}-windows-x64.exe"
        if platform == "windows-x64"
        else f"InfernuxHubInstaller-{target}-linux-x64"
    )
    return {
        "installer": {
            "name": installer,
            "url": f"https://example.invalid/{installer}",
            "fallback_url": f"https://github.invalid/{installer}",
        },
        "update": {
            "name": full,
            "url": f"https://example.invalid/{full}",
            "fallback_url": f"https://github.invalid/{full}",
            "size": size,
        },
        "manifest": {
            "name": manifest_asset_name(platform),
            "url": f"https://example.invalid/{manifest_asset_name(platform)}",
            "fallback_url": f"https://github.invalid/{manifest_asset_name(platform)}",
        },
    }


def _catalog(
    target: str,
    *platforms: str,
    minimum_updatable_version: str = "0.0.0",
) -> dict:
    return {
        "$schema": "infernux.hub_catalog",
        "stable": target,
        "releases": [
            {
                "version": target,
                "channel": "stable",
                "published_at": "2026-09-03T00:00:00Z",
                "release_url": "https://example.invalid/release",
                "minimum_updatable_version": minimum_updatable_version,
                "platforms": {
                    platform: _platform_release(target, platform)
                    for platform in platforms
                },
            }
        ],
    }


def test_packaged_hub_version_requires_the_current_document(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(hub_updater, "is_frozen", lambda: True)
    monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(tmp_path))
    (tmp_path / "hub-version.json").write_text(
        '{"version": "0.4.0", "old_version": "0.3.7"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="current schema"):
        hub_updater.current_hub_version()


def test_unpublished_release_does_not_offer_downloads(monkeypatch):
    release = _catalog("0.4.0", "windows-x64", "linux-x64")
    release["releases"][0]["published_at"] = None
    monkeypatch.setattr(hub_updater, "_request_bytes", lambda *_args: json.dumps(release).encode())

    result = check_for_update("0.3.7", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.CATALOG_INVALID
    assert "not published" in result.detail
    assert result.update is None


def test_check_selects_the_standalone_update(monkeypatch):
    release = _catalog("1.1.0", "windows-x64", "linux-x64")
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("1.0.0", platform_id="windows-x64")
    update = result.update

    assert result.status is HubUpdateStatus.UPDATE_AVAILABLE
    assert update is not None
    assert update.asset_name == "InfernuxHub-1.1.0-windows-x64-full.zip"
    assert update.asset_url.endswith(update.asset_name)
    assert update.manifest_url.endswith("InfernuxHub-windows-x64-manifest.json")
    assert update.platform == "windows-x64"


def test_check_selects_the_linux_release(monkeypatch):
    release = _catalog("1.1.0", "windows-x64", "linux-x64")
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("1.0.0", platform_id="linux-x64")
    update = result.update

    assert result.status is HubUpdateStatus.UPDATE_AVAILABLE
    assert update is not None
    assert update.asset_name == "InfernuxHub-1.1.0-linux-x64-full.zip"
    assert update.manifest_url.endswith("InfernuxHub-linux-x64-manifest.json")
    assert update.platform == "linux-x64"


def test_asset_mirror_metadata_is_optional(monkeypatch):
    release = _catalog("1.1.0", "windows-x64")
    for asset in release["releases"][0]["platforms"]["windows-x64"].values():
        del asset["fallback_url"]
    monkeypatch.setattr(hub_updater, "_request_bytes", lambda *_args: json.dumps(release).encode())

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.UPDATE_AVAILABLE
    assert result.update.asset_fallback_url == ""
    assert result.update.manifest_fallback_url == ""


def test_check_requires_the_current_platform_release(monkeypatch):
    release = _catalog("1.1.0", "windows-x64")
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("1.0.0", platform_id="linux-x64")

    assert result.status is HubUpdateStatus.CATALOG_INVALID
    assert "no linux-x64 release" in result.detail


def test_check_rejects_a_malformed_catalog(monkeypatch):
    release = _catalog("1.1.0", "windows-x64")
    del release["releases"][0]["published_at"]
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.CATALOG_INVALID
    assert "current contract" in result.detail


def test_check_returns_none_when_current_is_latest(monkeypatch):
    release = _catalog("1.1.0", "windows-x64")
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("1.1.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.UP_TO_DATE
    assert result.update is None


def test_check_reports_network_unavailable(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(hub_updater, "_request_bytes", fail)

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.NETWORK_UNAVAILABLE
    assert "offline" in result.detail


def test_request_preserves_the_actual_network_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("connection timed out")

    monkeypatch.setattr(hub_updater.urllib.request, "urlopen", fail)

    with pytest.raises(OSError, match="connection timed out"):
        hub_updater._request_bytes("https://example.invalid/catalog")


def test_catalog_uses_the_website_before_its_static_mirror(monkeypatch):
    requested = []
    release = _catalog("1.1.0", "windows-x64")

    def fetch(request, *, timeout):
        requested.append(request.full_url)
        assert timeout == 15
        if len(requested) == 1:
            raise OSError("website unavailable")
        return io.BytesIO(json.dumps(release).encode())

    monkeypatch.setattr(hub_updater.urllib.request, "urlopen", fetch)

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.UPDATE_AVAILABLE
    assert requested == [
        "https://infernux-engine.com/hub-catalog.json",
        "https://raw.githubusercontent.com/ChenlizheMe/Infernux/master/docs/hub-catalog.json",
    ]


@pytest.mark.parametrize("body, status", [
    (b"not json", HubUpdateStatus.CATALOG_INVALID),
    (json.dumps(_catalog("1.0.0", "windows-x64")).encode(), HubUpdateStatus.UP_TO_DATE),
])
def test_a_successful_catalog_response_does_not_query_another_authority(
    monkeypatch, body, status,
):
    requested = []

    def fetch(request, *, timeout):
        requested.append(request.full_url)
        return io.BytesIO(body)

    monkeypatch.setattr(hub_updater.urllib.request, "urlopen", fetch)

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is status
    assert requested == [hub_updater.HUB_CATALOG_URL]


def test_catalog_network_failures_are_bounded_and_visible(monkeypatch):
    requested = []

    def fetch(request, *, timeout):
        requested.append(request.full_url)
        raise OSError("offline")

    monkeypatch.setattr(hub_updater.urllib.request, "urlopen", fetch)

    result = check_for_update("1.0.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.NETWORK_UNAVAILABLE
    assert "offline" in result.detail
    assert len(requested) == 2


@pytest.mark.parametrize("current, target, status", [
    ("1.1.0-rc.1", "1.1.0", HubUpdateStatus.UPDATE_AVAILABLE),
    ("1.1.0-rc.2", "1.1.0-rc.10", HubUpdateStatus.UPDATE_AVAILABLE),
    ("1.1.0", "1.1.0-rc.1", HubUpdateStatus.UP_TO_DATE),
    ("1.9.9", "1.10.0", HubUpdateStatus.UPDATE_AVAILABLE),
])
def test_version_ordering_includes_release_candidates(monkeypatch, current, target, status):
    release = _catalog(target, "windows-x64")
    monkeypatch.setattr(hub_updater, "_request_bytes", lambda *_args: json.dumps(release).encode())

    assert check_for_update(current, platform_id="windows-x64").status is status


def test_download_uses_a_bounded_timeout_and_one_asset_mirror(tmp_path, monkeypatch):
    release = _catalog("1.1.0", "windows-x64")
    monkeypatch.setattr(hub_updater, "_request_bytes", lambda *_args: json.dumps(release).encode())
    update = check_for_update("1.0.0", platform_id="windows-x64").update
    requested = []

    def fetch(request, *, timeout):
        requested.append(request.full_url)
        assert timeout == 60
        if len(requested) == 1:
            raise OSError("primary interrupted")
        response = io.BytesIO(b"x")
        response.headers = {"Content-Length": "1"}
        return response

    monkeypatch.setattr(hub_updater.urllib.request, "urlopen", fetch)
    destination = tmp_path / update.asset_name

    hub_updater._download(update, destination)

    assert destination.read_bytes() == b"x"
    assert requested == [update.asset_url, update.asset_fallback_url]


def test_update_into_versioned_runtime_hub_is_required(monkeypatch):
    release = _catalog(
        "0.4.0",
        "windows-x64",
        minimum_updatable_version="0.4.0",
    )
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("0.3.7", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.UNSUPPORTED_CURRENT_VERSION
    assert result.update is None
    assert result.installer_url.endswith("windows-x64.exe")


def test_updates_after_runtime_catalog_migration_are_optional(monkeypatch):
    release = _catalog(
        "0.4.1",
        "windows-x64",
        minimum_updatable_version="0.4.0",
    )
    monkeypatch.setattr(
        hub_updater,
        "_request_bytes",
        lambda *_args, **_kwargs: json.dumps(release).encode(),
    )

    result = check_for_update("0.4.0", platform_id="windows-x64")

    assert result.status is HubUpdateStatus.UPDATE_AVAILABLE
    assert result.update is not None


def test_packaged_updater_requests_elevation(tmp_path: Path, monkeypatch):
    staged = tmp_path / "update"
    (staged / "stage").mkdir(parents=True)
    (staged / "hub-update.json").write_text("{}", encoding="utf-8")
    observed = {}

    monkeypatch.setattr(hub_updater.sys, "platform", "win32")
    monkeypatch.setattr(hub_updater, "is_frozen", lambda: True)
    monkeypatch.setattr(
        hub_updater, "get_app_dir", lambda: str(tmp_path / "installed")
    )

    def launch(script, arguments, working_directory):
        observed["script"] = script
        observed["arguments"] = arguments
        observed["working_directory"] = working_directory

    monkeypatch.setattr(hub_updater, "_launch_elevated_powershell", launch)

    hub_updater.launch_external_updater(staged, is_dark=True)

    assert observed["script"] == staged / "apply-update.ps1"
    assert observed["working_directory"] == staged
    assert "-InstallDir" in observed["arguments"]
    assert "-StageDir" in observed["arguments"]
    assert "-MetadataPath" in observed["arguments"]
    script = observed["script"].read_text(encoding="utf-8-sig")
    assert "#191919" in script
    assert "@BG_BASE@" not in script


def test_packaged_linux_updater_uses_the_managed_runtime(tmp_path: Path, monkeypatch):
    staged = tmp_path / "update"
    (staged / "stage").mkdir(parents=True)
    (staged / "hub-update.json").write_text("{}", encoding="utf-8")
    installed = tmp_path / "installed"
    updater = installed / "InfernuxHubData" / "updater" / "hub_update_apply.py"
    updater.parent.mkdir(parents=True)
    updater.write_text("", encoding="utf-8")
    observed = {}

    class RuntimeManager:
        def get_runtime_path(self):
            return "/managed/python"

    monkeypatch.setattr(hub_updater.sys, "platform", "linux")
    monkeypatch.setattr(hub_updater, "is_frozen", lambda: True)
    monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(installed))
    monkeypatch.setattr(python_runtime, "PythonRuntimeManager", RuntimeManager)
    monkeypatch.setattr(
        hub_updater.subprocess,
        "Popen",
        lambda arguments, **kwargs: observed.update(
            arguments=arguments, kwargs=kwargs
        ),
    )

    hub_updater.launch_external_updater(staged, is_dark=True)

    assert observed["arguments"][0] == "/managed/python"
    assert observed["arguments"][1] == str(updater)
    assert observed["kwargs"] == {"cwd": staged, "start_new_session": True}


def test_stage_update_uses_the_archive_and_current_manifests(
    tmp_path: Path, monkeypatch
):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "Infernux Hub.exe").write_bytes(b"new executable")
    (payload / "lib.dll").write_bytes(b"new library")
    platform = "windows-x64"
    manifest = create_manifest(payload, "1.1.0", platform)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    archive_path = tmp_path / "full.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in payload.iterdir():
            archive.write(file, file.name)

    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "Infernux Hub.exe").write_bytes(b"old executable")
    (installed / "removed.dll").write_bytes(b"removed")
    write_manifest(
        create_manifest(installed, "1.0.0", platform),
        installed / manifest_asset_name(platform),
    )
    update = HubUpdate(
        current_version="1.0.0",
        target_version="1.1.0",
        release_url="",
        asset_name=archive_path.name,
        asset_url="",
        asset_fallback_url="",
        size=archive_path.stat().st_size,
        manifest_url="",
        manifest_fallback_url="",
        platform=platform,
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(installed / "InfernuxHubData/Shared"))
    monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(installed))
    monkeypatch.setattr(
        hub_updater,
        "_download",
        lambda _update, destination, _progress=None: shutil.copy2(
            archive_path, destination
        ),
    )
    monkeypatch.setattr(
        hub_updater, "_request_bytes", lambda *_args, **_kwargs: manifest_bytes
    )

    staged = stage_update(update)

    assert (staged / "stage" / "Infernux Hub.exe").read_bytes() == b"new executable"
    assert (staged / "stage" / "lib.dll").read_bytes() == b"new library"
    assert (
        staged / "stage" / manifest_asset_name(platform)
    ).read_bytes() == manifest_bytes
    metadata = json.loads(
        (staged / "hub-update.json").read_text(encoding="utf-8")
    )
    assert metadata["target_version"] == "1.1.0"
    assert metadata["delete"] == ["removed.dll"]
    assert {entry["path"] for entry in metadata["files"]} == {
        "Infernux Hub.exe",
        "lib.dll",
        manifest_asset_name(platform),
    }


def test_stage_update_rejects_an_unowned_archive_member(tmp_path: Path, monkeypatch):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "Infernux Hub.exe").write_bytes(b"new executable")
    platform = "windows-x64"
    manifest = create_manifest(payload, "1.1.0", platform)
    manifest_bytes = json.dumps(manifest).encode()
    archive_path = tmp_path / "full.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(payload / "Infernux Hub.exe", "Infernux Hub.exe")
        archive.writestr("unexpected.dll", b"unexpected")

    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "Infernux Hub.exe").write_bytes(b"old executable")
    write_manifest(
        create_manifest(installed, "1.0.0", platform),
        installed / manifest_asset_name(platform),
    )
    update = HubUpdate(
        current_version="1.0.0",
        target_version="1.1.0",
        release_url="",
        asset_name=archive_path.name,
        asset_url="",
        asset_fallback_url="",
        size=archive_path.stat().st_size,
        manifest_url="",
        manifest_fallback_url="",
        platform=platform,
    )
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(hub_updater, "get_app_dir", lambda: str(installed))
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(installed / "InfernuxHubData/Shared"))
    monkeypatch.setattr(
        hub_updater,
        "_download",
        lambda _update, destination, _progress=None: shutil.copy2(
            archive_path, destination
        ),
    )
    monkeypatch.setattr(
        hub_updater, "_request_bytes", lambda *_args, **_kwargs: manifest_bytes
    )

    with pytest.raises(ValueError, match="does not match its manifest"):
        stage_update(update)
