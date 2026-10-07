"""Hub VersionManager — issue #43 regression tests.

Covers: atomic cancellable downloads, corrupted-wheel healing, unique temp
files under concurrency, and version listing that ignores broken installs.

These tests are pure-Python (no Qt, no network): GitHub access and the HTTP
stream are monkeypatched.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "packaging",
))

import version_manager as vm_mod
from version_manager import DownloadCancelled, VersionManager, _merge_release_catalogs


def test_catalog_network_failures_are_not_an_empty_version_list(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path)
    def fail(*args, **kwargs):
        raise urllib.error.URLError("TLS certificate verification failed")
    monkeypatch.setattr(vm_mod.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="Unable to fetch engine versions") as error:
        VersionManager().list_versions()
    assert "pypi:" in str(error.value) and "github:" in str(error.value)
    assert "certificate" in str(error.value)


def _make_wheel_bytes(version="9.9.9", *, build=None, python="312") -> bytes:
    """Minimal wheel with the real distribution and target identity documents."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("infernux/__init__.py", "")
        z.writestr(f"infernux-{version}.dist-info/METADATA",
                   f"Metadata-Version: 2.1\nName: infernux\nVersion: {version}\n")
        z.writestr(f"infernux-{version}.dist-info/WHEEL",
                   "Wheel-Version: 1.0\n" + (f"Build: {build}\n" if build else "")
                   + f"Tag: cp{python}-cp{python}-win_amd64\n")
    return buf.getvalue()


def test_rebuilds_keep_independent_project_version_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path)
    names = [f"infernux-0.4.1-{build}-cp313-cp313-win_amd64.whl" for build in (1, 2, 10)]
    releases = _merge_release_catalogs([
        {"tag_name": f"v0.4.1-v{build}", "assets": [{"name": name, "browser_download_url": "https://example.invalid/" + name}]}
        for build, name in zip((1, 2, 10), names)
    ], {})
    manager = VersionManager(_RuntimeInventory("3.13"))
    monkeypatch.setattr(manager, "_fetch_releases", lambda: releases)
    cache = tmp_path / "0.4.1"
    cache.mkdir()
    (cache / names[0]).write_bytes(_make_wheel_bytes("0.4.1", build=1, python="313"))
    versions = manager.list_versions()
    assert [entry.version for entry in versions] == ["0.4.1-v10", "0.4.1-v2", "0.4.1"]
    assert not any(entry.update_available for entry in versions)
    assert versions[0].wheel_url.endswith(names[2])
    # Directory enumeration / lexicographic order must not choose build 2 over 10.
    (cache / names[2]).write_bytes(_make_wheel_bytes("0.4.1", build=10, python="313"))
    (cache / names[1]).write_bytes(_make_wheel_bytes("0.4.1", build=2, python="313"))
    assert Path(manager.get_wheel_path("0.4.1")).name == names[0]
    assert Path(manager.get_wheel_path("0.4.1-v10")).name == names[2]
    assert not manager.list_versions()[0].update_available


def test_channel_retry_never_installs_older_build_under_new_name(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path)
    manager = VersionManager(_RuntimeInventory("3.13"))
    old = "infernux-0.4.1-1-cp313-cp313-win_amd64.whl"
    new = "infernux-0.4.1-2-cp313-cp313-win_amd64.whl"
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [{"tag_name": "v0.4.1-v2", "assets": [
        {"name": old, "browser_download_url": "https://example.invalid/old"},
        {"name": new, "browser_download_url": "https://example.invalid/new"},
    ]}])
    requests = []
    def fail(request, *, timeout):
        requests.append(request.full_url)
        raise urllib.error.URLError("unavailable")
    monkeypatch.setattr(vm_mod.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        manager.download_version("0.4.1-v2")
    assert requests == ["https://example.invalid/new"]
    assert not list(tmp_path.rglob("*.whl"))


class _FakeResponse:
    def __init__(self, payload: bytes, chunk: int = 7):
        self._data = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self._chunk = chunk

    def read(self, n):
        return self._data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _RuntimeInventory:
    def __init__(self, *versions: str):
        self._versions = list(versions)

    def installed_versions(self):
        return list(self._versions)

    def has_runtime(self, version):
        return str(version) in self._versions


@pytest.fixture()
def vm(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    manager = VersionManager(_RuntimeInventory("3.12"))
    release = {
        "tag_name": "v9.9.9",
        "prerelease": False,
        "published_at": "2026-01-01T00:00:00Z",
        "assets": [{
            "name": "infernux-9.9.9-cp312-cp312-win_amd64.whl",
            "browser_download_url": "https://example.invalid/infernux-9.9.9-cp312-cp312-win_amd64.whl",
            "size": len(_make_wheel_bytes()),
        }],
    }
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [release])
    return manager


@pytest.fixture(autouse=True)
def _windows_wheel_fixtures(monkeypatch):
    monkeypatch.setattr(
        vm_mod,
        "supported_wheel_platforms",
        lambda: frozenset({"win_amd64"}),
    )


class TestDownload:
    def test_successful_download_is_valid_wheel(self, vm, monkeypatch):
        payload = _make_wheel_bytes()
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(payload))
        path = vm.download_version("9.9.9")
        assert os.path.isfile(path)
        assert zipfile.is_zipfile(path)
        assert vm.is_installed("9.9.9")

    def test_cancel_leaves_no_residue(self, vm, monkeypatch):
        # Payload spans many 64 KB chunks so the per-chunk cancel check
        # actually triggers mid-transfer.
        payload = _make_wheel_bytes() + b"\0" * (64 * 1024 * 6)
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(payload))
        calls = {"n": 0}

        def cancel_after_two_chunks():
            calls["n"] += 1
            return calls["n"] > 2

        with pytest.raises(DownloadCancelled):
            vm.download_version("9.9.9", should_cancel=cancel_after_two_chunks)

        ver_dir = vm_mod._VERSIONS_DIR / "9.9.9"
        # No partial wheel, no temp files, and ideally no empty dir at all.
        if ver_dir.exists():
            assert list(ver_dir.iterdir()) == []
        assert not vm.is_installed("9.9.9")
        assert "9.9.9" not in vm.installed_versions()

    def test_cancel_then_reinstall_succeeds(self, vm, monkeypatch):
        cancel_payload = _make_wheel_bytes() + b"\0" * (64 * 1024 * 6)
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(cancel_payload))
        flag = {"n": 0}

        def cancel_once():
            flag["n"] += 1
            return flag["n"] > 1

        with pytest.raises(DownloadCancelled):
            vm.download_version("9.9.9", should_cancel=cancel_once)

        # Second attempt with no cancellation must produce a valid install.
        good_payload = _make_wheel_bytes()
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(good_payload))
        path = vm.download_version("9.9.9")
        assert zipfile.is_zipfile(path)
        assert vm.is_installed("9.9.9")

    def test_truncated_transfer_rejected(self, vm, monkeypatch):
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(b"not-a-zip"))
        with pytest.raises(ValueError, match="not a valid wheel"):
            vm.download_version("9.9.9")
        assert not vm.is_installed("9.9.9")

    def test_existing_corrupted_wheel_is_replaced(self, vm, monkeypatch):
        ver_dir = vm_mod._VERSIONS_DIR / "9.9.9"
        ver_dir.mkdir(parents=True)
        bad = ver_dir / "infernux-9.9.9-cp312-cp312-win_amd64.whl"
        bad.write_bytes(b"garbage from an interrupted install")

        payload = _make_wheel_bytes()
        monkeypatch.setattr(vm_mod.urllib.request, "urlopen",
                            lambda req, *, timeout: _FakeResponse(payload))
        path = vm.download_version("9.9.9")
        assert zipfile.is_zipfile(path)

    def test_pypi_transport_failure_uses_matching_github_asset(self, vm, monkeypatch):
        filename = "infernux-9.9.9-cp312-cp312-win_amd64.whl"
        release = {
            "tag_name": "v9.9.9",
            "assets": [
                {
                    "name": filename,
                    "browser_download_url": f"https://files.pythonhosted.org/pypi/{filename}",
                    "size": len(_make_wheel_bytes()),
                    "source": "pypi",
                },
                {
                    "name": filename,
                    "browser_download_url": f"https://github.com/github/{filename}",
                    "size": len(_make_wheel_bytes()),
                    "source": "github",
                },
            ],
        }
        monkeypatch.setattr(vm, "_fetch_releases", lambda: [release])
        requested = []

        def open_asset(request, *, timeout):
            requested.append(request.full_url)
            if "pythonhosted" in request.full_url:
                raise urllib.error.URLError("PyPI unavailable")
            return _FakeResponse(_make_wheel_bytes())

        monkeypatch.setattr(vm_mod.urllib.request, "urlopen", open_asset)

        assert zipfile.is_zipfile(vm.download_version("9.9.9"))
        assert requested == [
            f"https://files.pythonhosted.org/pypi/{filename}",
            f"https://github.com/github/{filename}",
        ]

    def test_newer_pypi_build_tag_wins_within_same_engine_version(
        self, vm, monkeypatch
    ):
        release = {
            "tag_name": "v9.9.9",
            "assets": [
                {
                    "name": "infernux-9.9.9-cp312-cp312-win_amd64.whl",
                    "browser_download_url": "https://files.pythonhosted.org/old/infernux-9.9.9-cp312-cp312-win_amd64.whl",
                    "size": len(_make_wheel_bytes()),
                    "source": "pypi",
                },
                {
                    "name": "infernux-9.9.9-1-cp312-cp312-win_amd64.whl",
                    "browser_download_url": "https://files.pythonhosted.org/new/infernux-9.9.9-1-cp312-cp312-win_amd64.whl",
                    "size": len(_make_wheel_bytes(build=1)),
                    "source": "pypi",
                },
                {
                    "name": "infernux-9.9.9-1-cp312-cp312-win_amd64.whl",
                    "browser_download_url": "https://github.com/new/infernux-9.9.9-1-cp312-cp312-win_amd64.whl",
                    "size": len(_make_wheel_bytes(build=1)),
                    "source": "github",
                },
            ],
        }
        monkeypatch.setattr(vm, "_fetch_releases", lambda: [release])
        requested = []

        def open_asset(request, *, timeout):
            requested.append(request.full_url)
            return _FakeResponse(_make_wheel_bytes(build=1))

        monkeypatch.setattr(vm_mod.urllib.request, "urlopen", open_asset)

        vm.download_version("9.9.9")
        assert requested == ["https://files.pythonhosted.org/new/infernux-9.9.9-1-cp312-cp312-win_amd64.whl"]

        wheels = vm_mod._find_wheel_assets(release)
        assert [(wheel.filename, wheel.source) for wheel in wheels[:2]] == [
            ("infernux-9.9.9-1-cp312-cp312-win_amd64.whl", "pypi"),
            ("infernux-9.9.9-1-cp312-cp312-win_amd64.whl", "github"),
        ]

    def test_invalid_pypi_wheel_does_not_change_source(self, vm, monkeypatch):
        filename = "infernux-9.9.9-cp312-cp312-win_amd64.whl"
        release = {
            "tag_name": "v9.9.9",
            "assets": [
                {
                    "name": filename,
                    "browser_download_url": f"https://files.pythonhosted.org/pypi/{filename}",
                    "size": len(_make_wheel_bytes()),
                    "source": "pypi",
                },
                {
                    "name": filename,
                    "browser_download_url": f"https://github.com/github/{filename}",
                    "size": len(_make_wheel_bytes()),
                    "source": "github",
                },
            ],
        }
        monkeypatch.setattr(vm, "_fetch_releases", lambda: [release])
        requested = []

        def open_asset(request, *, timeout):
            requested.append(request.full_url)
            return _FakeResponse(b"not a wheel")

        monkeypatch.setattr(vm_mod.urllib.request, "urlopen", open_asset)

        with pytest.raises(ValueError, match="not a valid wheel"):
            vm.download_version("9.9.9")
        assert requested == [f"https://files.pythonhosted.org/pypi/{filename}"]


class TestListingHealsCorruption:
    def test_corrupted_wheel_not_listed_as_installed(self, vm):
        ver_dir = vm_mod._VERSIONS_DIR / "1.2.3"
        ver_dir.mkdir(parents=True)
        (ver_dir / "infernux-1.2.3-cp312-cp312-win_amd64.whl").write_bytes(b"junk")

        assert vm.get_wheel_path("1.2.3") is None
        assert "1.2.3" not in vm.installed_versions()
        # healing removed the junk file
        assert not list(ver_dir.glob("*.whl"))

    def test_valid_wheel_listed(self, vm):
        ver_dir = vm_mod._VERSIONS_DIR / "2.0.0"
        ver_dir.mkdir(parents=True)
        (ver_dir / "infernux-2.0.0-cp312-cp312-win_amd64.whl").write_bytes(_make_wheel_bytes("2.0.0"))

        assert vm.get_wheel_path("2.0.0") is not None
        assert "2.0.0" in vm.installed_versions()

    def test_remove_version(self, vm):
        ver_dir = vm_mod._VERSIONS_DIR / "2.0.0"
        ver_dir.mkdir(parents=True)
        (ver_dir / "infernux-2.0.0-cp312-cp312-win_amd64.whl").write_bytes(_make_wheel_bytes("2.0.0"))
        assert vm.remove_version("2.0.0") is True
        assert not ver_dir.exists()
        assert vm.remove_version("2.0.0") is False


def test_local_engine_install_requires_its_exact_python_runtime(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    manager = VersionManager(_RuntimeInventory("3.12"))
    wheel = tmp_path / "infernux-0.4.0-cp313-cp313-win_amd64.whl"
    wheel.write_bytes(_make_wheel_bytes())

    with pytest.raises(
        ValueError,
        match=r"Infernux 0\.4\.0 requires Python 3\.13.*install Python 3\.13",
    ):
        manager.install_local_wheel(str(wheel))

    assert not (tmp_path / "versions" / "0.4.0").exists()


def test_legacy_wheel_requires_its_legacy_runtime_before_local_import(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    wheel = tmp_path / "infernux-0.3.7-cp312-cp312-win_amd64.whl"
    wheel.write_bytes(_make_wheel_bytes("0.3.7"))

    current_only = VersionManager(_RuntimeInventory("3.13"))
    with pytest.raises(
        ValueError,
        match=r"Infernux 0\.3\.7 requires Python 3\.12.*install Python 3\.12",
    ):
        current_only.install_local_wheel(str(wheel))
    assert not (tmp_path / "versions" / "0.3.7").exists()

    with_legacy_runtime = VersionManager(_RuntimeInventory("3.13", "3.12"))
    installed_version = with_legacy_runtime.install_local_wheel(str(wheel))

    assert installed_version == "0.3.7"
    assert with_legacy_runtime.get_wheel_path("0.3.7", "3.12") is not None


def test_release_with_conflicting_python_abis_is_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    manager = VersionManager(_RuntimeInventory("3.12"))
    release = {
        "tag_name": "v0.4.0",
        "assets": [
            {
                "name": "infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "browser_download_url": "https://example.invalid/infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "size": 313,
            },
            {
                "name": "infernux-0.4.0-cp312-cp312-win_amd64.whl",
                "browser_download_url": "https://example.invalid/infernux-0.4.0-cp312-cp312-win_amd64.whl",
                "size": 312,
            },
        ],
    }
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [release])

    [engine] = manager.list_versions()

    assert engine.python_version == ""
    assert [wheel.python_version for wheel in engine.wheel_options] == [
        "3.13",
        "3.12",
    ]
    assert "must target exactly one Python minor version" in engine.compatibility_error


def test_online_engine_stays_visible_but_install_is_blocked_without_python(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    manager = VersionManager(_RuntimeInventory("3.12"))
    release = {
        "tag_name": "v0.4.0",
        "assets": [
            {
                "name": "infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "browser_download_url": "https://example.invalid/infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "size": 313,
            }
        ],
    }
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [release])

    [engine] = manager.list_versions()

    assert engine.version == "0.4.0"
    assert engine.python_version == "3.13"
    assert manager.installation_block_reason(engine) == (
        "Infernux 0.4.0 requires Python 3.13. "
        "Please install Python 3.13 first."
    )
    monkeypatch.setattr(
        vm_mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a blocked engine install must not start a download")
        ),
    )
    with pytest.raises(ValueError, match=r"requires Python 3\.13"):
        manager.download_version("0.4.0")


def test_cached_engine_wheels_are_resolved_by_exact_python_abi(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    version_dir = tmp_path / "versions" / "0.4.0"
    version_dir.mkdir(parents=True)
    cp312 = version_dir / "infernux-0.4.0-cp312-cp312-win_amd64.whl"
    cp313 = version_dir / "infernux-0.4.0-cp313-cp313-win_amd64.whl"
    cp312.write_bytes(_make_wheel_bytes("0.4.0", python="312"))
    cp313.write_bytes(_make_wheel_bytes("0.4.0", python="313"))
    manager = VersionManager(_RuntimeInventory("3.12", "3.13"))

    assert manager.get_wheel_path("0.4.0", "3.12") == str(cp312)
    assert manager.get_wheel_path("0.4.0", "3.13") == str(cp313)
    assert manager.installed_python_versions("0.4.0") == ["3.13", "3.12"]
    with pytest.raises(ValueError, match="conflicting Python ABIs"):
        manager.python_version_for_engine("0.4.0")


def test_release_assets_are_filtered_by_host_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    monkeypatch.setattr(
        vm_mod,
        "supported_wheel_platforms",
        lambda: frozenset({"win_amd64"}),
    )
    manager = VersionManager(_RuntimeInventory("3.13"))
    release = {
        "tag_name": "v0.4.0",
        "assets": [
            {
                "name": "infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "browser_download_url": "https://example.invalid/infernux-0.4.0-cp313-cp313-win_amd64.whl",
                "size": 101,
            },
            {
                "name": "infernux-0.4.0-cp313-cp313-manylinux_2_28_x86_64.whl",
                "browser_download_url": "https://example.invalid/infernux-0.4.0-cp313-cp313-manylinux_2_28_x86_64.whl",
                "size": 202,
            },
        ],
    }
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [release])

    [engine] = manager.list_versions()

    assert [wheel.filename for wheel in engine.wheel_options] == [
        "infernux-0.4.0-cp313-cp313-win_amd64.whl"
    ]
    assert engine.wheel_url == "https://example.invalid/infernux-0.4.0-cp313-cp313-win_amd64.whl"
    assert engine.python_version == "3.13"


def test_pypi_and_github_catalogs_merge_with_pypi_preferred(tmp_path, monkeypatch):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    filename = "infernux-0.4.0-cp313-cp313-win_amd64.whl"
    releases = _merge_release_catalogs(
        [
            {
                "tag_name": "v0.4.0",
                "published_at": "2026-09-01T00:00:00Z",
                "assets": [
                    {
                        "name": filename,
                        "browser_download_url": "https://github.com/wheel",
                        "size": 20,
                    }
                ],
            }
        ],
        {
            "releases": {
                "0.4.0": [
                    {
                        "filename": filename,
                        "url": "https://files.pythonhosted.org/wheel",
                        "size": 21,
                        "packagetype": "bdist_wheel",
                        "yanked": False,
                    }
                ]
            }
        },
    )
    manager = VersionManager(_RuntimeInventory("3.13"))
    monkeypatch.setattr(manager, "_fetch_releases", lambda: releases)

    [engine] = manager.list_versions()

    assert engine.sources == ("pypi", "github")
    assert engine.wheel_url == "https://files.pythonhosted.org/wheel"
    assert [wheel.source for wheel in engine.wheel_options] == ["pypi", "github"]


def test_local_engine_install_rejects_foreign_platform_before_copy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vm_mod, "_VERSIONS_DIR", tmp_path / "versions")
    monkeypatch.setattr(
        vm_mod,
        "supported_wheel_platforms",
        lambda: frozenset({"win_amd64"}),
    )
    manager = VersionManager(_RuntimeInventory("3.13"))
    wheel = tmp_path / "infernux-0.4.0-cp313-cp313-manylinux_2_28_x86_64.whl"
    wheel.write_bytes(_make_wheel_bytes())

    with pytest.raises(ValueError, match="not compatible with this platform"):
        manager.install_local_wheel(str(wheel))

    assert not (tmp_path / "versions" / "0.4.0").exists()
