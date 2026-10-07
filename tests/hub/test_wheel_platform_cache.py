"""Platform, ABI and byte identity must survive catalog/cache/project boundaries."""
import hashlib
import io
import json
import os
import time
import urllib.error
import zipfile
from types import SimpleNamespace

import pytest

import version_manager as vm
import model.project_model as pm


WIN = "infernux-0.4.1-3-cp313-cp313-win_amd64.whl"
LINUX = "infernux-0.4.1-3-cp313-cp313-manylinux_2_35_x86_64.whl"


def payload(name=WIN, source="good"):
    stream = io.BytesIO()
    tag = name.split("-3-")[1][:-4]
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("infernux/__init__.py", source)
        archive.writestr("infernux-0.4.1.dist-info/METADATA", "Name: infernux\nVersion: 0.4.1\n")
        archive.writestr("infernux-0.4.1.dist-info/WHEEL", f"Wheel-Version: 1.0\nBuild: 3\nTag: {tag}\n")
    return stream.getvalue()


def release(*names):
    return {"tag_name": "v0.4.1-v3", "assets": [
        {"name": name, "browser_download_url": "https://example.invalid/" + name}
        for name in names
    ]}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({"win_amd64"}))
    monkeypatch.setattr(vm, "_VERSIONS_DIR", tmp_path / "Engines")
    runtime = SimpleNamespace(has_runtime=lambda version: version == "3.13", installed_versions=lambda: ["3.13"])
    return vm.VersionManager(runtime)


@pytest.mark.parametrize("platform,tag,expected", [
    ("win32", "win_amd64", WIN), ("linux", "manylinux_2_35_x86_64", LINUX),
])
def test_catalog_platform_does_not_depend_on_asset_order(manager, monkeypatch, platform, tag, expected):
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({tag}))
    for names in ((LINUX, WIN), (WIN, LINUX)):
        monkeypatch.setattr(manager, "_fetch_releases", lambda: [release(*names)])
        entry = manager.list_versions()[0]
        assert entry.wheel_url.endswith(expected)
        assert entry.python_version == "3.13"
        assert [w.filename for w in entry.wheel_options] == [expected]


def test_cross_build_tags_cannot_select_linux_on_windows(manager, monkeypatch):
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({"manylinux_2_35_x86_64", "any"}))
    assert not vm.wheel_platform_compatible(LINUX)
    assert not vm.wheel_platform_compatible("infernux-0.4.1-3-cp313-cp313-any.whl")


@pytest.mark.parametrize("platform,tag,expected", [
    ("win32", "win_amd64", WIN), ("linux", "manylinux_2_35_x86_64", LINUX),
])
@pytest.mark.parametrize("online", [True, False])
def test_shared_disk_catalog_is_filtered_again_for_host(manager, monkeypatch, online, platform, tag, expected):
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({tag}))
    manager._cache_file.write_text(json.dumps({"_ts": time.time() if online else 0, "releases": [release(LINUX, WIN)]}))
    def unavailable(*args, **kwargs):
        if online:
            pytest.fail("fresh catalog should not need network")
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(vm.urllib.request, "urlopen", unavailable)
    assert manager.list_versions()[0].wheel_url.endswith(expected)


@pytest.mark.parametrize("platform,tag,compatible,foreign", [
    ("win32", "win_amd64", WIN, LINUX), ("linux", "manylinux_2_35_x86_64", LINUX, WIN),
])
def test_foreign_cache_is_not_an_installed_engine(manager, monkeypatch, platform, tag, compatible, foreign):
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({tag}))
    folder = vm._VERSIONS_DIR / "0.4.1"
    folder.mkdir()
    (folder / foreign).write_bytes(payload(foreign))
    assert not manager.is_installed("0.4.1-v3")
    assert manager.installed_python_versions("0.4.1-v3") == []
    (folder / compatible).write_bytes(payload(compatible))
    assert manager.get_wheel_path("0.4.1-v3") == str(folder / compatible)
    assert manager.get_wheel_path("0.4.1-v3", "3.12") is None
    assert (folder / foreign).is_file()  # Other host's cache is not deleted.


@pytest.mark.parametrize("name,url", [
    ("../" + WIN, WIN), ("..\\" + WIN, WIN), (WIN, LINUX),
    (WIN, LINUX.replace("manylinux", "%6danylinux")),
    (WIN.replace("cp313-cp313", "cp313-cp312"), WIN.replace("cp313-cp313", "cp313-cp312")),
])
def test_catalog_cannot_redirect_platform_or_cache_path(manager, name, url):
    document = release(name)
    document["assets"][0]["browser_download_url"] = "https://example.invalid/" + url
    assert vm._find_wheel_assets(document) == ()


def test_catalog_preserves_hashes_from_both_channels(manager):
    digest = hashlib.sha256(payload()).hexdigest()
    github = release(WIN)
    github["assets"][0]["digest"] = "sha256:" + digest
    pypi = {"releases": {"0.4.1": [{"filename": WIN, "url": "https://example.invalid/" + WIN,
                                     "packagetype": "bdist_wheel", "digests": {"sha256": digest}}]}}
    [merged] = vm._merge_release_catalogs([github], pypi)
    wheels = vm._find_wheel_assets(merged)
    assert len(wheels) == 2
    assert [w.sha256 for w in wheels] == [digest, digest]


@pytest.mark.parametrize("bad", ["linux", "mislabeled", "digest", "size", "cancel"])
def test_wrong_download_never_becomes_cached(manager, monkeypatch, bad):
    good = payload()
    document = release(WIN)
    document["assets"][0].update(size=len(good), digest="sha256:" + hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [document])
    content = payload(LINUX) if bad in {"linux", "mislabeled"} else payload(source="evil") if bad == "digest" else good
    if bad == "mislabeled":
        # A valid Windows metadata envelope can still contain the wrong bytes.
        content = payload(source="different native payload")
    if bad == "size":
        content += b"garbage"
    class Response(io.BytesIO):
        headers = {}
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *args, **kwargs: Response(content))
    with pytest.raises(vm.DownloadCancelled if bad == "cancel" else ValueError):
        manager.download_version("0.4.1-v3", should_cancel=lambda: bad == "cancel")
    assert not list(vm._VERSIONS_DIR.rglob("*.whl"))
    assert not list(vm._VERSIONS_DIR.rglob("*.tmp-*"))


def test_bad_cached_bytes_are_replaced_only_after_verified_download(manager, monkeypatch):
    good = payload()
    document = release(WIN)
    document["assets"][0].update(size=len(good), digest="sha256:" + hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [document])
    path = vm._VERSIONS_DIR / "0.4.1" / WIN
    path.parent.mkdir()
    bad = payload(source="evil")
    path.write_bytes(bad)
    class Response(io.BytesIO):
        headers = {}
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *args, **kwargs: Response(bad))
    with pytest.raises(ValueError, match="SHA-256"):
        manager.download_version("0.4.1-v3")
    assert path.read_bytes() == bad
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *args, **kwargs: Response(good))
    assert manager.download_version("0.4.1-v3") == str(path)
    assert path.read_bytes() == good
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("verified cache hit"))
    manager.download_version("0.4.1-v3")


def test_offline_cache_checks_persisted_release_digest(manager, monkeypatch):
    good = payload()
    document = release(WIN)
    document["assets"][0]["digest"] = "sha256:" + hashlib.sha256(good).hexdigest()
    manager._cache_file.write_text(json.dumps({"_ts": 0, "releases": [document]}))
    path = vm._VERSIONS_DIR / "0.4.1" / WIN
    path.parent.mkdir()
    path.write_bytes(payload(source="evil"))
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("cache read must not fetch"))
    assert manager.get_wheel_path("0.4.1-v3") is None
    assert manager.installed_python_versions("0.4.1-v3") == []
    path.write_bytes(good)
    assert manager.get_wheel_path("0.4.1-v3") == str(path)


def test_local_import_cannot_overwrite_verified_release_with_different_bytes(manager, tmp_path):
    good = payload()
    document = release(WIN)
    document["assets"][0]["digest"] = "sha256:" + hashlib.sha256(good).hexdigest()
    manager._cache_file.write_text(json.dumps({"_ts": 0, "releases": [document]}))
    path = vm._VERSIONS_DIR / "0.4.1" / WIN
    path.parent.mkdir()
    path.write_bytes(good)
    incoming = tmp_path / WIN
    incoming.write_bytes(payload(source="evil"))
    with pytest.raises(ValueError, match="SHA-256"):
        manager.install_local_wheel(str(incoming))
    assert path.read_bytes() == good
    assert not list(path.parent.glob("*.tmp-*"))


def test_conflicting_source_digests_do_not_choose_arbitrary_bytes(manager, monkeypatch):
    document = release(WIN, WIN)
    for index, asset in enumerate(document["assets"]):
        asset["digest"] = "sha256:" + str(index) * 64
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [document])
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("inconsistent release"))
    with pytest.raises(ValueError, match="sources disagree"):
        manager.download_version("0.4.1-v3")


@pytest.mark.parametrize("cached", [False, True])
def test_other_source_digest_cannot_be_bypassed(manager, monkeypatch, cached):
    good = payload()
    bad = payload(source="evil")
    document = release(WIN, WIN)
    document["assets"][0]["source"] = "pypi"
    document["assets"][1]["digest"] = "sha256:" + hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [document])
    destination = vm._VERSIONS_DIR / "0.4.1" / WIN
    if cached:
        destination.parent.mkdir()
        destination.write_bytes(bad)
    class Response(io.BytesIO):
        headers = {}
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda *a, **k: Response(bad))
    with pytest.raises(ValueError, match="SHA-256"):
        manager.download_version("0.4.1-v3")
    if cached:
        assert destination.read_bytes() == bad
    else:
        assert not destination.exists()
    assert not list(vm._VERSIONS_DIR.rglob("*.tmp-*"))


@pytest.mark.parametrize("document", [[], {"_ts": "yesterday"}, {"_ts": 0, "releases": [None]}, "truncated"])
def test_broken_catalog_is_refetched_atomically(manager, monkeypatch, document):
    manager._cache_file.write_text(document if isinstance(document, str) else json.dumps(document))
    class Response(io.BytesIO):
        pass
    monkeypatch.setattr(vm.urllib.request, "urlopen", lambda request, **kw: Response(
        json.dumps({"releases": {}} if "pypi.org" in request.full_url else [release(LINUX, WIN)]).encode()))
    assert manager.list_versions()[0].wheel_url.endswith(WIN)
    assert json.loads(manager._cache_file.read_text())["releases"]
    assert not list(vm._VERSIONS_DIR.glob("*.tmp-*"))


@pytest.mark.parametrize("bad", ["platform", "abi", "runtime", "identity"])
def test_project_install_rechecks_contract_before_touching_files(manager, tmp_path, monkeypatch, bad):
    project = tmp_path / "Project"
    site = project / ".runtime/site-packages"
    package = site / "infernux"
    package.mkdir(parents=True)
    existing = package / "__init__.py"
    existing.write_text("KEEP")
    executable = project / ".runtime/python.exe"
    executable.touch()
    name = LINUX if bad == "platform" else WIN
    wheel = tmp_path / name
    wheel.write_bytes(payload(LINUX if bad == "identity" else name))
    monkeypatch.setattr(pm, "is_frozen", lambda: True)
    monkeypatch.setattr(pm.ProjectModel, "_get_project_python", lambda *args: str(executable))
    monkeypatch.setattr(pm.ProjectModel, "_get_site_packages", lambda *args: str(site))
    monkeypatch.setattr(pm, "_project_python_version", lambda *args: "3.12" if bad == "abi" else "3.13")
    monkeypatch.setattr(pm, "_run_hidden", lambda *args, **kw: SimpleNamespace(stdout="3.12\n" if bad == "runtime" else "3.13\n"))
    model = pm.ProjectModel(None, SimpleNamespace(get_wheel_path=lambda *args: str(wheel)), object())
    with pytest.raises((RuntimeError, ValueError), match="Incompatible|ABI mismatch|identity"):
        model._install_infernux_in_runtime(str(project), "0.4.1-v3", validate_current=False)
    assert existing.read_text() == "KEEP"


def test_install_fingerprint_detects_same_size_and_mtime_change(tmp_path):
    wheel = tmp_path / WIN
    wheel.write_bytes(payload(source="good"))
    timestamp = wheel.stat()
    before = pm._wheel_install_fingerprint(str(wheel))
    wheel.write_bytes(payload(source="evil"))
    os.utime(wheel, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert wheel.stat().st_size == timestamp.st_size
    assert pm._wheel_install_fingerprint(str(wheel)) != before


@pytest.mark.parametrize("member", ["../../escaped", "C:/escaped", "/escaped", "infernux/file:stream"])
def test_unsafe_wheel_preserves_existing_install(manager, tmp_path, member):
    wheel = tmp_path / WIN
    wheel.write_bytes(payload())
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, "bad")
    site = tmp_path / "site-packages"
    package = site / "infernux"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("KEEP")
    with pytest.raises(ValueError, match="Unsafe"):
        pm._install_wheel_direct(str(wheel), str(site), "infernux")
    assert (package / "__init__.py").read_text() == "KEEP"
