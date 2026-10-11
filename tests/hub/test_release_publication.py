"""Exercise publication over local HTTP and the actual PowerShell release entry."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
import urllib.error

import pytest

ROOT = Path(__file__).resolve().parents[2]
CATALOG_KEY = "plugins/official-registry.json"
PACKAGE_KEY = "plugins/fixture.platform/1.2.3/fixture.platform.inxpkg"


@pytest.fixture
def channel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/release"))
    hub = importlib.import_module("publish_hub_objects")
    plugins = importlib.import_module("publish_plugin_objects")
    objects, requests, uploads = {}, [], {}
    failures = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_object(self):
            requests.append((self.command, self.path))
            key = self.path.removeprefix("/public/")
            if key not in objects:
                self.send_error(404)
                return
            data = objects[key]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(data)

        do_GET = read_object
        do_HEAD = read_object

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            key = body["key"]
            requests.append((self.command, self.path, key))
            if (self.path, key) in failures:
                self.send_error(503)
            elif self.path == "/upload/start":
                uploads[key] = []
                self.send_json({"uploadId": "fixture"})
            elif self.path == "/upload/complete":
                objects[key] = b"".join(uploads.pop(key))
                self.send_json({"size": len(objects[key])})
            elif self.path == "/upload/abort":
                uploads.pop(key, None)
                self.send_json({"aborted": True})
            else:
                self.send_error(404)

        def do_PUT(self):
            query = parse_qs(urlparse(self.path).query)
            key = query["key"][0]
            data = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.command, "/upload/part", key))
            if ("/upload/part", key) in failures:
                self.send_error(503)
                return
            uploads[key].append(data)
            self.send_json({"partNumber": int(query["partNumber"][0]), "etag": "fixture"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(hub, "PUBLIC_ENDPOINT", url + "/public")
    monkeypatch.setattr(hub, "UPLOAD_ENDPOINT", url + "/upload")
    # Force multiple parts so failure tests exercise an unfinished publication.
    monkeypatch.setattr(hub, "PART_SIZE", 16)
    payload = tmp_path / "fixture.platform.inxpkg"
    payload.write_bytes(b"INXPKG\0\0local package publication fixture")
    catalog = tmp_path / "official-registry.json"
    catalog.write_text(json.dumps({"packages": [{
        "reference": "fixture/platform", "version": "1.2.3", "category": "platform_build",
        "source": {"location": url + "/public/" + PACKAGE_KEY},
    }, {
        "reference": "fixture/editor", "category": "editor",
        "source": {"location": url + "/public/plugins/editor.inxpkg"},
    }]}), encoding="utf-8")
    objects["plugins/editor.inxpkg"] = b"existing editor package"
    monkeypatch.setattr(plugins, "ROOT", tmp_path)
    monkeypatch.setattr(plugins, "REGISTRY", catalog)
    # Only the GitHub download is supplied locally. Publisher and HTTP are real.
    monkeypatch.setattr(plugins, "stage", lambda *_: (payload, PACKAGE_KEY))
    monkeypatch.setattr(sys, "argv", ["publish_plugin_objects.py"])
    monkeypatch.setenv("R2_UPLOAD_TOKEN", "local-fixture-token")
    try:
        yield hub, plugins, objects, requests, failures, catalog, payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("existing", ["missing", "identical", "older"])
def test_registry_advances_only_after_all_packages_are_public(channel, existing):
    _, plugins, objects, requests, _, catalog, payload = channel
    if existing != "missing":
        objects[CATALOG_KEY] = catalog.read_bytes() if existing == "identical" else b'{"packages": []}'
    plugins.main()
    assert objects[CATALOG_KEY] == catalog.read_bytes()
    assert objects[PACKAGE_KEY] == payload.read_bytes()
    completion = requests.index(("POST", "/upload/complete", CATALOG_KEY))
    for key in [PACKAGE_KEY, "plugins/editor.inxpkg"]:
        assert requests.index(("HEAD", "/public/" + key)) < completion


@pytest.mark.parametrize("key", [PACKAGE_KEY, "hub/0.4.1/build-3/payload.zip", "hub/0.4.1/build-3/manifest.json"])
def test_versioned_objects_cannot_be_replaced_even_if_json(channel, key):
    hub, _, objects, requests, _, catalog, _ = channel
    objects[key] = b"published release"
    with pytest.raises(RuntimeError, match="Immutable published object differs"):
        hub.Publisher("fixture").upload(catalog, key)
    assert objects[key] == b"published release"
    assert requests == [("GET", "/public/" + key)]


def test_identical_versioned_payload_is_not_uploaded_twice(channel):
    hub, _, objects, requests, _, _, payload = channel
    objects[PACKAGE_KEY] = payload.read_bytes()
    hub.Publisher("fixture").upload(payload, PACKAGE_KEY)
    assert requests == [("GET", "/public/" + PACKAGE_KEY)]


@pytest.mark.parametrize("contents", [None, b""])
def test_unavailable_advertised_package_preserves_catalog(channel, contents):
    _, plugins, objects, requests, _, _, _ = channel
    objects[CATALOG_KEY] = b"old catalog"
    if contents is None:
        del objects["plugins/editor.inxpkg"]
    else:
        objects["plugins/editor.inxpkg"] = contents
    with pytest.raises((urllib.error.HTTPError, ValueError)):
        plugins.main()
    assert objects[CATALOG_KEY] == b"old catalog"
    assert ("POST", "/upload/start", CATALOG_KEY) not in requests


@pytest.mark.parametrize("key", [PACKAGE_KEY, CATALOG_KEY])
@pytest.mark.parametrize("step", ["part", "complete"])
def test_failed_multipart_publication_preserves_previous_catalog(channel, key, step):
    _, plugins, objects, requests, failures, _, _ = channel
    objects[CATALOG_KEY] = b"old catalog"
    failures.add(("/upload/" + step, key))
    with pytest.raises(urllib.error.HTTPError, match="503"):
        plugins.main()
    assert objects[CATALOG_KEY] == b"old catalog"
    assert ("POST", "/upload/abort", key) in requests


def test_single_reference_does_not_publish_registry(channel, monkeypatch):
    _, plugins, objects, requests, _, _, _ = channel
    objects[CATALOG_KEY] = b"old catalog"
    monkeypatch.setattr(sys, "argv", ["publish_plugin_objects.py", "--reference", "fixture/platform"])
    plugins.main()
    assert objects[CATALOG_KEY] == b"old catalog"
    assert ("POST", "/upload/start", CATALOG_KEY) not in requests


@pytest.mark.skipif(os.name != "nt", reason="Windows local release entry")
@pytest.mark.parametrize("build,inventory,success", [
    (1, "current", True), (3, "current", True), (3, "stale_hub", False),
    (3, "stale_wheel", False), (3, "current_and_old", True),
    (3, "stale_manifest", False), (3, "empty_wheel", False),
    (3, "wrong_platform", False), (3, "version_mismatch", False),
    (3, "cmake_failure", False),
])
def test_local_release_matches_authoritative_build(tmp_path, build, inventory, success):
    shell = shutil.which("powershell.exe")
    assert shell, "Windows release tests require the OS PowerShell executable"
    # Exercise paths with spaces and non-ASCII text, as supported by the entry.
    root = tmp_path / "发布 project"
    scripts = root / "scripts/release"
    scripts.mkdir(parents=True)
    for name in ["release_hub.ps1", "local_release_identity.py"]:
        source = ROOT / "scripts/release" / name
        shutil.copyfile(source, scripts / name)
    (root / "packaging").mkdir()
    shutil.copyfile(ROOT / "packaging/hub_release.py", root / "packaging/hub_release.py")
    (root / "pyproject.toml").write_text('[project]\nversion = "0.4.1"\n', encoding="utf-8")
    (root / "python/infernux").mkdir(parents=True)
    (root / "python/infernux/version.py").write_text(f"ENGINE_BUILD_NUMBER = {build}\n", encoding="utf-8")
    output = root / "dist/releases/0.4.1"
    output.mkdir(parents=True)
    hub_version = "0.4.1" if build == 1 else f"0.4.1-{build}"
    asset_hub = "0.4.1" if inventory == "stale_hub" else hub_version
    wheel_build = 2 if inventory == "stale_wheel" else build
    names = [f"infernux-0.4.1-{wheel_build}-cp313-cp313-win_amd64.whl",
             f"InfernuxHubInstaller-{asset_hub}-windows-x64.exe",
             f"InfernuxHub-{asset_hub}-windows-x64-full.zip"]
    if inventory == "current_and_old":
        names.extend(["infernux-0.4.1-2-cp313-cp313-win_amd64.whl", "InfernuxHubInstaller-0.4.1-windows-x64.exe"])
    for name in names:
        (output / name).write_bytes(b"inventory fixture, not a release binary")
    if inventory == "empty_wheel":
        (output / names[0]).write_bytes(b"")
    (output / "InfernuxHub-windows-x64-manifest.json").write_text(json.dumps({
        "$schema": "infernux.hub_update", "product": "InfernuxHub",
        "version": "0.4.1" if inventory == "stale_manifest" else hub_version,
        "platform": "linux-x64" if inventory == "wrong_platform" else "windows-x64", "files": [],
    }), encoding="utf-8")
    runner = tmp_path / "invoke.ps1"
    runner.write_text(
        "param([string]$Script, [string]$Version, [int]$CMakeExit)\n"
        "function global:cmake { $global:LASTEXITCODE=$CMakeExit; Write-Output 'CMAKE_BOUNDARY' }\n"
        "& $Script -Version $Version\n", encoding="utf-8")
    result = subprocess.run([
        shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(runner),
        "-Script", str(scripts / "release_hub.ps1"),
        "-Version", "0.4.2" if inventory == "version_mismatch" else "0.4.1",
        "-CMakeExit", "7" if inventory == "cmake_failure" else "0",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    expected_calls = 0 if inventory == "version_mismatch" else 1 if inventory == "cmake_failure" else 3
    assert result.stdout.count("CMAKE_BOUNDARY") == expected_calls, result.stdout + result.stderr
