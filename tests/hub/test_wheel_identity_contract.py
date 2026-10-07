"""Metadata identity is checked before an artifact becomes selectable."""
import io
import zipfile
from types import SimpleNamespace

import pytest

import version_manager as vm


NAME = "infernux-0.4.1-3-cp313-cp313-win_amd64.whl"


def archive_bytes(*, name="infernux", version="0.4.1", build="3", tag="cp313-cp313-win_amd64", extra_metadata=False):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("infernux/__init__.py", "")
        archive.writestr("infernux-0.4.1.dist-info/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr("infernux-0.4.1.dist-info/WHEEL", f"Wheel-Version: 1.0\nBuild: {build}\nTag: {tag}\n")
        if extra_metadata:
            archive.writestr("other-1.0.dist-info/METADATA", "Name: other\nVersion: 1.0\n")
    return stream.getvalue()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_VERSIONS_DIR", tmp_path / "cache")
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({"win_amd64"}))
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform="win32"))
    manager = vm.VersionManager()
    monkeypatch.setattr(manager, "_require_installed_python", lambda *args, **kwargs: None)
    return manager


@pytest.mark.parametrize("change", [dict(name="other"), dict(version="0.4.0"), dict(build="2"),
                                   dict(tag="cp312-cp312-win_amd64"), dict(extra_metadata=True)])
def test_mislabeled_local_wheel_is_not_published(manager, tmp_path, change):
    source = tmp_path / NAME
    source.write_bytes(archive_bytes(**change))
    with pytest.raises(ValueError, match="identity"):
        manager.install_local_wheel(str(source))
    assert not manager.is_installed("0.4.1-v3")
    assert not list((tmp_path / "cache").rglob("*.whl"))


def test_manually_copied_mislabeled_cache_is_not_available(manager, tmp_path):
    path = tmp_path / "cache/0.4.1" / NAME
    path.parent.mkdir(parents=True)
    content = archive_bytes(build="2")
    path.write_bytes(content)
    assert not manager.is_installed("0.4.1-v3")
    assert manager.installed_python_versions("0.4.1-v3") == []
    assert manager.installed_versions() == []
    assert path.read_bytes() == content


def test_download_identity_failure_does_not_try_another_channel(manager, tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_fetch_releases", lambda: [{"tag_name": "v0.4.1-v3", "assets": [
        {"name": NAME, "browser_download_url": "https://example.invalid/primary", "source": "pypi"},
        {"name": NAME, "browser_download_url": "https://example.invalid/secondary", "source": "github"},
    ]}])
    seen = []

    class Response(io.BytesIO):
        headers = {}

    def download(request, **_kwargs):
        seen.append(request.full_url)
        return Response(archive_bytes(build="2"))

    monkeypatch.setattr(vm.urllib.request, "urlopen", download)
    with pytest.raises(ValueError, match="identity"):
        manager.download_version("0.4.1-v3")
    assert seen == ["https://example.invalid/primary"]
    assert not list((tmp_path / "cache").rglob("*.whl"))
    assert not list((tmp_path / "cache").rglob("*.tmp-*"))


def test_valid_wheel_can_be_imported_and_selected(manager, tmp_path):
    source = tmp_path / NAME
    source.write_bytes(archive_bytes())
    assert manager.install_local_wheel(str(source)) == "0.4.1-v3"
    assert manager.is_installed("0.4.1-v3")


def test_failed_local_import_preserves_the_installed_artifact(manager, tmp_path):
    source = tmp_path / NAME
    content = archive_bytes()
    source.write_bytes(content)
    manager.install_local_wheel(str(source))
    installed = manager.get_wheel_path("0.4.1-v3")
    source.write_bytes(archive_bytes(version="0.4.0"))
    with pytest.raises(ValueError, match="identity"):
        manager.install_local_wheel(str(source))
    from pathlib import Path
    assert Path(installed).read_bytes() == content
    assert manager.is_installed("0.4.1-v3")
    assert not list((tmp_path / "cache").rglob("*.tmp-*"))


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("case", ["bad_zip", "truncated_zip", "missing_metadata", "wrong_identity",
                                   "copy_interrupted", "replace_denied", "missing_source", "valid"])
def test_local_wheel_publication_is_atomic(manager, tmp_path, monkeypatch, existing, case):
    from pathlib import Path
    import shutil

    source = tmp_path / "中文 空格 & incoming" / NAME
    source.parent.mkdir()
    cached = tmp_path / "cache/0.4.1" / NAME
    cached.parent.mkdir(parents=True)
    old = archive_bytes()
    if existing:
        cached.write_bytes(old)
    # Another exact revision must remain selectable through every outcome.
    other = cached.with_name(NAME.replace('-3-cp313', '-2-cp313'))
    other_bytes = archive_bytes(build="2")
    other.write_bytes(other_bytes)
    new = io.BytesIO(archive_bytes())
    with zipfile.ZipFile(new, 'a') as archive:
        archive.writestr('infernux/new.txt', b'new payload')
    incoming = new.getvalue()
    if case == 'bad_zip':
        incoming = b'interrupted download'
    elif case == 'truncated_zip':
        incoming = incoming[:-30]
    elif case == 'missing_metadata':
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('infernux/payload', 'payload')
        incoming = data.getvalue()
    elif case == 'wrong_identity':
        incoming = archive_bytes(build='2')
    if case != 'missing_source':
        source.write_bytes(incoming)
    if case == 'copy_interrupted':
        def interrupted_copy(src, dst):
            Path(dst).write_bytes(Path(src).read_bytes()[:20])
            raise OSError('controlled copy failure')
        monkeypatch.setattr(shutil, 'copyfile', interrupted_copy)
    if case == 'replace_denied':
        def replace_denied(src, dst):
            assert Path(dst) == cached
            raise PermissionError('controlled publication conflict')
        monkeypatch.setattr(vm.os, 'replace', replace_denied)

    if case == 'valid':
        assert manager.install_local_wheel(str(source)) == '0.4.1-v3'
        assert cached.read_bytes() == incoming
    else:
        with pytest.raises((ValueError, OSError)):
            manager.install_local_wheel(str(source))
        assert cached.exists() == existing
        if existing:
            assert cached.read_bytes() == old
    assert manager.get_wheel_path('0.4.1-v3') == (str(cached) if existing or case == 'valid' else None)
    assert manager.get_wheel_path('0.4.1-v2') == str(other)
    assert other.read_bytes() == other_bytes
    assert not list(cached.parent.glob('*.tmp-*'))


def test_local_wheel_reimport_accepts_the_selected_cache_file(manager, tmp_path):
    source = tmp_path / NAME
    payload = archive_bytes()
    source.write_bytes(payload)
    manager.install_local_wheel(str(source))
    from pathlib import Path
    installed = Path(manager.get_wheel_path('0.4.1-v3'))
    assert manager.install_local_wheel(str(installed)) == '0.4.1-v3'
    assert installed.read_bytes() == payload
    assert not list(installed.parent.glob('*.tmp-*'))


@pytest.mark.parametrize("change", [None, dict(version="0.4.0"), dict(name="other_distribution"), dict(build="2"),
                                   dict(tag="cp313-cp313-manylinux_2_35_x86_64"), dict(tag="cp312-cp312-win_amd64")])
def test_real_http_catalog_download_cache_chain(manager, monkeypatch, change):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    payload = archive_bytes(**(change or {}))
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            if self.path == "/pypi":
                content = b'{"releases":{}}'
            elif self.path.startswith("/releases?"):
                content = json.dumps([{"tag_name":"v0.4.1-v3", "assets":[{
                    "name": NAME, "browser_download_url": f"{origin}/{NAME}", "size": len(payload),
                }]}]).encode()
            elif self.path == f"/{NAME}":
                content = payload
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(vm, "_API_BASE", origin)
    monkeypatch.setattr(vm, "_PYPI_API", origin + "/pypi")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        [entry] = manager.list_versions()
        assert entry.version == "0.4.1-v3" and not entry.installed
        if change:
            with pytest.raises(ValueError, match="identity"):
                manager.download_version(entry.version)
            assert not manager.is_installed(entry.version)
        else:
            manager.download_version(entry.version)
            assert manager.is_installed(entry.version)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert requests == ["/pypi", "/releases?per_page=50", f"/{NAME}"]
    if change is None:
        offline = vm.VersionManager()
        assert offline.list_versions()[0].installed
        assert offline.get_wheel_path("0.4.1-v3")
