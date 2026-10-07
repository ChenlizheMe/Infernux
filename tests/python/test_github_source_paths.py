"""GitHub source snapshots use portable archive identities before file writes."""
from __future__ import annotations

import io
import json
import tarfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from infernux.plugins.github_releases import download_github_source


@pytest.fixture
def source_download(tmp_path, monkeypatch):
    payload = b""
    commit = "a" * 40
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = json.dumps({"sha": commit}).encode() if self.path == "/revision" else payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    real_open = urllib.request.urlopen

    def local_open(request, *, timeout):
        if request.full_url.startswith("https://api.github.com/"):
            path = "/revision"
        else:
            assert request.full_url.startswith("https://codeload.github.com/")
            path = "/archive"
        return real_open(f"http://127.0.0.1:{server.server_port}{path}", timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", local_open)

    def download(entries, *, subdirectory=""):
        nonlocal payload
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, kind in entries:
                info = tarfile.TarInfo(name)
                if kind == "directory":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                elif kind in {"symlink", "hardlink"}:
                    info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                    info.linkname = "../escape.txt"
                    archive.addfile(info)
                else:
                    content = kind.encode("utf-8")
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
        payload = buffer.getvalue()
        return download_github_source(
            "https://github.com/test/PortablePlugin", str(tmp_path / "download"),
            revision="main", subdirectory=subdirectory,
        )

    try:
        yield download, requests
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()


def _assert_no_extraction(tmp_path):
    checkout = tmp_path / "download" / "checkout"
    assert not checkout.exists() or not list(checkout.iterdir())
    assert not (tmp_path / "escape.txt").exists()


def test_source_drive_path_cannot_escape_checkout(tmp_path, source_download):
    download, _ = source_download
    # The only potential escaped destination is still inside this test's temp
    # directory, including when exercising the unfixed Windows implementation.
    escaped = (tmp_path / "escape.txt").as_posix()
    with pytest.raises(ValueError, match="source snapshot path"):
        download([("repo/valid.txt", "valid"), (f"repo/{escaped}", "escaped")])
    _assert_no_extraction(tmp_path)


@pytest.mark.parametrize("path", [
    "repo/../escape.txt", "/repo/escape.txt", "repo/./escape.txt", "repo//escape.txt",
    "repo/folder\\escape.txt", "repo/file.txt:stream", "repo/trailing. /file.txt",
    "repo/CON.txt", "repo/folder/AUX", "repo/folder/COM1.txt",
    "repo/C:relative.txt", "repo/folder/LPT².txt", "repo/name\x01.txt",
])
def test_source_nonportable_paths_are_rejected_before_writes(tmp_path, source_download, path):
    download, _ = source_download
    with pytest.raises(ValueError, match="source snapshot path"):
        download([("repo/valid.txt", "valid"), (path, "invalid")])
    _assert_no_extraction(tmp_path)


@pytest.mark.parametrize("second", ["repo/same.txt", "repo/SAME.txt", "other/different.txt", "repo/same.txt/child"])
def test_source_ambiguous_identity_is_rejected_before_writes(tmp_path, source_download, second):
    download, _ = source_download
    with pytest.raises(ValueError, match="source snapshot"):
        download([("repo/same.txt", "first"), (second, "second")])
    _assert_no_extraction(tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_source_links_are_rejected_before_writes(tmp_path, source_download, kind):
    download, _ = source_download
    with pytest.raises(RuntimeError, match="unsupported entry"):
        download([("repo/valid.txt", "valid"), ("repo/link", kind)])
    _assert_no_extraction(tmp_path)


@pytest.mark.parametrize("selected", ["../plugin", "/plugin", "C:/plugin", "plugin/../sibling"])
def test_source_subdirectory_must_be_relative_before_download(tmp_path, source_download, selected):
    download, requests = source_download
    with pytest.raises(ValueError, match="source snapshot path"):
        download([("repo/plugin/source.py", "source")], subdirectory=selected)
    assert requests == []
    assert not (tmp_path / "download").exists()


def test_source_portable_subdirectory_keeps_unicode(tmp_path, source_download):
    download, requests = source_download
    result = download([
        ("repo/", "directory"), ("repo/plugins/", "directory"),
        ("repo/plugins/海图 tools/", "directory"),
        ("repo/plugins/海图 tools/Runtime/boot.py", "value = 42\n"),
        ("repo/plugins/海图 tools/Textures/海.png", "image"),
        ("repo/ignored/link", "symlink"),
        ("repo/README.md", "outside selection"),
    ], subdirectory="plugins\\海图 tools")
    root = Path(result.root)
    assert (root / "Runtime/boot.py").read_text() == "value = 42\n"
    assert (root / "Textures/海.png").read_text() == "image"
    assert not (root / "README.md").exists()
    assert result.commit == "a" * 40
    assert requests == ["/revision", "/archive"]
    assert not (tmp_path / "download/source.tar.gz").exists()


@pytest.mark.parametrize("entries", [
    [("repo/dir/first.txt", "first"), ("repo/DIR/second.txt", "second")],
    [("repo/dir/first.txt", "first"), ("repo/dir", "file over implicit directory")],
    [("repo/dir/", "directory"), ("repo/dir", "file over explicit directory")],
])
def test_source_parent_alias_or_file_overlap_is_rejected(tmp_path, source_download, entries):
    download, _ = source_download
    with pytest.raises(ValueError, match="source snapshot"):
        download(entries)
    _assert_no_extraction(tmp_path)
