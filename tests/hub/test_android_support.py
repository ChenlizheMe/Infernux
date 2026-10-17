from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGING_DIR = Path(__file__).resolve().parents[2] / "packaging"

import android_support


@pytest.mark.parametrize(
    "host,code,release_after,expected_calls",
    [
        ("nt", 5, 24, 25),
        ("nt", 32, 24, 25),
        ("nt", 32, None, 52),
        ("posix", 5, None, 1),
        ("nt", 87, None, 1),
        ("nt", None, None, 1),
    ],
)
def test_directory_publication_bounds_only_windows_sharing_errors(
    monkeypatch,
    host,
    code,
    release_after,
    expected_calls,
):
    error = PermissionError("SDK directory is held open")
    error.winerror = code
    clock, calls = [0.0], []

    def replace(source, destination):
        calls.append((source, destination))
        if release_after is None or len(calls) <= release_after:
            raise error

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(android_support, "os", SimpleNamespace(name=host, replace=replace))
    monkeypatch.setattr(android_support, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    if release_after is None:
        with pytest.raises(PermissionError) as raised:
            android_support._replace_support_directory("staging", "installed")
        assert raised.value is error
    else:
        android_support._replace_support_directory("staging", "installed")
    assert calls == [("staging", "installed")] * expected_calls
    assert clock[0] < 5.11


@pytest.fixture(autouse=True)
def isolated_hub_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(android_support, "get_hub_shared_data_dir", lambda: str(tmp_path / "shared"))


def _create_support(root: Path) -> None:
    sdk = root / "sdk"
    for directory in (
        sdk / "platforms/android-36",
        sdk / "build-tools/36.0.0",
        sdk / "cmake/3.30.5",
    ):
        directory.mkdir(parents=True)
        (directory / "source.properties").write_text("installed", encoding="utf-8")
    (sdk / "licenses").mkdir(parents=True)
    (sdk / "licenses/android-sdk-license").write_text("accepted", encoding="utf-8")
    toolchain = sdk / "ndk/29.0.14206865/build/cmake/android.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_text("toolchain", encoding="utf-8")
    adb = sdk / "platform-tools" / ("adb.exe" if sys.platform == "win32" else "adb")
    adb.parent.mkdir(parents=True)
    adb.write_bytes(b"adb")

    jdk = root / "jdk"
    java = jdk / "bin" / ("java.exe" if sys.platform == "win32" else "java")
    java.parent.mkdir(parents=True)
    java.write_bytes(b"java")
    (jdk / "release").write_text('JAVA_VERSION="17.0.12"\n', encoding="utf-8")
    gradle = root / "gradle/bin" / ("gradle.bat" if sys.platform == "win32" else "gradle")
    gradle.parent.mkdir(parents=True)
    gradle.write_bytes(b"gradle")

    for abi in android_support.ANDROID_PYTHON_ABIS:
        prefix = root / "python" / abi
        for directory in (
            prefix / "include/python3.13",
            prefix / "lib/python3.13",
        ):
            directory.mkdir(parents=True)
            (directory / ".keep").write_bytes(b"runtime")
        (prefix / "infernux-android-python.json").write_text(
            json.dumps(
                {
                    "$schema": "infernux.android_python_runtime",
                    "kind": "infernux-android-cpython",
                    "target": {"abi": abi},
                    "cpython": {"version": "3.13.15"},
                    "toolchain": {"ndk_version": android_support.ANDROID_PYTHON_NDK},
                }
            ),
            encoding="utf-8",
        )

    (root / android_support.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "$schema": android_support.ANDROID_SUPPORT_SCHEMA,
                "kind": android_support.ANDROID_SUPPORT_KIND,
                "version": android_support.ANDROID_SUPPORT_VERSION,
                "host": android_support.host_id(),
                "requirements": {
                    "android_api": android_support.ANDROID_API,
                    "build_tools": android_support.ANDROID_BUILD_TOOLS,
                    "cmake": android_support.ANDROID_CMAKE,
                    "ndk": android_support.ANDROID_NDK,
                    "python_ndk": android_support.ANDROID_PYTHON_NDK,
                    "jdk_major": android_support.ANDROID_JDK_MAJOR,
                    "gradle": android_support.GRADLE_VERSION,
                    "python": android_support.ANDROID_PYTHON_SERIES,
                    "python_abis": list(android_support.ANDROID_PYTHON_ABIS),
                },
                "paths": {
                    "sdk": "sdk",
                    "jdk": "jdk",
                    "gradle": "gradle",
                    "python": {
                        "arm64-v8a": "python/arm64-v8a",
                        "x86_64": "python/x86_64",
                    },
                },
                "total_bytes": 128,
            }
        ),
        encoding="utf-8",
    )


def _archive_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def test_installs_one_shared_android_support_archive_and_exports_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _create_support(source)
    archive = tmp_path / "android.inxkit"
    _archive_tree(source, archive)
    manager = android_support.AndroidSupportManager(tmp_path / "managed")
    monkeypatch.delenv("INFERNUX_ANDROID_SUPPORT_ROOT", raising=False)

    assert not manager.status().installed
    installed = manager.install_archive(archive)

    assert installed == str(manager.root)
    assert manager.status().installed
    assert Path(os.environ["INFERNUX_ANDROID_SUPPORT_ROOT"]) == manager.root
    environment = android_support.android_support_environment(manager.root)
    assert environment["ANDROID_SDK_ROOT"] == str(manager.root / "sdk")
    assert environment["INFERNUX_ANDROID_PYTHON_PREFIX_ARM64"].endswith(
        "python\\arm64-v8a" if sys.platform == "win32" else "python/arm64-v8a"
    )


def test_invalid_reinstall_preserves_the_current_platform_kit(tmp_path: Path) -> None:
    current = tmp_path / "managed"
    _create_support(current)
    before = (current / android_support.MANIFEST_NAME).read_bytes()
    broken = tmp_path / "broken.inxkit"
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr(android_support.MANIFEST_NAME, "{}")

    manager = android_support.AndroidSupportManager(current)
    with pytest.raises(android_support.AndroidSupportError, match="manifest"):
        manager.install_archive(broken)

    assert (current / android_support.MANIFEST_NAME).read_bytes() == before
    assert manager.status().installed


@pytest.mark.skipif(os.name != "posix", reason="POSIX executable permissions")
def test_installed_kit_preserves_executables_without_privileged_mode_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _create_support(source)
    java = source / "jdk/bin/java"
    java.write_text("#!/bin/sh\nprintf 'kit-java-ready\\n'\n", encoding="utf-8")
    java.chmod(0o4755)
    plain = source / "jdk/release"
    plain.chmod(0o644)
    archive = tmp_path / "android.inxkit"
    _archive_tree(source, archive)
    manager = android_support.AndroidSupportManager(tmp_path / "managed")
    # Environment publication is covered separately; do not leak this test's kit.
    monkeypatch.setattr(manager, "activate_environment", lambda: True)

    manager.install_archive(archive)

    executable = manager.root / "jdk/bin/java"
    assert stat.S_IMODE(executable.stat().st_mode) == 0o755
    assert stat.S_IMODE((manager.root / "jdk/release").stat().st_mode) == 0o644
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout == "kit-java-ready\n"


@pytest.mark.parametrize("received", [b"trunc", b"modified"])
def test_channel_download_rejects_incomplete_or_changed_content(tmp_path, monkeypatch, received):
    expected = b"complete"
    manager = android_support.AndroidSupportManager(tmp_path / "managed")
    _create_support(manager.root)
    manifest = (manager.root / android_support.MANIFEST_NAME).read_bytes()
    monkeypatch.setattr(
        manager,
        "_release_asset",
        lambda: ("https://example.invalid/android.inxkit", hashlib.sha256(expected).hexdigest(), len(expected)),
    )
    response = io.BytesIO(received)
    response.status = 206
    response.headers = {"Content-Range": f"bytes 0-{len(expected) - 1}/{len(expected)}"}
    monkeypatch.setattr(android_support.urllib.request, "urlopen", lambda *_a, **_k: response)
    with pytest.raises(android_support.AndroidSupportError, match=f"received {len(received)}/{len(expected)} bytes"):
        manager.install()
    assert (manager.root / android_support.MANIFEST_NAME).read_bytes() == manifest
    partials = list((tmp_path / "shared").rglob("*.part"))
    if len(received) < len(expected):
        assert len(partials) == 1 and partials[0].read_bytes() == received
    else:
        assert partials == []


def test_channel_download_uses_contiguous_ranges_and_one_complete_digest(tmp_path, monkeypatch):
    content = b"a" * (32 * 1024 * 1024) + b"last range"
    manager = android_support.AndroidSupportManager(tmp_path / "managed")
    monkeypatch.setattr(
        manager,
        "_release_asset",
        lambda: ("https://example.invalid/android.inxkit", hashlib.sha256(content).hexdigest(), len(content)),
    )
    requests = []
    progress = []

    def open_range(request, timeout):
        assert timeout == 120
        header = request.get_header("Range")
        requests.append(header)
        start, end = map(int, header.removeprefix("bytes=").split("-"))
        response = io.BytesIO(content[start : end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(content)}"}
        return response

    def install_archive(path):
        assert Path(path).read_bytes() == content
        return "installed"

    monkeypatch.setattr(android_support.urllib.request, "urlopen", open_range)
    monkeypatch.setattr(manager, "install_archive", install_archive)
    assert manager.install(on_progress=lambda done, total: progress.append((done, total))) == "installed"
    assert requests == ["bytes=0-33554431", f"bytes=33554432-{len(content) - 1}"]
    assert progress[-1] == (len(content), len(content))
    assert all(a[0] < b[0] for a, b in zip(progress, progress[1:]))
    assert not list((tmp_path / "shared").rglob("*.part"))


def test_explicit_install_retry_resumes_interrupted_release(tmp_path, monkeypatch):
    content = b"prefix and remaining bytes"
    manager = android_support.AndroidSupportManager(tmp_path / "managed")
    monkeypatch.setattr(
        manager,
        "_release_asset",
        lambda: ("https://example.invalid/android.inxkit", hashlib.sha256(content).hexdigest(), len(content)),
    )
    requests = []

    class InterruptedResponse(io.BytesIO):
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}"}

        def read(self, count=-1):
            if self.tell() == 6:
                raise TimeoutError("connection interrupted")
            return super().read(count)

    def open_range(request, **kwargs):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return InterruptedResponse(content[:6])
        response = io.BytesIO(content[6:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 6-{len(content) - 1}/{len(content)}"}
        return response

    def install_archive(path):
        assert Path(path).read_bytes() == content
        return "installed"

    monkeypatch.setattr(android_support.urllib.request, "urlopen", open_range)
    monkeypatch.setattr(manager, "install_archive", install_archive)
    with pytest.raises(TimeoutError, match="interrupted"):
        manager.install()
    assert len(requests) == 1
    partials = list((tmp_path / "shared/Cache/Downloads/Android").glob("*.part"))
    assert len(partials) == 1 and partials[0].read_bytes() == content[:6]
    assert not manager.root.exists()
    assert manager.install() == "installed"
    assert requests == [f"bytes=0-{len(content) - 1}", f"bytes=6-{len(content) - 1}"]
    assert not partials[0].exists()


def test_release_packager_creates_the_exact_hub_installable_asset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _create_support(source)
    output = tmp_path / android_support.archive_name()
    command = [
        sys.executable,
        str(PACKAGING_DIR.parent / "scripts/setup/package_android_support.py"),
        "--sdk",
        str(source / "sdk"),
        "--jdk",
        str(source / "jdk"),
        "--gradle",
        str(source / "gradle"),
        "--python-arm64",
        str(source / "python/arm64-v8a"),
        "--python-x86-64",
        str(source / "python/x86_64"),
        "--output",
        str(output),
    ]

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command.insert(1, "-I")
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    first = output.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert output.read_bytes() == first
    manager = android_support.AndroidSupportManager(tmp_path / "installed")
    manager.install_archive(output)
    assert manager.status().installed
