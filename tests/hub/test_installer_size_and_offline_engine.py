"""Installer size reductions and the offline first-project engine seed."""

from __future__ import annotations

import fnmatch
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_hub
import bundled_engine
import embed_runtime_manager
import stage_bundled_python_runtime as stage
import version_manager as vm
from installer.install_application import HubInstallTransaction
from installer.payload import (
    BUNDLED_ENGINES_DIR,
    HUB_EXECUTABLE,
    HUB_PAYLOAD_ARCHIVE,
    create_payload_archive,
)
from private_python_runtime import PYTHON_VERSION, runtime_archive_for_machine
from python_runtime_catalog import DEFAULT_PYTHON_RUNTIME


WHEEL = "infernux-0.4.1-3-cp313-cp313-win_amd64.whl"


def _wheel_bytes(*, version="0.4.1", build="3", tag="cp313-cp313-win_amd64") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("infernux/__init__.py", "")
        archive.writestr(
            f"infernux-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: infernux\nVersion: {version}\n",
        )
        build_line = f"Build: {build}\n" if build else ""
        archive.writestr(
            f"infernux-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\n{build_line}Tag: {tag}\n",
        )
    return stream.getvalue()


@pytest.fixture
def windows_wheels(monkeypatch):
    """Evaluate wheel platform rules as a Windows x64 Hub on every CI host."""
    monkeypatch.setattr(vm, "supported_wheel_platforms", lambda: frozenset({"win_amd64"}))
    monkeypatch.setattr(vm, "sys", SimpleNamespace(platform="win32"))


# -- Runtime bundle ---------------------------------------------------------


def _staged_runtime(root: Path) -> Path:
    runtime = root / "runtime" / DEFAULT_PYTHON_RUNTIME.directory_name
    (runtime / "Lib" / "site-packages" / "numpy").mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"private python" * 64)
    (runtime / "Lib" / "os.py").write_text("import sys\n" * 200, encoding="utf-8")
    (runtime / "Lib" / "site-packages" / "numpy" / "__init__.pyi").write_text("x: int\n")
    (runtime / ".infernux-private-python-runtime.json").write_text("{}\n")
    return runtime


def test_runtime_bundle_is_lzma_and_seeds_the_managed_runtime(tmp_path, monkeypatch):
    runtime = _staged_runtime(tmp_path / "stage")
    stage._create_runtime_bundle(str(runtime))
    bundle = runtime.parent / "runtime_bundle.zip"

    with zipfile.ZipFile(bundle) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        assert {info.compress_type for info in members} == {zipfile.ZIP_LZMA}
        assert all(info.filename.startswith("python313/") for info in members)
        assert "python313/Lib/site-packages/numpy/__init__.pyi" in archive.namelist()

    shipped = tmp_path / "hub" / "InfernuxHubData" / "runtime"
    shipped.mkdir(parents=True)
    os.replace(bundle, shipped / "runtime_bundle.zip")
    manager = embed_runtime_manager.PythonRuntimeManager(runtime_dir=str(tmp_path / "managed"))
    monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [str(shipped)])
    monkeypatch.setattr(embed_runtime_manager, "is_current_private_runtime_root", lambda *a, **kw: True)
    monkeypatch.setattr(manager, "_prepare_candidate", lambda *a, **kw: None)

    assert manager._seed_runtime_from_bundle(version="3.13") == manager.private_runtime_python("3.13")
    managed = Path(manager.private_runtime_root("3.13"))
    assert (managed / "python.exe").read_bytes() == b"private python" * 64
    assert (managed / "Lib" / "os.py").read_text(encoding="utf-8") == "import sys\n" * 200


def test_staged_runtime_prunes_debug_files_but_keeps_build_inputs(tmp_path):
    root = tmp_path / "python313"
    kept = [
        "Lib/site-packages/numpy/__init__.pyi",
        "Lib/site-packages/nuitka/build/static_src/MainProgram.c",
        "include/Python.h",
        "libs/python313.lib",
        "DLLs/libcrypto-3-x64.dll",
        "Lib/venv/scripts/nt/venvlauncher.exe",
    ]
    pruned = [
        "DLLs/libcrypto-3-x64.pdb",
        "Lib/venv/scripts/nt/venvlauncher.pdb",
        "Lib/site-packages/PyWin32.chm",
        "Lib/site-packages/numpy/tests/test_core.py",
        "Lib/__pycache__/os.cpython-313.pyc",
    ]
    for relative in kept + pruned:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    stage._prune_runtime_root(str(root))

    assert all((root / relative).is_file() for relative in kept)
    assert not any((root / relative).exists() for relative in pruned)


def _runtime_marker() -> dict[str, str]:
    archive = runtime_archive_for_machine()
    return {
        "owner": "Infernux Hub",
        "kind": "private-python-runtime",
        "python_version": PYTHON_VERSION,
        "python_series": DEFAULT_PYTHON_RUNTIME.series,
        "source_archive": archive.name,
        "written_at": "2026-10-01T16:51:42Z",
    }


# -- Installer payload ------------------------------------------------------


def _hub_dir(root: Path) -> Path:
    runtime = root / "InfernuxHubData" / "runtime"
    runtime.mkdir(parents=True)
    (root / HUB_EXECUTABLE).write_bytes(b"hub executable" * 100)
    (root / "qt6core.dll").write_bytes(b"qt core" * 100)
    with zipfile.ZipFile(runtime / "runtime_bundle.zip", "w", compression=zipfile.ZIP_LZMA) as bundle:
        bundle.writestr("python313/python.exe", b"python" * 100)
    return root


def test_installed_hub_contains_the_bundled_engine(tmp_path):
    hub = _hub_dir(tmp_path / "hub")
    wheel = tmp_path / WHEEL
    wheel.write_bytes(_wheel_bytes())
    payload = tmp_path / "payload"
    payload.mkdir()
    create_payload_archive(hub, payload / HUB_PAYLOAD_ARCHIVE, {(BUNDLED_ENGINES_DIR / WHEEL).as_posix(): wheel})
    install_dir = tmp_path / "Infernux Hub"

    with HubInstallTransaction(payload, install_dir) as installation:
        installation.prepare()
        installation.activate()
        installation.commit()

    assert (
        bundled_engine.bundled_engines_dir(install_dir / "InfernuxHubData").joinpath(WHEEL).read_bytes()
        == wheel.read_bytes()
    )


# -- Qt trimming -------------------------------------------------------------


_TRIMMED = [
    "qt6pdf.dll",
    "PySide6/qt6pdf.dll",
    "PySide6/qt-plugins/imageformats/qpdf.dll",
    "PySide6/qt-plugins/imageformats/qjpeg.dll",
    "PySide6/qt-plugins/imageformats/qwebp.dll",
    "PySide6/qt-plugins/imageformats/qtiff.dll",
    "PySide6/qt-plugins/imageformats/qgif.dll",
    "PySide6/qt-plugins/imageformats/qicns.dll",
    "PySide6/qt-plugins/imageformats/qtga.dll",
    "PySide6/qt-plugins/imageformats/qwbmp.dll",
    "PySide6/qt-plugins/platforms/qdirect2d.dll",
]
_WINDOWS_ONLY_TRIMMED = [
    "qt6svg.dll",
    "PySide6/qt-plugins/imageformats/qsvg.dll",
    "PySide6/qt-plugins/iconengines/qsvgicon.dll",
]
_KEPT = [
    "qt6core.dll",
    "qt6gui.dll",
    "qt6network.dll",
    "qt6widgets.dll",
    "PySide6/qt-plugins/platforms/qwindows.dll",
    "PySide6/qt-plugins/imageformats/qico.dll",
    "PySide6/qt-plugins/styles/qmodernwindowsstyle.dll",
    "PySide6/qt-plugins/tls/qschannelbackend.dll",
    "PySide6/qt-plugins/tls/qopensslbackend.dll",
    "PySide6/qt-plugins/tls/qcertonlybackend.dll",
    "resources/icon.png",
]


def _nuitka_matches(dest_path: str, patterns: list[str]) -> bool:
    # Mirrors Nuitka's --noinclude-dlls check: fnmatch on the destination path.
    return any(fnmatch.fnmatch(dest_path, pattern) for pattern in patterns)


def _excluded_patterns(command: list[str]) -> list[str]:
    prefix = "--noinclude-dlls="
    return [argument[len(prefix) :] for argument in command if argument.startswith(prefix)]


# -- Engine wheel embedding ----------------------------------------------------


def test_installer_embeds_only_the_exact_host_engine_wheel(tmp_path, windows_wheels):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    for name, content in {
        WHEEL: _wheel_bytes(),
        "infernux-0.4.1-2-cp313-cp313-win_amd64.whl": _wheel_bytes(build="2"),
        "infernux-0.4.0-cp313-cp313-win_amd64.whl": _wheel_bytes(version="0.4.0", build=""),
        "infernux-0.4.1-3-cp313-cp313-manylinux_2_28_x86_64.whl": _wheel_bytes(tag="cp313-cp313-manylinux_2_28_x86_64"),
        "other-0.4.1-3-cp313-cp313-win_amd64.whl": b"not ours",
    }.items():
        (release_dir / name).write_bytes(content)

    assert build_hub._bundled_engine_wheel(release_dir, "0.4.1-v3") == release_dir / WHEEL
    assert build_hub._bundled_engine_wheel(release_dir, "0.4.2") is None
    assert build_hub._bundled_engine_wheel(tmp_path / "missing", "0.4.1-v3") is None


def _installer_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    build = tmp_path / "build"
    package = tmp_path / "package"
    _hub_dir(package / "hub")
    commands = []
    filename = "InfernuxHubInstaller.exe" if build_hub.os.name == "nt" else "InfernuxHubInstaller"

    def compile_installer(command, **_kwargs):
        commands.append(command)
        (build / "nuitka" / filename).write_bytes(b"installer")

    monkeypatch.setattr(build_hub, "_common_nuitka_command", lambda *a, **kw: ["nuitka"])
    monkeypatch.setattr(build_hub, "_run", compile_installer)
    monkeypatch.setattr(build_hub, "_validate_msvc_reports", lambda *a: [])
    monkeypatch.setattr(build_hub, "_validate_windows_pe", lambda *a: None)
    monkeypatch.setattr(build_hub, "_project_version", lambda *a: "0.4.1-3")
    monkeypatch.setattr(build_hub, "_project_engine_release", lambda *a: "0.4.1-v3")
    return source, build, package, commands


def test_installer_payload_carries_the_release_engine_wheel(tmp_path, monkeypatch, windows_wheels):
    source, build, package, commands = _installer_fixture(tmp_path, monkeypatch)
    release = tmp_path / "release"
    release.mkdir()
    (release / WHEEL).write_bytes(_wheel_bytes())

    build_hub._build_installer(source, build, package, release_dir=release, build_env={})

    with zipfile.ZipFile(build / HUB_PAYLOAD_ARCHIVE) as archive:
        info = archive.getinfo(f"InfernuxHubData/engines/{WHEEL}")
        assert info.compress_type == zipfile.ZIP_STORED
        assert archive.read(info) == (release / WHEEL).read_bytes()
    assert any(argument.endswith(f"=payload/{HUB_PAYLOAD_ARCHIVE}") for argument in commands[0])
    assert f"--include-data-dir={source / 'packaging' / 'resources' / 'fonts'}=resources/fonts" in commands[0]


# -- Seeding the Engines cache --------------------------------------------------


class _RuntimeManager:
    def __init__(self, installed=("3.13",)):
        self.installed = set(installed)

    def has_runtime(self, version):
        return str(version) in self.installed

    def installed_versions(self):
        return sorted(self.installed)


@pytest.fixture
def engines(tmp_path, monkeypatch, windows_wheels):
    monkeypatch.setattr(vm, "_VERSIONS_DIR", tmp_path / "Shared" / "Engines")
    hub_data = tmp_path / "app" / "InfernuxHubData"
    (hub_data / "engines").mkdir(parents=True)
    (hub_data / "engines" / WHEEL).write_bytes(_wheel_bytes())
    return hub_data


def test_bundled_engine_is_seeded_once(engines, tmp_path):
    manager = vm.VersionManager(runtime_manager=_RuntimeManager())

    assert bundled_engine.seed_bundled_engines(manager, hub_data_dir=engines) == ["0.4.1-v3"]
    assert manager.is_installed("0.4.1-v3", "3.13")
    state = vm._VERSIONS_DIR / bundled_engine.SEEDED_STATE_FILENAME
    assert json.loads(state.read_text(encoding="utf-8")) == {"seeded": [WHEEL]}
    assert manager.installed_versions() == ["0.4.1-v3"]

    # A user who removes the engine must not see it silently come back.
    assert manager.remove_version("0.4.1-v3")
    assert bundled_engine.seed_bundled_engines(manager, hub_data_dir=engines) == []
    assert not manager.is_installed("0.4.1-v3")


def test_seeding_errors_are_logged_and_retried(engines, caplog):
    (engines / "engines" / WHEEL).write_bytes(_wheel_bytes(build="2"))
    manager = vm.VersionManager(runtime_manager=_RuntimeManager())

    with caplog.at_level("ERROR", logger="bundled_engine"):
        assert bundled_engine.seed_bundled_engines(manager, hub_data_dir=engines) == []
    assert "Could not install bundled engine wheel" in caplog.text
    assert not (vm._VERSIONS_DIR / bundled_engine.SEEDED_STATE_FILENAME).exists()
