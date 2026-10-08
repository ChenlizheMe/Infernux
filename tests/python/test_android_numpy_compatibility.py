"""Validate wheel selection and staging, without executing Android binaries."""

import importlib
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def exporter(monkeypatch):
    editor = Path(__file__).resolve().parents[2] / "external/plugins/infernux_android/package/editor"
    monkeypatch.syspath_prepend(str(editor))
    for architecture in ("ARM64", "X86_64"):
        monkeypatch.delenv("INFERNUX_ANDROID_NUMPY_WHEEL_" + architecture, raising=False)
    return importlib.import_module("infernux_android.exporter")


def wheel(root, platform, *, version="2.5.2", interpreter="cp313", abi="cp313"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"numpy-{version}-{interpreter}-{abi}-{platform}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("numpy/__init__.py", "VALUE = 42\n")
    return path


def request(path=None):
    options = {} if path is None else {"android_numpy_wheel": str(path)}
    return SimpleNamespace(profile=SimpleNamespace(options=options), report=lambda *args: None)


@pytest.mark.parametrize("abi,architecture", [("x86_64", "x86_64"), ("arm64-v8a", "arm64_v8a")])
@pytest.mark.parametrize("api", [21, 24, 26, 27, 35, "bad", "26_wrong"])
def test_wheel_api_tag_matches_player_minimum(exporter, tmp_path, abi, architecture, api):
    selected = wheel(tmp_path / "wheels", f"android_{api}_{architecture}")
    if isinstance(api, int) and api <= 26:
        assert exporter._find_android_numpy_wheel(request(), tmp_path, abi, "3.13") == selected
    else:
        with pytest.raises(ValueError, match="API 26"):
            exporter._find_android_numpy_wheel(request(), tmp_path, abi, "3.13")


@pytest.mark.parametrize("configured", ["automatic", "option", "environment"])
def test_newer_incompatible_wheel_does_not_override_compatible_version(exporter, tmp_path, monkeypatch, configured):
    compatible = wheel(tmp_path / "wheels", "android_26_arm64_v8a", version="2.4.0")
    incompatible = wheel(tmp_path / "wheels", "android_35_arm64_v8a")
    build = request(incompatible if configured == "option" else None)
    if configured == "environment":
        monkeypatch.setenv("INFERNUX_ANDROID_NUMPY_WHEEL_ARM64", str(incompatible))
    if configured == "automatic":
        assert exporter._find_android_numpy_wheel(build, tmp_path, "arm64-v8a", "3.13") == compatible
    else:
        # An explicit choice is rejected; a different wheel is never substituted.
        with pytest.raises(ValueError, match="API 26"):
            exporter._find_android_numpy_wheel(build, tmp_path, "arm64-v8a", "3.13")


@pytest.mark.parametrize("platform,interpreter,abi", [
    ("android_26_x86_64", "cp313", "cp313"),
    ("android_26_arm64_v8a", "cp312", "cp312"),
    ("android_26_arm64_v8a", "cp313", "cp313t"),
])
def test_wheel_requires_exact_python_and_architecture(exporter, tmp_path, platform, interpreter, abi):
    wheel(tmp_path / "wheels", platform, interpreter=interpreter, abi=abi)
    with pytest.raises(ValueError):
        exporter._find_android_numpy_wheel(request(), tmp_path, "arm64-v8a", "3.13")


@pytest.mark.parametrize("api", [24, 26, 35])
def test_staging_validates_wheel_before_mutating_existing_runtime(exporter, tmp_path, api):
    manifest = importlib.import_module("infernux_android.runtime_manifest")
    prefix = tmp_path / "prefix"
    include = prefix / "include/python3.13"
    stdlib = prefix / "lib/python3.13/encodings"
    include.mkdir(parents=True)
    stdlib.mkdir(parents=True)
    (include / "Python.h").write_text("// staging fixture", encoding="utf-8")
    (stdlib / "__init__.py").write_text("# staging fixture", encoding="utf-8")
    (prefix / "lib/libpython3.13.so").write_bytes(b"not executed: Android native fixture")
    manifest.create_runtime_manifest(prefix, abi="arm64-v8a", python_version="3.13.15",
        cpython_android_api=24, minimum_android_api=26, ndk_version="27.3.13750724",
        source_url="https://fixture.invalid/python-source")
    selected = wheel(tmp_path / "provided", f"android_{api}_arm64_v8a")
    staging = tmp_path / "staging"
    sentinel = staging / "app/src/main/assets/python/previous-runtime.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("previous successful build", encoding="utf-8")
    if api > 26:
        with pytest.raises(ValueError, match="API 26"):
            exporter._stage_python_runtime(request(selected), staging, prefix, "arm64-v8a")
        assert sentinel.read_text(encoding="utf-8") == "previous successful build"
        assert not (staging / "app/src/main/jniLibs").exists()
    else:
        assert exporter._stage_python_runtime(request(selected), staging, prefix, "arm64-v8a") == "3.13"
        assert (sentinel.parent / "site-packages/numpy/__init__.py").read_text() == "VALUE = 42\n"


def test_gradle_minimum_uses_same_contract_as_wheel_selection(exporter, tmp_path, monkeypatch):
    template = Path(exporter.__file__).parent / "templates/host"
    staging = tmp_path / "host"
    shutil.copytree(template, staging)
    monkeypatch.setattr(exporter, "_ANDROID_MINIMUM_API", 24)
    exporter._configure_project(staging, tmp_path / "sdk", "arm64-v8a")
    assert "minSdk 24" in (staging / "app/build.gradle").read_text(encoding="utf-8")
    wheel(tmp_path / "prefix/wheels", "android_26_arm64_v8a")
    with pytest.raises(ValueError, match="API 24"):
        exporter._find_android_numpy_wheel(request(), tmp_path / "prefix", "arm64-v8a", "3.13")
