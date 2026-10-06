from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest

from infernux.plugins.platform_support import (
    ANDROID_SUPPORT_REQUIRED_MESSAGE,
    android_support_available,
    plugin_install_block_reason,
    require_plugin_support,
)
from infernux.plugins import InxPackage, PluginManager


def _host_id() -> str:
    if os.name == "nt":
        return "windows-x64"
    return "linux-x64" if platform.system().casefold() == "linux" else ""


def _support_root(root: Path) -> None:
    sdk = root / "sdk"
    (sdk / "platforms/android-36").mkdir(parents=True)
    (sdk / "build-tools/36.0.0").mkdir(parents=True)
    toolchain = sdk / "ndk/29.0.14206865/build/cmake/android.toolchain.cmake"
    toolchain.parent.mkdir(parents=True)
    toolchain.write_bytes(b"toolchain")
    adb = sdk / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
    adb.parent.mkdir(parents=True)
    adb.write_bytes(b"adb")
    java = root / "jdk/bin" / ("java.exe" if os.name == "nt" else "java")
    java.parent.mkdir(parents=True)
    java.write_bytes(b"java")
    gradle = root / "gradle/bin" / ("gradle.bat" if os.name == "nt" else "gradle")
    gradle.parent.mkdir(parents=True)
    gradle.write_bytes(b"gradle")
    for abi in ("arm64-v8a", "x86_64"):
        runtime = root / "python" / abi / "infernux-android-python.json"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("{}", encoding="utf-8")
    (root / "infernux-android-support.json").write_text(
        json.dumps(
            {
                "$schema": "infernux.android_support",
                "kind": "infernux-android-support",
                "version": "0.1.0",
                "host": _host_id(),
                "requirements": {},
                "paths": {
                    "sdk": "sdk",
                    "jdk": "jdk",
                    "gradle": "gradle",
                    "python": {
                        "arm64-v8a": "python/arm64-v8a",
                        "x86_64": "python/x86_64",
                    },
                },
                "total_bytes": 1,
            }
        ),
        encoding="utf-8",
    )


def test_android_plugin_is_blocked_before_hub_support_is_installed(tmp_path: Path) -> None:
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(tmp_path / "missing")}

    assert plugin_install_block_reason("vendor/other", environment) == ""
    assert (
        plugin_install_block_reason("infernux/platform-android", environment)
        == ANDROID_SUPPORT_REQUIRED_MESSAGE
    )
    with pytest.raises(RuntimeError, match="Infernux Hub"):
        require_plugin_support("infernux/platform-android", environment)


def test_editor_uses_hub_shared_storage_separately_from_preferences(tmp_path, monkeypatch):
    from infernux.plugins.cache import package_cache_root
    from infernux.plugins.platform_support import android_support_root

    shared = tmp_path / "Hub/InfernuxHubData/Shared"
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(shared))
    monkeypatch.setenv("INFERNUX_DATA_ROOT", str(tmp_path / "preferences"))
    monkeypatch.delenv("INFERNUX_ANDROID_SUPPORT_ROOT", raising=False)
    monkeypatch.delenv("INFERNUX_PACKAGE_CACHE_ROOT", raising=False)
    assert Path(package_cache_root()) == shared / "Library/Plugins"
    assert android_support_root() == shared / "PlatformKits/android/0.1.0" / _host_id()


def test_android_plugin_is_unblocked_only_by_a_complete_hub_support_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "android"
    root.mkdir()
    _support_root(root)
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(root)}

    assert android_support_available(environment)
    assert plugin_install_block_reason("infernux/platform-android", environment) == ""

    (root / "sdk/platform-tools" / ("adb.exe" if os.name == "nt" else "adb")).unlink()
    assert not android_support_available(environment)


def test_support_install_gate_rechecks_manifest_inside_a_panel_frame(tmp_path):
    from infernux.core.file_read_cache import read_model_frame

    root = tmp_path / "android"
    root.mkdir()
    _support_root(root)
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(root)}
    manifest = root / "infernux-android-support.json"
    original = manifest.read_bytes()
    with read_model_frame():
        assert plugin_install_block_reason("infernux/platform-android", environment) == ""
        manifest.unlink()
        with pytest.raises(RuntimeError, match="Infernux Hub"):
            require_plugin_support("infernux/platform-android", environment)
        manifest.write_bytes(original)
        require_plugin_support("infernux/platform-android", environment)
        manifest.write_text("{broken", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Infernux Hub"):
            require_plugin_support("infernux/platform-android", environment)


def test_support_layout_reuses_parsing_but_checks_every_required_file(tmp_path, monkeypatch):
    from infernux.plugins import platform_support as support
    from infernux.core.file_read_cache import read_model_frame

    root = tmp_path / "android"
    root.mkdir()
    _support_root(root)
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(root)}
    assert android_support_available(environment)

    def unexpected(*args, **kwargs):
        pytest.fail("Unchanged support layout was parsed or constructed again")

    monkeypatch.setattr(support, "_relative", unexpected)
    layout = support._support_layout(root)
    with read_model_frame():
        assert android_support_available(environment)
        # Installation decisions must not consume an earlier UI observation.
        for path in layout.files:
            contents = path.read_bytes()
            path.unlink()
            with pytest.raises(RuntimeError, match="Infernux Hub"):
                require_plugin_support("infernux/platform-android", environment)
            path.write_bytes(contents)
            require_plugin_support("infernux/platform-android", environment)
        for path in layout.directories:
            path.rmdir()
            assert not android_support_available(environment)
            path.mkdir()
            assert android_support_available(environment)


def test_support_environment_uses_one_validated_manifest_revision(tmp_path, monkeypatch):
    from infernux.plugins.platform_support import android_support_environment

    root = tmp_path / "android"
    root.mkdir()
    _support_root(root)
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(root)}
    manifest = root / "infernux-android-support.json"
    read_text = Path.read_text
    reads = []

    def read_and_replace(path, *args, **kwargs):
        value = read_text(path, *args, **kwargs)
        if path == manifest:
            reads.append(1)
            manifest.write_text('{"paths": {"sdk": "unvalidated"}}', encoding="utf-8")
        return value

    monkeypatch.setattr(Path, "read_text", read_and_replace)
    result = android_support_environment(environment)
    assert result["ANDROID_SDK_ROOT"] == str(root / "sdk")
    assert result["JAVA_HOME"] == str(root / "jdk")
    assert len(reads) == 1
    assert android_support_environment(environment) == {}


@pytest.mark.parametrize("document", [[], None, {"paths": []}, {"paths": {"python": []}}])
def test_malformed_support_manifest_never_produces_an_environment(tmp_path, document):
    from infernux.plugins.platform_support import android_support_environment

    root = tmp_path / "android"
    root.mkdir()
    _support_root(root)
    environment = {"INFERNUX_ANDROID_SUPPORT_ROOT": str(root)}
    assert android_support_available(environment)
    (root / "infernux-android-support.json").write_text(json.dumps(document), encoding="utf-8")
    assert not android_support_available(environment)
    assert android_support_environment(environment) == {}


def test_plugin_manager_cannot_bypass_the_hub_android_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "inx_package.json").write_text(
        json.dumps(
            {
                "reference": "infernux/platform-android",
                "name": "Android",
                "version": "0.1.0",
                "engine": "",
            }
        ),
        encoding="utf-8",
    )
    (source / "editor").mkdir()
    (source / "editor/plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    package = tmp_path / "android.inxpkg"
    InxPackage.export_source(str(source), str(package))
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv(
        "INFERNUX_ANDROID_SUPPORT_ROOT", str(tmp_path / "missing-support")
    )

    with pytest.raises(RuntimeError, match="Infernux Hub"):
        PluginManager(str(project)).install_package(
            str(package), install_dependencies=False
        )

    assert not (project / "Packages/infernux/platform-android").exists()


def test_plugin_panel_marks_android_import_unavailable_before_click(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infernux.engine.ui.plugin_panel import PluginPanel

    class _Registry:
        @staticmethod
        def installed_metadata():
            return []

        @staticmethod
        def available():
            return [
                {
                    "reference": "infernux/platform-android",
                    "name": "Android",
                    "source": {"type": "github", "official": True},
                }
            ]

    class _Manager:
        registry = _Registry()
        states = {}

        @staticmethod
        def cached_reference_path(_reference):
            return ""

    monkeypatch.setenv(
        "INFERNUX_ANDROID_SUPPORT_ROOT", str(tmp_path / "missing-support")
    )
    rows = PluginPanel()._visible_rows(_Manager())

    assert rows[0]["_install_block_reason"] == ANDROID_SUPPORT_REQUIRED_MESSAGE
