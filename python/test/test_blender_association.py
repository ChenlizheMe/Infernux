from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.engine.model_import import _association as association, toolchain


def test_windows_default_association_is_used_without_managed_blender(monkeypatch, tmp_path):
    executable = tmp_path / "艺术 工具" / "blender.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(association.sys, "platform", "win32")
    monkeypatch.setattr(association, "_windows_executable", lambda: str(executable))
    monkeypatch.setattr(toolchain, "PreferencesStore", lambda: SimpleNamespace(get=lambda *args: ""))
    monkeypatch.setenv("INFERNUX_BLENDER_EXECUTABLE", str(tmp_path / "missing.exe"))
    assert toolchain.get_blender_executable() == str(executable.resolve())
    executable.rename(executable.with_name("asset-browser.exe"))
    monkeypatch.setattr(association, "_windows_executable", lambda: str(executable.with_name("asset-browser.exe")))
    assert association.find_associated_blender() == ""


def test_linux_association_uses_xdg_precedence_and_quoted_exec(monkeypatch, tmp_path):
    binary = tmp_path / "Blender 工具" / "blender"
    binary.parent.mkdir()
    binary.touch()
    binary.chmod(0o755)
    user, system = tmp_path / "user", tmp_path / "system"
    for root in (user, system):
        (root / "applications").mkdir(parents=True)
    desktop = user / "applications" / "blender.desktop"
    desktop.write_text(f'[Desktop Entry]\nType=Application\nExec="{binary.as_posix()}" %f\n', encoding="utf-8")
    (system / "applications" / "blender.desktop").write_text('[Desktop Entry]\nType=Application\nExec=/wrong/blender %f\n')
    monkeypatch.setenv("XDG_DATA_HOME", str(user))
    monkeypatch.setenv("XDG_DATA_DIRS", str(system))
    monkeypatch.setattr(association.shutil, "which", lambda command: command)
    calls = []
    def query(args, **kwargs):
        calls.append(args)
        assert not kwargs.get("shell")
        return SimpleNamespace(stdout="blender.desktop\n")
    monkeypatch.setattr(association.subprocess, "run", query)
    assert association._linux_executable() == binary.as_posix()
    assert calls == [["xdg-mime", "query", "default", "application/x-blender"]]
    desktop.write_text('[Desktop Entry]\nType=Application\nHidden=true\nExec=blender %f\n')
    assert association._linux_executable() == ""


@pytest.mark.parametrize("command", ['flatpak run org.blender.Blender %f', 'env MODE=test blender %f', 'blender --python injected.py %f'])
def test_launcher_arguments_are_never_executed_as_blender(monkeypatch, tmp_path, command):
    desktop = tmp_path / "blender.desktop"
    desktop.write_text(f'[Desktop Entry]\nType=Application\nExec={command}\n')
    assert association._desktop_executable(desktop) == ""


def test_clearing_override_reconfigures_database_to_default(monkeypatch, tmp_path):
    from infernux.core.assets import AssetManager
    state = {"blender_executable": "override"}
    store = SimpleNamespace(get=lambda key, default="": state.get(key, default),
                            set=lambda key, value: state.__setitem__(key, value))
    calls = []
    monkeypatch.setattr(toolchain, "PreferencesStore", lambda: store)
    monkeypatch.setattr(toolchain, "_default_blender_executable", lambda: "associated/blender")
    monkeypatch.setattr(AssetManager, "require_asset_database", lambda: SimpleNamespace(configure_blender_import=lambda *args: calls.append(args)))
    toolchain.set_blender_executable("")
    assert calls == [("associated/blender", toolchain.export_script())]
    assert state["blender_executable"] == ""
