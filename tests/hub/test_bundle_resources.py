"""Standalone Hub data and onefile Installer data have different owners."""

from pathlib import Path
import sys

import pytest

import hub_resources
import hub_utils
import installer_gui


@pytest.mark.parametrize("compiled", [False, True])
def test_installer_reads_extracted_resources_and_payload(monkeypatch, tmp_path, compiled):
    module_dir = tmp_path / ("extracted" if compiled else "checkout/packaging")
    monkeypatch.setattr(installer_gui, "__file__", str(module_dir / "installer_gui.py"))
    # The onefile executable lives elsewhere; its directory must never win.
    monkeypatch.setattr(sys, "executable", str(tmp_path / "downloads/Installer.exe"))
    monkeypatch.delitem(installer_gui.__dict__, "__compiled__", raising=False)
    if compiled:
        monkeypatch.setitem(installer_gui.__dict__, "__compiled__", object())
    assert Path(installer_gui._resource_dir()) == module_dir / "resources"
    expected = module_dir / "payload" if compiled else module_dir.parent / "out/package/hub"
    assert Path(installer_gui._payload_dir()).resolve() == expected.resolve()


@pytest.mark.parametrize("marker", ["source", "module", "main"])
def test_hub_resources_follow_standalone_launch_context(monkeypatch, tmp_path, marker):
    source = tmp_path / "checkout/packaging"
    installed = tmp_path / "installed"
    monkeypatch.setattr(sys, "executable", str(installed / "Hub.exe"))
    for module in (hub_resources, hub_utils):
        monkeypatch.setattr(module, "__file__", str(source / f"{module.__name__}.py"))
        monkeypatch.delitem(module.__dict__, "__compiled__", raising=False)
        if marker == "module":
            monkeypatch.setitem(module.__dict__, "__compiled__", object())
    monkeypatch.delattr(sys.modules["__main__"], "__compiled__", raising=False)
    if marker == "main":
        monkeypatch.setattr(sys.modules["__main__"], "__compiled__", object(), raising=False)
    expected = source if marker == "source" else installed
    assert Path(hub_resources._resource_dir()) == expected / "resources"
    assert Path(hub_utils.get_bundle_dir()) == expected


def test_checkout_ships_the_hub_icon_and_fonts():
    for resource in (hub_resources.ICON_PATH, *hub_resources.FONT_PATHS):
        assert Path(resource).is_file(), resource
        assert Path(resource).stat().st_size > 0, resource
    # Fonts are redistributed under SIL OFL; their licenses travel with them.
    fonts = Path(hub_resources.FONTS_DIR)
    assert (fonts / "Space-Grotesk-OFL.txt").is_file()
    assert sum(Path(path).stat().st_size for path in hub_resources.FONT_PATHS) < 512 * 1024


def test_bundled_hub_typeface_loads_in_qt():
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    for resource in hub_resources.FONT_PATHS:
        font_id = QFontDatabase.addApplicationFont(resource)
        assert font_id >= 0, resource
        try:
            assert "Space Grotesk" in QFontDatabase.applicationFontFamilies(font_id)
        finally:
            QFontDatabase.removeApplicationFont(font_id)
    assert app is not None
