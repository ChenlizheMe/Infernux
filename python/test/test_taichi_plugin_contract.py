from __future__ import annotations

import json
from pathlib import Path

from Infernux.plugins import InxPackage


ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "external/plugins/infernux_taichi"


def test_taichi_plugin_is_a_guid_packaged_bridge() -> None:
    manifest = json.loads((PLUGIN / "package/inx_package.json").read_text(encoding="utf-8"))
    assert manifest["reference"] == "infernux/taichi"
    assert manifest["engine"] == ">=0.4,<0.5"

    bridge = (PLUGIN / "package/editor/infernux_taichi/bridge.py").read_text(encoding="utf-8")
    assert "Infernux._compiler.taichi" in bridge
    assert "load_native" in bridge
    assert "sys.path" not in bridge
    assert "Path(" not in bridge


def test_taichi_plugin_exports_and_inspects(tmp_path: Path) -> None:
    destination = tmp_path / "infernux.taichi.inxpkg"
    preview = InxPackage.export_source(str(PLUGIN), str(destination), profile="release")
    inspected = InxPackage.inspect(str(destination))
    assert inspected.metadata["reference"] == "infernux/taichi"
    assert inspected.metadata["version"] == "0.1.0"
    assert preview.metadata["control_guid"] == inspected.metadata["control_guid"]
    assert any(path.endswith("infernux_taichi/lifecycle.py") for path in inspected.logical_entries)


def test_taichi_plugin_is_in_official_catalog() -> None:
    catalog = json.loads((ROOT / "external/plugins/plugins.json").read_text(encoding="utf-8"))
    entry = next(item for item in catalog["plugins"] if item["path"] == "infernux_taichi")
    assert entry["repository"] == "https://github.com/ChenlizheMe/infernux_taichi"
    assert entry["category"] == "compute"
    assert entry["default"] is False
