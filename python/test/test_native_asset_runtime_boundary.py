from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_asset_runtime_has_explicit_source_and_symbol_ownership() -> None:
    sources = (ROOT / "cmake" / "InfernuxSources.cmake").read_text(encoding="utf-8")
    targets = (ROOT / "cmake" / "InfernuxNativeTargets.cmake").read_text(encoding="utf-8")

    assert "set(INFERNUX_ASSET_RUNTIME_SOURCES" in sources
    assert "AssetDependencyGraph.cpp" in sources
    assert "AssetDatabase/AssetIndex.cpp" in sources
    assert "InxResource/InxResourceMeta.cpp" in sources
    assert "${INFERNUX_ASSET_RUNTIME_SOURCES}" in sources.split(
        "list(REMOVE_ITEM INFERNUX_RUNTIME_SOURCES", 1
    )[1]

    assert "add_library(InfernuxAssetRuntime SHARED ${INFERNUX_ASSET_RUNTIME_SOURCES})" in targets
    assert "target_link_libraries(InfernuxAssetRuntime PUBLIC InfernuxFoundation)" in targets
    assert "INFERNUX_ASSET_RUNTIME_EXPORTS=1" in targets

    legacy_exports = targets.split("set(_infernux_legacy_auto_export_dlls", 1)[1].split(")", 1)[0]
    assert "InfernuxAssetRuntime" not in legacy_exports


def test_asset_runtime_is_part_of_the_packaged_player_contract() -> None:
    abi = (ROOT / "python" / "Infernux" / "engine" / "python_abi.py").read_text(encoding="utf-8")

    assert '"InfernuxAssetRuntime.dll"' in abi
    assert '"libInfernuxAssetRuntime.so"' in abi
