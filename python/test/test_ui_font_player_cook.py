from __future__ import annotations

import json
from pathlib import Path

import pytest

from Infernux.application import Application
from Infernux.core.asset_ref import TextureRef, create_asset_ref
from Infernux.core.assets import AssetManager
from Infernux.engine.game_builder import GameBuilder
from Infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from Infernux.engine.project_context import (
    resolve_runtime_asset_guid,
    set_runtime_asset_resolver,
)
from Infernux.engine.runtime_artifact_catalog import build_catalog, runtime_artifact_id
from Infernux.ui import UIButton, UIText
from Infernux.ui.ui_render_dispatch import _extract_text_attrs


def _binding(guid: str, source_path: str) -> dict[str, object]:
    return {
        "source_guid": guid,
        "source_path": source_path,
        "source_fingerprint": {
            "size": 1,
            "modified_ns": 1,
            "content_hash": "0123456789abcdef",
        },
        "dependencies": [],
    }


def _runtime_record(
    guid: str,
    source_path: str,
    runtime_path: str,
    *,
    dependencies: tuple[str, ...] = (),
) -> dict[str, object]:
    artifact_id = runtime_artifact_id("Content.inxpkg", runtime_path)
    return {
        "guid": guid,
        "runtime_path": source_path,
        "resource_type": 3,
        "artifact_path": runtime_path,
        "runtime_artifact_ids": [artifact_id],
        "primary_runtime_artifact_id": artifact_id,
        "runtime_artifact_reason": "",
        "runtime_artifacts": [
            {
                "package": "Content.inxpkg",
                "runtime_path": runtime_path,
                "runtime_artifact_id": artifact_id,
            }
        ],
        "source_fingerprint": {
            "size": 1,
            "modified_ns": 1,
            "content_hash": "0123456789abcdef",
        },
        "metadata": {"metadata": {"guid": {"value": guid}}},
        "dependencies": [
            runtime_artifact_id("Content.inxpkg", path)
            for path in dependencies
        ],
    }


def test_player_ui_font_texture_and_chinese_text_use_only_cooked_guid_closure(
    tmp_path, monkeypatch,
):
    primary_guid = "11111111111111111111111111111111"
    fallback_guid = "22222222222222222222222222222222"
    texture_guid = "33333333333333333333333333333333"
    scene_guid = "44444444444444444444444444444444"
    primary_runtime = f"Library/Artifacts/Blob/{primary_guid}.ttf"
    fallback_runtime = f"Library/Artifacts/Blob/{fallback_guid}.otf"
    texture_runtime = f"Library/Artifacts/Texture/{texture_guid}.inxtex"
    scene_runtime = f"Library/Artifacts/Document/{scene_guid}.scene"
    source_paths = {
        primary_guid: "Assets/Fonts/Latin.ttf",
        fallback_guid: "Assets/Fonts/CJK.otf",
        texture_guid: "Assets/UI/Panel.png",
        scene_guid: "Assets/Scenes/Main.scene",
    }

    text = UIText()
    text.text = "Infernux 中文文本"
    text.font = create_asset_ref(
        "Font", guid=primary_guid, path_hint="C:/Windows/Fonts/host-only.ttf"
    )
    text.fallback_fonts = [
        create_asset_ref(
            "Font", guid=fallback_guid, path_hint="/usr/share/fonts/host-only.otf"
        )
    ]
    button = UIButton()
    button.label = "开始游戏"
    button.background_texture = TextureRef(
        guid=texture_guid, path_hint="D:/Author/Textures/stale.png"
    )
    scene_payload = json.dumps(
        {
            "objects": [
                {"components": [{"fields": text._serialize_fields_document()}]},
                {"components": [{"fields": button._serialize_fields_document()}]},
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")

    artifacts = {
        primary_runtime: b"project-primary-font",
        fallback_runtime: b"project-cjk-font",
        texture_runtime: b"INXTEXTURE-project-ui",
        scene_runtime: scene_payload,
    }
    package_entries = []
    for guid, source_path, runtime_path in (
        (primary_guid, source_paths[primary_guid], primary_runtime),
        (fallback_guid, source_paths[fallback_guid], fallback_runtime),
        (texture_guid, source_paths[texture_guid], texture_runtime),
        (scene_guid, source_paths[scene_guid], scene_runtime),
    ):
        payload = artifacts[runtime_path]
        entry = {
            "package": "Content.inxpkg",
            "runtime_path": runtime_path,
            "bytes": len(payload),
            "asset_binding": _binding(guid, source_path),
        }
        if runtime_path == scene_runtime:
            entry["payload"] = payload
        package_entries.append(entry)

    catalog_document = build_catalog(
        package_entries,
        player_host={"executable": "Game.exe"},
        package_records=[],
    )
    dependency_paths = (primary_runtime, fallback_runtime, texture_runtime)
    records_document = {
        "$schema": "infernux.runtime_asset_records",
        "entries": [
            _runtime_record(primary_guid, source_paths[primary_guid], primary_runtime),
            _runtime_record(fallback_guid, source_paths[fallback_guid], fallback_runtime),
            _runtime_record(texture_guid, source_paths[texture_guid], texture_runtime),
            _runtime_record(
                scene_guid,
                source_paths[scene_guid],
                scene_runtime,
                dependencies=dependency_paths,
            ),
        ],
    }
    for runtime_path, payload in artifacts.items():
        destination = tmp_path.joinpath(*runtime_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    catalog = PlayerRuntimeAssetCatalog.from_documents(
        str(tmp_path), catalog_document, records_document
    )
    scene_artifact = next(
        artifact
        for artifact in catalog_document["artifacts"]
        if artifact["runtime_path"] == scene_runtime
    )
    assert set(scene_artifact["dependencies"]) == {
        runtime_artifact_id("Content.inxpkg", path)
        for path in dependency_paths
    }
    assert scene_artifact["unresolved_dependencies"] == []

    cooked_paths = {
        guid: str(tmp_path.joinpath(*runtime_path.split("/")))
        for guid, runtime_path in (
            (primary_guid, primary_runtime),
            (fallback_guid, fallback_runtime),
            (texture_guid, texture_runtime),
            (scene_guid, scene_runtime),
        )
    }
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        AssetManager,
        "_asset_database",
        type(
            "ForbiddenPlayerFontDatabase",
            (),
            {
                "get_path_from_guid": lambda *_args: (_ for _ in ()).throw(
                    AssertionError("Player font resolution must not round-trip through paths")
                )
            },
        )(),
    )
    set_runtime_asset_resolver(catalog.resolve_guid)
    try:
        attrs = _extract_text_attrs(text)
        assert attrs["font_path"] == cooked_paths[primary_guid]
        assert attrs["fallback_font_paths"] == [cooked_paths[fallback_guid]]
        assert Path(attrs["font_path"]).read_bytes() == b"project-primary-font"
        assert Path(attrs["fallback_font_paths"][0]).read_bytes() == b"project-cjk-font"
        assert resolve_runtime_asset_guid(texture_guid) == cooked_paths[texture_guid]
    finally:
        set_runtime_asset_resolver(None)


def test_cook_dependency_scanner_ignores_path_only_font_fields():
    stale = {
        "$type": "asset_ref",
        "asset_type": "Font",
        "guid": "",
        "path_hint": "C:/Windows/Fonts/host-only.ttf",
    }
    current = {
        "$type": "asset_ref",
        "asset_type": "Font",
        "guid": "font-guid",
    }

    assert list(GameBuilder._asset_reference_values(stale)) == []
    assert list(GameBuilder._asset_reference_values(current)) == [
        ("font-guid", "")
    ]


def test_player_missing_font_guid_does_not_fall_back_to_host_or_default(
    monkeypatch,
):
    text = UIText()
    text.font = create_asset_ref(
        "Font",
        guid="missing-font-guid",
        path_hint="C:/Windows/Fonts/host-only.ttf",
    )
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    set_runtime_asset_resolver(lambda _guid: None)
    try:
        with pytest.raises(FileNotFoundError, match="missing-font-guid"):
            _extract_text_attrs(text)
    finally:
        set_runtime_asset_resolver(None)
