"""Tests for infernux.core.asset_types — enums, dataclasses, meta file helpers."""

from __future__ import annotations

import builtins
import json

import pytest

from infernux.core.asset_types import (
    AudioCompressionFormat,
    AudioImportSettings,
    FilterMode,
    MeshImportSettings,
    SpriteFrame,
    TextureImportSettings,
    TextureCompression,
    TextureCompressionQuality,
    TextureFormat,
    TextureType,
    WrapMode,
    _load_strict_meta_root,
)


def test_import_settings_preserve_a_concurrent_sidecar_update(tmp_path, monkeypatch):
    from infernux.core import asset_types

    asset = tmp_path / "texture.png"
    sidecar = tmp_path / "texture.png.meta"
    document = {
        "metadata": {
            "guid": {"type": "string", "value": "1" * 32},
            "file_path": {"type": "string", "value": "Assets/texture.png"},
            "srgb": {"type": "bool", "value": True},
        }
    }
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    read_root = asset_types._load_strict_meta_root
    external_bytes = b'{"metadata":{"guid":{"type":"string","value":"22222222222222222222222222222222"}}}\n'

    def concurrent_checkout(path):
        original = read_root(path)
        sidecar.write_bytes(external_bytes)
        return original

    monkeypatch.setattr(asset_types, "_load_strict_meta_root", concurrent_checkout)
    assert asset_types.write_meta_fields(str(asset), {"srgb": False}) is False
    assert sidecar.read_bytes() == external_bytes

    monkeypatch.setattr(asset_types, "_load_strict_meta_root", read_root)
    assert asset_types.write_meta_fields(str(asset), {"srgb": False}) is True
    updated = json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]
    assert updated["guid"]["value"] == "2" * 32
    assert updated["srgb"]["value"] is False


def test_queued_sidecar_edits_preserve_all_fields(tmp_path):
    from infernux.core import asset_types

    asset = tmp_path / "texture.png"
    sidecar = tmp_path / "texture.png.meta"
    sidecar.write_text('{"metadata":{}}', encoding="utf-8")
    edits = [asset_types.write_meta_fields_async(str(asset), {f"setting_{index}": index}) for index in range(16)]
    assert all(edit.result(timeout=5) is True for edit in edits)
    fields = json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]
    assert {name: field["value"] for name, field in fields.items()} == {
        f"setting_{index}": index for index in range(16)
    }


def test_missing_thread_executor_does_not_block_asset_model_imports(monkeypatch):
    from infernux.core import asset_types

    original_import = builtins.__import__

    def import_without_thread_executor(name, *args, **kwargs):
        if name == "concurrent.futures":
            raise ImportError("thread executor is unavailable")
        return original_import(name, *args, **kwargs)

    previous_pool = asset_types._io_pool
    asset_types._io_pool = None
    monkeypatch.setattr(builtins, "__import__", import_without_thread_executor)
    try:
        with pytest.raises(
            RuntimeError,
            match="Asynchronous asset writes are unavailable",
        ):
            asset_types._asset_io_pool()
    finally:
        asset_types._io_pool = previous_pool


def test_document_is_strict_and_transactional():
    from infernux.lib import ResourceMeta

    valid = {
        "metadata": {
            "guid": {"type": "string", "value": "strict-guid"},
            "resource_type": {
                "type": "enum infernux::ResourceType",
                "value": "DefaultText",
            },
        },
    }
    meta = ResourceMeta()
    meta.deserialize_document(valid)
    assert meta.serialize_document() == valid

    invalid_documents = []
    unknown_field = json.loads(json.dumps(valid))
    unknown_field["legacy_guid"] = "strict-guid"
    invalid_documents.append(unknown_field)
    wrong_value_type = json.loads(json.dumps(valid))
    wrong_value_type["metadata"]["guid"]["value"] = ["strict-guid"]
    invalid_documents.append(wrong_value_type)

    for invalid in invalid_documents:
        with pytest.raises(ValueError):
            meta.deserialize_document(invalid)
        assert meta.serialize_document() == valid


def test_texture_import_settings_roundtrip_copy_and_format_precedence():
    original = TextureImportSettings()
    assert (original.max_size, original.aniso_level, original.srgb) == (2048, -1, True)
    settings = TextureImportSettings(
        texture_type=TextureType.NORMAL_MAP,
        wrap_mode=WrapMode.CLAMP,
        filter_mode=FilterMode.TRILINEAR,
        generate_mipmaps=False,
        srgb=False,
        max_size=1024,
        aniso_level=4,
        compression=TextureCompression.BC5,
        compression_quality=TextureCompressionQuality.HIGH,
    )
    assert TextureImportSettings.from_dict(settings.to_dict()) == settings
    copied = settings.copy()
    copied.max_size = 512
    assert settings.max_size == 1024 and copied != settings
    assert TextureImportSettings().__eq__("other") is NotImplemented
    assert TextureImportSettings.from_dict(original.to_dict()).aniso_level == -1
    document = original.to_dict()
    document.update(aniso_level=1, texture_format="rgba4444", texture_compression="bc1")
    loaded = TextureImportSettings.from_dict(document)
    assert loaded.aniso_level == 1
    assert loaded.format == TextureFormat.RGBA4444
    assert loaded.compression == TextureCompression.NONE


def test_sprite_import_requires_unique_persisted_frame_identities(tmp_path):
    frame = SpriteFrame(name="idle", x=0, y=0, w=64, h=64)
    assert len(frame.stable_id) == 32
    assert SpriteFrame.from_dict(frame.to_dict()) == frame
    with pytest.raises(ValueError, match="at least one sprite frame"):
        TextureImportSettings(texture_type=TextureType.SPRITE).to_dict()
    settings = TextureImportSettings(
        texture_type=TextureType.SPRITE,
        sprite_frames=[
            frame,
            SpriteFrame(stable_id=frame.stable_id, name="duplicate", x=64, w=64, h=64),
        ],
    )
    with pytest.raises(ValueError, match="must be unique"):
        settings.to_dict()
    meta = tmp_path / "sprite.png.meta"
    meta.write_text(
        json.dumps(
            {
                "metadata": {
                    "sprite_frames": {"type": "string", "value": "[]"},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="sprite_frames must use json_array"):
        _load_strict_meta_root(str(meta))


def test_sprite_grid_covers_every_pixel_and_preserves_existing_frame_identity():
    from infernux.engine.ui.asset_details_renderer import _SpriteEditorState, _auto_slice

    view = _SpriteEditorState()
    view.tex_w, view.tex_h = 7, 5
    view.slice_cols, view.slice_rows = 3, 2
    settings = TextureImportSettings(texture_type=TextureType.SPRITE)
    _auto_slice(settings, view)
    pixels = [(x, y) for frame in settings.sprite_frames
              for x in range(frame.x, frame.x + frame.w)
              for y in range(frame.y, frame.y + frame.h)]
    assert len(pixels) == 35
    assert set(pixels) == {(x, y) for x in range(7) for y in range(5)}
    settings.sprite_frames[-1].name = "last frame"
    settings.sprite_frames[-1].pivot_x = 0.25
    before = settings.to_dict()
    _auto_slice(settings, view)
    assert settings.to_dict() == before


def test_audio_import_settings_roundtrip_copy_and_load_policy():
    defaults = AudioImportSettings()
    assert not defaults.force_mono and defaults.quality == 1
    assert defaults.compression_format == AudioCompressionFormat.PCM
    settings = AudioImportSettings(
        force_mono=True,
        quality=0.5,
        compression_format=AudioCompressionFormat.VORBIS,
        load_type="streaming",
    )
    assert AudioImportSettings.from_dict(settings.to_dict()) == settings == settings.copy()
    copied = settings.copy()
    copied.force_mono = False
    assert settings.force_mono and copied != settings
    document = settings.to_dict()
    del document["load_type"]
    assert AudioImportSettings.from_dict(document).load_type == "decompress_on_load"
    document["load_type"] = "automatic_guess"
    with pytest.raises(ValueError, match="load_type"):
        AudioImportSettings.from_dict(document)


def test_mesh_import_settings_roundtrip_copy_and_current_schema():
    settings = MeshImportSettings(
        normal_smoothing_angle=30,
        weld_vertices=False,
        sort_hierarchy_by_name=True,
        import_visibility=False,
        import_blend_shapes=False,
        mesh_compression="high",
        index_format="uint32",
        normal_mode="calculate",
        tangent_mode="source_only",
        normal_smoothing_source="source",
        flip_uvs=True,
        optimize_mesh=False,
    )
    assert MeshImportSettings.from_dict(settings.to_dict()) == settings
    copied = settings.copy()
    copied.optimize_mesh = True
    assert not settings.optimize_mesh and copied != settings
    document = settings.to_dict()
    document.update(guid="mesh-guid", bone_count=14, animation_names_csv="Armature|Action")
    assert MeshImportSettings.from_dict(document) == settings
    del document["weld_vertices"]
    with pytest.raises(ValueError, match="complete current field set"):
        MeshImportSettings.from_dict(document)


def test_public_meta_write_preserves_typed_values_and_rejects_partial_invalid_updates(tmp_path):
    from infernux.core.asset_types import read_meta_guid, write_meta_fields

    asset = tmp_path / "asset.png"
    meta = tmp_path / "asset.png.meta"
    meta.write_text('{"metadata":{}}', encoding="utf-8")
    assert write_meta_fields(
        str(asset),
        {
            "guid": "abc123",
            "srgb": True,
            "width": 42,
            "scale": 0.5,
            "label": "authored",
        },
    )
    fields = json.loads(meta.read_text(encoding="utf-8"))["metadata"]
    assert {key: value["type"] for key, value in fields.items()} == {
        "guid": "string",
        "srgb": "bool",
        "width": "int",
        "scale": "float",
        "label": "string",
    }
    assert read_meta_guid(str(asset)) == "abc123"
    saved = meta.read_bytes()
    assert not write_meta_fields(str(asset), {"label": "must not publish", "bad": None})
    assert meta.read_bytes() == saved
    meta.write_text('{"guid":"legacy-root-guid"}', encoding="utf-8")
    assert read_meta_guid(str(asset)) == ""
