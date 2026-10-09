"""Cook deterministic assets; retain encoded audio and custom bytes unchanged."""
import json
import struct
import wave
from pathlib import Path

import pytest

from infernux.application import Application
from infernux.core.asset_document import encode_asset_document, read_asset_document
from infernux.core.material import Material
from infernux.lib import InxMaterial, AudioClip, _Infernux as native


def wave_source(path):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(struct.pack("<8h", 1000, 3000, 2000, 4000, -1000, 1000, 0, 2000))


def test_cooked_document_roundtrip_and_bad_payload(tmp_path):
    document = {"name": "材质", "parameters": [True, None, 42, -2, .375], "nested": {"guid": "a" * 32}}
    payload = encode_asset_document(document)
    assert payload.startswith(b"INXDOCUMENT")
    assert encode_asset_document(document) == payload
    path = tmp_path / "asset.inxdoc"
    path.write_bytes(payload)
    assert read_asset_document(str(path)) == document
    for raw in (payload[:-1], b"{}", b"INXDOCUMENT"):
        with pytest.raises(RuntimeError):
            native._decode_asset_document(raw)


@pytest.mark.parametrize("kind", ["2d", "3d", "fsm", "timeline"])
def test_animation_loads_cooked_document(tmp_path, kind):
    from infernux.core.animation_clip import AnimationClip
    from infernux.core.animation_clip3d import AnimationClip3D
    from infernux.core.anim_state_machine import AnimStateMachine
    from infernux.core.animation_timeline import AnimationTimeline

    cls = {"2d": AnimationClip, "3d": AnimationClip3D, "fsm": AnimStateMachine,
           "timeline": AnimationTimeline}[kind]
    source = cls()
    path = tmp_path / "Authored.inxdoc"
    path.write_bytes(encode_asset_document(source.to_dict()))
    restored = cls.load(str(path))
    assert restored is not None
    document = restored.to_dict()
    expected = source.to_dict()
    # Public resource names derive from the filename in both Editor and Player.
    document.pop("name", None)
    expected.pop("name", None)
    assert document == expected
    before = path.read_bytes()
    assert not restored.save()
    assert path.read_bytes() == before


@pytest.mark.parametrize("player,play", [(True, False), (False, True)])
def test_material_runtime_changes_and_pending_flush_never_write(tmp_path, monkeypatch, player, play):
    path = tmp_path / "Material.mat"
    material = Material(InxMaterial("Runtime", "Unlit"))
    assert material.save(str(path))
    before = path.read_bytes()
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: player))
    monkeypatch.setattr(Material, "_suppress_auto_save", play)
    material._save_pending = True
    Material._pending_saves[id(material)] = material
    material.set_float("roughness", .375)
    Material.flush_all_pending()
    material.flush()
    assert material.get_float("roughness") == pytest.approx(.375)
    assert path.read_bytes() == before
    assert id(material) not in Material._pending_saves


def test_cooked_material_cannot_be_overwritten_by_native_save(tmp_path):
    material = InxMaterial("Cooked", "Unlit")
    path = tmp_path / "Material.inxdoc"
    payload = encode_asset_document(material.serialize_document())
    path.write_bytes(payload)
    material.file_path = str(path)
    material.set_float("roughness", .75)
    assert not material.save()
    assert not Material(material).save(str(path))
    assert path.read_bytes() == payload


def test_effect_runtime_changes_do_not_schedule_or_flush_source(tmp_path, monkeypatch):
    from infernux.renderstack.render_effect import RenderEffect
    from infernux.renderstack.render_effect_asset import RenderEffectAsset
    from infernux.core.assets import AssetManager

    path = tmp_path / "Bloom.effect"
    path.write_text("original", encoding="utf-8")
    effect = RenderEffect(RenderEffectAsset(feature_type="infernux.post.bloom", parameters={"intensity": .5}),
                          file_path=str(path))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(AssetManager, "schedule_asset_save", lambda *a, **k: pytest.fail("Player scheduled save"))
    monkeypatch.setattr(AssetManager, "flush_scheduled_saves", lambda *a, **k: pytest.fail("Player flushed save"))
    effect.set_float("intensity", .75)
    effect.flush()
    assert effect.get_float("intensity") == pytest.approx(.75)
    assert path.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("suffix", [".neuraltexture", ".json", ".bin"])
def test_unknown_payload_is_verbatim_through_cook_and_real_package(tmp_path, suffix):
    from infernux.engine.game_builder import GameBuilder
    from infernux.engine.player_package_native import read_entry, read_manifest
    from test_game_builder_asset_closure import _entry, _write_asset_index

    project = tmp_path / "Project"
    source = project / f"Assets/Custom{suffix}"
    source.parent.mkdir(parents=True)
    # Engine-looking fields must not cause a custom format to be rewritten.
    payload = b'{ "ref": {"$type":"asset_ref","path_hint":"C:/Private/Assets/raw.jpg"}, "weights": [ 1, 2 ] }\n'
    source.write_bytes(payload)
    entry = _entry("opaque", "Assets/" + source.name)
    scene = project / "Assets/Main.scene"
    scene.write_text(json.dumps({"identity_format": "guid-v1", "name": "Main", "isPlaying": False,
                                 "objects": []}), encoding="utf-8")
    settings = project / "ProjectSettings"
    settings.mkdir()
    (settings / "BuildSettings.json").write_text(json.dumps({"scene_guids": ["scene"]}), encoding="utf-8")
    _write_asset_index(project, [_entry("scene", "Assets/Main.scene"), entry])
    output = tmp_path / "Build"
    data = output / "Custom_Data"
    builder = GameBuilder(str(project), str(output), game_name="Custom")
    builder._copy_game_data(str(output))
    (output / "Data").rename(data)
    runtime = f"Library/Artifacts/Blob/opaque{suffix}"
    assert (data / runtime).read_bytes() == payload
    builder._pack_content_archive(str(output))
    package = data / "Content.inxpkg"
    assert read_entry(package, runtime) == payload
    assert source.read_bytes() == payload
    assert not any(item["path"].startswith("Assets/") for item in read_manifest(package)["files"])


def test_opaque_payload_does_not_invent_catalog_dependencies():
    from infernux.engine.runtime_artifact_catalog import build_catalog

    payload = b'{"ref":{"$type":"asset_ref","guid":"not-an-engine-reference"}}'
    catalog = build_catalog([{"package": "Content.inxpkg", "runtime_path": "Library/Artifacts/Blob/custom.json",
                              "bytes": len(payload), "payload": payload}], player_host={}, package_records=[])
    assert catalog["artifacts"][0]["dependencies"] == []
    assert catalog["artifacts"][0]["unresolved_dependencies"] == []


def test_existing_effect_artifact_loads_without_compiling_or_writing(tmp_path, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.engine import project_context
    from infernux.renderstack.render_effect import RenderEffect
    from infernux.renderstack.render_effect_asset import RenderEffectAsset, dump_render_effect_document
    from infernux.renderstack.render_effect_compiler import RenderEffectArtifactRegistry

    with project_context.using_project_root(str(tmp_path)):
        source = tmp_path / "Bloom.effect"
        source.write_text(dump_render_effect_document(RenderEffectAsset(
            feature_type="infernux.post.bloom", parameters={"intensity": .5})), encoding="utf-8")
        artifact, _ = RenderEffectArtifactRegistry.compile_and_publish(str(source), guid="cooked-effect")
        cooked = Path(artifact.artifact_path)
        before = cooked.read_bytes()
        source.unlink()
        monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
        monkeypatch.setattr(RenderEffectArtifactRegistry, "compile_and_publish",
                            lambda *a, **k: pytest.fail("Player attempted source compilation"))
        effect = AssetManager._load_by_type(str(cooked), RenderEffect)
        assert isinstance(effect, RenderEffect)
        assert effect.get_float("intensity") == .5
        effect.set_float("intensity", .75)
        effect.flush()
        assert cooked.read_bytes() == before


def test_real_import_pack_and_source_less_player(tmp_path):
    import os
    import subprocess
    import sys
    import infernux

    for action in ("cook", "player"):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), action],
            env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
            capture_output=True, text=True, encoding="utf-8", timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def exercise_cooked_player(root, action):
    import os
    import shutil
    from PIL import Image
    from infernux.core.asset_types import (
        read_texture_import_settings, write_texture_import_settings, TextureCompression,
        read_audio_import_settings, write_audio_import_settings, read_meta_file,
    )
    from infernux.core.assets import AssetManager
    from infernux.core.animation_clip import AnimationClip
    from infernux.engine.engine import Engine
    from infernux.engine.game_builder import GameBuilder
    from infernux.engine.player_package_native import extract_pack, read_entry, read_manifest
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.runtime_artifact_catalog import build_catalog, load_asset_index
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
    from infernux.engine import project_context
    from infernux.renderstack.render_effect import RenderEffect
    from infernux.renderstack.render_effect_asset import (
        RenderEffectAsset, RenderEffectGroupAsset, RenderEffectGroupEntry, EffectAssetReference,
        dump_render_effect_document,
    )
    from infernux.lib import AssetRegistry, LogLevel, ResourceType, RuntimeMode, SceneManager

    PreferencesStore()._path = str(root / "preferences.json")
    project = root / "Author"
    relocated = root / "NewMachine" / "Game_Data"
    if action == "player":
        assert not project.exists()
        os.environ["_INFERNUX_PLAYER_MODE"] = "1"
        expected = json.loads((root / "expected.json").read_text(encoding="utf-8"))
        engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
        try:
            engine.init_headless(str(relocated))
            Application._bind_engine(engine, "player")
            records = json.loads((relocated / "Library/RuntimeAssetRecords.json").read_text(encoding="utf-8"))
            records["entries"] = [entry for entry in records["entries"] if entry["guid"] in expected.values()]
            runtime = PlayerRuntimeAssetCatalog.from_documents(
                str(relocated), json.loads((root / "catalog.json").read_text(encoding="utf-8")), records)
            project_context.set_runtime_asset_resolver(runtime.resolve_guid)
            project_context.set_runtime_asset_extension_resolver(runtime.source_extension_for_guid)
            project_context.set_runtime_asset_query(runtime.query_asset_guids)
            registry = AssetRegistry.instance()
            texture = registry.load_texture_by_guid(expected["texture"])
            assert texture is not None
            assert (texture.pixel_width, texture.pixel_height) == (512, 512)
            assert "bc1" in texture.pixel_format.lower() and texture.mip_count == 10
            material = AssetManager.load_by_guid(expected["material"])
            assert isinstance(material, Material)
            material_path = Path(material._native.file_path)
            before = material_path.read_bytes()
            material.set_float("roughness", .875)
            material.flush()
            assert material_path.read_bytes() == before
            assert material.get_float("roughness") == .875
            scene = SceneManager.instance().get_active_scene()
            owner = scene.create_game_object("Original encoded audio")
            source = owner.add_component("AudioSource")._require_cpp_component()
            source.play_on_awake = False
            for key, guid in expected.items():
                if not key.startswith("audio:"):
                    continue
                _, extension, mono, streaming = key.split(":")
                audio_path = relocated / f"Library/Artifacts/Audio/{guid}.{extension}"
                assert not Path(str(audio_path) + ".meta").exists()
                clip = AudioClip()
                assert clip.load_from_file(str(audio_path))
                assert clip.channels == (1 if mono == "1" else 2), key
                assert clip.is_streaming == (streaming == "1"), key
                managed = AssetManager.load_by_guid(guid)
                assert managed and managed.channels == clip.channels and managed.is_streaming == clip.is_streaming, key
                source.set_track_clip_by_guid(0, guid)
                registered = source.get_track_clip(0)
                assert registered and registered.channels == clip.channels, key
                assert registered.is_streaming == clip.is_streaming, key
            scene.destroy_game_object(owner)
            animation = AssetManager.load_by_guid(expected["animation"])
            assert isinstance(animation, AnimationClip)
            from infernux.core.asset_ref import RenderEffectRef
            from infernux.renderstack.render_effect_compiler import expand_render_effect_reference

            effect = AssetManager.load_by_guid(expected["effect"])
            assert isinstance(effect, RenderEffect) and effect.get_float("intensity") == .5
            before = Path(effect.file_path).read_bytes()
            effect.set_float("intensity", .75)
            effect.flush()
            assert Path(effect.file_path).read_bytes() == before
            expanded = expand_render_effect_reference(RenderEffectRef(guid=expected["group"]))
            assert len(expanded) == 1 and expanded[0].get_float("intensity") == .25
        finally:
            engine.exit()
        return

    assets = project / "Assets"
    assets.mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        database = engine.get_asset_database()
        scene = assets / "Main.scene"
        scene.write_text(json.dumps(SceneManager.instance().get_active_scene().serialize_asset_document()), encoding="utf-8")
        imported_scene = database.import_asset(str(scene))
        assert imported_scene, imported_scene.error
        (project / "ProjectSettings/BuildSettings.json").write_text(
            json.dumps({"scene_guids": [imported_scene.guid]}), encoding="utf-8")

        source = assets / "Large.jpg"
        with Image.new("RGB", (8192, 8192), (30, 100, 180)) as image:
            image.save(source)
        imported = database.import_asset(str(source))
        assert imported, imported.error
        expected = {"texture": imported.guid}
        settings = read_texture_import_settings(str(source))
        settings.max_size = 512
        settings.compression = TextureCompression.BC1
        settings.generate_mipmaps = True
        assert write_texture_import_settings(str(source), settings)
        imported = AssetManager.reimport_asset(str(source), database=database)
        assert imported, imported.error
        metadata = read_meta_file(str(source))
        assert (metadata["artifact_width"], metadata["artifact_height"]) == (512, 512)
        assert "bc1" in metadata["artifact_format"].lower()
        artifact = Path(database.get_runtime_artifact_path(expected["texture"], ResourceType.Texture))
        cooked_texture = artifact.read_bytes()

        documentation_guids = set()
        for relative in ("plugin_pages/screenshot.png", "runtime/deep/plugin_pages/images/screenshot.png"):
            documentation = project / "Packages/vendor/docs" / relative
            documentation.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGBA", (3001, 3), (30, 100, 180, 128)) as image:
                image.save(documentation)
            result = database.import_asset(str(documentation))
            assert result, result.error
            documentation_guids.add(result.guid)
            metadata = read_meta_file(str(documentation))
            assert metadata["artifact_width"] == 3001 and metadata["artifact_format"] == "rgba8_srgb"
            assert Path(database.get_runtime_artifact_path(result.guid, ResourceType.Texture)).is_file()

        material = Material(InxMaterial("Surface", "Unlit"))
        material.set_float("roughness", .25)
        assert material.save(str(assets / "Surface.mat"))
        imported = database.import_asset(str(assets / "Surface.mat"))
        assert imported, imported.error
        expected["material"] = imported.guid
        audio_payloads = {}
        for extension in ("wav", "mp3", "ogg", "flac"):
            for mono in (False, True):
                for streaming in (False, True):
                    audio = assets / f"Sound-{mono}-{streaming}.{extension}"
                    if extension == "wav":
                        wave_source(audio)
                    else:
                        shutil.copyfile(Path(__file__).parents[1] / f"native/fixtures/audio/stream_probe.{extension}", audio)
                    imported = database.import_asset(str(audio))
                    assert imported, imported.error
                    expected[f"audio:{extension}:{int(mono)}:{int(streaming)}"] = imported.guid
                    audio_payloads[f"Library/Artifacts/Audio/{imported.guid}.{extension}"] = audio.read_bytes()
                    audio_settings = read_audio_import_settings(str(audio))
                    audio_settings.force_mono = mono
                    audio_settings.load_type = "streaming" if streaming else "decompress_on_load"
                    assert write_audio_import_settings(str(audio), audio_settings)
                    imported = AssetManager.reimport_asset(str(audio), database=database)
                    assert imported, imported.error
        animation = assets / "Motion.animclip2d"
        assert AnimationClip().save(str(animation))
        imported = database.import_asset(str(animation))
        assert imported, imported.error
        expected["animation"] = imported.guid
        effect = assets / "Bloom.effect"
        effect.write_text(dump_render_effect_document(RenderEffectAsset(
            feature_type="infernux.post.bloom", parameters={"intensity": .5})), encoding="utf-8")
        imported = database.import_asset(str(effect))
        assert imported, imported.error
        expected["effect"] = imported.guid
        group = assets / "Post.effectgroup"
        group.write_text(dump_render_effect_document(RenderEffectGroupAsset(entries=(
            RenderEffectGroupEntry("bloom", EffectAssetReference(guid=imported.guid), overrides={"intensity": .25}),
        ))), encoding="utf-8")
        imported = database.import_asset(str(group))
        assert imported, imported.error
        expected["group"] = imported.guid
        database.flush_derived_index()
        output = root / "Build"
        builder = GameBuilder(str(project), str(output), game_name="Game")
        builder.freeze_asset_index_entries(load_asset_index(project))
        builder._copy_game_data(str(output))
        builder._write_runtime_asset_records(str(output))
        assert documentation_guids.isdisjoint(builder._cooked_asset_entries)
        catalog_entries = []
        for relative, binding in builder._runtime_asset_identity_bindings.items():
            if binding["source_guid"] in expected.values():
                payload = (output / "Data" / relative).read_bytes()
                catalog_entries.append({"package": "Content.inxpkg", "runtime_path": relative,
                                        "bytes": len(payload), "payload": payload, "asset_binding": binding})
        catalog = build_catalog(catalog_entries, player_host={}, package_records=[])
        (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        data = output / "Game_Data"
        (output / "Data").rename(data)
        builder._pack_content_archive(str(output))
        package = data / "Content.inxpkg"
        manifest = read_manifest(package)
        entries = [entry["path"] for entry in manifest["files"]]
        for path, original_bytes in audio_payloads.items():
            assert read_entry(package, path) == original_bytes
            if not path.endswith(".wav"):
                assert next(entry for entry in manifest["files"] if entry["path"] == path)["codec"] == "store"
        assert not any("plugin_pages" in entry or any(guid in entry for guid in documentation_guids)
                       for entry in entries)
        records = json.loads(read_entry(package, "Library/RuntimeAssetRecords.json"))
        assert documentation_guids.isdisjoint(entry["guid"] for entry in records["entries"])
        textures = [entry for entry in entries if entry.endswith(".inxtex") and expected["texture"] in entry]
        assert len(textures) == 1, textures
        assert read_entry(package, textures[0]) == cooked_texture
        assert not any(Path(entry).suffix.lower() in {
            ".jpg", ".inxaudio", ".mat", ".animclip2d", ".effect", ".effectgroup", ".meta",
        } for entry in entries)
        with Image.open(source) as original_image:
            assert original_image.size == (8192, 8192)
        extract_pack(package, relocated)
        (root / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    finally:
        engine.exit()
    project.rename(root / "OriginalProjectNotAvailableToPlayer")


if __name__ == "__main__":
    import sys
    exercise_cooked_player(Path(sys.argv[1]), sys.argv[2])
