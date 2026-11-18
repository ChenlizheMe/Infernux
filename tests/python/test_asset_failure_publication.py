"""Current resource failures retire new requests and remain repairable in one session."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
import wave

import pytest

from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry, InxMaterial, ResourceType


@pytest.fixture
def assets(engine, monkeypatch):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", registry)
    monkeypatch.setattr(AssetManager, "_native_engine", classmethod(lambda _cls: engine))
    return database, registry


@pytest.mark.parametrize("suffix,loader,getter,valid", [
    (".mat", "load_material_by_guid", "get_material", None),
    (".obj", "load_mesh_by_guid", "get_mesh", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
    (".ppm", "load_texture_by_guid", "get_texture_asset", b"P3\n1 1\n255\n255 0 0\n"),
    (".physicmaterial", "load_physic_material_by_guid", "get_physic_material",
     b'{"friction":0.5,"bounciness":0.1,"friction_combine":0,"bounce_combine":0}'),
])
def test_failed_source_cannot_reacquire_retained_runtime_payload(assets, tmp_path, suffix, loader, getter, valid):
    database, registry = assets
    source = tmp_path / ("Current" + suffix)
    valid = valid or InxMaterial.create_default_lit().serialize().encode()
    source.write_bytes(valid)
    imported = database.import_asset(str(source))
    assert imported, imported.error
    guid = imported.guid
    load, get = getattr(registry, loader), getattr(registry, getter)
    retained = load(guid)
    assert retained is not None
    original_meta = Path(str(source) + ".meta").read_bytes()
    try:
        source.write_bytes(b"broken current source")
        failed = database.reimport_asset(str(source))
        assert not failed and failed.error
        assert source.read_bytes() == b"broken current source"
        assert Path(str(source) + ".meta").read_bytes() == original_meta
        assert database.get_guid_from_path(str(source)) == guid
        assert database.get_import_error(guid) == failed.error
        assert not registry.is_loaded(guid)
        for _ in range(3):
            assert get(guid) is None
            assert load(guid) is None
        # A retained instance is an old lease, never the current catalog payload.
        assert retained is not None
        source.write_bytes(valid)
        repaired = database.reimport_asset(str(source))
        assert repaired, repaired.error
        assert repaired.guid == guid and not database.get_import_error(guid)
        assert load(guid) is not None
    finally:
        database.delete_asset(str(source))


def _wav(path, frames=16):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(bytes(frames * 2))


@pytest.mark.parametrize("entrypoint", ["database", "asset_manager"])
@pytest.mark.parametrize("suffix,loader", [(".mat", "load_material_by_guid"),
                                           (".physicmaterial", "load_physic_material_by_guid"),
                                           (".wav", "load_audio_by_guid")])
def test_reimport_publishes_in_place_payload_once(assets, tmp_path, entrypoint, suffix, loader):
    database, registry = assets
    source = tmp_path / ("InPlace" + suffix)
    document = (json.loads(InxMaterial.create_default_lit().serialize()) if suffix == ".mat" else
                {"friction": 0.5, "bounciness": 0.1, "friction_combine": 0, "bounce_combine": 0})
    if suffix == ".wav":
        _wav(source)
    else:
        source.write_text(json.dumps(document), encoding="utf-8")
    imported = database.import_asset(str(source))
    assert imported, imported.error
    load = getattr(registry, loader)
    retained = load(imported.guid)
    assert retained is not None
    before = registry.get_asset_residency(imported.guid).runtime_version
    try:
        if suffix == ".wav":
            _wav(source, frames=32)
        else:
            if suffix == ".mat":
                document["properties"]["metallic"]["value"] = 0.75
            else:
                document["friction"] = 0.125
            source.write_text(json.dumps(document), encoding="utf-8")
        result = (database.reimport_asset(str(source)) if entrypoint == "database" else
                  AssetManager.reimport_asset(str(source), database=database))
        assert result, result.error
        assert load(imported.guid) is retained
        assert registry.get_asset_residency(imported.guid).runtime_version == before + 1
        if suffix == ".wav":
            assert retained.sample_count == 32
        elif suffix == ".mat":
            assert retained.get_float("metallic") == pytest.approx(0.75)
        else:
            assert retained.friction == pytest.approx(0.125)
    finally:
        database.delete_asset(str(source))


@pytest.mark.parametrize("suffix,begin,error_text", [
    (".wav", "begin_load_audio_by_guid", "Audio"),
    (".frag", "begin_load_shader_by_guid", "Shader"),
])
def test_worker_failure_has_resource_error_and_releases_ticket(assets, engine, tmp_path, suffix, begin, error_text):
    database, registry = assets
    source = tmp_path / ("WorkerFailure" + suffix)
    source.write_bytes(b"invalid encoded resource")
    imported = database.import_asset(str(source))
    assert imported, imported.error  # Metadata acceptance does not claim successful runtime decoding.
    try:
        ticket = getattr(registry, begin)(imported.guid)
        ticket.wait()
        assert ticket.complete
        with pytest.raises(RuntimeError, match=error_text):
            registry.try_commit_asset_load(ticket)
        assert not registry.is_loaded(imported.guid)
        assert error_text in database.get_import_error(imported.guid)
        # A normal frame can finish after the failure; no failed ticket owns work capacity.
        for _ in range(3):
            engine.tick(0.016)
        if suffix == ".wav":
            _wav(source)
            assert database.reimport_asset(str(source))
            replacement = registry.begin_load_audio_by_guid(imported.guid)
            replacement.wait()
            assert registry.try_commit_asset_load(replacement)
            assert registry.get_audio(imported.guid).is_loaded
            assert not database.get_import_error(imported.guid)
    finally:
        database.delete_asset(str(source))


@pytest.mark.parametrize("location,admission", [("project", "cold"), ("project", "worker"),
                                               ("external", "cold"), ("external", "worker")])
@pytest.mark.parametrize("suffix,resource_type,loader,valid", [
    (".ppm", ResourceType.Texture, "load_texture_by_guid", b"P3\n1 1\n255\n0 255 0\n"),
    (".obj", ResourceType.Mesh, "load_mesh_by_guid", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
])
def test_damaged_library_product_rebuilds_from_current_source(assets, tmp_path, suffix, resource_type, loader, valid,
                                                             location, admission):
    database, registry = assets
    root = Path(database.assets_root) / tmp_path.name if location == "project" else tmp_path
    root.mkdir(parents=True, exist_ok=True)
    source = root / ("Rebuild" + suffix)
    source.write_bytes(valid)
    imported = database.import_asset(str(source))
    assert imported, imported.error
    artifact = Path(database.get_runtime_artifact_path(imported.guid, resource_type))
    artifact.write_bytes(artifact.read_bytes()[:20])
    try:
        if admission == "worker":
            ticket = (registry.begin_load_texture_by_guid(imported.guid) if resource_type == ResourceType.Texture
                      else registry.begin_load_mesh_by_guid(imported.guid))
            ticket.wait()
            assert registry.try_commit_asset_load(ticket)
        loaded = getattr(registry, loader)(imported.guid)
        assert loaded is not None, database.get_import_error(imported.guid)
        assert len(artifact.read_bytes()) > 20
        assert source.read_bytes() == valid
        assert database.get_guid_from_path(str(source)) == imported.guid
        assert not database.get_import_error(imported.guid)
    finally:
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        if location == "project":
            root.rmdir()


@pytest.mark.parametrize("suffix", [
    ".mat", ".physicmaterial", ".rendertexture", ".effect", ".effectgroup", ".particlegraph", ".inxdata", ".prefab",
])
def test_structured_resource_failure_keeps_source_and_reports_path(assets, tmp_path, monkeypatch, suffix):
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors

    database, _registry = assets
    monkeypatch.setattr(AssetLoadErrors, "_instance", None)
    errors = AssetLoadErrors(database, None)
    source = tmp_path / ("BadDocument" + suffix)
    source.write_bytes(b"{ broken current document")
    try:
        result = AssetManager.import_asset(str(source), database=database)
        assert not result and result.error
        assert source.read_bytes() == b"{ broken current document"
        errors.poll()
        assert errors.error(str(source))
    finally:
        if database.contains_path(str(source)):
            database.delete_asset(str(source))


def test_font_failure_before_guid_is_visible_and_fixed_source_recovers(assets, engine, tmp_path, monkeypatch):
    from infernux.engine.ui.asset_load_errors import AssetLoadErrors

    database, _registry = assets
    monkeypatch.setattr(AssetLoadErrors, "_instance", None)
    errors = AssetLoadErrors(database, None)
    source = tmp_path / "Font.ttf"
    source.write_bytes(b"not a font")
    renderer = engine.get_screen_ui_renderer()
    assert renderer.measure_text("Font", 24, 0, str(source)) == (0., 0.)
    result = AssetManager.import_asset(str(source), database=database)
    assert not result and not result.guid
    assert "font" in errors.error(str(source)).lower()
    assert source.read_bytes() == b"not a font"
    assert not Path(str(source) + ".meta").exists()
    font = Path(__file__).resolve().parents[2] / "external/imgui_for_infernux/misc/fonts/Roboto-Medium.ttf"
    shutil.copyfile(font, source)
    try:
        repaired = AssetManager.import_asset(str(source), database=database)
        assert repaired, repaired.error
        errors.poll()
        assert not errors.error(str(source))
        assert renderer.measure_text("Font", 24, 0, str(source))[0] > 0
        source.write_bytes(font.read_bytes()[:64])
        failed = AssetManager.reimport_asset(str(source), database=database)
        assert not failed and database.get_import_error(repaired.guid)
        assert renderer.measure_text("Font", 24, 0, str(source)) == (0., 0.)
        shutil.copyfile(font, source)
        restored = AssetManager.reimport_asset(str(source), database=database)
        assert restored, restored.error
        assert renderer.measure_text("Font", 24, 0, str(source))[0] > 0
    finally:
        if database.contains_path(str(source)):
            database.delete_asset(str(source))


def test_missing_font_stops_only_the_affected_text(assets, engine, scene):
    from infernux.core.asset_ref import create_asset_ref
    from infernux.lib import ScreenUIList
    from infernux.ui import UIText
    from infernux.ui.ui_render_dispatch import dispatch, resolve_text_layout

    bad = scene.create_game_object("Missing Font text").add_py_component(UIText())
    bad.font = create_asset_ref("Font", guid="e" * 32)
    good = scene.create_game_object("Independent text").add_py_component(UIText())
    good.text = "Independent"
    renderer = engine.get_screen_ui_renderer()
    assert resolve_text_layout(bad, renderer.measure_text) is False
    resolve_text_layout(good, renderer.measure_text)
    renderer.begin_frame(64, 64)
    renderer.begin_command_packet()
    try:
        arguments = dict(renderer=renderer, ui_list=ScreenUIList.Overlay,
                         sx=0, sy=0, sw=64, sh=24, ref_w=64, ref_h=64,
                         scale_x=1, scale_y=1, text_scale=1, get_tex_id=lambda _texture: 0)
        assert dispatch(bad, "runtime", **arguments) is False
        assert dispatch(good, "runtime", **arguments) is True
        assert renderer.end_command_packet() is not None
    except BaseException:
        renderer.abort_command_packet()
        raise


def test_blender_process_failure_is_an_asset_failure(assets, tmp_path):
    database, _registry = assets
    source = tmp_path / "Broken.blend"
    source.write_bytes(b"invalid blend source")
    script = tmp_path / "export.py"
    script.write_text("", encoding="utf-8")
    # A real process exits with an unsupported Blender option; no Blender install is needed.
    database.configure_blender_import(sys.executable, str(script))
    try:
        result = database.import_asset(str(source))
        assert not result
        assert "Blender import failed" in result.error
        assert database.get_import_error(result.guid) == result.error
        assert source.read_bytes() == b"invalid blend source"
    finally:
        database.configure_blender_import("", "")
        if database.contains_path(str(source)):
            database.delete_asset(str(source))


def test_failed_import_has_no_preview_from_previous_generation(assets, engine, tmp_path):
    database, _registry = assets
    source = tmp_path / "Preview.ppm"
    source.write_bytes(b"P3\n1 1\n255\n255 0 0\n")
    imported = database.import_asset(str(source))
    assert imported, imported.error
    key = "failed-current-preview"
    deadline = time.monotonic() + 10
    try:
        while not engine.query_or_schedule_texture_preview(key, str(source), 1)[0]:
            assert time.monotonic() < deadline
            engine.tick(0.016)
        source.write_bytes(b"broken current image")
        assert not database.reimport_asset(str(source))
        for _ in range(3):
            assert engine.query_or_schedule_texture_preview(key, str(source), 2) == (0, 0, 0)
            assert engine.get_texture_preview_texture_id(key) == 0
            engine.tick(0.016)
    finally:
        engine.invalidate_texture_preview_task(key)
        database.delete_asset(str(source))


def test_dependency_repair_clears_only_runtime_failure(assets, tmp_path):
    database, registry = assets
    shader = tmp_path / "Dependency.vert"
    valid_shader = '#version 450\nShaderInfo { Name "ResourceDependencyRecovery" }\nvoid main() { gl_Position=vec4(0.0); }\n'
    shader.write_text(valid_shader, encoding="utf-8")
    shader_result = database.import_asset(str(shader))
    assert shader_result, shader_result.error
    shader_guid = shader_result.guid
    document = json.loads(InxMaterial.create_default_lit().serialize())
    document["shaders"]["vertex"] = {
        "guid": shader_guid,
        "shader_id": database.get_meta_by_guid(shader_guid).get_string("shader_id"),
        "path_hint": str(shader),
    }
    dependent = tmp_path / "RuntimeDependent.mat"
    broken_source = tmp_path / "SourceDependent.mat"
    material_guids = []
    for source in (dependent, broken_source):
        source.write_text(json.dumps(document), encoding="utf-8")
        result = database.import_asset(str(source))
        assert result, result.error
        material_guids.append(result.guid)
    try:
        shader.write_text(valid_shader.replace("vec4(0.0)", "invalid_glsl_token"), encoding="utf-8")
        assert database.reimport_asset(str(shader))
        ticket = registry.begin_load_shader_by_guid(shader_guid)
        ticket.wait()
        with pytest.raises(RuntimeError):
            registry.try_commit_asset_load(ticket)
        registry.invalidate_asset(material_guids[0])
        assert registry.load_material_by_guid(material_guids[0]) is None
        assert database.get_import_error(material_guids[0])
        broken_source.write_text("{ damaged authoring source", encoding="utf-8")
        rejected = database.reimport_asset(str(broken_source))
        assert not rejected
        source_error = database.get_import_error(material_guids[1])
        assert source_error and not database.clear_runtime_load_failure(material_guids[1])
        shader.write_text(valid_shader, encoding="utf-8")
        repaired = database.reimport_asset(str(shader))
        assert repaired, repaired.error
        assert not database.get_import_error(material_guids[0])
        assert registry.load_material_by_guid(material_guids[0]) is not None
        assert database.get_import_error(material_guids[1]) == source_error
        assert registry.load_material_by_guid(material_guids[1]) is None
    finally:
        for source in (dependent, broken_source, shader):
            database.delete_asset(str(source))
