"""Pipeline author parameters survive unavailable providers and typed reloads."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from infernux.components.fields import FieldType, get_raw_field_value, list_field, serialized_field
from infernux.components.serializable_object import SerializableObject
from infernux.components.value_codec import VALUE_CODECS
from infernux.core.asset_ref import TextureRef
from infernux.math import vector2 as Vector2, vector3 as Vector3, vector4 as Vector4
from infernux.renderstack import RenderPipeline, RenderStack
from infernux.renderstack.default_forward_pipeline import MSAASamples


class ParameterSettings(SerializableObject):
    gain: float = serialized_field(default=1.0)


class ParameterDocumentPipeline(RenderPipeline):
    name = "Parameter Document Test"
    offset: Vector3 = serialized_field(default=Vector3(0, 0, 0))
    extent: Vector2 = serialized_field(default=Vector2(1, 1))
    tint: Vector4 = serialized_field(default=Vector4(1, 1, 1, 1))
    samples: MSAASamples = serialized_field(default=MSAASamples.X4)
    weights: list = list_field(element_type=FieldType.FLOAT, default=[1.0])
    settings: ParameterSettings = serialized_field(default=ParameterSettings())
    texture: TextureRef = serialized_field(default=None, asset_type="Texture")
    quality: float = serialized_field(default=1.0, range=(0.0, 2.0))


@pytest.fixture
def typed_stack(monkeypatch):
    original = RenderStack.discover_pipelines
    monkeypatch.setattr(RenderStack, "discover_pipelines", staticmethod(
        lambda: {**original(), ParameterDocumentPipeline.name: ParameterDocumentPipeline}))
    stack = RenderStack()
    stack.set_pipeline(ParameterDocumentPipeline.name)
    return stack


@pytest.mark.parametrize("inactive", [False, True])
@pytest.mark.parametrize("field,factory", [
    ("offset", lambda: Vector3(4, 5, 6)), ("extent", lambda: Vector2(2, 3)),
    ("tint", lambda: Vector4(.1, .2, .3, .4)), ("samples", lambda: MSAASamples.X2),
    ("weights", lambda: [2.0, 4.0]), ("settings", lambda: ParameterSettings(gain=3.5)),
    ("texture", lambda: TextureRef(guid="a" * 32)),
])
def test_pipeline_parameters_roundtrip_through_shared_codec(typed_stack, inactive, field, factory):
    stack = typed_stack
    value = factory()
    if inactive:
        stack.set_pipeline(RenderStack.DEFAULT_PIPELINE_NAME)
    stack.set_pipeline_parameter(field, value, pipeline_class_name=ParameterDocumentPipeline.name)
    document = json.loads(json.dumps(stack._serialize_fields_document(), allow_nan=False))
    encoded = json.loads(document["pipeline_params_json"])[ParameterDocumentPipeline.name][field]
    assert encoded == VALUE_CODECS.encode(value)
    restored = RenderStack()
    restored._deserialize_fields_document(document)
    restored.set_pipeline(ParameterDocumentPipeline.name)
    actual = get_raw_field_value(restored.pipeline, field)
    assert type(actual) is type(value)
    assert VALUE_CODECS.encode(actual) == VALUE_CODECS.encode(value)


def test_rejected_pipeline_value_does_not_publish_live_or_saved_state(typed_stack):
    stack = typed_stack
    stack.set_pipeline_parameter("offset", Vector3(4, 5, 6))
    before = stack._serialize_fields_document()
    with pytest.raises((ValueError, TypeError)):
        stack.set_pipeline_parameter("offset", Vector3(float("nan"), 1, 2))
    assert tuple(stack.pipeline.offset) == (4, 5, 6)
    assert stack._serialize_fields_document() == before


@pytest.mark.parametrize("value,expected", [(-5.0, 0.0), (8.0, 2.0)])
@pytest.mark.parametrize("inactive", [False, True])
def test_parameter_range_is_applied_before_document_publication(typed_stack, value, expected, inactive):
    stack = typed_stack
    if inactive:
        stack.set_pipeline(RenderStack.DEFAULT_PIPELINE_NAME)
    stack.set_pipeline_parameter("quality", value, pipeline_class_name=ParameterDocumentPipeline.name)
    assert json.loads(stack.pipeline_params_json)[ParameterDocumentPipeline.name]["quality"] == expected
    stack.set_pipeline(ParameterDocumentPipeline.name)
    assert stack.pipeline.quality == expected


@pytest.mark.parametrize("event", ["catalog", "module"])
def test_provider_retirement_snapshots_current_live_parameters(typed_stack, monkeypatch, event):
    from types import SimpleNamespace

    stack = typed_stack
    stack._sync_pipeline_catalog()
    stack.pipeline.offset = Vector3(9, 8, 7)
    if event == "catalog":
        providers = stack.discover_pipelines()
        providers.pop(ParameterDocumentPipeline.name)
        monkeypatch.setattr(RenderStack, "discover_pipelines", staticmethod(lambda: providers))
        with pytest.warns(RuntimeWarning, match="was removed"):
            stack._sync_pipeline_catalog()
    else:
        name = "_retired_parameter_provider_test"
        monkeypatch.delitem(sys.modules, name, raising=False)
        stack._pipeline_module = SimpleNamespace(__name__=name)
        stack._on_pipeline_file_changed("retired.py")
    assert stack._pipeline is None
    assert json.loads(stack._serialize_fields_document()["pipeline_params_json"])[
        ParameterDocumentPipeline.name]["offset"] == [9, 8, 7]


def test_empty_replacement_document_clears_previous_parameters(typed_stack):
    stack = typed_stack
    stack.set_pipeline_parameter("offset", Vector3(4, 5, 6))
    document = stack._serialize_fields_document()
    document["pipeline_params_json"] = ""
    stack._deserialize_fields_document(document)
    assert tuple(stack.pipeline.offset) == (0, 0, 0)


def test_invalid_restoration_never_publishes_partial_pipeline(typed_stack):
    stack = typed_stack
    stack._pipeline_param_store = {ParameterDocumentPipeline.name: {"offset": [9, 8, 7], "extent": [1]}}
    with pytest.raises((ValueError, TypeError)):
        _ = stack.pipeline
    assert stack._pipeline is None


@pytest.mark.parametrize("field,initial,value", [
    ("offset", lambda: Vector3(0, 0, 0), [4, 5, 6]),
    ("texture", lambda: TextureRef(guid="a" * 32), {"guid": "b" * 32}),
])
def test_typed_parameter_editor_undo_redo_survives_projection_rebuild(typed_stack, scene, field, initial, value):
    from infernux.engine.interaction import EditorInteractionCore
    from infernux.engine.undo import UndoManager

    old_core, old_undo = EditorInteractionCore._instance, UndoManager._instance
    core = EditorInteractionCore()
    undo = UndoManager(core.action_journal)
    stack = scene.create_game_object("Typed authoring").add_component(RenderStack)
    stack.set_pipeline(ParameterDocumentPipeline.name)
    stack.set_pipeline_parameter(field, initial())
    before = VALUE_CODECS.encode(get_raw_field_value(stack.pipeline, field))
    try:
        core.render_stacks.set_pipeline_parameter(stack, field, value)
        after = VALUE_CODECS.encode(get_raw_field_value(stack.pipeline, field))
        assert after != before
        saved = stack._serialize_fields_document()
        stack._deserialize_fields_document(saved)
        undo.undo()
        assert VALUE_CODECS.encode(get_raw_field_value(stack.pipeline, field)) == before
        stack._pipeline = None
        undo.redo()
        assert VALUE_CODECS.encode(get_raw_field_value(stack.pipeline, field)) == after
        assert stack._serialize_fields_document() == saved
    finally:
        core.shutdown()
        EditorInteractionCore._instance, UndoManager._instance = old_core, old_undo


def test_typed_pipeline_survives_scene_disk_roundtrip(typed_stack, scene, tmp_path):
    from infernux.engine.scene_authoring import decode_scene_document
    from infernux.engine.component_restore import deserialize_scene_document_transactionally

    stack = scene.create_game_object("Typed scene settings").add_component(RenderStack)
    stack.set_pipeline(ParameterDocumentPipeline.name)
    stack.set_pipeline_parameter("offset", Vector3(4, 5, 6))
    stack.set_pipeline_parameter("samples", MSAASamples.X2)
    stack.set_pipeline_parameter("settings", ParameterSettings(gain=3.5))
    path = tmp_path / "Typed.scene"
    assert scene.save_to_file(str(path))
    document = decode_scene_document(json.loads(path.read_text(encoding="utf-8")))
    assert deserialize_scene_document_transactionally(scene, document)
    restored = scene.find("Typed scene settings").get_component(RenderStack)
    assert tuple(restored.pipeline.offset) == (4, 5, 6)
    assert restored.pipeline.samples is MSAASamples.X2
    assert restored.pipeline.settings.gain == 3.5


@pytest.mark.parametrize("selected", [False, True])
def test_missing_provider_scene_save_reopen_and_restore(tmp_path, selected):
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
        str(tmp_path), str(int(selected))], env=os.environ.copy(), capture_output=True,
        text=True, encoding="utf-8", timeout=60)
    (tmp_path / "provider-roundtrip.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROVIDER_AUTHORING_PRESERVED" in result.stdout


def exercise_missing_provider(project, selected):
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.scene_authoring import decode_scene_document
    from infernux.lib import LogLevel, RuntimeMode, SceneManager
    from infernux.renderstack.discovery import discover_pipelines, invalidate_discovery_cache
    from infernux.engine.component_restore import deserialize_scene_document_transactionally

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    source = project / "Assets" / "AuthorPipeline.py"
    source.write_text(
        "from infernux.renderstack import RenderPipeline\n"
        "from infernux.components.fields import serialized_field\n"
        "class AuthorPipeline(RenderPipeline):\n"
        "    name = 'Restored Author Pipeline'\n"
        "    exposure: float = serialized_field(default=1.0)\n", encoding="utf-8")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        invalidate_discovery_cache()
        name = "Restored Author Pipeline"
        assert name in discover_pipelines()
        scene = SceneManager.instance().get_active_scene()
        owner = scene.create_game_object("Rendering author state")
        stack = owner.add_component(RenderStack)
        stack.set_pipeline(name)
        stack.set_pipeline_parameter("exposure", 7.5)
        if not selected:
            stack.set_pipeline(RenderStack.DEFAULT_PIPELINE_NAME)
        expected = json.loads(stack._serialize_fields_document()["pipeline_params_json"])
        scene_path = project / "Assets" / "Authored.scene"
        assert scene.save_to_file(str(scene_path))
        source.rename(source.with_suffix(".unavailable"))
        invalidate_discovery_cache()
        assert name not in discover_pipelines()
        document = decode_scene_document(json.loads(scene_path.read_text(encoding="utf-8")))
        assert deserialize_scene_document_transactionally(scene, document)
        stack = scene.find("Rendering author state").get_component(RenderStack)
        assert json.loads(stack.pipeline_params_json) == expected
        if selected:
            with pytest.raises(RuntimeError, match="unavailable"):
                _ = stack.pipeline
        assert scene.save_to_file(str(scene_path))
        document = decode_scene_document(json.loads(scene_path.read_text(encoding="utf-8")))
        assert deserialize_scene_document_transactionally(scene, document)
        source.with_suffix(".unavailable").rename(source)
        invalidate_discovery_cache()
        assert name in discover_pipelines()
        stack = scene.find("Rendering author state").get_component(RenderStack)
        stack.set_pipeline(name)
        assert stack.pipeline.exposure == 7.5
        print("PROVIDER_AUTHORING_PRESERVED")
    finally:
        engine.exit()


@pytest.mark.parametrize("pipeline_name,field", [("__default__", "msaa_samples"), (ParameterDocumentPipeline.name, "samples")])
def test_previous_renderstack_enum_save_is_upgraded_on_load(typed_stack, pipeline_name, field):
    saved = typed_stack._serialize_fields_document()
    saved["pipeline_class_name"] = RenderStack.DEFAULT_PIPELINE_NAME if pipeline_name == "__default__" else pipeline_name
    saved["pipeline_params_json"] = json.dumps({pipeline_name: {field: {"__enum_name__": "X2"}}})
    restored = RenderStack()
    restored._deserialize_fields_document(saved)
    assert getattr(restored.pipeline, field) is MSAASamples.X2
    canonical = json.loads(restored._serialize_fields_document()["pipeline_params_json"])
    assert canonical[pipeline_name][field] == VALUE_CODECS.encode(MSAASamples.X2)


def test_previous_enum_save_with_unknown_member_is_rejected(typed_stack):
    saved = typed_stack._serialize_fields_document()
    saved["pipeline_params_json"] = json.dumps({ParameterDocumentPipeline.name: {"samples": {"__enum_name__": "X16"}}})
    with pytest.raises(ValueError, match="unknown legacy enum member"):
        RenderStack()._deserialize_fields_document(saved)


if __name__ == "__main__":
    exercise_missing_provider(Path(sys.argv[1]), bool(int(sys.argv[2])))
