from dataclasses import dataclass

import pytest

from infernux.components.fields import FieldMetadata, FieldType
from infernux.components.value_codec import ValueCodecDescriptor, ValueCodecRegistry
from infernux.components.value_document import (
    TYPE_KEY,
    COMPONENT_REF,
    ASSET_REF,
)
from infernux.core.asset_ref import (
    GenericAssetRef,
    create_asset_ref,
)
from infernux.graph.ramp import AnimationCurve, Gradient, GradientKey, Keyframe


def _meta(field_type: FieldType, name: str = "field") -> FieldMetadata:
    return FieldMetadata(name=name, field_type=field_type, default=None)


def test_nested_encode_error_contains_full_path():
    codec = ValueCodecRegistry()

    with pytest.raises(TypeError, match=r"Root\.items\[1\]\.payload"):
        codec.encode({"items": [1, {"payload": object()}]}, "Root")


@pytest.mark.parametrize("value", [[True, 2]])
def test_vectors_reject_implicit_numeric_conversion(value):
    codec = ValueCodecRegistry()

    with pytest.raises(TypeError, match="Position: vector values must be numbers"):
        codec.decode(value, _meta(FieldType.VEC2), "Position")


@pytest.mark.parametrize(
    ("value", "field_type"),
    [
        (
            AnimationCurve(
                (Keyframe(0.0, 0.25), Keyframe(1.0, 1.5)),
                "repeat",
                "ping_pong",
            ),
            FieldType.ANIMATION_CURVE,
        ),
        (
            Gradient(
                (
                    GradientKey(0.0, (1.0, 0.0, 0.0, 1.0)),
                    GradientKey(1.0, (0.0, 0.0, 1.0, 0.5)),
                ),
                "fixed",
            ),
            FieldType.GRADIENT,
        ),
    ],
)
def test_ramp_value_codecs_round_trip(value, field_type):
    codec = ValueCodecRegistry()

    document = codec.encode(value, "Ramp")
    restored = codec.decode(document, _meta(field_type), "Ramp")

    assert restored == value


def test_custom_codecs_are_explicit_and_path_aware():
    @dataclass
    class Point:
        x: int

    custom_type = object()
    codec = ValueCodecRegistry()
    decoded = []

    def validate_point(value, meta, path, registry):
        if not isinstance(value, dict) or type(value.get("point")) is not int:
            raise TypeError(f"{path}: point requires an integer")

    def decode_point(value, meta, path, registry):
        decoded.append(path)
        return Point(value["point"])

    descriptor = ValueCodecDescriptor(
        name="point",
        can_encode=lambda value: isinstance(value, Point),
        can_decode=lambda field: field is custom_type,
        encode=lambda value, path, registry: {"point": value.x, "path": path},
        validate=validate_point,
        decode=decode_point,
    )
    codec.register_codec(descriptor)
    with pytest.raises(ValueError, match="already registered"):
        codec.register_codec(descriptor)
    assert codec.encode(Point(3), "Owner.location") == {"point": 3, "path": "Owner.location"}
    codec.validate({"point": 4}, custom_type, "Owner.location")
    assert decoded == []
    assert codec.decode({"point": 4}, custom_type, "Owner.location") == Point(4)
    assert decoded == ["Owner.location"] and codec.descriptors[0].identity == "point"


def test_custom_encoder_must_return_document_data():
    codec = ValueCodecRegistry()
    codec.register_codec(
        ValueCodecDescriptor(
            name="bad-output",
            can_encode=lambda value: value is Ellipsis,
            can_decode=lambda field: False,
            encode=lambda value, path, registry: object(),
            validate=lambda value, meta, path, registry: None,
            decode=lambda value, meta, path, registry: value,
        )
    )

    with pytest.raises(TypeError, match="codec produced non-document value"):
        codec.encode(Ellipsis, "Owner.payload")


@pytest.mark.parametrize(
    "document,error",
    [
        ({TYPE_KEY: COMPONENT_REF, "game_object_id": -1, "component_type": "Mover"}, "non-negative integer"),
        ({TYPE_KEY: ASSET_REF, "asset_type": "Material", "guid": 42, "path_hint": ""}, "values must be strings"),
    ],
)
def test_reference_documents_are_exact(document, error):
    codec = ValueCodecRegistry()

    with pytest.raises((TypeError, ValueError), match=error):
        codec.decode(document, FieldType.UNKNOWN, "Target")


def test_asset_reference_clipboard_ignores_extra_fields_and_keeps_builtins():
    from infernux.core.asset_reference_types import AssetReferenceCodec

    payload = AssetReferenceCodec.decode(
        'infernux.asset_reference {"asset_type":"Texture","builtin":"white",'
        '"guid":"","path_hint":"","legacy":"ignored"}'
    )

    assert payload == {
        "asset_type": "Texture",
        "builtin": "white",
        "guid": "",
        "path_hint": "",
    }


def test_structured_path_only_asset_reference_is_empty_and_cannot_select_union_type():
    from infernux.core.asset_reference_types import AssetReferenceCodec

    payload = AssetReferenceCodec.normalize(
        "Texture.Sampled",
        {"path_hint": "Assets/Textures/legacy.rendertexture"},
    )

    assert payload["guid"] == ""
    assert payload["path_hint"] == ""
    assert payload["asset_type"] == "Texture.Sampled"


def test_structured_asset_reference_keeps_explicit_concrete_type_without_path_inference():
    from infernux.core.asset_reference_types import AssetReferenceCodec

    payload = AssetReferenceCodec.normalize(
        "Texture.Sampled",
        {
            "asset_type": "RenderTexture",
            "guid": "deleted-render-texture",
            "path_hint": "Assets/Textures/not-the-truth.png",
        },
    )

    assert payload["asset_type"] == "RenderTexture"
    assert payload["guid"] == "deleted-render-texture"
    assert payload["path_hint"] == ""


def test_raw_project_relative_editor_path_can_resolve_to_guid(monkeypatch):
    from infernux.core.asset_reference_types import AssetReferenceCodec
    from infernux.core.assets import AssetManager
    import infernux.engine.project_context as project_context
    import os

    class Database:
        def get_guid_from_path(self, path):
            expected = os.path.join("C:\\Project", "Assets/Textures/current.png")
            return "texture-guid" if path == expected else ""

        def get_path_from_guid(self, _guid):
            return ""

    monkeypatch.setattr(AssetManager, "_asset_database", Database())
    monkeypatch.setattr(project_context, "get_project_root", lambda: "C:\\Project")

    payload = AssetReferenceCodec.normalize("Texture", "Assets/Textures/current.png")

    assert payload["guid"] == "texture-guid"


def test_registered_asset_without_specialized_wrapper_roundtrips():
    codec = ValueCodecRegistry()
    metadata = FieldMetadata(
        name="mesh",
        field_type=FieldType.ASSET,
        default=None,
        asset_type="Mesh",
    )
    reference = create_asset_ref(
        "Mesh",
        guid="mesh-guid",
        path_hint="Assets/Models/Probe.fbx",
    )

    document = codec.encode(reference, "Probe.mesh")
    decoded = codec.decode(document, metadata, "Probe.mesh")

    assert document == {
        TYPE_KEY: ASSET_REF,
        "asset_type": "Mesh",
        "guid": "mesh-guid",
    }
    assert isinstance(decoded, GenericAssetRef)
    assert decoded.asset_type == "Mesh"
    assert decoded.guid == "mesh-guid"
    incomplete = dict(document)
    del incomplete["asset_type"]
    with pytest.raises(ValueError, match="requires asset_type and guid"):
        codec.decode(incomplete, metadata, "Probe.mesh")
    with pytest.raises(ValueError, match="unknown asset reference type"):
        codec.decode(dict(document, asset_type="RetiredMesh"), metadata, "Probe.mesh")


def test_asset_list_codec_preserves_parent_element_contract():
    codec = ValueCodecRegistry()
    metadata = FieldMetadata(
        name="clips", field_type=FieldType.LIST, default=[], element_type=FieldType.ASSET, asset_type="AudioClip"
    )
    document = [
        {TYPE_KEY: ASSET_REF, "asset_type": "AudioClip", "guid": "audio-guid", "path_hint": "Assets/Audio/Probe.wav"}
    ]
    decoded = codec.decode(document, metadata, "Probe.clips")
    assert len(decoded) == 1 and decoded[0].guid == "audio-guid" and decoded[0].path_hint == ""
    document[0] = {TYPE_KEY: ASSET_REF, "asset_type": "Mesh", "guid": "mesh-guid"}
    with pytest.raises(TypeError, match="requires AudioClip"):
        codec.decode(document, metadata, "Probe.clips")


@pytest.mark.parametrize("enum_name,member_name", [("CameraProjection", "Orthographic")])
def test_native_enum_uses_the_shared_enum_document(enum_name, member_name):
    from infernux import lib
    from infernux.components.fields import FieldMetadata, FieldType
    from infernux.components.value_codec import VALUE_CODECS

    enum_type = getattr(lib, enum_name)
    value = enum_type.__members__[member_name]
    metadata = FieldMetadata("mode", FieldType.ENUM, value, enum_type=enum_type)
    document = VALUE_CODECS.encode(value)
    assert document == {"$type": "enum", "enum_type": enum_name, "name": member_name}
    assert VALUE_CODECS.decode(document, metadata) == value


def test_undeclared_native_enum_value_cannot_be_serialized():
    from infernux.lib import CameraProjection
    from infernux.components.value_codec import VALUE_CODECS

    with pytest.raises(ValueError, match="unknown.*enum member"):
        VALUE_CODECS.encode(CameraProjection(99))
