"""Annotation-driven Unity-style serialized field declarations.

Covers:
- bare typed defaults (``health: int = 100``)
- Annotated[] markers (Range/Tooltip/Header/HideInInspector/NonSerialized/...)
- Optional[X] and ``X | None`` unwrapping
- enum annotations
- Color fields
- asset/component reference annotations
- annotation + serialized_field() composition
- CDS backing for annotated numeric fields
"""

from __future__ import annotations

import enum
from typing import Annotated, Optional

import pytest

from infernux.components import (
    InxComponent,
    serialized_field,
    FieldType,
    get_serialized_fields,
    Range,
    Tooltip,
    Group,
    Multiline,
    HideInInspector,
    NonSerialized,
    HDR,
    Color,
    VALUE_CODECS,
)
from infernux.components.ref_wrappers import GameObjectRef


# ── annotation unwrapping ────────────────────────────────────────────────


# ── build_field_from_annotation ──────────────────────────────────────────


# ── full component declarations ──────────────────────────────────────────


class _Phase(enum.Enum):
    IDLE = 0
    RUN = 1


class SerializedFieldShowcase(InxComponent):
    # Unity-style: annotation + plain default
    health: int = 100
    speed: Annotated[float, Range(0, 20), Tooltip("m/s")] = 5.0
    name_tag: Annotated[str, Multiline] = "hello"
    armor: float = 3  # int literal, float annotation → coerced
    phase: _Phase = _Phase.RUN
    tint: Color = Color(1, 0, 0)
    secret: Annotated[int, HideInInspector] = 9
    scratch: Annotated[int, NonSerialized] = 0
    opt_count: Optional[int] = 4
    # references
    mat: "Material" = None  # noqa: F821  (string annotation; resolves via registry)
    target: Optional[GameObjectRef] = None
    grouped_value: Annotated[float, Group("Advanced")] = serialized_field(default=1.5)


class TestSerializedFieldShowcase:
    def test_declared_fields_preserve_metadata_and_authored_document(self):
        fields = get_serialized_fields(SerializedFieldShowcase)
        assert fields["armor"].field_type == FieldType.FLOAT
        assert isinstance(fields["armor"].default, float)
        assert fields["health"].field_type == FieldType.INT and fields["health"].default == 100
        assert fields["speed"].range == (0, 20) and fields["speed"].tooltip == "m/s"
        assert fields["name_tag"].multiline
        assert fields["phase"].field_type == FieldType.ENUM
        assert fields["phase"].enum_type is _Phase and fields["phase"].default == _Phase.RUN
        assert fields["tint"].field_type == FieldType.COLOR
        assert fields["tint"].default == [1.0, 0.0, 0.0, 1.0] and not fields["tint"].hdr
        assert fields["secret"].hidden and "scratch" not in fields
        assert fields["opt_count"].field_type == FieldType.INT and fields["opt_count"].default == 4
        assert fields["mat"].field_type == FieldType.MATERIAL
        assert fields["grouped_value"].group == "Advanced"
        assert fields["grouped_value"].default == 1.5
        authored, restored = SerializedFieldShowcase(), SerializedFieldShowcase()
        try:
            assert authored.health == 100 and authored.speed == pytest.approx(5.0)
            authored.health, authored.speed, authored.secret = 55, 9.5, 1234
            document = authored._serialize_fields_document()
            assert document["secret"] == 1234 and "scratch" not in document
            restored._deserialize_fields_document(document)
            assert restored._serialize_fields_document() == document
            assert restored.health == 55 and restored.speed == pytest.approx(9.5)
            assert restored.phase is _Phase.RUN and restored.secret == 1234
        finally:
            authored._call_on_destroy()
            restored._call_on_destroy()

    def test_annotated_numeric_fields_share_scalar_and_batch_storage(self):
        import numpy as np
        from infernux.batch import batch_read, batch_write

        components = [SerializedFieldShowcase(), SerializedFieldShowcase()]
        try:
            np.testing.assert_array_equal(batch_read(components, "health"), [100, 100])
            components[0].health = 55
            np.testing.assert_array_equal(batch_read(components, "health"), [55, 100])
            batch_write(components, np.array([7.0, 9.5]), "speed")
            assert [component.speed for component in components] == pytest.approx([7.0, 9.5])
            np.testing.assert_allclose(batch_read(components, "speed"), [7.0, 9.5])
        finally:
            for component in components:
                component._call_on_destroy()


class TestStrictSerializationFailures:
    @pytest.mark.parametrize(
        "mutate, error",
        [
            (lambda data: data.__setitem__("__type_name__", "Other"), "type mismatch"),
            (lambda data: data.__setitem__("health", "bad"), "requires an integer"),
        ],
    )
    def test_invalid_field_document_is_rejected_without_partial_assignment(self, mutate, error):
        import json

        class StrictFields(InxComponent):
            health: int = 10
            title: str = "ready"

        source = StrictFields()
        document = json.loads(source._serialize_fields())
        mutate(document)

        target = StrictFields()
        target.health = 77
        target.title = "unchanged"
        with pytest.raises((TypeError, ValueError), match=error):
            target._deserialize_fields(json.dumps(document))
        assert target.health == 77
        assert target.title == "unchanged"

    def test_component_document_reopen_uses_current_fields_and_defaults(self, tmp_path):
        import json

        class CurrentFields(InxComponent):
            health: int = 10
            title: str = "ready"
            tags: list[str] = serialized_field(default=[])

        source = CurrentFields()
        source.health, source.title, source.tags = 42, "authored", ["saved"]
        document = json.loads(source._serialize_fields())
        document.pop("health")
        document.pop("tags")
        document["hp"] = 42
        document["removed_debug_value"] = True
        document["__unknown__"] = 1
        path = tmp_path / "component-fields.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        restored = CurrentFields()
        try:
            restored._deserialize_fields(path.read_text(encoding="utf-8"))
            assert restored.health == 10 and restored.title == "authored" and restored.tags == []
            current = restored._serialize_fields_document()
            assert "hp" not in current and "removed_debug_value" not in current and "__unknown__" not in current
        finally:
            source._call_on_destroy()
            restored._call_on_destroy()

    def test_all_fields_validate_before_any_decode(self, monkeypatch):
        class ValidateFirst(InxComponent):
            count: int = 1
            label: str = "ready"

        document = ValidateFirst()._serialize_fields_document()
        document["label"] = 42
        decode_calls = []
        original_decode = VALUE_CODECS.decode

        def track_decode(value, field_meta_or_type, path="value"):
            decode_calls.append(path)
            return original_decode(value, field_meta_or_type, path)

        monkeypatch.setattr(VALUE_CODECS, "decode", track_decode)

        with pytest.raises(TypeError, match="STRING field requires a string"):
            ValidateFirst()._deserialize_fields_document(document)

        assert decode_calls == []


class TestColorClass:
    def test_undo_snapshot_rgba(self):
        from infernux.engine.undo._base import _snapshot_value

        src = Color(0.2, 0.4, 0.6)
        snap = _snapshot_value(src)
        assert isinstance(snap, list)
        assert snap == [0.2, 0.4, 0.6, 1.0]
        assert snap is not src


class TestEffectColorFieldsAllowHdr:
    def test_mixin_color_annotation_keeps_hdr(self):
        from infernux.renderstack._serialized_field_mixin import (
            SerializedFieldCollectorMixin,
        )

        class BloomTint(SerializedFieldCollectorMixin):
            tint: Annotated[Color, HDR] = Color(1.0, 1.0, 1.0, 1.0)

        meta = get_serialized_fields(BloomTint)["tint"]
        assert meta.field_type == FieldType.COLOR
        assert meta.hdr is True


@pytest.mark.parametrize("storage", ["cds", "python", "data"])
def test_serialized_vector_reads_are_values_for_each_storage_owner(storage):
    from infernux import Vector2, Vector3, vec4f
    from infernux.components import SerializableObject

    owner_type = type("VectorValues_" + storage, (SerializableObject if storage == "data" else InxComponent,), {
        "__module__": __name__,
        "_uses_component_data_store": storage == "cds",
        "point2": serialized_field(default=Vector2(1, 2)),
        "point3": serialized_field(default=Vector3(1, 2, 3)),
        "point4": serialized_field(default=vec4f(1, 2, 3, 4)),
    })
    owner = owner_type()
    if storage != "data":
        assert (owner._cds_slot is not None) == (storage == "cds")
    for name, vector_type, expected in (
        ("point2", Vector2, (1, 2)),
        ("point3", Vector3, (1, 2, 3)),
        ("point4", vec4f, (1, 2, 3, 4)),
    ):
        value = getattr(owner, name)
        assert isinstance(value, vector_type)
        value.x = 99
        assert tuple(getattr(owner, name)) == expected
        setattr(owner, name, value)
        value.x = 100
        assert tuple(getattr(owner, name)) == (99, *expected[1:])
    owner.point3 += Vector3(1, 1, 1)
    assert tuple(owner.point3) == (100, 3, 4)
