"""Tests for infernux.components.fields — field types, metadata, descriptors.

Merges tests from test_component_annotation_defaults.py.
"""

import weakref

from infernux.components import InxComponent
from infernux.components.fields import (
    FieldType,
    FieldMetadata,
    SerializedFieldDescriptor,
    get_serialized_fields,
    serialized_field,
)


def test_null_serialized_reference_uses_annotation_and_retains_constraints():
    from infernux.lib import GameObject
    from infernux.components.ref_wrappers import GameObjectRef

    class Reporter(InxComponent):
        target: GameObject = serialized_field(
            default=None,
            header="References",
            tooltip="Select a renderer",
            required_component="MeshRenderer",
        )

    metadata = get_serialized_fields(Reporter)["target"]
    assert metadata.field_type == FieldType.GAME_OBJECT
    assert isinstance(metadata.default, GameObjectRef)
    assert metadata.header == "References"
    assert metadata.tooltip == "Select a renderer"
    assert metadata.required_component == "MeshRenderer"
    assert Reporter().target is None
    assert Reporter._field_schemas_["target"].value_type == "FieldType.GAME_OBJECT"


class TestSerializedFieldFactory:
    @staticmethod
    def _owner_type():
        descriptor = SerializedFieldDescriptor(
            FieldMetadata(name="value", field_type=FieldType.STRING, default="")
        )

        class Owner:
            value = descriptor

        return Owner

    def test_values_are_owned_by_the_instance(self):
        Owner = self._owner_type()
        first = Owner()
        second = Owner()

        first.value = "first"
        second.value = "second"
        assert first.value == "first"
        assert second.value == "second"

        del first.value
        assert first.value == ""
        assert second.value == "second"

    def test_collected_instance_does_not_affect_a_live_sibling(self):
        import gc

        Owner = self._owner_type()
        live = Owner()
        live.value = "kept"
        dying = Owner()
        dying.value = "temporary"
        reference = weakref.ref(dying)

        del dying
        gc.collect()

        assert reference() is None
        assert live.value == "kept"
