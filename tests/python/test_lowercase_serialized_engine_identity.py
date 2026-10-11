import pytest


@pytest.mark.parametrize("kind", ["entry", "argument", "effect"])
def test_legacy_engine_serializable_identity_reads_and_writes_canonical_package(kind):
    from infernux.components.serializable_object import SerializableObject, get_serializable_type_id
    from infernux.ui.ui_event_entry import UIEventArgument, UIEventEntry
    from infernux.renderstack.effect_slot import EffectSlot

    value = {"entry": UIEventEntry, "argument": UIEventArgument, "effect": EffectSlot}[kind]()
    document = value._serialize()
    current = get_serializable_type_id(value)
    assert current.startswith("infernux.")
    document["type_id"] = "Infernux." + current.removeprefix("infernux.")
    restored = SerializableObject._deserialize(document)
    assert type(restored) is type(value)
    assert restored._serialize() == value._serialize()


def test_legacy_button_binding_preserves_nested_arguments_and_target(scene):
    from infernux.ui import UIButton
    from infernux.ui.ui_event_entry import UIEventEntry, UIEventArgument

    target = scene.create_game_object("LegacyEventTarget")
    button_owner = scene.create_game_object("LegacyButton")
    button = UIButton()
    button_owner.add_py_component(button)
    argument = UIEventArgument()
    argument.kind, argument.name, argument.int_value = "int", "count", 7
    entry = UIEventEntry()
    entry.target = target
    entry.component_name, entry.method_name = "SceneActions", "load_level_one"
    entry.arguments = [argument]
    button.on_click_entries = [entry]
    canonical = button._serialize_fields_document()
    import copy
    legacy = copy.deepcopy(canonical)
    binding = legacy["on_click_entries"][0]
    binding["type_id"] = "Infernux.ui.ui_event_entry:UIEventEntry"
    binding["fields"]["arguments"][0]["type_id"] = "Infernux.ui.ui_event_entry:UIEventArgument"
    button.on_click_entries = []
    button._deserialize_fields_document(legacy)
    assert button._serialize_fields_document() == canonical
    restored, = button.on_click_entries
    assert restored.target is target
    assert restored.arguments[0].int_value == 7


def test_serializable_namespace_migration_does_not_guess_other_package_names():
    from infernux.components.serializable_object import get_serializable_class
    from infernux.ui.ui_event_entry import UIEventEntry

    assert get_serializable_class("INFERNUX.ui.ui_event_entry:UIEventEntry") is None
    assert get_serializable_class("Infernux.ui.ui_event_entry:MissingType") is None
    assert get_serializable_class("infernux.ui.ui_event_entry:UIEventEntry") is UIEventEntry
