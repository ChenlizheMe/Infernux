from __future__ import annotations

import pytest

from infernux.components import InxComponent, serialized_field, GameObjectRef
from infernux.components.builtin import BoxCollider, Collider
from infernux.components.ref_wrappers import ComponentRef
from infernux.components.fields import get_raw_field_value
from infernux.ui import ui_event_entry
from infernux.ui.ui_event_system import _canvas_raycast


def test_event_parameter_reflection_failure_is_not_an_empty_signature(monkeypatch):
    class Target:
        def invoke(self, value: int) -> None:
            del value

    def reject_signature(_callback):
        raise ValueError("signature unavailable")

    monkeypatch.setattr(ui_event_entry.inspect, "signature", reject_signature)

    with pytest.raises(ValueError, match="signature unavailable"):
        ui_event_entry.get_method_parameter_specs(Target(), "invoke")


def test_event_parameter_type_hint_failure_is_not_treated_as_untyped(monkeypatch):
    class Target:
        def invoke(self, value: "MissingType") -> None:
            del value

    def reject_hints(*_args, **_kwargs):
        raise NameError("MissingType")

    monkeypatch.setattr(ui_event_entry, "get_type_hints", reject_hints)

    with pytest.raises(NameError, match="MissingType"):
        ui_event_entry.get_method_parameter_specs(Target(), "invoke")


def test_canvas_event_raycast_requires_current_canvas_contract():
    with pytest.raises(RuntimeError, match=r"does not provide raycast\(\)"):
        _canvas_raycast(object(), 10.0, 20.0)


def test_canvas_event_raycast_uses_authoritative_canvas_method_once():
    calls = []
    target = object()

    class Canvas:
        def raycast(self, x, y):
            calls.append((x, y))
            return target

    assert _canvas_raycast(Canvas(), 10.0, 20.0) is target
    assert calls == [(10.0, 20.0)]


class EventArgumentTarget(InxComponent):
    marker: int = serialized_field(default=0)


class EventArgumentReceiver(InxComponent):
    def accept_box(self, selected: BoxCollider):
        pass

    def accept_script(self, selected: EventArgumentTarget):
        pass

    def accept_any(self, selected: ComponentRef):
        pass

    def accept_base(self, selected: Collider):
        pass


def _bound_event(scene, method):
    from infernux.ui import UIButton

    owner = scene.create_game_object("Callback owners")
    receiver = owner.add_py_component(EventArgumentReceiver())
    if method == "accept_script":
        owner.add_py_component(EventArgumentTarget())
        selected = owner.add_py_component(EventArgumentTarget())
        selected.marker = 27
    else:
        owner.add_component("BoxCollider")
        selected = owner.add_component("BoxCollider")
        selected.is_trigger = True
    specs = ui_event_entry.get_method_parameter_specs(receiver, method)
    argument = ui_event_entry.UIEventArgument(
        kind="component", name="selected", component_type=specs[0].component_type,
        component=ComponentRef(selected),
    )
    entry = ui_event_entry.UIEventEntry(
        target=GameObjectRef(owner), component_name="EventArgumentReceiver",
        method_name=method, arguments=[argument],
    )
    button = scene.create_game_object("Button").add_py_component(UIButton())
    button.on_click_entries = [entry]
    return owner, receiver, selected, specs, entry, button


@pytest.mark.parametrize("method", ["accept_box", "accept_script", "accept_any"])
def test_inspector_normalization_and_scene_reopen_preserve_exact_event_argument(
    scene, tmp_path, monkeypatch, method,
):
    from infernux.lib import SceneManager
    from infernux.engine.scene_document_transaction import SceneDocumentTransaction
    from infernux.engine.ui import inspector_ui_components as inspector

    owner, receiver, selected, specs, entry, button = _bound_event(scene, method)
    before = entry._serialize()
    applied = []
    monkeypatch.setattr(inspector, "_apply_if_changed", lambda *args: applied.append(args))
    monkeypatch.setattr(inspector, "render_compact_section_header", lambda *_a, **_kw: False)
    for _ in range(3):
        entries, changed = inspector._render_onclick_arguments(
            None, button, button.on_click_entries, 0, entry, 100, receiver, method,
        )
        assert not changed and not applied
        assert entries[0]._serialize() == before
        normalized = ui_event_entry.normalize_event_arguments(entry.arguments, specs)
        reference = get_raw_field_value(normalized[0], "component")
        assert reference.component_id == selected.component_id
        assert reference.resolve().component_id == selected.component_id

    path = tmp_path / "event-argument.scene"
    assert scene.save_to_file(str(path))
    saved = path.read_bytes()
    reopened = SceneManager.instance().create_scene("Reopened event arguments")
    transaction = SceneDocumentTransaction(reopened, path=str(path), clear_registries=False)
    assert transaction.run_to_completion(), transaction.error
    loaded_button = reopened.find("Button").get_py_component(type(button))
    loaded_receiver = reopened.find("Callback owners").get_py_component(EventArgumentReceiver)
    loaded_entry = loaded_button.on_click_entries[0]
    arguments = ui_event_entry.materialize_event_arguments(loaded_entry, loaded_receiver)
    assert len(arguments) == 1
    loaded = arguments[0]
    assert loaded.game_object.id != owner.id
    assert loaded.component_id != selected.component_id
    if method == "accept_script":
        assert loaded.marker == 27
    else:
        assert loaded.is_trigger
    normalized = ui_event_entry.normalize_event_arguments(loaded_entry.arguments, specs)
    assert get_raw_field_value(normalized[0], "component").component_id == loaded.component_id
    assert reopened.save_to_file(str(path))
    assert path.read_bytes() == saved


@pytest.mark.parametrize("method", ["accept_box", "accept_script", "accept_any"])
def test_missing_event_argument_keeps_identity_and_never_selects_survivor(scene, method):
    owner, receiver, selected, specs, entry, button = _bound_event(scene, method)
    selected_id = selected.component_id
    if method == "accept_script":
        owner.remove_py_component(selected)
    else:
        owner.remove_component(selected)
    normalized = ui_event_entry.normalize_event_arguments(entry.arguments, specs)
    reference = get_raw_field_value(normalized[0], "component")
    assert reference.go_id == owner.id
    assert reference.component_id == selected_id
    assert reference.resolve() is None


def test_incompatible_event_signature_change_clears_selection_instead_of_retargeting(scene):
    owner, receiver, selected, specs, entry, button = _bound_event(scene, "accept_box")
    owner.add_component("SphereCollider")
    specs = [ui_event_entry.UIEventMethodParameter("selected", "component", "SphereCollider")]
    normalized = ui_event_entry.normalize_event_arguments(entry.arguments, specs)
    reference = get_raw_field_value(normalized[0], "component")
    assert reference.go_id == reference.component_id == 0
    assert reference.component_type == "SphereCollider"
    assert get_raw_field_value(entry.arguments[0], "component").component_id == selected.component_id


def test_compatible_event_signature_change_keeps_selected_derived_component(scene):
    owner, receiver, selected, specs, entry, button = _bound_event(scene, "accept_box")
    specs = ui_event_entry.get_method_parameter_specs(receiver, "accept_base")
    normalized = ui_event_entry.normalize_event_arguments(entry.arguments, specs)
    assert get_raw_field_value(normalized[0], "component").component_id == selected.component_id
