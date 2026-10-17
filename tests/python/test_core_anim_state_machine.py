"""Public animation authoring workflows and invalid document boundaries."""

from __future__ import annotations


import pytest


from infernux.core.anim_state_machine import (
    AnimStateMachine,
    AnimState,
    AnimTransition,
    AnimParameter,
    AnimCondition,
    AnimConditionError,
    AnimClipEntry,
    evaluate_anim_condition,
)


from infernux.graph import TypeRef, ValueType


def test_condition_function_call_rejected():
    with pytest.raises(AnimConditionError):
        evaluate_anim_condition("foo()", {})


def test_fsm_rejects_condition_with_unknown_parameter_id():
    condition = AnimCondition(parameter_id="missing-parameter")
    document = AnimStateMachine(
        states=[
            AnimState(
                name="Idle",
                transitions=[AnimTransition(target_state="Run", conditions=[condition])],
            ),
            AnimState(name="Run"),
        ],
        default_state="Idle",
    ).to_dict()

    with pytest.raises(ValueError, match="declared parameters"):
        AnimStateMachine.from_dict(document)


def test_gameplay_condition_uses_flags_numeric_ranges_and_state():
    context = {"speed": 1.0, "grounded": True, "state": "run"}
    expression = 'grounded and 0 < speed <= 2 and state == "run"'
    assert evaluate_anim_condition(expression, context)
    context["grounded"] = False
    assert not evaluate_anim_condition(expression, context)
    assert evaluate_anim_condition("not grounded or speed < -1", context)
    assert not evaluate_anim_condition("missing", context)
    assert not evaluate_anim_condition("", context)


def test_fsm_authored_document_roundtrip_copy_and_file(tmp_path):
    fsm = AnimStateMachine(name="Player", mode="3d")
    assert not fsm.save()
    speed = AnimParameter(name="speed", value_type=TypeRef(ValueType.F32), default=1.0)
    fsm.parameters = [
        speed,
        AnimParameter(name="jump", value_type=TypeRef(ValueType.BOOL), default=True),
        AnimParameter(name="hp", value_type=TypeRef(ValueType.I32), default=7),
    ]
    idle = fsm.add_state("Idle")
    blend = fsm.add_state("Blend")
    blend.kind = "group"
    blend.clips = [AnimClipEntry("clip-a", 0.3), AnimClipEntry("clip-b", 0.7), AnimClipEntry("clip-c", 1.0)]
    sequence = fsm.add_state("Sequence")
    sequence.kind, sequence.timeline_guid = "timeline", "timeline-guid"
    idle.transitions.append(
        AnimTransition(
            target_state="Blend",
            duration=0.25,
            conditions=[AnimCondition(parameter_id=speed.stable_id, operator=">", threshold=0.5)],
        )
    )
    document = fsm.to_dict()
    restored = AnimStateMachine.from_dict(document)
    assert restored.to_dict() == document
    assert restored.default_state == "Idle"
    assert [item.weight for item in restored.get_state("Blend").clips] == pytest.approx([0.3, 0.7, 1.0])
    assert restored.get_state("Sequence").timeline_guid == "timeline-guid"
    clone = fsm.copy()
    clone.file_path = "a-different-local-path.animfsm"
    assert clone == fsm
    clone.states[0].name = "Changed"
    clone.parameters[0] = AnimParameter(name="speed", value_type=TypeRef(ValueType.F32), default=9.0)
    assert fsm.states[0].name == "Idle" and fsm.parameters[0].default == 1.0
    assert clone != fsm
    path = str(tmp_path / "Player.animfsm")
    assert fsm.save(path)
    assert AnimStateMachine.load(path).to_dict() == document


def test_removing_states_reassigns_default_and_cleans_transition_targets():
    fsm = AnimStateMachine()
    first = fsm.add_state("A")
    second = fsm.add_state("B")
    first.transitions.append(AnimTransition(target_state="B"))
    second.transitions.append(AnimTransition(target_state="A"))
    assert fsm.get_state("A") is first and fsm.get_state("missing") is None
    assert fsm.remove_state("A")
    assert fsm.default_state == "B" and second.transitions == []
    assert not fsm.remove_state("missing")
    assert fsm.remove_state("B")
    assert fsm.default_state == "" and fsm.state_count == 0


@pytest.mark.parametrize("contents", ["{"])
def test_fsm_load_rejects_unreadable_documents(tmp_path, contents):
    path = tmp_path / "Invalid.animfsm"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert AnimStateMachine.load(str(path)) is None
