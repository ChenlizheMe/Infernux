"""Persistent booleans and explicit triggers through real state transitions."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["runtime", "action", "spirit", "skeletal"])
@pytest.mark.parametrize("case", ["default", "bool", "trigger", "overwrite-bool", "overwrite-value", "unrelated", "reset"])
def test_parameter_lifetime_through_native_assets(tmp_path, kind, case):
    run_case(tmp_path, kind, case)


@pytest.mark.parametrize("case", ["hot-bool", "hot-trigger"])
def test_controller_hot_reload_preserves_parameter_kind(tmp_path, case):
    run_case(tmp_path, "spirit", case)


def run_case(tmp_path, kind, case):
    import infernux
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(tmp_path), kind, case],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARAMETER_LIFETIME_OK" in result.stdout


def exercise(project, kind, case):
    from infernux.core.animation_timeline import AnimationTimeline
    from infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition, AnimParameter, AnimCondition
    from infernux.core.timeline_fsm_runtime import TimelineFSMRuntime
    from infernux.core.assets import AssetManager
    from infernux.core.asset_ref import AnimStateMachineRef, TimelineFSMRef
    from infernux.components.timeline_action import TimelineAction
    from infernux.components.spirit_animator import SpiritAnimator
    from infernux.components.skeletal_animator import SkeletalAnimator
    from infernux.components.builtin.sprite_renderer import SpriteRenderer
    from infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.graph import TypeRef, ValueType
    from infernux.lib import SceneManager, RuntimeMode, LogLevel

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)

    def save(asset, name):
        path = project / "Assets" / name
        assert asset.save(str(path))
        handle = AssetManager.import_asset(str(path))
        assert handle and handle.guid
        return handle.guid

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        manager.set_active_scene(manager.create_scene("Parameters"))
        timeline = save(AnimationTimeline(duration=1.0), "Motion.animtimeline")
        powered = AnimParameter(name="powered", value_type=TypeRef(ValueType.BOOL), default=case in {"default", "reset"})
        spare = AnimParameter(name="spare", value_type=TypeRef(ValueType.BOOL), default=False)
        states = [AnimState(name=name, kind="timeline", timeline_guid=timeline, exit_time_normalized=0.0)
                  for name in ("A", "B", "C")]
        for state, target, parameter in ((states[0], "B", powered), (states[1], "C", spare)):
            state.transitions = [AnimTransition(target_state=target, conditions=[
                AnimCondition(parameter_id=parameter.stable_id, operator="==", threshold=1),
            ])]
        mode = {"spirit":"2d", "skeletal":"3d"}.get(kind, "timeline")
        fsm = AnimStateMachine(mode=mode, states=states, parameters=[powered, spare], default_state="A")
        guid = save(fsm, "Controller.timelinefsm" if mode == "timeline" else "Controller.animfsm")
        if kind == "runtime":
            actor = TimelineFSMRuntime()
            actor.set_fsm(fsm)
            assert actor.play()
        else:
            owner = manager.get_active_scene().create_game_object("Animator")
            if kind in {"spirit", "skeletal"}:
                owner.add_component(SpriteRenderer if kind == "spirit" else SkinnedMeshRenderer)
            actor = owner.add_component({"action":TimelineAction, "spirit":SpiritAnimator, "skeletal":SkeletalAnimator}[kind])
            actor.controller = TimelineFSMRef(guid=guid) if kind == "action" else AnimStateMachineRef(guid=guid)
            actor.awake()
            actor.start()
        assert actor.current_state == "A"
        if case in {"trigger", "overwrite-bool", "overwrite-value", "reset", "hot-trigger"}:
            actor.set_trigger("powered")
        if case in {"bool", "overwrite-bool", "unrelated", "hot-bool"}:
            actor.set_bool("powered", True)
        if case == "overwrite-value":
            (actor._runtime if kind == "action" else actor).set_parameter("powered", True)
        if case == "reset":
            if kind == "runtime":
                actor.set_fsm(fsm)
                assert actor.play()
            else:
                actor.reload_controller()
        if case == "unrelated":
            actor.set_trigger("spare")
        if case.startswith("hot-"):
            path = project / "Assets/Controller.animfsm"
            fsm.states[1].speed = 2.0
            assert fsm.save(str(path))
            assert actor._reload_controller_asset(str(path))
        assert actor.get_bool("powered")
        actor.update(0.1)
        assert actor.current_state == "B"
        assert actor.get_bool("powered") is (case not in {"trigger", "hot-trigger"}), (kind, case)
        if case == "unrelated":
            assert actor.get_bool("spare")
            actor.update(0.1)
            assert actor.current_state == "C"
            assert actor.get_bool("spare") is False
            assert actor.get_bool("powered") is True
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    print("PARAMETER_LIFETIME_OK")
