"""Playback contracts with real assets, attached animators and native transforms."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


CASES = [
    *(f"timeline-{kind}-{mode}" for kind in ("spirit", "skeletal")
      for mode in ("transition", "hold", "loop")),
    "sprite-shared", "sprite-distinct", "sprite-restart",
    "sprite-terminal-exact", "sprite-terminal-overshoot",
    "events-long", "events-partitioned", "events-boundary",
]


@pytest.mark.parametrize("case", CASES)
def test_animator_playback_boundary(tmp_path, case):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), case],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ANIMATOR_BOUNDARY_OK" in result.stdout


def exercise(project, case):
    from infernux.components import InxComponent
    from infernux.components.spirit_animator import SpiritAnimator
    from infernux.components.skeletal_animator import SkeletalAnimator
    from infernux.components.builtin.sprite_renderer import SpriteRenderer
    from infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
    from infernux.core.animation_timeline import AnimationTimeline, TimelineKeyframe, APPLY_ABSOLUTE
    from infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition
    from infernux.core.animation_clip import AnimationClip, AnimationFrame
    from infernux.core.animation_event import AnimationEvent
    from infernux.core.asset_ref import AnimStateMachineRef
    from infernux.core.assets import AssetManager
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, SceneManager

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)

    def save(name, suffix, asset):
        path = project / "Assets" / (name + suffix)
        assert asset.save(str(path))
        handle = AssetManager.import_asset(str(path))
        assert handle and handle.guid
        return handle.guid

    def attach(fsm, kind="spirit"):
        guid = save("Controller", ".animfsm", fsm)
        owner = SceneManager.instance().get_active_scene().create_game_object("Animator")
        owner.add_component(SpriteRenderer if kind == "spirit" else SkinnedMeshRenderer)
        actor = owner.add_component(SpiritAnimator if kind == "spirit" else SkeletalAnimator)
        actor.controller = AnimStateMachineRef(guid=guid)
        actor.awake()
        actor.start()
        assert actor.current_state == "A" and actor.is_playing
        return owner, actor

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        manager.set_active_scene(manager.create_scene("Playback"))
        if case.startswith("timeline-"):
            _, kind, mode = case.split("-")
            timeline = AnimationTimeline(duration=1.0, apply_mode=APPLY_ABSOLUTE, keyframes=[
                TimelineKeyframe(time=0.0, position=[0.0, 0.0, 0.0]),
                TimelineKeyframe(time=1.0, position=[8.0, 0.0, 0.0]),
            ])
            guid = save("Motion", ".animtimeline", timeline)
            first = AnimState(name="A", kind="timeline", timeline_guid=guid, loop=mode == "loop")
            second = AnimState(name="B", kind="timeline", timeline_guid=guid, loop=False)
            if mode == "transition":
                first.transitions = [AnimTransition(target_state="B")]
            owner, actor = attach(AnimStateMachine(
                mode="2d" if kind == "spirit" else "3d", default_state="A", states=[first, second],
            ), kind)
            actor.update(0.5)
            assert actor.normalized_time == pytest.approx(0.5)
            assert owner.transform.local_position.x == pytest.approx(4.0)
            actor.update(0.5)
            if mode == "transition":
                assert actor.current_state == "B" and actor.is_playing
                assert actor.normalized_time == 0.0
                assert owner.transform.local_position.x == 0.0
            else:
                assert actor.is_playing is (mode == "loop")
                assert actor.normalized_time == (0.0 if mode == "loop" else 1.0)
                assert owner.transform.local_position.x == (0.0 if mode == "loop" else 8.0)
            return

        from PIL import Image
        from infernux.core.asset_types import (
            TextureImportSettings, TextureType, SpriteFrame, write_texture_import_settings,
        )
        texture_path = project / "Assets" / "Frames.png"
        Image.new("RGBA", (3, 1), (255, 255, 255, 255)).save(texture_path)
        texture = AssetManager.import_asset(str(texture_path))
        ids = [f"{index + 1:032x}" for index in range(3)]
        assert write_texture_import_settings(str(texture_path), TextureImportSettings(
            texture_type=TextureType.SPRITE, sprite_frames=[
                SpriteFrame(stable_id=value, name=str(index), x=index, y=0, w=1, h=1)
                for index, value in enumerate(ids)
            ],
        ))

        def clip(name, frames, *, events=()):
            return save(name, ".animclip2d", AnimationClip(
                authoring_texture_guid=texture.guid,
                frames=[AnimationFrame(sprite_frame_id=value) for value in frames],
                fps=2.0, events=list(events),
            ))

        if case.startswith("events-"):
            calls = []

            class Receiver(InxComponent):
                def on_animation_event(self, function, string_arg, number_arg):
                    calls.append(function)

            # Deliberately unsorted authored order. Dispatch follows playback time.
            events = [AnimationEvent(0.75, "late"), AnimationEvent(0.0, "zero"),
                      AnimationEvent(1.0, "end"), AnimationEvent(0.25, "early")]
            guid = clip("Loop", [ids[0], ids[1]], events=events)
            owner, actor = attach(AnimStateMachine(default_state="A", states=[
                AnimState(name="A", clip_guid=guid, loop=True),
            ]))
            owner.add_component(Receiver)
            actor.update(0.5)
            assert calls == ["early"]
            calls.clear()
            deltas = ([2.0] if case == "events-long" else
                      [0.25] * 8 if case == "events-partitioned" else [0.5])
            for delta in deltas:
                actor.update(delta)
            expected = (["late", "end", "zero", "early"] * 2
                        if case != "events-boundary" else ["late", "end", "zero"])
            assert calls == expected, (case, calls, expected)
            assert actor.normalized_time == pytest.approx(0.0 if case == "events-boundary" else 0.5)
            return

        terminal = case.startswith("sprite-terminal")
        restart = case == "sprite-restart"
        first = clip("First", [ids[1], ids[0]] if restart else [ids[0], ids[2] if terminal else ids[0]])
        second = clip("Second", [ids[1], ids[2] if case == "sprite-distinct" else ids[0]])
        owner, actor = attach(AnimStateMachine(default_state="A", states=[
            AnimState(name="A", clip_guid=first, loop=not terminal, restart_same_clip=restart),
            AnimState(name="B", clip_guid=second),
        ]))
        renderer = owner.get_component(SpriteRenderer)
        if terminal:
            actor.update(1.0 if case.endswith("exact") else 1.5)
            assert not actor.is_playing and actor.normalized_time == 1.0
            assert renderer.frame_id == ids[2]
            actor.update(5.0)
            assert renderer._get_bound_native_component().frame_id == ids[2]
        else:
            actor.update(0.6)
            assert renderer.frame_id == ids[0]
            assert actor.play("A" if restart else "B")
            assert renderer.frame_id == ids[1]
            actor.update(0.6)
            expected = ids[2] if case == "sprite-distinct" else ids[0]
            assert renderer.frame_id == renderer._get_bound_native_component().frame_id == expected
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
    print("ANIMATOR_BOUNDARY_OK")
