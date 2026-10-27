"""Animation playback, authored assets and event publication contracts."""

from __future__ import annotations


import pytest

from infernux.core.anim_state_machine import (
    AnimCondition,
    AnimParameter,
    AnimState,
    AnimStateMachine,
    AnimTransition,
    AnimClipEntry,
)
from infernux.graph import TypeRef, ValueType
from infernux.components.skeletal_animator import SkeletalAnimator
from infernux.components.sprite_animator import SpriteAnimator


@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("frame_count,fps", [(3, 2.0), (29, 23.976), (61, 29.97)])
def test_uniform_sprite_frame_sampling_holds_or_wraps_at_the_clip_end(loop, frame_count, fps):
    import math

    from infernux.core.animation_clip import animation_frame_index

    duration = frame_count / fps
    assert animation_frame_index(-1.0, frame_count, fps, loop) == 0
    assert animation_frame_index(1.0 / fps, frame_count, fps, loop) == 1
    assert animation_frame_index(math.nextafter(duration, -math.inf), frame_count, fps, loop) == frame_count - 1
    assert animation_frame_index(duration, frame_count, fps, loop) == (0 if loop else frame_count - 1)
    assert animation_frame_index(duration + 2.5 / fps, frame_count, fps, loop) == (2 if loop else frame_count - 1)
    assert animation_frame_index(1.0, 0, fps, loop) is None


def _make_animator() -> SkeletalAnimator:
    """Bare SkeletalAnimator with a hand-built FSM (no renderer / no scene)."""
    anim = SkeletalAnimator()
    anim._parameters = {}
    anim._duration_cache = {}
    anim._clip_cache = {}
    return anim


def _condition(parameter: AnimParameter, operator: str = "==", threshold: float = 1.0):
    return AnimCondition(
        parameter_id=parameter.stable_id,
        operator=operator,
        threshold=threshold,
    )


class _FakeClip:
    def __init__(
        self,
        take_name="Walk",
        duration_hint=2.0,
        source_model_guid="",
        default_loop=True,
        apply_root_motion=False,
        bone_mask=(),
        curves=(),
    ):
        self.take_name = take_name
        self.duration_hint = duration_hint
        self.source_model_guid = source_model_guid
        self.source_model_path = ""
        self.default_loop = default_loop
        self.apply_root_motion = apply_root_motion
        self.bone_mask = list(bone_mask)
        self.curves = list(curves)


class _RendererBinding:
    def __init__(self, native):
        self.native = native

    def _get_bound_native_component(self):
        return self.native


class _RendererOwner:
    def __init__(self, renderer):
        self.renderer = renderer
        self.lookups = 0

    def get_component(self, _component_type):
        self.lookups += 1
        return self.renderer

    def get_children(self):
        return []


class _HierarchyOwner:
    def __init__(self, renderer=None, children=()):
        self.renderer = renderer
        self.children = list(children)

    def get_component(self, _component_type):
        return self.renderer

    def get_children(self):
        return list(self.children)


class _NativePoseRecorder:
    def __init__(self):
        self.calls = []
        self.pose_stacks = []

    def submit_animation_pose(self, *args):
        self.calls.append(args)

    def submit_pose_stack(self, layers):
        self.pose_stacks.append(layers)


def test_skeletal_restart_and_rejected_transition_preserve_playback(monkeypatch):
    from infernux.core.animation_event import AnimationEvent

    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator._fsm = AnimStateMachine(default_state="Walk", states=[
        AnimState(name="Walk"), AnimState(name="Missing"),
        AnimState(name="MissingTimeline", kind="timeline"),
    ])
    animator._clip_cache = {"Walk": _FakeClip()}
    calls = []

    class Receiver:
        def start_event(self):
            calls.append("start")

    owner.get_py_components = lambda: [Receiver()]
    animator._clip_cache["Walk"].events = [AnimationEvent(0.0, "start_event")]
    assert animator.play("Walk")
    animator.update(0.5)
    assert calls == ["start"]
    revision = animator._playback_revision
    assert animator.play("Walk")
    assert animator._elapsed == 0.5 and animator._playback_revision == revision
    for missing in ("Missing", "MissingTimeline"):
        assert not animator.play(missing)
        assert animator.current_state == "Walk" and animator.is_playing
        assert animator._elapsed == 0.5 and animator._playback_revision == revision
    assert animator.play("Walk", restart=True)
    assert animator._elapsed == 0.0 and animator._playback_revision == revision + 1
    animator.update(0.25)
    assert calls == ["start", "start"]
    assert native.calls[-1][1] == pytest.approx(0.25)
    layers = animator.get_pose_layers()
    assert layers[0]["time"] == pytest.approx(0.25) and layers[0]["take_name"] == "Walk"
    layers[0]["bone_mask"].append("Arm")
    assert animator.get_pose_layers()[0]["bone_mask"] == []


@pytest.mark.parametrize("phase", ["switch", "terminal", "restart"])
def test_skeletal_event_state_change_keeps_the_new_playback_cursor(monkeypatch, phase):
    from infernux.core.animation_clip3d import AnimationClip3D
    from infernux.core.animation_event import AnimationEvent

    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    calls = []

    class Receiver:
        def on_animation_event(self, function, string_arg, number_arg):
            calls.append(function)
            if len(calls) == 1:
                assert animator.play("A" if phase == "restart" else "B")

    receiver = Receiver()
    owner.get_py_components = lambda: [receiver]
    animator._fsm = AnimStateMachine(default_state="A", states=[
        AnimState(name="A", loop=phase != "terminal", restart_same_clip=phase == "restart"),
        AnimState(name="B"),
    ])
    animator._clip_cache = {
        "A": AnimationClip3D(take_name="First", duration_hint=1.0, events=[
            AnimationEvent(1.0 if phase == "terminal" else 0.25, "first"),
        ]),
        "B": AnimationClip3D(take_name="Second", duration_hint=1.0, events=[AnimationEvent(0.25, "second")]),
    }
    assert animator.play("A")
    animator.update(1.0 if phase == "terminal" else 0.5)
    assert animator.current_state == ("A" if phase == "restart" else "B") and animator.is_playing
    assert animator.normalized_time == 0.0
    animator.update(0.3)
    assert calls == ["first", "first" if phase == "restart" else "second"]
    assert native.calls[-1][1] == pytest.approx(0.3)


def test_imported_root_motion_updates_owner_transform_once_per_frame(monkeypatch):
    from infernux.lib import Vector3, quatf

    class Transform:
        def __init__(self):
            self.translations = []
            self.local_rotation = quatf.identity()

        def translate_local(self, value):
            self.translations.append(value)

    class Native(_NativePoseRecorder):
        def __init__(self):
            super().__init__()
            self.root_calls = []

        def get_root_motion_delta(self, *args):
            self.root_calls.append(args)
            return Vector3(3.0 if args[0] == "Run" else 1.0, 0.0, 0.0), quatf.euler(0.0, 90.0, 0.0)

        def get_animation_duration_seconds(self, *_args):
            return 2.0

    native = Native()
    renderer = _RendererBinding(native)
    transform = Transform()
    owner = _HierarchyOwner(renderer=renderer)
    owner.transform = transform
    animator = _make_animator()
    animator._fsm = AnimStateMachine(states=[AnimState(name="Walk", loop=True)], default_state="Walk")
    animator._current_state = animator._fsm.get_state("Walk")
    animator._current_clip = _FakeClip("Walk", duration_hint=2.0, source_model_guid="d" * 32, apply_root_motion=True)
    animator._elapsed = 0.5
    animator._playing = True
    animator._last_native_pose_key = None
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))

    animator.update(0.25)

    assert native.root_calls == [("Walk", 0.5, 0.75, True, "d" * 32)]
    assert transform.translations[0].x == pytest.approx(1.0)
    assert quatf.angle(transform.local_rotation, quatf.euler(0.0, 90.0, 0.0)) < 0.01
    assert len(native.calls) == 1
    animator._apply_root_motion([(animator._current_clip, .25, True), (_FakeClip("Run"), .75, True)], .5, .75)
    assert transform.translations[1].x == pytest.approx(2.5)
    assert [call[0] for call in native.root_calls] == ["Walk", "Walk", "Run"]


def test_imported_non_looping_clip_cannot_be_forced_to_wrap_by_fsm(monkeypatch):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    state = AnimState(name="Once", loop=True)
    animator._fsm = AnimStateMachine(states=[state, AnimState(name="Next")], default_state="Once")
    animator._current_state = animator._fsm.get_state("Once")
    animator._current_clip = _FakeClip("Once", duration_hint=1.0, default_loop=False)
    animator._clip_cache = {"Once": animator._current_clip, "Next": _FakeClip("Next")}
    animator._elapsed = 0.9
    animator._playing = True

    animator.update(0.2)

    assert animator._elapsed == pytest.approx(1.0)
    assert animator.is_playing is False
    assert native.calls[-1][6] is False

    state.transitions = [AnimTransition(target_state="Next", duration=0.5)]
    assert animator.play("Once")
    animator.update(1.0)
    assert animator.current_state == "Next" and animator.is_playing
    assert animator._blend_from_elapsed == 1.0
    animator.update(0.25)
    outgoing, incoming = native.pose_stacks[-1]
    assert outgoing["take_name"] == "Once" and outgoing["time"] == 1.0
    assert outgoing["loop"] is False
    assert incoming["take_name"] == "Next" and incoming["time"] == pytest.approx(0.25)
    assert outgoing["weight"] == incoming["weight"] == pytest.approx(0.5)


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_completed_state_can_transition_when_landing_arrives_later(monkeypatch, explicit_stop):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    grounded = AnimParameter(name="grounded", value_type=TypeRef(ValueType.BOOL), default=False)
    jump = AnimState(name="Jump", loop=False, transitions=[AnimTransition(
        target_state="Idle", duration=.2, conditions=[_condition(grounded)],
    )])
    animator._fsm = AnimStateMachine(
        states=[jump, AnimState(name="Idle")], default_state="Jump", parameters=[grounded],
    )
    animator._clip_cache = {"Jump": _FakeClip("Jump", duration_hint=1.0), "Idle": _FakeClip("Idle")}
    assert animator.play("Jump")
    animator.update(1.1)
    assert animator.current_state == "Jump" and not animator.is_playing
    animator.update(.2)
    assert animator._elapsed == 1.0
    if explicit_stop:
        animator.stop()
    animator.set_bool("grounded", True)
    animator.update(.1)
    assert animator.current_state == ("Jump" if explicit_stop else "Idle")
    assert animator.is_playing is not explicit_stop
    if not explicit_stop:
        assert animator._blend_from_elapsed == 1.0


def test_skeletal_animator_reacquires_renderer_after_scene_replacement(monkeypatch):
    animator = _make_animator()
    stale = _RendererBinding(None)
    live = _RendererBinding(object())
    owner = _RendererOwner(live)
    animator._skinned_renderers = [stale]
    monkeypatch.setattr(
        SkeletalAnimator,
        "game_object",
        property(lambda _self: owner),
    )

    assert animator._resolve_native_renderers() == [live.native]
    assert animator._skinned_renderers == [live]
    assert owner.lookups == 1


def test_skeletal_animator_submits_same_pose_to_all_descendant_renderers(monkeypatch):
    animator = _make_animator()
    body_native = _NativePoseRecorder()
    visor_native = _NativePoseRecorder()
    owner = _HierarchyOwner(
        children=[
            _HierarchyOwner(renderer=_RendererBinding(body_native)),
            _HierarchyOwner(renderer=_RendererBinding(visor_native)),
        ]
    )
    monkeypatch.setattr(
        SkeletalAnimator,
        "game_object",
        property(lambda _self: owner),
    )
    animator._fsm = AnimStateMachine(
        states=[AnimState(name="Walk", loop=True)],
        default_state="Walk",
    )
    animator._current_state = animator._fsm.get_state("Walk")
    animator._current_clip = _FakeClip("Walk", duration_hint=2.0)
    animator._elapsed = 0.5
    animator._playing = True
    animator._last_native_pose_key = None

    animator._sync_native_runtime_playback()

    assert len(body_native.calls) == 1
    assert visor_native.calls == body_native.calls


def test_skeletal_animator_submits_clip_source_separately_from_render_model(monkeypatch):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator._fsm = AnimStateMachine(
        states=[AnimState(name="Walk", loop=True)],
        default_state="Walk",
    )
    animator._current_state = animator._fsm.get_state("Walk")
    animator._current_clip = _FakeClip("Walk", source_model_guid="d" * 32)
    animator._elapsed = 0.5
    animator._playing = True
    animator._last_native_pose_key = None

    animator._sync_native_runtime_playback()

    assert len(native.calls) == 1
    assert native.calls[0][7] == "d" * 32


def test_imported_bone_mask_uses_native_pose_stack(monkeypatch):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator._fsm = AnimStateMachine(states=[AnimState(name="Upper", loop=True)], default_state="Upper")
    animator._current_state = animator._fsm.get_state("Upper")
    animator._current_clip = _FakeClip("Wave", bone_mask=["Spine", "Arm"])
    animator._elapsed = 0.5
    animator._playing = True
    animator._last_native_pose_key = None

    animator._sync_native_runtime_playback()

    assert native.calls == []
    assert native.pose_stacks == [
        [
            {
                "take_name": "Wave",
                "source_model_guid": "",
                "time": 0.5,
                "weight": 1.0,
                "loop": True,
                "bone_mask": ["Spine", "Arm"],
            }
        ]
    ]


def test_clip_group_pose_cache_weights_masks_and_loop_policy(monkeypatch):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    clip_b = _FakeClip("Run", bone_mask=["Leg"], default_loop=False)
    animator._current_clip = _FakeClip("Walk", bone_mask=["Spine"])
    animator._group_clips = [animator._current_clip, clip_b, _FakeClip("Recoil", bone_mask=["Arm"])]
    animator._group_weights = [1.0, 1.0, 0.2]
    animator._elapsed = 0.5
    animator._current_state = AnimState(name="Locomotion", kind="group")

    animator._sync_native_runtime_playback()
    assert [layer["loop"] for layer in native.pose_stacks[-1]] == [True, False, True]
    animator._current_clip.bone_mask = ["Arm"]
    animator._sync_native_runtime_playback()

    assert len(native.pose_stacks) == 2
    assert native.pose_stacks[-1][0]["bone_mask"] == ["Arm"]
    assert native.pose_stacks[-1][1]["bone_mask"] == ["Leg"]
    clip_b.default_loop = True
    animator._sync_native_runtime_playback()
    assert len(native.pose_stacks) == 3
    assert [layer["loop"] for layer in native.pose_stacks[-1]] == [True, True, True]
    animator.set_clip_weight(2, 0.0)
    assert [layer["weight"] for layer in native.pose_stacks[-1]] == [1.0, 1.0]
    animator._sync_native_runtime_playback()
    assert len(native.pose_stacks) == 4


def test_clip_group_events_independent_loop_clocks_and_whole_group_crossfade(monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.core.animation_clip3d import ImportedFloatCurve

    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    clips = {"legs": _FakeClip("Run", duration_hint=1.0, bone_mask=["Leg"]),
             "arms": _FakeClip("Reload", duration_hint=3.0, default_loop=False, bone_mask=["Arm"]),
             "head": _FakeClip("Look", duration_hint=2.0, bone_mask=["Head"])}
    clips["legs"].curves = [ImportedFloatCurve("FootPlant", ((0.0, 2.0), (1.0, 2.0)))]
    clips["arms"].curves = [ImportedFloatCurve("FootPlant", ((0.0, 4.0), (1.0, 4.0)))]
    monkeypatch.setattr(AssetManager, "load_by_guid", lambda guid, _kind: clips.get(guid))
    animator._fsm = AnimStateMachine(default_state="Group", states=[
        AnimState(name="Group", kind="group", clips=[AnimClipEntry(guid) for guid in clips]),
        AnimState(name="Idle"), AnimState(name="Incomplete", kind="group", clips=[AnimClipEntry("missing")]),
    ])
    animator._clip_cache["Idle"] = _FakeClip("Idle")
    received = []
    monkeypatch.setattr(animator, "_dispatch_clip_events", lambda clip, *args, **kwargs: received.append((clip.take_name, args)))
    assert animator.play("Group")
    animator.update(2.5)
    assert animator._elapsed == 2.5 and animator.normalized_time == pytest.approx(2.5 / 3)
    assert len(received) == 3
    assert animator.get_curve_value("FootPlant") == pytest.approx(3.0)
    assert received[0][1] == (None, 0.5, True)
    assert received[1][1] == (None, 2.5 / 3, False)
    assert not animator.play("Incomplete")
    assert animator.current_state == "Group" and animator._elapsed == 2.5
    assert animator.cross_fade("Idle", .2)
    animator.update(.1)
    layers = native.pose_stacks[-1]
    assert [layer["take_name"] for layer in layers] == ["Run", "Reload", "Look", "Idle"]
    assert [layer["weight"] for layer in layers] == [1.0] * 4
    assert [layer["group_weight"] for layer in layers] == pytest.approx([.5] * 4)
    assert len({layer["blend_group"] for layer in layers[:3]}) == 1
    assert layers[-1]["blend_group"] != layers[0]["blend_group"]
    assert [layer["time"] for layer in layers[:3]] == pytest.approx([2.6] * 3)
    animator.update(.2)
    assert native.calls[-1][0] == "Idle"


def test_imported_float_curve_drives_animator_float_parameter():
    from infernux.core.animation_clip3d import ImportedFloatCurve

    animator = _make_animator()
    clip = _FakeClip(curves=[ImportedFloatCurve("FootPlant", ((0.0, 0.0), (1.0, 2.0)))])
    animator._sample_imported_curves(clip, 0.25)

    assert animator.get_curve_value("FootPlant") == pytest.approx(0.5)
    assert animator.get_float("FootPlant") == pytest.approx(0.5)


def test_imported_events_dispatch_every_occurrence_across_multiple_wraps(monkeypatch):
    from infernux.core.animation_event import AnimationEvent

    calls = []

    class Receiver:
        def step(self, text, number):
            calls.append((text, number))

    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    owner.get_py_components = lambda: [Receiver()]
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator = _make_animator()
    animator._fsm = AnimStateMachine(states=[AnimState(name="Loop", loop=True)], default_state="Loop")
    animator._current_state = animator._fsm.get_state("Loop")
    animator._current_clip = _FakeClip("Loop", duration_hint=1.0)
    animator._current_clip.events = [AnimationEvent(0.25, "step", "L", 1.0)]
    animator._elapsed = 0.9
    animator._prev_event_norm = 0.9
    animator._playing = True

    animator.update(2.2)

    assert calls == [("L", 1.0), ("L", 1.0)]


def test_native_pose_submission_failure_propagates(monkeypatch):
    animator = _make_animator()

    class Native:
        @staticmethod
        def submit_animation_pose(*_args):
            raise RuntimeError("native pose rejected")

    owner = _HierarchyOwner(renderer=_RendererBinding(Native()))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator._fsm = AnimStateMachine(states=[AnimState(name="Walk")], default_state="Walk")
    animator._current_state = animator._fsm.get_state("Walk")
    animator._current_clip = _FakeClip("Walk")

    with pytest.raises(RuntimeError, match="native pose rejected"):
        animator._sync_native_runtime_playback()


class TestTriggerConsumption:
    def test_sprite_animator_same_semantics(self):
        sp = SpriteAnimator()
        attack = AnimParameter(name="attack", value_type=TypeRef(ValueType.BOOL), default=False)
        attacking = AnimParameter(name="is_attacking", value_type=TypeRef(ValueType.BOOL), default=False)
        sp._fsm = AnimStateMachine(parameters=[attack, attacking])
        sp.set_trigger("attack")
        sp._consume_triggers(AnimTransition(conditions=[_condition(attacking)]))
        assert sp._parameters["attack"] is True


@pytest.mark.parametrize("animator_type", [SpriteAnimator, SkeletalAnimator])
def test_runtime_state_before_awake_is_not_shared_between_instances(animator_type):
    first, second = animator_type(), animator_type()
    first.set_bool("grounded", True)
    first._clip_cache["Walk"] = None
    assert second.get_bool("grounded") is False
    assert "Walk" not in second._clip_cache


class TestEmptyConditionTransition:
    def _animator_with_state(self, loop: bool, duration: float):
        anim = _make_animator()
        fsm = AnimStateMachine(name="t", mode="3d")
        state = AnimState(name="S", loop=loop)
        fsm.states.append(state)
        anim._fsm = fsm
        anim._current_state = anim._fsm.get_state("S")
        anim._current_clip = _FakeClip("S_take", duration_hint=duration)
        return anim, state


# ── Safe condition evaluator (replaces eval()) ──────────────────────────────


# ── Animation events ────────────────────────────────────────────────────────


class _EventSink:
    def __init__(self):
        self.calls = []
        self.footsteps = 0

    def on_animation_event(self, name, string_arg, number_arg):
        self.calls.append((name, string_arg, number_arg))

    def footstep(self, string_arg, number_arg):
        self.footsteps += 1


class _FakeGameObject:
    def __init__(self, comps):
        self._comps = comps

    def get_py_components(self):
        return list(self._comps)


class TestAnimationEventWindowing:
    def test_first_sample_includes_zero_once_and_retains_loop_boundary_order(self):
        from infernux.core.animation_event import AnimationEvent, collect_crossed_events

        start = AnimationEvent(0.0, "start")
        end = AnimationEvent(1.0, "end")
        events = [start, end]
        assert collect_crossed_events(events, None, 0.0, False) == [start]
        assert collect_crossed_events(events, 0.0, 0.1, False) == []
        assert collect_crossed_events(events, None, 0.1, True, wrap_count=2) == [start, end, start, end, start]
        assert collect_crossed_events(events, 0.9, 0.1, True) == [end, start]

    def test_dispatch_uses_precomputed_arity_without_signature_reflection(self, monkeypatch):
        from infernux.core import animation_event
        from infernux.core.animation_event import AnimationEvent
        from infernux.engine import runtime_dispatch
        from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch

        class AritySink:
            def __init__(self):
                self.calls = []

            def one_arg(self, value):
                self.calls.append(("one", value))

            def broken(self, _value, _number):
                self.calls.append("broken")
                raise TypeError("handler body")

        sink = AritySink()
        publication = publish_runtime_dispatch_epoch((AritySink,))
        publication.commit()
        try:
            descriptor = runtime_dispatch.current_runtime_epoch().descriptor_for(AritySink)
            assert descriptor is not None
            assert descriptor.methods["one_arg"].accepted_arg_counts == (1,)

            def reject_signature(_callback):
                raise AssertionError("event dispatch must not inspect signatures")

            monkeypatch.setattr(runtime_dispatch.inspect, "signature", reject_signature)
            go = _FakeGameObject([sink])
            animation_event.dispatch_animation_events(
                go,
                [
                    AnimationEvent(0.5, "one_arg", "L", 2.0),
                    AnimationEvent(0.5, "broken", "R", 3.0),
                ],
                0.4,
                0.6,
                looped=False,
            )
            assert sink.calls == [("one", "L"), "broken"]
        finally:
            publication.rollback()

    def test_generic_sink_and_named_alias_are_invoked_once(self):
        from infernux.core.animation_event import AnimationEvent, dispatch_animation_events
        from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch

        class AliasSink:
            def __init__(self):
                self.calls = 0

            def on_animation_event(self, _name, _text, _number):
                self.calls += 1

            alias = on_animation_event

        sink = AliasSink()
        publication = publish_runtime_dispatch_epoch((AliasSink,))
        publication.commit()
        try:
            dispatch_animation_events(
                _FakeGameObject([sink]),
                [AnimationEvent(0.5, "alias", "", 0.0)],
                0.4,
                0.6,
                looped=False,
            )
            assert sink.calls == 1
        finally:
            publication.rollback()


# ── Serialization round-trips for new fields ────────────────────────────────


def _write_texture_asset_metadata(
    texture_path,
    *,
    guid: str,
    texture_type,
    sprite_frame_ids=(),
):
    import json

    from infernux.core.asset_types import SpriteFrame, TextureImportSettings

    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(b"test texture")
    settings = TextureImportSettings(
        texture_type=texture_type,
        sprite_frames=[
            SpriteFrame(stable_id=stable_id, name=f"frame_{index}", w=16, h=16)
            for index, stable_id in enumerate(sprite_frame_ids)
        ],
    )

    def tagged(value):
        if type(value) is bool:
            tag = "bool"
        elif type(value) is int:
            tag = "int"
        elif type(value) is list:
            tag = "json_array"
        else:
            tag = "string"
        return {"type": tag, "value": value}

    metadata = {"guid": tagged(guid)}
    metadata.update({key: tagged(value) for key, value in settings.to_dict().items()})
    texture_path.with_name(texture_path.name + ".meta").write_text(
        json.dumps({"metadata": metadata}, indent=2),
        encoding="utf-8",
    )


class TestAnimationClipSpriteFrameReferences:
    FRAME_A = "1" * 32
    FRAME_B = "2" * 32
    TEXTURE_GUID = "a" * 32

    def test_valid_sprite_frame_references_resolve_guid_mapping(self, tmp_path):
        from infernux.core.animation_clip import AnimationClip, AnimationFrame
        from infernux.core.asset_types import TextureType

        project = tmp_path / "project"
        texture = project / "Assets" / "Sprites" / "sheet.png"
        _write_texture_asset_metadata(
            texture,
            guid=self.TEXTURE_GUID,
            texture_type=TextureType.SPRITE,
            sprite_frame_ids=(self.FRAME_A, self.FRAME_B),
        )
        clip = AnimationClip(
            authoring_texture_guid=self.TEXTURE_GUID,
            frames=[AnimationFrame(sprite_frame_id=self.FRAME_B)],
        )

        resolved = clip.validate_sprite_frame_references(
            project_root=str(project),
            guid_paths={self.TEXTURE_GUID: "Assets/Sprites/sheet.png"},
        )

        assert resolved == str(texture)

    def test_guid_survives_file_round_trip_and_resolves_moved_texture(self, tmp_path, monkeypatch):
        from infernux.core.animation_clip import AnimationClip, AnimationFrame
        from infernux.core.asset_types import TextureType
        from infernux.core.assets import AssetManager

        project = tmp_path / "project"
        texture = project / "Assets" / "Sprites" / "sheet.png"
        clip_path = project / "Assets" / "Animations" / "walk.animclip2d"
        clip_path.parent.mkdir(parents=True)
        _write_texture_asset_metadata(
            texture,
            guid=self.TEXTURE_GUID,
            texture_type=TextureType.SPRITE,
            sprite_frame_ids=(self.FRAME_A,),
        )
        clip = AnimationClip(
            name="walk",
            authoring_texture_guid=self.TEXTURE_GUID,
            frames=[AnimationFrame(sprite_frame_id=self.FRAME_A)],
        )
        database = type(
            "Database",
            (),
            {"get_path_from_guid": staticmethod(lambda guid: str(texture) if guid == self.TEXTURE_GUID else "")},
        )()
        monkeypatch.setattr(AssetManager, "_asset_database", database)
        assert clip.save(str(clip_path)) is True

        loaded = AnimationClip.load(str(clip_path))

        assert loaded is not None
        assert loaded.authoring_texture_guid == self.TEXTURE_GUID
        assert loaded.validate_sprite_frame_references(
            project_root=str(project),
            guid_paths={self.TEXTURE_GUID: str(texture)},
        ) == str(texture)


class TestSpriteAnimatorAssetReload:
    @staticmethod
    def _clip(frame_count: int, fps: float):
        from infernux.core.animation_clip import AnimationClip, AnimationFrame

        return AnimationClip(
            frames=[AnimationFrame(sprite_frame_id=f"{index + 1:032x}") for index in range(frame_count)],
            fps=fps,
        )

    def test_clip_hot_reload_preserves_state_progress_and_playback(self, tmp_path, monkeypatch):
        import json
        from infernux.components import sprite_animator as animator_module

        path = tmp_path / "walk.animclip2d"
        clip_guid = "a" * 32
        old_clip = self._clip(4, 4.0)
        path.write_text(json.dumps(old_clip.to_dict()), encoding="utf-8")
        database = type(
            "Database",
            (),
            {"get_path_from_guid": staticmethod(lambda guid: str(path) if guid == clip_guid else "")},
        )()
        monkeypatch.setattr(animator_module, "_get_asset_database", lambda: database)

        state = AnimState(name="Walk", clip_guid=clip_guid)
        animator = SpriteAnimator()
        animator._fsm = AnimStateMachine(
            states=[state],
            default_state="Walk",
        )
        animator._clip_cache = {"Walk": old_clip}
        animator._current_state_name = "Walk"
        animator._current_clip = old_clip
        animator._elapsed = old_clip.duration * 0.5
        animator._playing = True
        animator._parameters = {"speed": 3.0}

        class Renderer:
            frame_id = ""
            sync_count = 0

            def sync_visual(self):
                self.sync_count += 1

        renderer = Renderer()
        animator._sprite_renderer = renderer

        replacement = self._clip(8, 8.0)
        path.write_text(json.dumps(replacement.to_dict()), encoding="utf-8")
        from infernux.engine.interaction import AssetMutation, AssetMutationKind

        animator._on_asset_changed(
            AssetMutation(
                AssetMutationKind.MODIFIED,
                f"{path}.meta",
                guid=clip_guid,
            )
        )

        assert animator._current_state_name == "Walk"
        assert animator._current_clip is animator._clip_cache["Walk"]
        assert animator._current_clip.frame_count == 8
        assert animator.normalized_time == pytest.approx(0.5)
        assert animator._playing is True
        assert animator._parameters == {"speed": 3.0}
        assert renderer.frame_id == replacement.frames[4].sprite_frame_id
        assert renderer.sync_count == 1

    def test_invalid_clip_hot_reload_keeps_previous_runtime_clip(self, tmp_path, monkeypatch):
        import json
        from infernux.components import sprite_animator as animator_module

        path = tmp_path / "walk.animclip2d"
        clip_guid = "b" * 32
        clip = self._clip(2, 2.0)
        path.write_text(json.dumps(clip.to_dict()), encoding="utf-8")
        database = type(
            "Database",
            (),
            {"get_path_from_guid": staticmethod(lambda guid: str(path) if guid == clip_guid else "")},
        )()
        monkeypatch.setattr(animator_module, "_get_asset_database", lambda: database)
        state = AnimState(name="Walk", clip_guid=clip_guid)
        animator = SpriteAnimator()
        animator._fsm = AnimStateMachine(states=[state], default_state="Walk")
        animator._clip_cache = {"Walk": clip}
        animator._current_state_name = "Walk"
        animator._current_clip = clip
        animator._elapsed = 0.25
        animator._playing = True
        animator._sprite_renderer = None

        path.write_text("not json", encoding="utf-8")
        from infernux.engine.interaction import AssetMutation, AssetMutationKind

        animator._on_asset_changed(
            AssetMutation(
                AssetMutationKind.MODIFIED,
                str(path),
                guid=clip_guid,
            )
        )

        assert animator._current_clip is clip
        assert animator._clip_cache["Walk"] is clip
        assert animator._elapsed == pytest.approx(0.25)
        assert animator._playing is True


def test_trigger_consumption_uses_exact_declared_identifiers():
    animator = _make_animator()
    attack = AnimParameter(name="attack", value_type=TypeRef(ValueType.BOOL), default=False)
    attacking = AnimParameter(name="is_attacking", value_type=TypeRef(ValueType.BOOL), default=False)
    jump = AnimParameter(name="jump", value_type=TypeRef(ValueType.BOOL), default=False)
    animator._fsm = AnimStateMachine(parameters=[attack, attacking, jump])
    animator.set_trigger("attack")
    animator.set_trigger("jump")
    animator.set_bool("idle", True)
    animator._consume_triggers(AnimTransition(conditions=[_condition(attacking)]))
    assert animator._parameters["attack"] and animator._parameters["jump"]
    animator._consume_triggers(AnimTransition(conditions=[_condition(attack), _condition(jump)]))
    assert not animator._parameters["attack"] and not animator._parameters["jump"]
    assert animator._parameters["idle"]


def test_reconfiguring_crossfade_duration_preserves_override_and_hard_cut_semantics():
    animator = _make_animator()
    animator.cross_fade_duration = 0.25
    a, b = _FakeClip("A"), _FakeClip("B")
    animator._start_blend_if_needed(a, 0.5, 1.0, b, fade_duration=0.6)
    assert animator._blend_duration == pytest.approx(0.6) and animator._blend_from_take_name == "A"
    animator._start_blend_if_needed(b, 0.2, 1.0, a, fade_duration=None)
    assert animator._blend_duration == pytest.approx(0.25) and animator._blend_from_take_name == "B"
    animator._start_blend_if_needed(a, 1.2, 1.0, a, fade_duration=None)
    assert animator._blend_duration == pytest.approx(0.25) and animator._blend_from_take_name == "A"
    animator._start_blend_if_needed(a, 0.0, 1.0, b, fade_duration=0.0)
    assert animator._blend_duration == 0.0 and animator._blend_from_clip is None


@pytest.mark.parametrize("state_loop,clip_loop", [(True, True), (False, True), (True, False)])
def test_crossfade_can_sync_phase_then_restart_the_return_state(monkeypatch, state_loop, clip_loop):
    animator = _make_animator()
    native = _NativePoseRecorder()
    owner = _HierarchyOwner(renderer=_RendererBinding(native))
    monkeypatch.setattr(SkeletalAnimator, "game_object", property(lambda _self: owner))
    animator._fsm = AnimStateMachine(
        states=[AnimState(name="Walk", loop=state_loop), AnimState(name="Run", loop=True)], default_state="Walk"
    )
    animator._clip_cache = {
        "Walk": _FakeClip("Walk", duration_hint=2.0, default_loop=clip_loop),
        "Run": _FakeClip("Run", duration_hint=4.0),
    }
    animator._current_state = animator._fsm.get_state("Walk")
    animator._current_clip = animator._clip_cache["Walk"]
    animator._elapsed, animator._playing = 1.5, True
    assert animator.cross_fade("Run", 2.0, preserve_phase=True)
    assert animator._elapsed == pytest.approx(3.0)
    assert animator._prev_event_norm == pytest.approx(0.75)
    assert animator._blend_from_elapsed == pytest.approx(1.5)
    animator._advance_blend(1.0)
    animator._sync_native_runtime_playback()
    outgoing, incoming = native.pose_stacks[-1]
    assert outgoing["time"] == pytest.approx(0.5 if state_loop and clip_loop else 2.0)
    assert outgoing["loop"] is (state_loop and clip_loop)
    assert incoming["loop"] is True
    assert outgoing["weight"] == incoming["weight"] == pytest.approx(0.5)
    assert animator.cross_fade("Walk", 0.2)
    assert animator._elapsed == 0.0 and animator._prev_event_norm is None


def test_non_looping_clip_end_condition_follows_clamped_progress():
    animator = _make_animator()
    state = AnimState(name="Once", loop=False)
    animator._fsm = AnimStateMachine(states=[state], default_state=state.name)
    animator._current_state = state
    animator._current_clip = _FakeClip("Once", duration_hint=1.0)
    transition = AnimTransition(target_state="Next")
    animator._elapsed = 0.4
    assert not animator._evaluate_condition(transition)
    animator._elapsed = 1.5
    assert animator._evaluate_condition(transition) and animator.normalized_time == 1.0
