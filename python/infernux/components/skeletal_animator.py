"""
SkeletalAnimator — runtime 3D animation state machine controller.

Mirrors :class:`SpriteAnimator` (2D) for skeletal assets: bridge from
``.animfsm`` / ``.animclip3d`` to :class:`SkinnedMeshRenderer`, advancing FSM
state and publishing animation poses to the native GPU skinning path.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional

from infernux.components.component import InxComponent, per_instance
from infernux.components.fields import serialized_field
from infernux.components.decorators import disallow_multiple, add_component_menu
from infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
from infernux.core.anim_state_machine import (
    AnimStateMachine, AnimState, AnimTransition,
)
from infernux.core.animation_clip3d import AnimationClip3D
from infernux.core.asset_ref import AnimStateMachineRef
from infernux.debug import Debug
from infernux.graph.types import ValueType


def _animation_source_guid(clip: Optional[AnimationClip3D]) -> str:
    """Return the runtime model GUID that owns *clip*'s animation take."""
    if clip is None:
        return ""
    return str(clip.source_model_guid or "").strip()


def _get_asset_database():
    from infernux.core.assets import AssetManager

    return AssetManager.require_asset_database()


def _resolve_timeline_path(state: AnimState) -> Optional[str]:
    """Resolve a timeline state's ``.animtimeline`` asset to a disk path."""
    if not state.timeline_guid:
        return None
    return _get_asset_database().get_path_from_guid(state.timeline_guid) or None


def _clip_duration_hint(clip: Optional[AnimationClip3D]) -> float:
    if clip is None:
        return 0.0
    return max(float(clip.duration_hint), 0.0)


def _clip_should_loop(state: Optional[AnimState], clip: Optional[AnimationClip3D]) -> bool:
    """Combine the imported clip contract with the FSM's stop override."""
    imported = bool(getattr(clip, "default_loop", True)) if clip is not None else True
    return imported and (bool(state.loop) if state is not None else True)


def _clip_bone_mask(clip: Optional[AnimationClip3D]) -> list[str]:
    return list(getattr(clip, "bone_mask", ()) or ()) if clip is not None else []


# When importer/meta leaves duration unknown (e.g. embedded FBX takes), use this for
# normalized_time and looping so native/runtime hooks see monotonic [0,1) progress.
_DEFAULT_PLAYBACK_SEC_WHEN_UNKNOWN_DURATION = 1.0


@disallow_multiple
@add_component_menu("Animation/Skeletal Animator")
class SkeletalAnimator(InxComponent):
    """Drives a SkinnedMeshRenderer from a 3D AnimStateMachine (``.animfsm``)."""

    controller: AnimStateMachineRef = serialized_field(
        default=None,
        asset_type="AnimStateMachine",
        tooltip="3D AnimStateMachine controller (.animfsm)",
    )

    playback_speed: float = serialized_field(
        default=1.0,
        range=(0.0, 10.0),
        tooltip="Global playback speed multiplier",
    )

    auto_play: bool = serialized_field(
        default=True,
        tooltip="Start playing the default state on start",
    )

    cross_fade_duration: float = serialized_field(
        default=0.15,
        range=(0.0, 2.0),
        tooltip="Seconds used to blend between 3D animation states",
    )

    _parameters: Dict[str, object] = per_instance(dict)
    _pending_triggers: frozenset[str] = frozenset()
    _curve_values: Dict[str, float] = per_instance(dict)

    _fsm: Optional[AnimStateMachine] = None
    _skinned_renderers: list[SkinnedMeshRenderer] = per_instance(list)
    _native_renderers: list = per_instance(list)
    native_pose_submission: bool = True
    # Optional fixed animation update interval in seconds. Zero keeps the
    # historical per-frame update; a positive value advances in accumulated
    # steps so remote characters can sample at the rate their poses are used.
    update_interval: float = 0.0
    _clip_cache: dict[object, Optional[AnimationClip3D]] = per_instance(dict)

    _current_state: Optional[AnimState] = None
    _current_clip: Optional[AnimationClip3D] = None
    _group_clips: list[AnimationClip3D] = per_instance(list)
    _group_weights: list[float] = per_instance(list)
    _elapsed: float = 0.0
    _playing: bool = False
    _playback_revision: int = 0
    _blend_from_clip: Optional[AnimationClip3D] = None
    _blend_from_take_name: str = ""
    _blend_from_elapsed: float = 0.0
    _blend_from_speed: float = 1.0
    _blend_from_loop: bool = True
    _blend_from_group: list = per_instance(list)
    _blend_elapsed: float = 0.0
    _blend_duration: float = 0.0
    _last_native_take_name: str = ""
    _last_native_pose_key = None
    _duration_cache: Dict[tuple[str, str], float] = per_instance(dict)
    _current_timeline = None
    _timeline_cache: Dict[str, object] = per_instance(dict)
    _timeline_base = None

    def awake(self):
        self._parameters = {}
        self._pending_triggers = frozenset()
        self._curve_values = {}
        self._clip_cache = {}
        self._duration_cache = {}
        self._timeline_cache = {}
        self._current_timeline = None
        self._timeline_base = None
        self._last_native_take_name = ""
        self._last_native_pose_key = None
        self._current_state = None
        self._current_clip = None
        self._group_clips = []
        self._group_weights = []
        self._elapsed = 0.0
        self._playing = False
        self._update_accum = 0.0
        self._skinned_renderers = []
        self._native_renderers = []
        self._clear_blend_state()

    def start(self):
        # Play-mode scene replacement can invalidate an edit-scene wrapper after
        # the Python component has already been restored.  Resolve through the
        # live GameObject instead of trusting a serialized/cached wrapper.
        self._skinned_renderers = []
        renderers = self._resolve_native_renderers(force=True)
        if not renderers:
            Debug.log_warning(
                "[SkeletalAnimator] No SkinnedMeshRenderer found on this GameObject or its descendants."
            )
            return

        self._load_controller()

        if self.auto_play and self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def update(self, delta_time: float):
        interval = float(self.update_interval)
        if interval > 0.0:
            self._update_accum += delta_time
            if self._update_accum < interval:
                return
            delta_time = self._update_accum
            self._update_accum = 0.0
        if self._current_timeline is not None:
            self._update_timeline(delta_time)
            return
        if not self._playing or not self._current_clip:
            return

        if self._current_state.kind == "group":
            self._update_group(delta_time)
            return

        state = self._get_current_state()
        clip = self._current_clip
        playback_revision = self._playback_revision
        speed = self.playback_speed * (state.speed if state else 1.0)
        previous_elapsed = self._elapsed
        self._elapsed += delta_time * speed
        self._advance_blend(delta_time)

        prev_norm = getattr(self, "_prev_event_norm", 0.0)
        duration = self._clip_duration(clip)
        if duration > 0.0 and self._elapsed >= duration:
            should_loop = _clip_should_loop(state, clip)
            self._apply_imported_root_motion(clip, previous_elapsed, self._elapsed, should_loop)
            if should_loop:
                post = self._elapsed % duration
                post_norm = post / duration
                wrap_count = max(1, math.floor(self._elapsed / duration))
                self._dispatch_clip_events_wrapped(clip, prev_norm, post_norm, wrap_count)
                if self._playback_revision != playback_revision:
                    return
                self._prev_event_norm = post_norm
                self._try_auto_transition()
                if self._playback_revision != playback_revision:
                    return
                if self._playing:
                    self._elapsed = post
            else:
                self._elapsed = duration
                self._dispatch_clip_events(clip, prev_norm, 1.0, False)
                if self._playback_revision != playback_revision:
                    return
                self._prev_event_norm = 1.0
                self._sample_imported_curves(clip, 1.0)
                self._try_auto_transition()
                if self._playback_revision == playback_revision:
                    self._playing = False
                self._sync_native_runtime_playback()
                return
        else:
            self._apply_imported_root_motion(clip, previous_elapsed, self._elapsed, _clip_should_loop(state, clip))
            curr_norm = (self._elapsed / duration) if duration > 0.0 else 0.0
            self._dispatch_clip_events(clip, prev_norm, curr_norm, False)
            if self._playback_revision != playback_revision:
                return
            self._prev_event_norm = curr_norm

        self._sample_imported_curves(clip, self.normalized_time)

        self._try_auto_transition()
        self._sync_native_runtime_playback()

    def _apply_imported_root_motion(
        self, clip: AnimationClip3D, previous_time: float, current_time: float, loop: bool,
    ) -> None:
        if not bool(getattr(clip, "apply_root_motion", False)):
            return
        self._apply_root_motion([(clip, 1.0, loop)], previous_time, current_time)

    def _apply_root_motion(self, sources, previous_time: float, current_time: float) -> None:
        if not sources or current_time == previous_time:
            return
        from infernux.lib import Vector3, quatf

        renderers = self._resolve_native_renderers()
        if not renderers:
            return
        cpp = renderers[0]
        translation, rotation_sum, total, reference = Vector3(), [0.0] * 4, 0.0, None
        for clip, weight, loop in sources:
            delta, rotation = cpp.get_root_motion_delta(
                clip.take_name, float(previous_time), float(current_time), bool(loop), _animation_source_guid(clip),
            )
            translation += delta * weight
            if reference is None:
                reference = rotation
            signed = -weight if quatf.dot(reference, rotation) < 0.0 else weight
            rotation_sum = [value + rotation[index] * signed for index, value in enumerate(rotation_sum)]
            total += weight
        translation /= max(total, 1.0)
        rotation = quatf.slerp(quatf.identity(), quatf(*rotation_sum).normalized, min(total, 1.0))
        transform = self.game_object.transform
        transform.translate_local(translation)
        transform.local_rotation = transform.local_rotation * rotation

    def _dispatch_clip_events(
        self, clip, prev_norm: float, curr_norm: float, looped: bool, *, wrap_count: int = 1,
    ):
        """Fire any animation events on *clip* crossed this frame."""
        events = getattr(clip, "events", None)
        if not events:
            return
        try:
            from infernux.core.animation_event import dispatch_animation_events
            dispatch_animation_events(
                self.game_object, events, prev_norm, curr_norm, looped, wrap_count=wrap_count,
            )
        except Exception as exc:
            Debug.log_warning(f"[SkeletalAnimator] event dispatch error: {exc}")

    def _dispatch_clip_events_wrapped(
        self, clip, prev_norm: float, curr_norm: float, wrap_count: int,
    ) -> None:
        """Dispatch every crossed occurrence, including multiple wraps in one update."""
        self._dispatch_clip_events(clip, prev_norm, curr_norm, True, wrap_count=wrap_count)

    def _sample_imported_curves(self, clip: Optional[AnimationClip3D], normalized_time: float) -> None:
        curves = getattr(clip, "curves", ()) if clip is not None else ()
        self._curve_values = {curve.name: float(curve.sample(normalized_time)) for curve in curves}

    def get_curve_value(self, name: str, default: float = 0.0) -> float:
        """Return the current value of an import-authored scalar curve."""
        return float(self._curve_values.get(str(name), default))

    @property
    def current_state(self) -> str:
        return self._current_state.name if self._current_state is not None else ""

    @property
    def current_take_name(self) -> str:
        if self._current_clip is None:
            return ""
        return str(getattr(self._current_clip, "take_name", "") or "")

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def normalized_time(self) -> float:
        if self._current_timeline is not None:
            dur = max(1e-6, float(self._current_timeline.duration))
            state = self._get_current_state()
            if state is None or state.loop:
                return (self._elapsed % dur) / dur
            return min(self._elapsed / dur, 1.0)
        duration = self._state_duration()
        if duration > 0.0:
            if self._current_state is not None and self._current_state.kind == "group" and self._current_state.loop:
                return (self._elapsed % duration) / duration
            return min(self._elapsed / duration, 1.0)
        # No duration in asset — assume a neutral loop period so time/normalized are not stuck.
        t = _DEFAULT_PLAYBACK_SEC_WHEN_UNKNOWN_DURATION
        return (self._elapsed % t) / t

    def play(self, state_name: str = "", *, restart: bool = False) -> bool:
        if not self._fsm:
            return False
        name = state_name or self._fsm.default_state
        if not name:
            return False
        return self._enter_state(name, restart=bool(restart))

    def cross_fade(
        self,
        state_name: str,
        duration: float,
        *,
        preserve_phase: bool = False,
        restart: bool = False,
    ) -> bool:
        """Blend into *state_name*, optionally preserving a looping gait phase.

        ``preserve_phase`` is intended for related cyclic states such as Walk
        and Run.  It maps the outgoing normalized time onto the incoming clip
        instead of restarting that clip at frame zero, preventing planted feet
        from snapping during locomotion transitions.
        """
        return self._enter_state(
            state_name,
            fade_duration=max(float(duration), 0.0),
            preserve_phase=bool(preserve_phase),
            restart=bool(restart),
        )

    def stop(self):
        self._playback_revision += 1
        self._playing = False
        self._clear_blend_state()
        self._last_native_pose_key = None
        self._sync_native_runtime_playback()

    def get_pose_layers(self) -> list[dict]:
        """Snapshot the sampled pose for composition with procedural or masked layers.

        Submit the combined layers to SkinnedMeshRenderer in late_update, after
        this animator has advanced its state and dispatched animation events.
        """
        clip = self._current_clip
        if clip is None:
            return []

        def layer(source, elapsed, weight, loop, group="", fade=1.0):
            return {"take_name": source.take_name, "source_model_guid": _animation_source_guid(source),
                    "time": float(elapsed), "weight": float(weight), "loop": loop,
                    "bone_mask": _clip_bone_mask(source), "blend_group": group, "group_weight": fade}

        state = self._get_current_state()
        layers = []
        weight = 0.0
        if self._blend_from_clip is not None and self._blend_duration > 0.0:
            weight = 1.0 - min(max(self._blend_elapsed / self._blend_duration, 0.0), 1.0)
            if weight > 0.0:
                sources = self._blend_from_group or [(self._blend_from_clip, 1.0, self._blend_from_loop)]
                layers.extend(layer(source, self._blend_from_elapsed, amount, loop,
                                    f"{self.component_id}:outgoing", weight)
                              for source, amount, loop in sources if amount > 0.0)
        if state is not None and state.kind == "group":
            layers.extend(layer(source, self._elapsed, amount, _clip_should_loop(state, source),
                                f"{self.component_id}:current", 1.0 - weight)
                          for source, amount in zip(self._group_clips, self._group_weights) if amount > 0.0)
        else:
            layers.append(layer(clip, self._elapsed, 1.0, _clip_should_loop(state, clip),
                                f"{self.component_id}:current", 1.0 - weight))
        return layers

    def set_clip_weight(self, index: int, weight: float) -> None:
        """Set a current group's clip weight; masks keep upper/lower-body coverage independent."""
        if self._current_state is None or self._current_state.kind != "group":
            raise ValueError("set_clip_weight requires an active animation clip group")
        value = float(weight)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("animation clip weight must be in [0, 1]")
        self._group_weights[index] = value
        self._sync_native_runtime_playback()

    def set_parameter(self, name: str, value: object):
        self._parameters[name] = value
        if name in self._pending_triggers:
            self._pending_triggers -= {name}

    def get_parameter(self, name: str, default: object = None) -> object:
        return self._parameters.get(name, default)

    def get_bool(self, name: str) -> bool:
        return bool(self._parameters.get(name, False))

    def set_bool(self, name: str, value: bool):
        self.set_parameter(name, bool(value))

    def get_float(self, name: str) -> float:
        # Unity-compatible imported float curves drive Animator float
        # parameters while their clip is active.
        if name in self._curve_values:
            return float(self._curve_values[name])
        return float(self._parameters.get(name, 0.0))

    def set_float(self, name: str, value: float):
        self.set_parameter(name, float(value))

    def get_int(self, name: str) -> int:
        return int(self._parameters.get(name, 0))

    def set_int(self, name: str, value: int):
        self.set_parameter(name, int(value))

    def set_trigger(self, name: str):
        self._parameters[name] = True
        self._pending_triggers |= {name}

    def reload_controller(self):
        self._load_controller()
        if self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def on_after_deserialize(self):
        self._clip_cache = {}
        self._duration_cache = {}
        self._timeline_cache = {}
        self._current_timeline = None
        self._timeline_base = None
        self._last_native_take_name = ""
        self._last_native_pose_key = None
        self._parameters = {}
        self._curve_values = {}
        self._pending_triggers = frozenset()
        self._clear_blend_state()

    def _load_controller(self):
        self.stop()
        self._current_state = None
        self._current_clip = None
        self._current_timeline = None
        self._elapsed = 0.0
        self._parameters = {}
        self._curve_values = {}
        self._pending_triggers = frozenset()
        self._fsm = None
        self._clip_cache = {}
        self._duration_cache = {}
        self._timeline_cache = {}

        fsm = self.controller
        if fsm is None:
            return

        if fsm.mode != "3d":
            Debug.log_warning(f"[SkeletalAnimator] Controller is mode='{fsm.mode}', expected '3d'.")
        self._fsm = fsm
        self._seed_parameters_from_fsm(fsm)
        for state in fsm.states:
            if state.kind == "group":
                self._resolve_group_clips(state)
            elif state.kind == "clip":
                self._resolve_clip(state)

    def _seed_parameters_from_fsm(self, fsm: AnimStateMachine) -> None:
        self._parameters = {}
        self._pending_triggers = frozenset()
        for p in fsm.parameters:
            if p.value_type.value_type is ValueType.BOOL:
                self._parameters[p.name] = bool(p.default)
            elif p.value_type.value_type is ValueType.I32:
                self._parameters[p.name] = int(p.default)
            else:
                self._parameters[p.name] = float(p.default)

    def _resolve_clip(self, state: AnimState) -> Optional[AnimationClip3D]:
        key = state.name
        if key in self._clip_cache:
            return self._clip_cache[key]

        from infernux.core.assets import AssetManager

        clip = AssetManager.load_by_guid(state.clip_guid, AnimationClip3D) if state.clip_guid else None
        if clip is None and state.clip_guid:
            Debug.log_warning(f"[SkeletalAnimator] Clip not found for state '{state.name}'")
        self._clip_cache[key] = clip
        return clip

    def _resolve_group_clips(self, state: AnimState) -> list[AnimationClip3D]:
        from infernux.core.assets import AssetManager

        clips = []
        for entry in state.clips:
            key = ("group", entry.clip_guid)
            if key not in self._clip_cache:
                self._clip_cache[key] = AssetManager.load_by_guid(entry.clip_guid, AnimationClip3D) if entry.clip_guid else None
            clip = self._clip_cache[key]
            if clip is None:
                return []
            clips.append(clip)
        return clips

    def _resolve_timeline(self, state: AnimState):
        """Resolve and cache the ``.animtimeline`` asset for a timeline state."""
        key = state.name
        if key in self._timeline_cache:
            return self._timeline_cache[key]
        tl = None
        path = _resolve_timeline_path(state)
        if path:
            from infernux.core.animation_timeline import AnimationTimeline
            tl = AnimationTimeline.load(path)
            if tl is None:
                raise RuntimeError(
                    f"SkeletalAnimator could not load timeline for state {state.name!r}: {path}"
                )
        self._timeline_cache[key] = tl
        return tl

    def _update_timeline(self, delta_time: float):
        """Advance + apply a timeline state, driving the owner GameObject transform."""
        tl = self._current_timeline
        if tl is None or not self._playing:
            return
        state = self._get_current_state()
        playback_revision = self._playback_revision
        # Looping is decided by the owning FSM state, not the timeline asset.
        loop = bool(state.loop) if state is not None else True
        if self._playing:
            speed = self.playback_speed * (state.speed if state else 1.0)
            self._elapsed += delta_time * speed
            dur = max(1e-6, float(tl.duration))
            if self._elapsed >= dur:
                # Evaluate exit-time transitions at the final pose before wrapping.
                self._apply_timeline(tl, dur)
                self._try_auto_transition()
                if self._playback_revision == playback_revision and self._playing:
                    if loop:
                        self._elapsed = self._elapsed % dur
                    else:
                        self._elapsed = dur
                        self._playing = False
                    self._apply_timeline(tl, self._elapsed)
                return
        self._apply_timeline(tl, self._elapsed)
        self._try_auto_transition()

    def _capture_timeline_base(self):
        """Snapshot the owner's local transform as the additive base for a timeline."""
        transform = self.game_object.transform
        position = transform.local_position
        rotation = transform.local_euler_angles
        scale = transform.local_scale
        self._timeline_base = (
            [float(position.x), float(position.y), float(position.z)],
            [float(rotation.x), float(rotation.y), float(rotation.z)],
            [float(scale.x), float(scale.y), float(scale.z)],
        )

    def _apply_timeline(self, tl, t: float):
        """Sample *tl* at time *t* and write the local transform of the owner."""
        sampled = tl.sample(t)
        if sampled is None:
            return
        pos, rot, scl = sampled
        if getattr(tl, "apply_mode", "additive") == "additive":
            bp, br, bs = self._timeline_base or ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            pos = [bp[0] + pos[0], bp[1] + pos[1], bp[2] + pos[2]]
            rot = [br[0] + rot[0], br[1] + rot[1], br[2] + rot[2]]
            scl = [bs[0] * scl[0], bs[1] * scl[1], bs[2] * scl[2]]
        transform = self.game_object.transform
        pose = tuple(float(value) for value in (*pos, *rot, *scl))
        if pose == getattr(self, "_last_timeline_pose", None):
            return
        transform.set_local_trs(*pose)
        self._last_timeline_pose = pose

    def _state_duration(self) -> float:
        clips = self._group_clips if self._current_state is not None and self._current_state.kind == "group" else [self._current_clip]
        return max((self._clip_duration(clip) for clip in clips), default=0.0)

    def _update_group(self, delta_time: float) -> None:
        state = self._current_state
        revision = self._playback_revision
        previous = self._elapsed
        self._elapsed += delta_time * self.playback_speed * state.speed
        duration = self._state_duration()
        if not state.loop:
            self._elapsed = min(self._elapsed, duration)
        self._advance_blend(delta_time)
        self._curve_values = {}
        curve_weights = {}
        for clip, weight in zip(self._group_clips, self._group_weights):
            if weight <= 0.0:
                continue
            clip_duration = self._clip_duration(clip)
            if clip_duration <= 0.0:
                continue
            loop = _clip_should_loop(state, clip)
            before, after = previous / clip_duration, self._elapsed / clip_duration
            wraps = math.floor(after) - math.floor(before) if loop else 0
            self._dispatch_clip_events(clip, before % 1.0 if loop else min(before, 1.0),
                                       after % 1.0 if loop else min(after, 1.0),
                                       wraps > 0, wrap_count=max(1, wraps))
            if self._playback_revision != revision:
                return
            for curve in clip.curves:
                value = curve.sample(after % 1.0 if loop else min(after, 1.0))
                self._curve_values[curve.name] = self._curve_values.get(curve.name, 0.0) + value * weight
                curve_weights[curve.name] = curve_weights.get(curve.name, 0.0) + weight
        for name, weight in curve_weights.items():
            self._curve_values[name] /= max(weight, 1.0)
        self._apply_root_motion(
            [(clip, weight, _clip_should_loop(state, clip))
             for clip, weight in zip(self._group_clips, self._group_weights)
             if weight > 0.0 and clip.apply_root_motion], previous, self._elapsed,
        )
        self._try_auto_transition()
        if self._playback_revision == revision and not state.loop and self._elapsed >= duration:
            self._playing = False
        self._sync_native_runtime_playback()

    def _enter_state(
        self,
        state_name: str,
        fade_duration: Optional[float] = None,
        *,
        preserve_phase: bool = False,
        restart: bool = False,
    ) -> bool:
        if not self._fsm:
            return False
        state = self._fsm.get_state(state_name)
        if state is None:
            Debug.log_warning(f"[SkeletalAnimator] State not found: '{state_name}'")
            return False

        if not restart and not getattr(state, "restart_same_clip", False):
            if self._playing and self._current_state is state:
                return True

        # Timeline state: drives the owner transform instead of a skeletal clip.
        if getattr(state, "kind", "clip") == "timeline":
            timeline = self._resolve_timeline(state)
            if timeline is None:
                return False
            self._playback_revision += 1
            self._clear_blend_state()
            self._current_state = state
            self._current_clip = None
            self._group_clips = []
            self._group_weights = []
            self._current_timeline = timeline
            self._elapsed = 0.0
            self._prev_event_norm = 0.0
            self._playing = True
            self._capture_timeline_base()  # additive deltas apply on top of this
            self._last_timeline_pose = None
            self._apply_timeline(self._current_timeline, 0.0)
            self._sync_native_runtime_playback()
            return True

        previous_state = self._get_current_state()
        previous_clip = self._current_clip if self._playing else None
        previous_elapsed = self._elapsed
        previous_speed = self._clip_effective_speed(previous_state, previous_clip)

        group = self._resolve_group_clips(state) if state.kind == "group" else []
        clip = (group[0] if group else None) if state.kind == "group" else self._resolve_clip(state)
        if clip is None:
            Debug.log_error(
                f"[SkeletalAnimator] Refusing state '{state.name}': animation clip is unavailable"
            )
            return False
        self._playback_revision += 1
        self._current_timeline = None
        next_elapsed = 0.0
        if preserve_phase and previous_clip is not None:
            previous_duration = self._state_duration()
            next_duration = max((self._clip_duration(item) for item in group), default=self._clip_duration(clip))
            if previous_duration > 0.0 and next_duration > 0.0:
                phase = max(float(previous_elapsed), 0.0) / previous_duration
                phase = phase % 1.0 if _clip_should_loop(previous_state, previous_clip) else min(phase, 1.0)
                next_elapsed = phase * next_duration
        self._start_blend_if_needed(previous_clip, previous_elapsed, previous_speed, clip,
                                    fade_duration=fade_duration)
        self._current_state = state
        self._current_clip = clip
        self._group_clips = group
        self._group_weights = [entry.weight for entry in state.clips]
        self._elapsed = next_elapsed
        self._prev_event_norm = (next_elapsed / self._clip_duration(clip)) if next_elapsed > 0.0 else 0.0
        self._playing = True
        self._sample_imported_curves(clip, self.normalized_time)
        self._sync_native_runtime_playback()
        return True

    def _clip_effective_speed(self, state: Optional[AnimState], clip: Optional[AnimationClip3D]) -> float:
        if clip is None:
            return 1.0
        return self.playback_speed * (state.speed if state else 1.0)

    def _clip_duration(self, clip: Optional[AnimationClip3D]) -> float:
        duration = _clip_duration_hint(clip)
        if duration > 0.0 or clip is None:
            return duration
        take_name = str(getattr(clip, "take_name", "") or "")
        if not take_name:
            return 0.0
        source_guid = _animation_source_guid(clip)
        cache_key = (source_guid, take_name)
        if cache_key in self._duration_cache:
            return self._duration_cache[cache_key]
        renderers = self._resolve_native_renderers()
        if not renderers:
            return 0.0
        duration = max(float(renderers[0].get_animation_duration_seconds(take_name, source_guid)), 0.0)
        self._duration_cache[cache_key] = duration
        return duration

    def _start_blend_if_needed(
        self,
        previous_clip: Optional[AnimationClip3D],
        previous_elapsed: float,
        previous_speed: float,
        next_clip: Optional[AnimationClip3D],
        fade_duration: Optional[float] = None,
    ) -> None:
        self._clear_blend_state()
        if previous_clip is None or next_clip is None:
            return
        prev_take = str(getattr(previous_clip, "take_name", "") or "")
        next_take = str(getattr(next_clip, "take_name", "") or "")
        # Per-transition duration (AnimTransition.duration / cross_fade()) wins;
        # the component-level cross_fade_duration is only the fallback.
        if fade_duration is not None:
            duration = max(float(fade_duration), 0.0)
        else:
            duration = max(float(getattr(self, "cross_fade_duration", 0.0) or 0.0), 0.0)
        # Same-take fades are supported natively (different sample times), so
        # only an actually-missing take disables blending.
        if duration <= 0.0 or not prev_take or not next_take:
            return
        self._blend_from_clip = previous_clip
        self._blend_from_take_name = prev_take
        self._blend_from_elapsed = max(float(previous_elapsed), 0.0)
        self._blend_from_speed = float(previous_speed)
        self._blend_from_loop = _clip_should_loop(self._get_current_state(), previous_clip)
        if self._current_state is not None and self._current_state.kind == "group":
            self._blend_from_group = [(clip, weight, _clip_should_loop(self._current_state, clip))
                                      for clip, weight in zip(self._group_clips, self._group_weights)]
        self._blend_elapsed = 0.0
        self._blend_duration = duration

    def _clear_blend_state(self) -> None:
        self._blend_from_clip = None
        self._blend_from_take_name = ""
        self._blend_from_elapsed = 0.0
        self._blend_from_speed = 1.0
        self._blend_from_loop = True
        self._blend_from_group = []
        self._blend_elapsed = 0.0
        self._blend_duration = 0.0

    def _advance_blend(self, delta_time: float) -> None:
        if self._blend_from_clip is None or self._blend_duration <= 0.0:
            return
        self._blend_elapsed += max(float(delta_time), 0.0)
        self._blend_from_elapsed += max(float(delta_time), 0.0) * self._blend_from_speed
        prev_duration = self._clip_duration(self._blend_from_clip)
        if not self._blend_from_group and prev_duration > 0.0 and self._blend_from_elapsed >= prev_duration:
            if self._blend_from_loop:
                self._blend_from_elapsed %= prev_duration
            else:
                self._blend_from_elapsed = prev_duration
        if self._blend_elapsed >= self._blend_duration:
            self._clear_blend_state()

    def _resolve_native_renderers(self, *, force: bool = False):
        if not force and self._native_renderers:
            return self._native_renderers

        self._skinned_renderers = []
        self._native_renderers = []
        try:
            owner = self.game_object
        except (ReferenceError, RuntimeError):
            return []
        if owner is None:
            return []

        native = []
        pending = deque([owner])
        while pending:
            owner = pending.popleft()
            renderer = owner.get_component(SkinnedMeshRenderer)
            if renderer is not None:
                cpp = renderer._get_bound_native_component()
                if cpp is not None:
                    self._skinned_renderers.append(renderer)
                    native.append(cpp)
            pending.extend(owner.get_children())
        self._native_renderers = native
        return native

    def _sync_native_runtime_playback(self) -> None:
        if not self.native_pose_submission:
            return
        native_renderers = self._resolve_native_renderers()
        if not native_renderers:
            return
        state = self._get_current_state()
        if (state is not None and state.kind == "group") or self._blend_from_group:
            layers = self.get_pose_layers()
            pose_key = ("group", self._playing, tuple(
                (layer["take_name"], layer["source_model_guid"], layer["time"], layer["weight"],
                 layer["loop"], tuple(layer["bone_mask"]), layer["blend_group"], layer["group_weight"])
                for layer in layers))
            if pose_key != self._last_native_pose_key:
                for cpp in native_renderers:
                    cpp.submit_pose_stack(layers)
                self._last_native_pose_key = pose_key
                self._last_native_take_name = self.current_take_name
            return
        # Continuous playback has one current native submission path. Explicit
        # runtime_animation_time assignment is reserved for discontinuous seek.
        has_clip = self._current_clip is not None and bool(self.current_take_name)
        take_name = self.current_take_name if has_clip else ""
        source_guid = _animation_source_guid(self._current_clip) if take_name else ""
        loop = _clip_should_loop(state, self._current_clip)
        normalized = float(self.normalized_time) if take_name else 0.0
        blend_take = ""
        blend_time = 0.0
        blend_weight = 0.0
        if self._blend_from_clip is not None and self._blend_duration > 0.0:
            progress = min(max(self._blend_elapsed / self._blend_duration, 0.0), 1.0)
            blend_take = self._blend_from_take_name
            blend_time = float(self._blend_from_elapsed)
            blend_weight = float(1.0 - progress)
        blend_source_guid = _animation_source_guid(self._blend_from_clip) if blend_take else ""
        current_mask = _clip_bone_mask(self._current_clip)
        blend_mask = _clip_bone_mask(self._blend_from_clip) if blend_take else []
        pose_key = (
            "clip", self._playing, take_name, float(self._elapsed), normalized,
            blend_take, blend_time, blend_weight, loop, self._blend_from_loop, source_guid, blend_source_guid,
            tuple(current_mask), tuple(blend_mask),
        )
        if pose_key == self._last_native_pose_key:
            return
        for cpp in native_renderers:
            if blend_take or current_mask or blend_mask:
                layers = []
                if blend_take and blend_weight > 0.0:
                    layers.append({"take_name": blend_take, "source_model_guid": blend_source_guid,
                                   "time": blend_time, "weight": blend_weight, "loop": self._blend_from_loop,
                                   "bone_mask": blend_mask})
                if take_name:
                    layers.append({"take_name": take_name, "source_model_guid": source_guid,
                                   "time": float(self._elapsed), "weight": 1.0 - blend_weight,
                                   "loop": loop, "bone_mask": current_mask})
                cpp.submit_pose_stack(layers)
            else:
                cpp.submit_animation_pose(
                    take_name,
                    float(self._elapsed) if take_name else 0.0,
                    normalized,
                    blend_take,
                    blend_time,
                    blend_weight,
                    loop,
                    source_guid,
                    blend_source_guid,
                )
        self._last_native_take_name = take_name
        self._last_native_pose_key = pose_key

    def _get_current_state(self) -> Optional[AnimState]:
        return self._current_state

    def _exit_time_gate_ok(self, state: AnimState) -> bool:
        if self._current_timeline is not None:
            dur = max(1e-6, float(self._current_timeline.duration))
            thr = max(0.0, min(1.0, float(getattr(state, "exit_time_normalized", 1.0))))
            progress = min(max(self._elapsed / dur, 0.0), 1.0)
            return progress + 1e-7 >= thr
        duration = self._state_duration()
        if duration <= 0.0:
            return True
        thr = float(getattr(state, "exit_time_normalized", 1.0))
        thr = max(0.0, min(1.0, thr))
        progress = min(max(self._elapsed / duration, 0.0), 1.0)
        return progress + 1e-7 >= thr

    def _try_auto_transition(self):
        state = self._get_current_state()
        if not state or not state.transitions:
            return
        if not self._exit_time_gate_ok(state):
            return
        for tr in state.transitions:
            if self._evaluate_condition(tr):
                # AnimTransition.duration (authored in the FSM editor) drives
                # this specific fade; <= 0 falls back to cross_fade_duration.
                tr_duration = float(getattr(tr, "duration", 0.0) or 0.0)
                entered = self._enter_state(tr.target_state,
                                  fade_duration=tr_duration if tr_duration > 0.0 else None,
                                  preserve_phase=bool(getattr(
                                      tr, "synchronize_normalized_time", False,
                                  )))
                if entered:
                    self._consume_triggers(tr)
                    return

    def _evaluate_condition(self, transition: AnimTransition) -> bool:
        if not transition.conditions:
            state = self._get_current_state()
            if self._current_timeline is not None:
                return bool(state and not state.loop and self._elapsed >= self._current_timeline.duration)
            duration = self._state_duration()
            if duration <= 0.0:
                return False
            should_loop = state.loop if state is not None and state.kind == "group" else _clip_should_loop(state, self._current_clip)
            if self._current_clip and not should_loop:
                return self._elapsed >= duration
            return False
        return bool(
            self._fsm
            and self._fsm.evaluate_transition_conditions(
                transition, self._parameters
            )
        )

    def _consume_triggers(self, transition: AnimTransition):
        if not self._pending_triggers or self._fsm is None:
            return
        consumed = self._pending_triggers.intersection(self._fsm.transition_parameter_names(transition))
        for name in consumed:
            self._parameters[name] = False
        self._pending_triggers -= consumed
