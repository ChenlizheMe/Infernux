"""
SkeletalAnimator — runtime 3D animation state machine controller.

Mirrors :class:`SpiritAnimator` (2D) for skeletal assets: bridge from
``.animfsm`` / ``.animclip3d`` to :class:`SkinnedMeshRenderer`, advancing FSM
state and pushing playback time to native code for an upcoming skinning path.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from Infernux.components.component import InxComponent
from Infernux.components.serialized_field import serialized_field
from Infernux.components.decorators import require_component, disallow_multiple, add_component_menu
from Infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
from Infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition
from Infernux.core.animation_clip3d import AnimationClip3D, resolve_disk_path_for_guid_string
from Infernux.core.asset_ref import AnimStateMachineRef
from Infernux.debug import Debug


def _normalize_guid_key(s: str) -> str:
    """Case-insensitive, hyphen-insensitive key for comparing asset GUIDs."""
    return (s or "").replace("-", "").strip().lower()


def _try_guid_for_model_path(db, path: str) -> str:
    """Resolve a model file path to its asset GUID via the database, if available."""
    if not db or not (path or "").strip():
        return ""
    p0 = str(path).strip()
    if not p0:
        return ""
    seen = set()
    cands: list = []
    for p in (p0, os.path.normpath(p0), os.path.normpath(os.path.abspath(p0))):
        if p and p not in seen:
            seen.add(p)
            cands.append(p)
    for p in cands:
        try:
            g = db.get_guid_from_path(p)
            if g and str(g).strip():
                return str(g).strip()
        except Exception:
            pass
    return ""


def _model_keys_for_source(db, source_guid: str, source_path: str) -> tuple:
    """
    Return (key, had_explicit_guid) where key is the normalized 32-hex id when known.
    Uses serialized GUID first, then path→GUID from the database.
    """
    g = (source_guid or "").strip()
    k = _normalize_guid_key(g)
    if k:
        return (k, True)
    g2 = _try_guid_for_model_path(db, source_path) if db else ""
    k2 = _normalize_guid_key(g2)
    if k2:
        return (k2, False)
    return ("", False)


def _skinned_mismatch_message(
    db,
    clip: AnimationClip3D,
    source_model_guid: str,
    source_model_path: str,
) -> str:
    """
    If clip and renderer are definitely different model assets, return a warning string.
    Prefers GUID and path→GUID; does not use raw path comparison when both GUID keys match.
    """
    p_clip = (getattr(clip, "source_model_path", "") or "").strip()
    g_clip = (getattr(clip, "source_model_guid", "") or "").strip()
    p_rend = (source_model_path or "").strip()
    g_rend = (source_model_guid or "").strip()

    k_clip, _ = _model_keys_for_source(db, g_clip, p_clip)
    k_rend, _ = _model_keys_for_source(db, g_rend, p_rend)

    if k_clip and k_rend and k_clip != k_rend:
        return (
            f"[SkeletalAnimator] Clip source model does not match SkinnedMeshRenderer "
            f"(clip guid≈{g_clip!r} path='{p_clip}' vs "
            f"renderer guid≈{g_rend!r} path='{p_rend}')."
        )
    if k_clip and k_rend:
        return ""

    if k_clip or k_rend:
        return ""

    p_clip_n = os.path.normcase(os.path.normpath(p_clip)) if p_clip else ""
    p_rend_n = os.path.normcase(os.path.normpath(p_rend)) if p_rend else ""
    if p_clip and p_rend and p_clip_n and p_rend_n and p_clip_n != p_rend_n:
        return (
            f"[SkeletalAnimator] Clip source path does not match renderer source, and "
            f"neither could be resolved to a GUID. clip='{p_clip}' vs renderer='{p_rend}'"
        )
    return ""


def _get_asset_database():
    try:
        from Infernux.core.assets import AssetManager
        if AssetManager._asset_database is not None:
            return AssetManager._asset_database
    except ImportError:
        pass
    try:
        from Infernux.engine.play_mode import PlayModeManager
        pm = PlayModeManager.instance()
        if pm and pm._asset_database is not None:
            return pm._asset_database
    except ImportError:
        pass
    return None


def _resolve_clip_path(state: AnimState) -> Optional[str]:
    if state.clip_guid:
        db = _get_asset_database()
        if db:
            try:
                path = resolve_disk_path_for_guid_string(db, state.clip_guid)
                if path:
                    return path
            except Exception:
                pass
    raw = (state.clip_path or "").strip()
    # Project panel: embedded FBX take as "<guid>::subanim:<index>" (not a file path).
    if raw and "::subanim:" in raw:
        return raw
    if raw and os.path.isfile(raw):
        return raw
    return None


def _clip_duration_hint(clip: Optional[AnimationClip3D]) -> float:
    if clip is None:
        return 0.0
    try:
        return max(float(getattr(clip, "duration_hint", 0.0) or 0.0), 0.0)
    except Exception:
        return 0.0


# When importer/meta leaves duration unknown (e.g. embedded FBX takes), use this for
# normalized_time and looping so native/runtime hooks see monotonic [0,1) progress.
_DEFAULT_PLAYBACK_SEC_WHEN_UNKNOWN_DURATION = 1.0


@require_component(SkinnedMeshRenderer)
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

    _parameters: Dict[str, object] = {}

    _fsm: Optional[AnimStateMachine] = None
    _skinned_renderer: Optional[SkinnedMeshRenderer] = None
    _clip_cache: Dict[str, Optional[AnimationClip3D]] = {}

    _current_state_name: str = ""
    _current_clip: Optional[AnimationClip3D] = None
    _elapsed: float = 0.0
    _playing: bool = False
    _blend_from_clip: Optional[AnimationClip3D] = None
    _blend_from_take_name: str = ""
    _blend_from_elapsed: float = 0.0
    _blend_from_speed: float = 1.0
    _blend_elapsed: float = 0.0
    _blend_duration: float = 0.0
    _last_native_take_name: str = ""
    _duration_cache: Dict[str, float] = {}

    def awake(self):
        self._parameters = {}
        self._clip_cache = {}
        self._duration_cache = {}
        self._last_native_take_name = ""
        self._current_state_name = ""
        self._current_clip = None
        self._elapsed = 0.0
        self._playing = False
        self._clear_blend_state()

    def start(self):
        self._skinned_renderer = self.game_object.get_component(SkinnedMeshRenderer)
        if not self._skinned_renderer:
            Debug.log_warning("[SkeletalAnimator] No SkinnedMeshRenderer found on this GameObject.")
            return

        self._load_controller()

        if self.auto_play and self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def update(self, delta_time: float):
        if not self._playing or not self._current_clip:
            self._sync_native_runtime_playback()
            return

        state = self._get_current_state()
        clip = self._current_clip
        speed = self.playback_speed * (state.speed if state else 1.0)
        self._elapsed += delta_time * speed
        self._advance_blend(delta_time)

        duration = self._clip_duration(clip)
        if duration > 0.0 and self._elapsed >= duration:
            should_loop = state.loop if state else True
            if should_loop:
                self._try_auto_transition()
                self._elapsed %= duration
            else:
                self._elapsed = duration
                self._playing = False
                self._try_auto_transition()
                self._sync_native_runtime_playback()
                return

        self._apply_active_take()
        self._try_auto_transition()
        self._sync_native_runtime_playback()

    @property
    def current_state(self) -> str:
        return self._current_state_name

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
        duration = self._clip_duration(self._current_clip)
        if duration > 0.0:
            return min(self._elapsed / duration, 1.0)
        # No duration in asset — assume a neutral loop period so time/normalized are not stuck.
        t = _DEFAULT_PLAYBACK_SEC_WHEN_UNKNOWN_DURATION
        return (self._elapsed % t) / t

    def play(self, state_name: str = "") -> bool:
        if not self._fsm:
            return False
        name = state_name or self._fsm.default_state
        if not name:
            return False
        return self._enter_state(name)

    def stop(self):
        self._playing = False
        self._clear_blend_state()
        self._sync_native_runtime_playback()

    def set_parameter(self, name: str, value: object):
        self._parameters[name] = value

    def get_parameter(self, name: str, default: object = None) -> object:
        return self._parameters.get(name, default)

    def get_bool(self, name: str) -> bool:
        return bool(self._parameters.get(name, False))

    def set_bool(self, name: str, value: bool):
        self._parameters[name] = bool(value)

    def get_float(self, name: str) -> float:
        return float(self._parameters.get(name, 0.0))

    def set_float(self, name: str, value: float):
        self._parameters[name] = float(value)

    def get_int(self, name: str) -> int:
        return int(self._parameters.get(name, 0))

    def set_int(self, name: str, value: int):
        self._parameters[name] = int(value)

    def set_trigger(self, name: str):
        self._parameters[name] = True

    def reload_controller(self):
        self._load_controller()
        if self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def on_after_deserialize(self):
        self._clip_cache = {}
        self._duration_cache = {}
        self._last_native_take_name = ""
        self._parameters = {}
        self._clear_blend_state()

    def _load_controller(self):
        self._fsm = None
        self._clip_cache = {}
        self._duration_cache = {}

        fsm = self.controller
        if fsm is None:
            return

        if fsm.mode != "3d":
            Debug.log_warning(f"[SkeletalAnimator] Controller is mode='{fsm.mode}', expected '3d'.")
        self._fsm = fsm
        self._seed_parameters_from_fsm(fsm)
        for state in fsm.states:
            self._resolve_clip(state)

    def _seed_parameters_from_fsm(self, fsm: AnimStateMachine) -> None:
        self._parameters = {}
        for p in fsm.parameters:
            if p.kind == "bool":
                self._parameters[p.name] = bool(p.default_bool)
            elif p.kind == "int":
                self._parameters[p.name] = int(p.default_int)
            else:
                self._parameters[p.name] = float(p.default_float)

    def _resolve_clip(self, state: AnimState) -> Optional[AnimationClip3D]:
        key = state.name
        if key in self._clip_cache:
            return self._clip_cache[key]

        clip_path = _resolve_clip_path(state)
        clip = None
        if clip_path:
            clip = AnimationClip3D.load(clip_path)
            if clip is None:
                Debug.log_warning(f"[SkeletalAnimator] Failed to load clip for state '{state.name}': {clip_path}")
        else:
            if state.clip_guid or state.clip_path:
                Debug.log_warning(
                    f"[SkeletalAnimator] Clip not found for state '{state.name}' "
                    f"(guid='{state.clip_guid}', path='{state.clip_path}')"
                )
        self._clip_cache[key] = clip
        return clip

    def _enter_state(self, state_name: str) -> bool:
        if not self._fsm:
            return False
        state = self._fsm.get_state(state_name)
        if state is None:
            Debug.log_warning(f"[SkeletalAnimator] State not found: '{state_name}'")
            return False

        if not getattr(state, "restart_same_clip", False):
            if self._playing and self._current_state_name == state_name:
                return True

        previous_state = self._get_current_state()
        previous_clip = self._current_clip if self._playing else None
        previous_elapsed = self._elapsed
        previous_speed = self._clip_effective_speed(previous_state, previous_clip)

        clip = self._resolve_clip(state)
        self._start_blend_if_needed(previous_clip, previous_elapsed, previous_speed, clip)
        self._current_state_name = state_name
        self._current_clip = clip
        self._elapsed = 0.0
        self._playing = True
        self._apply_active_take()
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
        r = self._skinned_renderer
        cpp = getattr(r, "_cpp_component", None) if r else None
        if cpp is None:
            return 0.0
        take_name = str(getattr(clip, "take_name", "") or "")
        if not take_name:
            return 0.0
        if take_name in self._duration_cache:
            return self._duration_cache[take_name]
        try:
            get_duration = getattr(cpp, "get_animation_duration_seconds", None)
            if callable(get_duration):
                duration = max(float(get_duration(take_name) or 0.0), 0.0)
                self._duration_cache[take_name] = duration
                return duration
        except Exception:
            return 0.0
        return 0.0

    def _start_blend_if_needed(
        self,
        previous_clip: Optional[AnimationClip3D],
        previous_elapsed: float,
        previous_speed: float,
        next_clip: Optional[AnimationClip3D],
    ) -> None:
        self._clear_blend_state()
        if previous_clip is None or next_clip is None:
            return
        prev_take = str(getattr(previous_clip, "take_name", "") or "")
        next_take = str(getattr(next_clip, "take_name", "") or "")
        duration = max(float(getattr(self, "cross_fade_duration", 0.0) or 0.0), 0.0)
        if duration <= 0.0 or not prev_take or not next_take or prev_take == next_take:
            return
        self._blend_from_clip = previous_clip
        self._blend_from_take_name = prev_take
        self._blend_from_elapsed = max(float(previous_elapsed), 0.0)
        self._blend_from_speed = float(previous_speed)
        self._blend_elapsed = 0.0
        self._blend_duration = duration

    def _clear_blend_state(self) -> None:
        self._blend_from_clip = None
        self._blend_from_take_name = ""
        self._blend_from_elapsed = 0.0
        self._blend_from_speed = 1.0
        self._blend_elapsed = 0.0
        self._blend_duration = 0.0

    def _advance_blend(self, delta_time: float) -> None:
        if self._blend_from_clip is None or self._blend_duration <= 0.0:
            return
        self._blend_elapsed += max(float(delta_time), 0.0)
        self._blend_from_elapsed += max(float(delta_time), 0.0) * self._blend_from_speed
        prev_duration = self._clip_duration(self._blend_from_clip)
        if prev_duration > 0.0 and self._blend_from_elapsed >= prev_duration:
            self._blend_from_elapsed %= prev_duration
        if self._blend_elapsed >= self._blend_duration:
            self._clear_blend_state()

    def _apply_active_take(self):
        if not self._skinned_renderer:
            return
        take_name = ""
        clip = self._current_clip
        if clip is not None:
            take_name = str(getattr(clip, "take_name", "") or "")
            if take_name == self._last_native_take_name:
                return

            r = self._skinned_renderer
            renderer_guid = str(getattr(r, "source_model_guid", "") or "")
            renderer_path = str(getattr(r, "source_model_path", "") or "")
            db = _get_asset_database()
            msg = _skinned_mismatch_message(db, clip, renderer_guid, renderer_path)
            if msg:
                Debug.log_warning(msg)

        self._skinned_renderer.active_take_name = take_name
        self._last_native_take_name = take_name

    def _sync_native_runtime_playback(self) -> None:
        r = self._skinned_renderer
        if not r:
            return
        cpp = getattr(r, "_cpp_component", None)
        if cpp is None:
            return
        submit_pose = getattr(cpp, "submit_animation_pose", None)
        if callable(submit_pose):
            take_name = self.current_take_name if self._playing and self._current_clip is not None else ""
            normalized = float(self.normalized_time) if take_name else 0.0
            blend_take = ""
            blend_time = 0.0
            blend_weight = 0.0
            if self._blend_from_clip is not None and self._blend_duration > 0.0:
                progress = min(max(self._blend_elapsed / self._blend_duration, 0.0), 1.0)
                blend_take = self._blend_from_take_name
                blend_time = float(self._blend_from_elapsed)
                blend_weight = float(1.0 - progress)
            submit_pose(
                take_name,
                float(self._elapsed) if take_name else 0.0,
                normalized,
                blend_take,
                blend_time,
                blend_weight,
            )
            self._last_native_take_name = take_name
            return

        if self._playing and self._current_clip is not None:
            cpp.runtime_animation_time = float(self._elapsed)
            cpp.runtime_animation_normalized_time = float(self.normalized_time)
            if self._blend_from_clip is not None and self._blend_duration > 0.0:
                progress = min(max(self._blend_elapsed / self._blend_duration, 0.0), 1.0)
                cpp.blend_take_name = self._blend_from_take_name
                cpp.blend_animation_time = float(self._blend_from_elapsed)
                cpp.blend_weight = float(1.0 - progress)
            else:
                clear = getattr(cpp, "clear_animation_blend", None)
                if callable(clear):
                    clear()
        else:
            cpp.runtime_animation_time = 0.0
            cpp.runtime_animation_normalized_time = 0.0
            clear = getattr(cpp, "clear_animation_blend", None)
            if callable(clear):
                clear()

    def _get_current_state(self) -> Optional[AnimState]:
        if self._fsm and self._current_state_name:
            return self._fsm.get_state(self._current_state_name)
        return None

    def _exit_time_gate_ok(self, state: AnimState) -> bool:
        duration = self._clip_duration(self._current_clip)
        if duration <= 0.0:
            return True
        thr = float(getattr(state, "exit_time_normalized", 1.0))
        thr = max(0.0, min(1.0, thr))
        progress = min(max(self._elapsed / duration, 0.0), 1.0)
        return progress + 1e-7 >= thr

    def _try_auto_transition(self):
        state = self._get_current_state()
        if not state:
            return
        if not self._exit_time_gate_ok(state):
            return
        for tr in state.transitions:
            if self._evaluate_condition(tr):
                self._consume_triggers(tr.condition)
                self._enter_state(tr.target_state)
                return

    def _evaluate_condition(self, transition: AnimTransition) -> bool:
        cond = transition.condition.strip()

        if not cond:
            duration = self._clip_duration(self._current_clip)
            if duration <= 0.0:
                return False
            state = self._get_current_state()
            should_loop = state.loop if state else True
            if self._current_clip and not should_loop:
                return self._elapsed >= duration
            return False

        ctx = dict(self._parameters)
        ctx["time"] = self._elapsed
        ctx["normalized_time"] = self.normalized_time
        ctx["state"] = self._current_state_name

        try:
            return bool(eval(cond, {"__builtins__": {}}, ctx))  # noqa: S307
        except Exception as exc:
            Debug.log_warning(
                f"[SkeletalAnimator] Condition eval error in '{self._current_state_name}': "
                f"'{cond}' -> {exc}"
            )
            return False

    def _consume_triggers(self, condition: str):
        for name, val in list(self._parameters.items()):
            if val is True and name in condition:
                self._parameters[name] = False
