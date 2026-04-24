"""
Animator3D — runtime 3D animation state machine controller.

This is the first structural bridge from `.animfsm` / `.animclip3d` authoring
to animated model instances. It currently drives a `SkinnedMeshRenderer` by
selecting the active take name and maintaining FSM/runtime state, while future
native milestones will add real clip sampling / skinning playback.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from Infernux.components.component import InxComponent
from Infernux.components.serialized_field import serialized_field
from Infernux.components.decorators import require_component, disallow_multiple, add_component_menu
from Infernux.components.builtin.skinned_mesh_renderer import SkinnedMeshRenderer
from Infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition
from Infernux.core.animation_clip3d import AnimationClip3D
from Infernux.core.asset_ref import AnimStateMachineRef
from Infernux.debug import Debug


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
                path = db.get_path_from_guid(state.clip_guid)
                if path and os.path.isfile(path):
                    return path
            except Exception:
                pass
    if state.clip_path and os.path.isfile(state.clip_path):
        return state.clip_path
    return None


def _clip_duration_hint(clip: Optional[AnimationClip3D]) -> float:
    if clip is None:
        return 0.0
    try:
        return max(float(getattr(clip, "duration_hint", 0.0) or 0.0), 0.0)
    except Exception:
        return 0.0


@require_component(SkinnedMeshRenderer)
@disallow_multiple
@add_component_menu("Animation/Animator 3D")
class Animator3D(InxComponent):
    """Runtime controller that drives a SkinnedMeshRenderer from a 3D AnimStateMachine."""

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

    _parameters: Dict[str, object] = {}

    _fsm: Optional[AnimStateMachine] = None
    _skinned_renderer: Optional[SkinnedMeshRenderer] = None
    _clip_cache: Dict[str, Optional[AnimationClip3D]] = {}

    _current_state_name: str = ""
    _current_clip: Optional[AnimationClip3D] = None
    _elapsed: float = 0.0
    _playing: bool = False

    def awake(self):
        self._parameters = {}
        self._clip_cache = {}
        self._current_state_name = ""
        self._current_clip = None
        self._elapsed = 0.0
        self._playing = False

    def start(self):
        self._skinned_renderer = self.game_object.get_component(SkinnedMeshRenderer)
        if not self._skinned_renderer:
            Debug.log_warning("[Animator3D] No SkinnedMeshRenderer found on this GameObject.")
            return

        self._load_controller()

        if self.auto_play and self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def update(self, delta_time: float):
        if not self._playing or not self._current_clip:
            return

        state = self._get_current_state()
        clip = self._current_clip
        speed = self.playback_speed * (state.speed if state else 1.0) * float(getattr(clip, "speed", 1.0) or 1.0)
        self._elapsed += delta_time * speed

        duration = _clip_duration_hint(clip)
        if duration > 0.0 and self._elapsed >= duration:
            should_loop = state.loop if state else bool(getattr(clip, "loop", True))
            if should_loop:
                self._try_auto_transition()
                self._elapsed %= duration
            else:
                self._elapsed = duration
                self._playing = False
                self._try_auto_transition()
                return

        self._apply_active_take()
        self._try_auto_transition()

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
        duration = _clip_duration_hint(self._current_clip)
        if duration > 0.0:
            return min(self._elapsed / duration, 1.0)
        return 0.0

    def play(self, state_name: str = "") -> bool:
        if not self._fsm:
            return False
        name = state_name or self._fsm.default_state
        if not name:
            return False
        return self._enter_state(name)

    def stop(self):
        self._playing = False

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
        self._parameters = {}

    def _load_controller(self):
        self._fsm = None
        self._clip_cache = {}

        fsm = self.controller
        if fsm is None:
            return

        if fsm.mode != "3d":
            Debug.log_warning(f"[Animator3D] Controller is mode='{fsm.mode}', expected '3d'.")
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
                Debug.log_warning(f"[Animator3D] Failed to load clip for state '{state.name}': {clip_path}")
        else:
            if state.clip_guid or state.clip_path:
                Debug.log_warning(
                    f"[Animator3D] Clip not found for state '{state.name}' "
                    f"(guid='{state.clip_guid}', path='{state.clip_path}')"
                )
        self._clip_cache[key] = clip
        return clip

    def _enter_state(self, state_name: str) -> bool:
        if not self._fsm:
            return False
        state = self._fsm.get_state(state_name)
        if state is None:
            Debug.log_warning(f"[Animator3D] State not found: '{state_name}'")
            return False

        if not getattr(state, "restart_same_clip", False):
            if self._playing and self._current_state_name == state_name:
                return True

        clip = self._resolve_clip(state)
        self._current_state_name = state_name
        self._current_clip = clip
        self._elapsed = 0.0
        self._playing = True
        self._apply_active_take()
        return True

    def _apply_active_take(self):
        if not self._skinned_renderer:
            return
        take_name = ""
        clip = self._current_clip
        if clip is not None:
            take_name = str(getattr(clip, "take_name", "") or "")

            renderer_guid = str(getattr(self._skinned_renderer, "source_model_guid", "") or "")
            renderer_path = str(getattr(self._skinned_renderer, "source_model_path", "") or "")
            clip_guid = str(getattr(clip, "source_model_guid", "") or "")
            clip_path = str(getattr(clip, "source_model_path", "") or "")
            if clip_guid and renderer_guid and clip_guid != renderer_guid:
                Debug.log_warning(
                    f"[Animator3D] Clip source GUID '{clip_guid}' does not match renderer source '{renderer_guid}'."
                )
            elif clip_path and renderer_path and os.path.normpath(clip_path) != os.path.normpath(renderer_path):
                Debug.log_warning(
                    f"[Animator3D] Clip source path '{clip_path}' does not match renderer source '{renderer_path}'."
                )

        self._skinned_renderer.active_take_name = take_name

    def _get_current_state(self) -> Optional[AnimState]:
        if self._fsm and self._current_state_name:
            return self._fsm.get_state(self._current_state_name)
        return None

    def _exit_time_gate_ok(self, state: AnimState) -> bool:
        duration = _clip_duration_hint(self._current_clip)
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
            duration = _clip_duration_hint(self._current_clip)
            if duration <= 0.0:
                return False
            state = self._get_current_state()
            should_loop = state.loop if state else bool(getattr(self._current_clip, "loop", True))
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
                f"[Animator3D] Condition eval error in '{self._current_state_name}': "
                f"'{cond}' -> {exc}"
            )
            return False

    def _consume_triggers(self, condition: str):
        for name, val in list(self._parameters.items()):
            if val is True and name in condition:
                self._parameters[name] = False
