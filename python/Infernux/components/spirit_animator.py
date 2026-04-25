"""
SpiritAnimator — runtime 2D animation state machine controller.

Drives a :class:`SpriteRenderer` by evaluating an :class:`AnimStateMachine`
every frame.  Loads the FSM from a ``.animfsm`` file, resolves each state's
animation clip, and advances the current clip's frame index on the
SpriteRenderer.

Usage::

    animator = game_object.add_component(SpiritAnimator)
    animator.controller = AnimStateMachineRef(path_hint="Assets/Animations/player.animfsm")
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from Infernux.components.component import InxComponent
from Infernux.components.serialized_field import serialized_field, FieldType
from Infernux.components.decorators import require_component, disallow_multiple, add_component_menu
from Infernux.components.builtin.sprite_renderer import SpriteRenderer
from Infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition
from Infernux.core.animation_clip import AnimationClip
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
    """Resolve an AnimState's clip reference to an absolute file path."""
    # Try GUID first
    if state.clip_guid:
        db = _get_asset_database()
        if db:
            try:
                path = db.get_path_from_guid(state.clip_guid)
                if path and os.path.isfile(path):
                    return path
            except Exception:
                pass
    # Fallback to stored path
    if state.clip_path and os.path.isfile(state.clip_path):
        return state.clip_path
    return None


@require_component(SpriteRenderer)
@disallow_multiple
@add_component_menu("Animation/Spirit Animator")
class SpiritAnimator(InxComponent):
    """Runtime controller that drives a SpriteRenderer from a 2D AnimStateMachine."""

    # ── Serialized fields (shown in Inspector) ──────────────────────

    controller: AnimStateMachineRef = serialized_field(
        default=None,
        asset_type="AnimStateMachine",
        tooltip="2D AnimStateMachine controller (.animfsm)",
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

    # ── Runtime parameters (user-settable, used in conditions) ──────

    _parameters: Dict[str, object] = {}

    # ── Private runtime state ───────────────────────────────────────

    _fsm: Optional[AnimStateMachine] = None
    _sprite_renderer: Optional[SpriteRenderer] = None
    _clip_cache: Dict[str, Optional[AnimationClip]] = {}

    _current_state_name: str = ""
    _current_clip: Optional[AnimationClip] = None
    _elapsed: float = 0.0
    _playing: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────

    def awake(self):
        self._parameters = {}
        self._clip_cache = {}
        self._current_state_name = ""
        self._current_clip = None
        self._elapsed = 0.0
        self._playing = False

    def start(self):
        self._sprite_renderer = self.game_object.get_component(SpriteRenderer)
        if not self._sprite_renderer:
            Debug.log_warning("[SpiritAnimator] No SpriteRenderer found on this GameObject.")
            return

        self._load_controller()

        if self.auto_play and self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    def update(self, delta_time: float):
        if not self._playing or not self._current_clip or not self._sprite_renderer:
            return

        state = self._get_current_state()
        speed = self.playback_speed * (state.speed if state else 1.0)

        # Advance elapsed time
        self._elapsed += delta_time * speed

        clip = self._current_clip
        if clip.fps <= 0 or clip.frame_count == 0:
            return

        duration = clip.duration

        # Handle looping / clip end
        if self._elapsed >= duration:
            state = self._get_current_state()
            should_loop = state.loop if state else clip.loop
            if should_loop:
                # Evaluate transitions at loop boundary while elapsed >= duration (progress 1.0)
                self._try_auto_transition()
                self._elapsed %= duration
            else:
                self._elapsed = duration
                self._playing = False
                self._try_auto_transition()
                return

        # Compute and apply frame index
        raw_frame = int(self._elapsed * clip.fps)
        raw_frame = min(raw_frame, clip.frame_count - 1)
        sprite_frame = clip.frame_indices[raw_frame]
        self._sprite_renderer.frame_index = sprite_frame
        self._sprite_renderer.sync_visual()

        # Check transitions every frame (for condition-driven transitions)
        self._try_auto_transition()

    # ── Public API ──────────────────────────────────────────────────

    @property
    def current_state(self) -> str:
        """Name of the active FSM state."""
        return self._current_state_name

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def normalized_time(self) -> float:
        """Current playback position in [0, 1]."""
        if self._current_clip and self._current_clip.duration > 0:
            return min(self._elapsed / self._current_clip.duration, 1.0)
        return 0.0

    def play(self, state_name: str = "") -> bool:
        """Transition immediately to *state_name* (or default state)."""
        if not self._fsm:
            return False
        name = state_name or self._fsm.default_state
        if not name:
            return False
        return self._enter_state(name)

    def stop(self):
        """Stop playback.  The current frame stays on screen."""
        self._playing = False

    def set_parameter(self, name: str, value: object):
        """Set a named parameter that transition conditions can reference."""
        self._parameters[name] = value

    def get_parameter(self, name: str, default: object = None) -> object:
        """Get a named parameter value."""
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
        """Set a trigger parameter (auto-clears after consumed by a transition)."""
        self._parameters[name] = True

    def reload_controller(self):
        """Force-reload the FSM from disk."""
        self._load_controller()
        if self._fsm and self._fsm.default_state:
            self.play(self._fsm.default_state)

    # ── Serialization hooks ─────────────────────────────────────────

    def on_after_deserialize(self):
        self._clip_cache = {}
        self._parameters = {}

    # ── Internals ───────────────────────────────────────────────────

    def _load_controller(self):
        """Load the AnimStateMachine from the *controller* asset reference."""
        self._fsm = None
        self._clip_cache = {}

        # self.controller auto-resolves the AnimStateMachineRef via the
        # descriptor, so *fsm* is already the loaded AnimStateMachine (or None).
        fsm = self.controller
        if fsm is None:
            return

        if fsm.mode != "2d":
            Debug.log_warning(
                f"[SpiritAnimator] Controller is mode='{fsm.mode}', expected '2d'."
            )
        self._fsm = fsm
        self._seed_parameters_from_fsm(fsm)
        # Pre-cache all clips
        for state in fsm.states:
            self._resolve_clip(state)

    def _seed_parameters_from_fsm(self, fsm: AnimStateMachine) -> None:
        """Expose FSM parameter defaults in condition eval (``eval`` ctx)."""
        self._parameters = {}
        for p in fsm.parameters:
            if p.kind == "bool":
                self._parameters[p.name] = bool(p.default_bool)
            elif p.kind == "int":
                self._parameters[p.name] = int(p.default_int)
            else:
                self._parameters[p.name] = float(p.default_float)

    def _resolve_clip(self, state: AnimState) -> Optional[AnimationClip]:
        """Resolve and cache the AnimationClip for an FSM state."""
        key = state.name
        if key in self._clip_cache:
            return self._clip_cache[key]

        clip_path = _resolve_clip_path(state)
        clip = None
        if clip_path:
            clip = AnimationClip.load(clip_path)
            if clip is None:
                Debug.log_warning(
                    f"[SpiritAnimator] Failed to load clip for state '{state.name}': {clip_path}"
                )
        else:
            if state.clip_guid or state.clip_path:
                Debug.log_warning(
                    f"[SpiritAnimator] Clip not found for state '{state.name}' "
                    f"(guid='{state.clip_guid}', path='{state.clip_path}')"
                )
        self._clip_cache[key] = clip
        return clip

    def _enter_state(self, state_name: str) -> bool:
        """Enter a state: load its clip and reset playback."""
        if not self._fsm:
            return False
        state = self._fsm.get_state(state_name)
        if state is None:
            Debug.log_warning(f"[SpiritAnimator] State not found: '{state_name}'")
            return False

        if not getattr(state, "restart_same_clip", False):
            if self._playing and self._current_state_name == state_name:
                return True

        clip = self._resolve_clip(state)
        self._current_state_name = state_name
        self._current_clip = clip
        self._elapsed = 0.0
        self._playing = True

        # Apply first frame immediately
        if clip and clip.frame_count > 0 and self._sprite_renderer:
            self._sprite_renderer.frame_index = clip.frame_indices[0]
            self._sprite_renderer.sync_visual()

        return True

    def _get_current_state(self) -> Optional[AnimState]:
        if self._fsm and self._current_state_name:
            return self._fsm.get_state(self._current_state_name)
        return None

    def _exit_time_gate_ok(self, state: AnimState) -> bool:
        """Require normalized clip progress >= state's exit_time before any outgoing transition."""
        if not self._current_clip or self._current_clip.duration <= 0:
            return True
        thr = float(getattr(state, "exit_time_normalized", 1.0))
        thr = max(0.0, min(1.0, thr))
        d = self._current_clip.duration
        progress = min(max(self._elapsed / d, 0.0), 1.0)
        return progress + 1e-7 >= thr

    def _try_auto_transition(self):
        """Evaluate outgoing transitions from the current state."""
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
        """Evaluate a transition's condition expression.

        Empty condition means "transition when clip finishes" (only fires
        when the clip is non-looping and has reached its end).
        """
        cond = transition.condition.strip()

        # Empty condition → "on clip finished"
        if not cond:
            state = self._get_current_state()
            should_loop = state.loop if state else (
                self._current_clip.loop if self._current_clip else False)
            if self._current_clip and not should_loop:
                return self._elapsed >= self._current_clip.duration
            return False

        # Build a safe evaluation context
        ctx = dict(self._parameters)
        ctx["time"] = self._elapsed
        ctx["normalized_time"] = self.normalized_time
        ctx["state"] = self._current_state_name

        try:
            return bool(eval(cond, {"__builtins__": {}}, ctx))  # noqa: S307
        except Exception as exc:
            Debug.log_warning(
                f"[SpiritAnimator] Condition eval error in '{self._current_state_name}': "
                f"'{cond}' -> {exc}"
            )
            return False

    def _consume_triggers(self, condition: str):
        """Reset any trigger parameters that were used in the condition."""
        for name, val in list(self._parameters.items()):
            if val is True and name in condition:
                self._parameters[name] = False
