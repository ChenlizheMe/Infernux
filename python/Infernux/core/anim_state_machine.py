"""
AnimStateMachine — data model for an animation finite-state machine.

An AnimStateMachine holds a graph of named states (each referencing an
animation clip) connected by transitions.  Serialized as ``.animfsm`` JSON
files.  Shared between 2D and 3D animation systems.

Usage::

    fsm = AnimStateMachine.load("Assets/Animations/player.animfsm")
    fsm.save("Assets/Animations/player.animfsm")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnimParameter:
    """Declared variable for transition conditions (matches runtime SpiritAnimator parameters)."""

    name: str = "NewVar"
    kind: str = "float"  # bool, float, int
    default_bool: bool = False
    default_float: float = 0.0
    default_int: int = 0

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "bool":
            out["default_bool"] = self.default_bool
        elif self.kind == "float":
            out["default_float"] = self.default_float
        elif self.kind == "int":
            out["default_int"] = self.default_int
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "AnimParameter":
        raw_kind = str(d.get("kind", d.get("param_type", "float")))
        if raw_kind == "trigger":
            kind = "bool"
        else:
            kind = raw_kind

        def _as_float(v: Any, fallback: float) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return fallback

        def _as_int(v: Any, fallback: int) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return fallback

        legacy_default = d.get("default")

        bool_v = d.get("default_bool", None)
        if bool_v is None and isinstance(legacy_default, bool):
            bool_v = legacy_default
        if bool_v is None:
            bool_v = False

        float_v = d.get("default_float", None)
        if float_v is None and isinstance(legacy_default, (int, float)) and not isinstance(legacy_default, bool):
            float_v = legacy_default

        int_v = d.get("default_int", None)
        if int_v is None and isinstance(legacy_default, int) and not isinstance(legacy_default, bool):
            int_v = legacy_default

        return cls(
            name=str(d.get("name", "NewVar")),
            kind=kind,
            default_bool=bool(bool_v),
            default_float=_as_float(float_v, 0.0),
            default_int=_as_int(int_v, 0),
        )


@dataclass
class AnimTransition:
    """A directed transition between two states."""

    target_state: str = ""
    condition: str = ""       # expression string evaluated at runtime
    duration: float = 0.0     # cross-fade / blend duration in seconds

    def to_dict(self) -> dict:
        return {
            "target_state": self.target_state,
            "condition": self.condition,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimTransition:
        return cls(
            target_state=str(d.get("target_state", "")),
            condition=str(d.get("condition", "")),
            duration=float(d.get("duration", 0.0)),
        )


@dataclass
class AnimState:
    """A single state inside the FSM, referencing a clip and holding outgoing transitions."""

    name: str = "New State"
    clip_guid: str = ""       # GUID of the referenced .animclip2d / .animclip3d
    clip_path: str = ""       # fallback path (editor-only hint)
    speed: float = 1.0
    # 0..1: minimum normalized clip progress before outgoing transitions are considered.
    # 1.0 = must reach end of current clip segment (default; matches "play full clip then transition").
    exit_time_normalized: float = 1.0
    loop: bool = True         # whether to loop the clip in this state
    # If True, SpiritAnimator.play(state) restarts the clip when already in that state.
    # If False, play() is a no-op while that state is already playing (e.g. safe to call every frame).
    restart_same_clip: bool = False
    transitions: List[AnimTransition] = field(default_factory=list)
    # Visual position in the node editor (editor-only, persisted for convenience)
    position: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "clip_guid": self.clip_guid,
            "clip_path": self.clip_path,
            "speed": self.speed,
            "exit_time_normalized": self.exit_time_normalized,
            "loop": self.loop,
            "restart_same_clip": self.restart_same_clip,
            "transitions": [t.to_dict() for t in self.transitions],
            "position": list(self.position),
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimState:
        return cls(
            name=str(d.get("name", "New State")),
            clip_guid=str(d.get("clip_guid", "")),
            clip_path=str(d.get("clip_path", "")),
            speed=float(d.get("speed", 1.0)),
            exit_time_normalized=max(
                0.0, min(1.0, float(d.get("exit_time_normalized", 1.0)))
            ),
            loop=bool(d.get("loop", True)),
            restart_same_clip=bool(d.get("restart_same_clip", False)),
            transitions=[AnimTransition.from_dict(t) for t in d.get("transitions", [])],
            position=list(d.get("position", [0.0, 0.0])),
        )


@dataclass
class AnimStateMachine:
    """A finite-state machine describing animation states and transitions."""

    name: str = "New State Machine"
    default_state: str = ""                          # name of entry state
    mode: str = "2d"                                 # "2d" or "3d"
    states: List[AnimState] = field(default_factory=list)
    parameters: List[AnimParameter] = field(default_factory=list)
    file_path: str = field(default="", repr=False, compare=False)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "default_state": self.default_state,
            "mode": self.mode,
            "states": [s.to_dict() for s in self.states],
            "parameters": [p.to_dict() for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AnimStateMachine:
        raw_params = d.get("parameters") or []
        params: List[AnimParameter] = []
        if isinstance(raw_params, list):
            for item in raw_params:
                if isinstance(item, dict):
                    params.append(AnimParameter.from_dict(item))
        return cls(
            name=str(d.get("name", "New State Machine")),
            default_state=str(d.get("default_state", "")),
            mode=str(d.get("mode", "2d")),
            states=[AnimState.from_dict(s) for s in d.get("states", [])],
            parameters=params,
        )

    def copy(self) -> AnimStateMachine:
        return AnimStateMachine.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnimStateMachine):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    # ── File I/O ──────────────────────────────────────────────────────

    def save(self, path: str = "") -> bool:
        target = path or self.file_path
        if not target:
            return False
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, path: str) -> Optional[AnimStateMachine]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fsm = cls.from_dict(data)
            fsm.file_path = path
            fsm.name = os.path.splitext(os.path.basename(path))[0]
            return fsm
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def state_count(self) -> int:
        return len(self.states)

    def get_state(self, name: str) -> Optional[AnimState]:
        for s in self.states:
            if s.name == name:
                return s
        return None

    def add_state(self, name: str = "") -> AnimState:
        if not name:
            name = f"State {self.state_count}"
        state = AnimState(name=name)
        self.states.append(state)
        if not self.default_state:
            self.default_state = name
        return state

    def remove_state(self, name: str) -> bool:
        for i, s in enumerate(self.states):
            if s.name == name:
                self.states.pop(i)
                # Clean up transitions pointing to removed state
                for other in self.states:
                    other.transitions = [
                        t for t in other.transitions if t.target_state != name
                    ]
                if self.default_state == name:
                    self.default_state = self.states[0].name if self.states else ""
                return True
        return False
