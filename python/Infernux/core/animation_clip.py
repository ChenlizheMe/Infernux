"""
AnimationClip — data model for a 2D sprite animation clip.

An AnimationClip describes a sequence of sprite frames, playback speed,
and looping behaviour.  Serialized as ``.animclip2d`` JSON files.

Usage::

    clip = AnimationClip.load("Assets/Animations/idle.animclip2d")
    clip.save("Assets/Animations/idle.animclip2d")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnimationClip:
    """A single animation clip — a sequence of sprite frames with timing."""

    name: str = "New Animation Clip"
    authoring_texture_guid: str = ""
    authoring_texture_path: str = ""
    frame_indices: List[int] = field(default_factory=list)
    fps: float = 12.0
    loop: bool = True
    file_path: str = field(default="", repr=False, compare=False)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "authoring_texture_guid": self.authoring_texture_guid,
            "authoring_texture_path": self.authoring_texture_path,
            "frame_indices": list(self.frame_indices),
            "fps": self.fps,
            "loop": self.loop,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AnimationClip:
        return cls(
            name=str(d.get("name", "New Animation Clip")),
            authoring_texture_guid=str(d.get("authoring_texture_guid", "")),
            authoring_texture_path=str(d.get("authoring_texture_path", "")),
            frame_indices=list(d.get("frame_indices", [])),
            fps=float(d.get("fps", 12.0)),
            loop=bool(d.get("loop", True)),
        )

    def copy(self) -> AnimationClip:
        return AnimationClip.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnimationClip):
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
    def load(cls, path: str) -> Optional[AnimationClip]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            clip = cls.from_dict(data)
            clip.file_path = path
            # Name always derives from filename
            clip.name = os.path.splitext(os.path.basename(path))[0]
            return clip
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None

    # ── Helpers ───────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def duration(self) -> float:
        if self.fps <= 0 or not self.frame_indices:
            return 0.0
        return len(self.frame_indices) / self.fps
