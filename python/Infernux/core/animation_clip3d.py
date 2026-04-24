"""
AnimationClip3D — data model for a 3D skeletal animation clip.

Serialized as ``.animclip3d`` JSON files.  This is the authoring-side
counterpart to 2D :class:`AnimationClip` — it references a source model
(typically ``.fbx``) and names an animation take embedded in that file.

Runtime sampling / skinning lives in C++ (future milestones); this asset
is intentionally simple and stable for Python workflows + AI tooling.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnimationClip3D:
    """A single 3D animation clip — references a model + named take."""

    name: str = "New Animation Clip 3D"
    schema_version: int = 1

    # Source skeletal model (FBX/GLTF/etc.) — GUID is authoritative when present.
    source_model_guid: str = ""
    source_model_path: str = ""

    # Animation take name as reported by Assimp / the importer metadata.
    take_name: str = ""

    # Optional: bind-pose bone names captured at import time (debug / tooling).
    # This is duplicated from the model `.meta` for cheap inspector UX.
    bind_pose_bone_names: List[str] = field(default_factory=list)

    speed: float = 1.0
    loop: bool = True

    # Optional seconds (authoring or tooling); 0.0 = unknown. Embedded takes may be unknown.
    duration_hint: float = 0.0

    file_path: str = field(default="", repr=False, compare=False)

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "source_model_guid": self.source_model_guid,
            "source_model_path": self.source_model_path,
            "take_name": self.take_name,
            "bind_pose_bone_names": list(self.bind_pose_bone_names),
            "speed": float(self.speed),
            "loop": bool(self.loop),
            "duration_hint": float(self.duration_hint),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnimationClip3D":
        bones = d.get("bind_pose_bone_names", [])
        if not isinstance(bones, list):
            bones = []
        return cls(
            name=str(d.get("name", "New Animation Clip 3D")),
            schema_version=int(d.get("schema_version", 1)),
            source_model_guid=str(d.get("source_model_guid", "")),
            source_model_path=str(d.get("source_model_path", "")),
            take_name=str(d.get("take_name", "")),
            bind_pose_bone_names=[str(x) for x in bones],
            speed=float(d.get("speed", 1.0)),
            loop=bool(d.get("loop", True)),
            duration_hint=float(d.get("duration_hint", 0.0) or 0.0),
        )

    def copy(self) -> "AnimationClip3D":
        return AnimationClip3D.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnimationClip3D):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def is_valid_reference(self) -> bool:
        return bool((self.source_model_guid or "").strip() or (self.source_model_path or "").strip())

    # ── File I/O ─────────────────────────────────────────────────────

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
    def load(cls, path: str) -> Optional["AnimationClip3D"]:
        if not path:
            return None
        # Project Panel virtual take: model.fbx::subanim:<index> (not a file on disk)
        if "::subanim:" in path:
            return cls.from_embedded_take_virtual_path(path)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            clip = cls.from_dict(data)
            clip.file_path = path
            # Name always derives from filename (matches 2D clip behaviour).
            clip.name = os.path.splitext(os.path.basename(path))[0]
            return clip
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_embedded_take_virtual_path(cls, virtual_path: str) -> Optional["AnimationClip3D"]:
        """Build a read-only clip view for ``model.fbx::subanim:<index>`` (Project Panel)."""
        token = "::subanim:"
        if token not in virtual_path:
            return None
        base, _, rest = virtual_path.partition(token)
        if not base.strip():
            return None
        try:
            idx = int(rest.strip())
        except ValueError:
            return None
        # Placeholder / overflow row from the project panel
        if idx < 0 or idx >= 999999:
            return None
        if not os.path.isfile(base):
            return None

        from Infernux.core.asset_types import read_meta_file

        meta = read_meta_file(base) or {}
        csv = (meta.get("animation_names_csv") or "")
        if isinstance(csv, str):
            names = [p.strip() for p in csv.split(",") if p.strip()]
        else:
            names = []

        if idx >= len(names):
            return None

        take_name = names[idx]
        guid = _read_asset_guid_from_meta_sidecar(base)
        bind_csv = (meta.get("bone_names_csv") or "")
        if isinstance(bind_csv, str):
            bind_names = [p.strip() for p in bind_csv.split(",") if p.strip()]
        else:
            bind_names = []

        clip = cls(
            name=take_name,
            source_model_guid=guid,
            source_model_path=os.path.normpath(base),
            take_name=take_name,
            bind_pose_bone_names=bind_names,
            speed=1.0,
            loop=True,
            duration_hint=0.0,
        )
        clip.file_path = virtual_path
        return clip


def _read_asset_guid_from_meta_sidecar(asset_path: str) -> str:
    """Return GUID from a ``.meta`` file's root (not the metadata{} map)."""
    meta_path = asset_path + ".meta"
    if not os.path.isfile(meta_path):
        return ""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            root = json.load(f)
        return str(root.get("guid", "") or "")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ""
