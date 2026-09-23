"""
AnimationClip3D — data model for a 3D skeletal animation clip.

Serialized as ``.animclip3d`` JSON files.  This is the authoring-side
counterpart to 2D :class:`AnimationClip` — it references a source model
(typically ``.fbx``) and names an animation take embedded in that file.

This asset is a *take pointer*, not a keyframe container: runtime sampling,
blending, and GPU skinning are implemented in C++ (``InxSkinnedMesh`` builds
the bone palette; the vertex shader applies 4-influence skinning). Keep this
class simple and stable for Python workflows + AI tooling.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Infernux.core.animation_event import AnimationEvent, events_from_list
from Infernux.engine.path_utils import resolved_path


@dataclass(frozen=True)
class ImportedFloatCurve:
    """A named scalar curve authored on an imported skeletal clip."""

    name: str
    keys: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict:
        return {"name": self.name, "keys": [
            {"time_normalized": time, "value": value} for time, value in self.keys
        ]}

    @classmethod
    def from_dict(cls, document: dict) -> "ImportedFloatCurve":
        if type(document) is not dict or set(document) != {"name", "keys"}:
            raise ValueError("imported float curve requires name and keys")
        if type(document["name"]) is not str or not document["name"] or type(document["keys"]) is not list:
            raise TypeError("imported float curve requires a non-empty name and key array")
        keys = []
        previous = -1.0
        for key in document["keys"]:
            if type(key) is not dict or set(key) != {"time_normalized", "value"}:
                raise ValueError("imported float curve key requires time_normalized and value")
            time, value = key["time_normalized"], key["value"]
            if (isinstance(time, bool) or not isinstance(time, (int, float))
                    or isinstance(value, bool) or not isinstance(value, (int, float))):
                raise TypeError("imported float curve keys must be numeric")
            time, value = float(time), float(value)
            if not math.isfinite(time) or not math.isfinite(value) or not 0.0 <= time <= 1.0 or time <= previous:
                raise ValueError("imported float curve keys must be finite, ordered and normalized")
            previous = time
            keys.append((time, value))
        return cls(document["name"], tuple(keys))

    def sample(self, normalized_time: float) -> float:
        if not self.keys:
            return 0.0
        time = min(max(float(normalized_time), 0.0), 1.0)
        if time <= self.keys[0][0]:
            return self.keys[0][1]
        for (left_time, left), (right_time, right) in zip(self.keys, self.keys[1:]):
            if time <= right_time:
                amount = (time - left_time) / (right_time - left_time)
                return left + (right - left) * amount
        return self.keys[-1][1]


def is_asset_guid_string(s: str) -> bool:
    """Return whether *s* is a current 32-character lowercase asset GUID."""
    return (
        isinstance(s, str)
        and len(s) == 32
        and all(character in "0123456789abcdef" for character in s)
    )


def resolve_disk_path_for_guid_string(adb, guid: str) -> Optional[str]:
    """Resolve a registered asset to its disk or imported animation document."""
    if not adb or not is_asset_guid_string(guid):
        return None
    path = adb.get_path_from_guid(guid)
    if path and "::subanim:" in path:
        return path
    return resolved_path(path) if path and os.path.isfile(path) else None


def resolve_model_disk_path_from_virtual_base(base: str) -> Optional[str]:
    """Map virtual clip prefix (asset GUID or absolute model file path) to a readable model file path."""
    b = (base or "").strip()
    if not b:
        return None
    if is_asset_guid_string(b):
        try:
            from Infernux.core.assets import AssetManager
            adb = getattr(AssetManager, "_asset_database", None)
            p = resolve_disk_path_for_guid_string(adb, b)
            return p
        except Exception:
            return None
    p = resolved_path(b)
    return p if os.path.isfile(p) else None


def embedded_take_descriptors(meta: dict) -> list[dict]:
    """Published clips, distinct from the source inventory and its ordering."""
    encoded = meta.get("model_animations")
    if encoded is None:
        return []
    return json.loads(encoded)


@dataclass
class AnimationClip3D:
    """A single 3D animation clip — references a model + named take."""

    name: str = "New Animation Clip 3D"

    # Source skeletal model (FBX/GLTF/etc.).
    source_model_guid: str = ""

    # Stable imported clip ID.
    take_name: str = ""

    # Optional: bind-pose bone names captured at import time (debug / tooling).
    # This is duplicated from the model `.meta` for cheap inspector UX.
    bind_pose_bone_names: List[str] = field(default_factory=list)

    # Optional seconds (authoring or tooling); 0.0 = unknown. Embedded takes may be unknown.
    duration_hint: float = 0.0

    # Import-authoritative playback semantics.  The FSM may stop a looping
    # clip, but cannot force an imported non-looping clip to wrap.
    default_loop: bool = True
    apply_root_motion: bool = False
    reference_pose: str = "bind_pose"

    curves: List[ImportedFloatCurve] = field(default_factory=list)
    bone_mask: List[str] = field(default_factory=list)

    # Animation events keyed by normalized time (0..1); dispatched at runtime.
    events: List[AnimationEvent] = field(default_factory=list)

    file_path: str = field(default="", repr=False, compare=False)

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_model_guid": self.source_model_guid,
            "take_name": self.take_name,
            "bind_pose_bone_names": list(self.bind_pose_bone_names),
            "duration_hint": float(self.duration_hint),
            "default_loop": bool(self.default_loop),
            "apply_root_motion": bool(self.apply_root_motion),
            "reference_pose": self.reference_pose,
            "curves": [curve.to_dict() for curve in self.curves],
            "events": [e.to_dict() for e in self.events],
            "bone_mask": list(self.bone_mask),
        }

    def serialize_document(self) -> dict:
        """Return the complete current editable document."""
        return self.to_dict()

    def deserialize_document(self, document: dict) -> bool:
        """Replace authoring state while preserving this asset's file identity."""
        try:
            replacement = type(self).from_dict(document)
        except (KeyError, TypeError, ValueError):
            return False
        self.name = replacement.name
        self.source_model_guid = replacement.source_model_guid
        self.take_name = replacement.take_name
        self.bind_pose_bone_names = replacement.bind_pose_bone_names
        self.duration_hint = replacement.duration_hint
        self.default_loop = replacement.default_loop
        self.apply_root_motion = replacement.apply_root_motion
        self.reference_pose = replacement.reference_pose
        self.curves = replacement.curves
        self.events = replacement.events
        self.bone_mask = replacement.bone_mask
        return True

    @classmethod
    def from_dict(cls, d: dict) -> "AnimationClip3D":
        expected = {
            "name",
            "source_model_guid",
            "take_name",
            "bind_pose_bone_names",
            "duration_hint",
            "default_loop",
            "apply_root_motion",
            "reference_pose",
            "curves",
            "events",
            "bone_mask",
        }
        if type(d) is not dict:
            raise ValueError("animation clip 3D must be a document")
        if not expected.issubset(d):
            raise ValueError(
                f"animation clip 3D fields mismatch; "
                f"missing={sorted(expected - set(d))}"
            )
        document = {name: d[name] for name in expected}
        string_fields = ("name", "source_model_guid", "take_name")
        if any(type(document[name]) is not str for name in string_fields):
            raise TypeError("animation clip 3D identity fields must be strings")
        source_model_guid = document["source_model_guid"]
        if source_model_guid and not is_asset_guid_string(source_model_guid):
            raise ValueError("source_model_guid must be a 32-character lowercase asset GUID")
        bones = document["bind_pose_bone_names"]
        if type(bones) is not list or any(type(value) is not str for value in bones):
            raise TypeError("bind_pose_bone_names must be an array of strings")
        if not isinstance(document["duration_hint"], (int, float)) or isinstance(document["duration_hint"], bool):
            raise TypeError("duration_hint must be numeric")
        duration_hint = float(document["duration_hint"])
        if not math.isfinite(duration_hint) or duration_hint < 0.0:
            raise ValueError("duration_hint must be finite and non-negative")
        if type(document["events"]) is not list:
            raise TypeError("events must be an array")
        raw_curves = document["curves"]
        bone_mask = document["bone_mask"]
        if type(raw_curves) is not list or type(bone_mask) is not list:
            raise TypeError("curves and bone_mask must be arrays")
        curves = [ImportedFloatCurve.from_dict(curve) for curve in raw_curves]
        if len({curve.name for curve in curves}) != len(curves):
            raise ValueError("imported float curve names must be unique")
        if any(type(bone) is not str or not bone for bone in bone_mask) or len(set(bone_mask)) != len(bone_mask):
            raise ValueError("bone_mask requires unique non-empty bone names")
        default_loop = document["default_loop"]
        apply_root_motion = document["apply_root_motion"]
        reference_pose = document["reference_pose"]
        if type(default_loop) is not bool or type(apply_root_motion) is not bool:
            raise TypeError("3D animation loop and root-motion settings must be booleans")
        if reference_pose not in {"bind_pose", "first_frame"}:
            raise ValueError("reference_pose must be bind_pose or first_frame")
        return cls(
            name=document["name"],
            source_model_guid=source_model_guid,
            take_name=document["take_name"],
            bind_pose_bone_names=list(bones),
            duration_hint=duration_hint,
            default_loop=default_loop,
            apply_root_motion=apply_root_motion,
            reference_pose=reference_pose,
            curves=curves,
            events=events_from_list(document["events"]),
            bone_mask=list(bone_mask),
        )

    def sample_curve(self, name: str, normalized_time: float) -> float:
        for curve in self.curves:
            if curve.name == name:
                return curve.sample(normalized_time)
        raise KeyError(name)

    def copy(self) -> "AnimationClip3D":
        return AnimationClip3D.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnimationClip3D):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def is_valid_reference(self) -> bool:
        return bool((self.source_model_guid or "").strip())

    # ── File I/O ─────────────────────────────────────────────────────

    def save(self, path: str = "") -> bool:
        target = path or self.file_path
        if not target:
            return False
        type(self).from_dict(self.to_dict())
        try:
            from Infernux.core.document_store import write_document_text
            write_document_text(target, json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")
            return True
        except (OSError, RuntimeError):
            return False

    @classmethod
    def load(cls, path: str) -> Optional["AnimationClip3D"]:
        if not path:
            return None
        # Project Panel virtual take: model.fbx::subanim:<id> (not a file on disk)
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
        """Resolve one published clip by its stable imported identity."""
        token = "::subanim:"
        if token not in virtual_path:
            return None
        base, _, rest = virtual_path.partition(token)
        base = base.strip()
        if not base:
            return None
        from Infernux.core.asset_types import read_asset_metadata

        metadata = read_asset_metadata(virtual_path)
        if metadata and "import_document" in metadata:
            clip = cls.from_dict(json.loads(metadata["import_document"]))
            clip.file_path = virtual_path
            return clip
        model_disk = resolve_model_disk_path_from_virtual_base(base)
        if not model_disk:
            return None

        from Infernux.core.asset_types import read_meta_file

        meta = read_meta_file(model_disk) or {}
        identifier = rest.strip()
        published = embedded_take_descriptors(meta)
        selected = next((item for item in published if item["id"] == identifier), None)
        if selected is None:
            return None
        take_name = selected["id"]
        meta_guid = _read_asset_guid_from_meta_sidecar(model_disk)
        if is_asset_guid_string(base):
            source_guid = base
        else:
            source_guid = meta_guid
        if not is_asset_guid_string(source_guid):
            return None
        bind_csv = (meta.get("bone_names_csv") or "")
        if isinstance(bind_csv, str):
            bind_names = [p.strip() for p in bind_csv.split(",") if p.strip()]
        else:
            bind_names = []

        try:
            clip = cls.from_dict({
                "name": selected["name"],
                "source_model_guid": source_guid,
                "take_name": take_name,
                "bind_pose_bone_names": bind_names,
                "duration_hint": selected["duration"],
                "default_loop": selected["default_loop"],
                "apply_root_motion": selected["apply_root_motion"],
                "reference_pose": selected["reference_pose"],
                "curves": selected["curves"],
                "events": selected["events"],
                "bone_mask": selected["bone_mask"],
            })
        except (KeyError, TypeError, ValueError):
            return None
        clip.file_path = virtual_path
        return clip


def _read_asset_guid_from_meta_sidecar(asset_path: str) -> str:
    """Return the canonical GUID from a ``.meta`` sidecar."""
    from Infernux.core.asset_types import read_meta_guid
    return read_meta_guid(asset_path)
