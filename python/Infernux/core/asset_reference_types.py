"""Authoritative asset-reference type descriptors and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import PurePath
from typing import Any, Iterable, Optional

from .asset_types import (
    ANIMCLIP3D_EXTENSIONS,
    ANIMCLIP_EXTENSIONS,
    ANIMFSM_EXTENSIONS,
    ANIMTIMELINE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    FONT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MATERIAL_EXTENSIONS,
    MESH_EXTENSIONS,
    PARTICLE_GRAPH_EXTENSIONS,
    PHYSIC_MATERIAL_EXTENSIONS,
    PREFAB_EXTENSIONS,
    RENDER_EFFECT_EXTENSIONS,
    TIMELINEFSM_EXTENSIONS,
)


def _asset_database():
    try:
        from Infernux.core.assets import AssetManager

        database = getattr(AssetManager, "_asset_database", None)
        if database is not None:
            return database
    except (AttributeError, ImportError, RuntimeError):
        pass
    try:
        from Infernux.lib import AssetRegistry

        return AssetRegistry.instance().get_asset_database()
    except (AttributeError, ImportError, RuntimeError):
        return None


def _resolve_guid_path(guid: str) -> str:
    token = str(guid or "").strip()
    if not token:
        return ""
    database = _asset_database()
    if database is None:
        return ""
    try:
        return str(database.get_path_from_guid(token) or "").strip()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return ""


def _resolve_path_guid(path: str) -> str:
    """Resolve an editor-selected raw path once, before it becomes a ref."""

    token = str(path or "").strip()
    if not token:
        return ""
    database = _asset_database()
    resolve_guid = getattr(database, "get_guid_from_path", None)
    if not callable(resolve_guid):
        return ""
    candidates = [token]
    if not os.path.isabs(token):
        try:
            from Infernux.engine.project_context import get_project_root

            project_root = str(get_project_root() or "")
        except (ImportError, RuntimeError):
            project_root = ""
        if project_root:
            candidates.append(os.path.join(project_root, token))
    for candidate in candidates:
        try:
            guid = str(resolve_guid(candidate) or "").strip()
        except (KeyError, RuntimeError, TypeError, ValueError):
            continue
        if guid:
            return guid
    return ""


def canonical_asset_reference_identity(guid: str, path_hint: str) -> tuple[str, str]:
    """Normalize stored reference fields without deriving identity from a path.

    Picker and drag/drop code owns path-to-GUID conversion before it creates a
    reference.  This shared persistence/runtime boundary deliberately keeps a
    path hint as display data only.
    """

    return str(guid or "").strip(), str(path_hint or "").strip()


@dataclass(frozen=True, slots=True)
class AssetReferenceType:
    type_id: str
    display_name: str
    extensions: frozenset[str]
    drag_types: tuple[str, ...]
    widget_prefix: str
    aliases: tuple[str, ...] = ()
    allow_structured_reference: bool = False
    virtual_path_markers: tuple[str, ...] = ()
    compatible_types: tuple[str, ...] = ()

    @property
    def patterns(self) -> tuple[str, ...]:
        return tuple(f"*{extension}" for extension in sorted(self.extensions))

    def incompatibility(self, payload: Any) -> str:
        """Return an error message, or an empty string when compatible."""

        if payload is None:
            return ""
        path = ""
        guid = ""
        builtin = ""
        if isinstance(payload, dict):
            source_type = str(payload.get("asset_type") or "").strip()
            accepted_types = {
                self.type_id.casefold(),
                self.display_name.casefold(),
                *(str(alias).strip().casefold() for alias in self.aliases),
                *(value.casefold() for value in self.compatible_types),
            }
            if source_type and source_type.casefold() not in accepted_types:
                return (
                    f"{self.display_name} reference rejects asset type "
                    f"'{source_type}'"
                )
            guid = str(payload.get("guid") or "").strip()
            path = str(payload.get("path_hint") or "").strip()
            builtin = str(payload.get("builtin") or "").strip()
            if builtin:
                if self.allow_structured_reference:
                    return ""
                return f"{self.display_name} reference rejects built-in '{builtin}'"
            if not guid:
                return f"{self.display_name} reference is empty"
            if payload.get("$type") and not guid and not path:
                return (
                    f"{self.display_name} reference rejects non-asset structured "
                    f"value '{payload.get('$type')}'"
                )
        elif isinstance(payload, str):
            path = payload.strip()
        else:
            guid = str(getattr(payload, "guid", "") or "").strip()
            path = str(
                getattr(payload, "path_hint", "")
                or getattr(payload, "file_path", "")
                or ""
            ).strip()
            builtin = str(getattr(payload, "builtin", "") or "").strip()
            if builtin:
                if self.allow_structured_reference:
                    return ""
                return f"{self.display_name} reference rejects built-in '{builtin}'"
            if not guid and not path:
                return (
                    f"{self.display_name} reference rejects non-asset value "
                    f"'{type(payload).__name__}'"
                )

        if guid:
            path = _resolve_guid_path(guid)
            if not path:
                return f"{self.display_name} reference uses unknown GUID '{guid}'"
        elif not isinstance(payload, str):
            path = ""
        if not path:
            return f"{self.display_name} reference is empty"
        portable_path = path.replace("\\", "/")
        for marker in self.virtual_path_markers:
            head, separator, tail = portable_path.partition(marker)
            if separator and head and tail:
                return ""
        extension = PurePath(portable_path).suffix.casefold()
        if not extension:
            resolved = _resolve_guid_path(path)
            if not resolved:
                return (
                    f"{self.display_name} reference '{path}' has no asset type "
                    "information"
                )
            portable_path = resolved.replace("\\", "/")
            for marker in self.virtual_path_markers:
                head, separator, tail = portable_path.partition(marker)
                if separator and head and tail:
                    return ""
        if not any(portable_path.casefold().endswith(item) for item in self.extensions):
            accepted = ", ".join(sorted(self.extensions))
            return (
                f"{self.display_name} reference rejects '{path}': expected "
                f"one of {accepted}"
            )
        return ""


class AssetReferenceCodec:
    """Canonical, LLM-readable clipboard codec for editor asset references."""

    PREFIX = "infernux.asset_reference "

    @classmethod
    def encode(cls, asset_type: str, value: Any) -> str:
        descriptor = asset_type_registry.require(asset_type)
        payload = cls.normalize(descriptor.type_id, value)
        if not any(payload[key] for key in ("guid", "builtin")):
            return ""
        # Clipboard data is an authoring transport, but it must still carry
        # the stable identity rather than snapshotting a renameable path.
        payload = {
            "asset_type": payload["asset_type"],
            "builtin": payload["builtin"],
            "guid": payload["guid"],
        }
        return cls.PREFIX + json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def decode(cls, text: str) -> dict[str, str]:
        token = str(text or "").strip()
        if not token.startswith(cls.PREFIX):
            raise ValueError("clipboard does not contain an Infernux asset reference")
        raw = json.loads(token[len(cls.PREFIX) :])
        if type(raw) is not dict:
            raise ValueError("asset reference clipboard payload must be an object")
        asset_type = raw.get("asset_type", "")
        if type(asset_type) is not str:
            raise TypeError("asset reference clipboard asset_type must be a string")
        descriptor = asset_type_registry.require(asset_type)
        payload = cls.normalize(descriptor.type_id, raw)
        return payload

    @staticmethod
    def normalize(asset_type: str, value: Any) -> dict[str, str]:
        guid = ""
        path_hint = ""
        builtin = ""
        structured = isinstance(value, dict)
        if structured:
            guid = str(value.get("guid") or "").strip()
            path_hint = str(value.get("path_hint") or "").strip()
            builtin = str(value.get("builtin") or "").strip()
        elif isinstance(value, str):
            path_hint = value.strip()
            # Raw strings occur only at editor file-selection/drop boundaries.
            # Convert them to GUID identity immediately when imported.
            guid = _resolve_path_guid(path_hint)
        elif value is not None:
            guid = str(getattr(value, "guid", "") or "").strip()
            path_hint = str(
                getattr(value, "path_hint", "")
                or getattr(value, "file_path", "")
                or ""
            ).strip()
            builtin = str(getattr(value, "builtin", "") or "").strip()
        guid, path_hint = canonical_asset_reference_identity(guid, path_hint)
        # Structured documents already crossed the authoring boundary. Their
        # path field is an obsolete display snapshot regardless of whether a
        # GUID is present; current display paths are always resolved by GUID.
        if structured:
            path_hint = ""
        descriptor = asset_type_registry.get(asset_type)
        if descriptor is not None and descriptor.compatible_types:
            # A field accepting several resource types is not itself a new
            # asset type. Clipboard references retain the concrete resource
            # kind, so a material's sampled image can be pasted into a static
            # image slot, or its render target into a Camera slot.
            path = _resolve_guid_path(guid)
            extension = PurePath(path.replace("\\", "/")).suffix.casefold()
            for type_id in descriptor.compatible_types:
                member = asset_type_registry.require(type_id)
                if extension in member.extensions:
                    asset_type = member.type_id
                    break
            else:
                # Deleted assets still retain their concrete kind in scene
                # documents. The field union is never a resource identity.
                from .asset_ref import AssetRefBase, get_asset_type_for_ref
                declared = (value.get("asset_type", "") if structured
                            else get_asset_type_for_ref(value) if isinstance(value, AssetRefBase)
                            else "")
                member = asset_type_registry.get(declared)
                if member is not None and member.type_id in descriptor.compatible_types:
                    asset_type = member.type_id
        return {
            "asset_type": str(asset_type or "").strip(),
            "builtin": builtin,
            "guid": guid,
            "path_hint": path_hint,
        }


def resolve_asset_reference_path(asset_type: str, value: Any) -> str:
    """Validate one reference and resolve its canonical path or virtual path."""

    descriptor = asset_type_registry.require(asset_type)
    error = descriptor.incompatibility(value)
    if error:
        raise ValueError(error)
    payload = AssetReferenceCodec.normalize(descriptor.type_id, value)
    guid = payload["guid"]
    if guid:
        resolved = _resolve_guid_path(guid)
        if resolved:
            return resolved
        raise ValueError(f"{descriptor.display_name} reference has no resolvable path")
    # A raw string is an explicit editor file-selection input.  A path_hint
    # carried by a reference document is not an identity fallback.
    if isinstance(value, str) and payload["path_hint"]:
        return payload["path_hint"]
    raise ValueError(f"{descriptor.display_name} reference has no resolvable path")


class AssetTypeRegistry:
    """Single registry used by serialization, drawers, pickers and codecs."""

    def __init__(self) -> None:
        self._types: dict[str, AssetReferenceType] = {}
        self._aliases: dict[str, str] = {}

    def register(self, descriptor: AssetReferenceType, *, replace: bool = False) -> None:
        type_id = str(descriptor.type_id or "").strip()
        if not type_id:
            raise ValueError("asset reference type id must not be empty")
        if type_id in self._types and not replace:
            raise ValueError(f"asset reference type already registered: {type_id}")
        if replace and type_id in self._types:
            previous = self._types[type_id]
            for alias in self._descriptor_aliases(previous):
                self._aliases.pop(alias, None)
        self._types[type_id] = descriptor
        for alias in self._descriptor_aliases(descriptor):
            owner = self._aliases.get(alias)
            if owner is not None and owner != type_id:
                raise ValueError(
                    f"asset reference alias '{alias}' belongs to both {owner} and {type_id}"
                )
            self._aliases[alias] = type_id

    def get(self, type_or_alias: str) -> Optional[AssetReferenceType]:
        token = str(type_or_alias or "").strip()
        if not token:
            return None
        direct = self._types.get(token)
        if direct is not None:
            return direct
        type_id = self._aliases.get(token.casefold())
        return self._types.get(type_id) if type_id is not None else None

    def require(self, type_or_alias: str) -> AssetReferenceType:
        descriptor = self.get(type_or_alias)
        if descriptor is None:
            raise KeyError(f"unknown asset reference type: {type_or_alias}")
        return descriptor

    def values(self) -> tuple[AssetReferenceType, ...]:
        return tuple(self._types.values())

    @staticmethod
    def _descriptor_aliases(descriptor: AssetReferenceType) -> Iterable[str]:
        yield descriptor.type_id.casefold()
        yield descriptor.display_name.casefold()
        for alias in descriptor.aliases:
            yield str(alias).strip().casefold()


asset_type_registry = AssetTypeRegistry()


def _register_builtin(
    type_id: str,
    display_name: str,
    extensions,
    drag_types: tuple[str, ...],
    prefix: str,
    *,
    aliases: tuple[str, ...] = (),
    structured: bool = False,
    virtual_path_markers: tuple[str, ...] = (),
    compatible_types: tuple[str, ...] = (),
) -> None:
    asset_type_registry.register(
        AssetReferenceType(
            type_id=type_id,
            display_name=display_name,
            extensions=frozenset(str(item).casefold() for item in extensions),
            drag_types=drag_types,
            widget_prefix=prefix,
            aliases=aliases,
            allow_structured_reference=structured,
            virtual_path_markers=virtual_path_markers,
            compatible_types=compatible_types,
        )
    )


_register_builtin("Material", "Material", MATERIAL_EXTENSIONS, ("MATERIAL_FILE",), "mat")
_register_builtin(
    "Texture", "Texture", IMAGE_EXTENSIONS, ("TEXTURE_GUID", "TEXTURE_FILE"), "tex",
    aliases=("Texture2D",), structured=True, virtual_path_markers=("::subtex:",),
)
_register_builtin(
    "Texture.SDF", "Signed Distance Field", {".inxsdf"},
    ("TEXTURE_GUID", "TEXTURE_FILE"), "sdf", structured=True,
)
_register_builtin(
    "Texture.Sampled", "Sampled Texture", {*IMAGE_EXTENSIONS, ".rendertexture"},
    ("TEXTURE_GUID", "TEXTURE_FILE", "RENDER_TEXTURE_FILE"), "sampled_tex",
    structured=True, compatible_types=("Texture", "Texture2D", "RenderTexture"), virtual_path_markers=("::subtex:",),
)
_register_builtin(
    "Texture.VectorField", "Vector Field", {".inxvfield"},
    ("TEXTURE_GUID", "TEXTURE_FILE"), "vfield", structured=True,
)
_register_builtin("Shader.Vertex", "Vertex Shader", {".vert"}, ("SHADER_FILE",), "vert", aliases=("Vert",))
_register_builtin("Shader.Fragment", "Fragment Shader", {".frag"}, ("SHADER_FILE",), "frag", aliases=("Frag",))
_register_builtin("Shader", "Shader", {".vert", ".frag"}, ("SHADER_FILE",), "shd")
_register_builtin(
    "Mesh", "Mesh", MESH_EXTENSIONS, ("MODEL_GUID", "MODEL_FILE"), "mesh",
    aliases=("Model",), structured=True, virtual_path_markers=("::submesh:",),
)
_register_builtin("AudioClip", "AudioClip", AUDIO_EXTENSIONS, ("AUDIO_FILE",), "aud", aliases=("Audio",))
_register_builtin("Font", "Font", FONT_EXTENSIONS, ("FONT_FILE",), "font")
_register_builtin("Prefab", "Prefab", PREFAB_EXTENSIONS, ("PREFAB_GUID", "PREFAB_FILE"), "prefab")
_register_builtin("PhysicMaterial", "PhysicMaterial", PHYSIC_MATERIAL_EXTENSIONS, ("PHYSIC_MATERIAL_FILE",), "pmat")
_register_builtin("AnimStateMachine", "AnimFSM", ANIMFSM_EXTENSIONS, ("ANIMFSM_FILE",), "fsm", aliases=("AnimFSM",))
_register_builtin(
    "ParticleGraph",
    "Particle Graph",
    set(PARTICLE_GRAPH_EXTENSIONS) | {".particle.py"},
    ("PARTICLE_GRAPH_FILE",),
    "particle",
)
_register_builtin("RenderEffect", "Render Effect / Group", RENDER_EFFECT_EXTENSIONS, ("RENDER_EFFECT_FILE",), "effect")
_register_builtin("AnimationClip", "AnimClip2D", ANIMCLIP_EXTENSIONS, ("ANIMCLIP_FILE",), "aclip")
_register_builtin(
    "AnimationClip3D",
    "AnimClip3D",
    ANIMCLIP3D_EXTENSIONS,
    ("ANIMCLIP3D_FILE",),
    "aclip3d",
    virtual_path_markers=("::subanim:",),
)
_register_builtin("AnimationTimeline", "Timeline", ANIMTIMELINE_EXTENSIONS, ("ANIMTIMELINE_FILE",), "atl")
_register_builtin("TimelineFSM", "TimelineFSM", TIMELINEFSM_EXTENSIONS, ("TIMELINEFSM_FILE",), "tlfsm")
_register_builtin("DataAsset", "Data Asset", {".inxdata"}, ("DATA_ASSET_FILE",), "data")
_register_builtin("RenderTexture", "Render Texture", {".rendertexture"}, ("RENDER_TEXTURE_FILE",), "rt")
