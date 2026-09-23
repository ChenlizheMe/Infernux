"""Strict source documents for reusable RenderEffect assets and groups."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

import os

from Infernux.engine.path_utils import portable_path


RENDER_EFFECT_EXTENSION = ".effect"
RENDER_EFFECT_GROUP_EXTENSION = ".effectgroup"
RENDER_EFFECT_SCHEMA = "infernux.render_effect"
RENDER_EFFECT_GROUP_SCHEMA = "infernux.render_effect_group"

_TYPE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


def _resolve_author_effect_asset_reference(path: str) -> str:
    """Resolve an explicit authoring selection to its current GUID.

    The path exists only at this editor boundary. Serialized effect documents
    retain the GUID, so moving the selected asset cannot rewrite their bytes.
    """
    hint = portable_path(str(path or "").strip())
    if not hint:
        return ""

    from Infernux.core.assets import AssetManager
    from Infernux.engine.project_context import get_project_root

    database = getattr(AssetManager, "_asset_database", None)
    resolve_guid = getattr(database, "get_guid_from_path", None)
    if not callable(resolve_guid):
        return ""
    candidates = [hint]
    project_root = str(get_project_root() or "")
    if project_root and not os.path.isabs(hint):
        candidates.insert(0, os.path.join(project_root, hint))
    for candidate in candidates:
        found = str(resolve_guid(candidate) or "").strip()
        if found:
            return found
    return ""


@dataclass(frozen=True)
class EffectAssetReference:
    """Current asset reference whose sole identity is a non-empty GUID."""

    guid: str
    def __post_init__(self) -> None:
        guid = str(self.guid or "").strip()
        if not guid:
            raise ValueError("effect asset reference requires a non-empty GUID")
        object.__setattr__(self, "guid", guid)

    @classmethod
    def from_author_path(cls, path: str) -> "EffectAssetReference":
        """Resolve an editor-authored path through the current asset catalog."""
        guid = _resolve_author_effect_asset_reference(path)
        if not guid:
            raise LookupError(f"effect asset path is not registered: {path}")
        return cls(guid=guid)

    def to_dict(self) -> dict[str, str]:
        return {"guid": self.guid}


@dataclass(frozen=True)
class RenderEffectAsset:
    """One reusable effect implementation and its default parameters."""

    feature_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dependencies: tuple[EffectAssetReference, ...] = ()

    def __post_init__(self) -> None:
        feature_type = str(self.feature_type or "").strip()
        if not _TYPE_ID_PATTERN.fullmatch(feature_type):
            raise ValueError("feature_type must be a lowercase namespaced identifier")
        _require_json_object(self.parameters, "parameters")
        object.__setattr__(self, "feature_type", feature_type)
        object.__setattr__(self, "parameters", _json_clone(dict(self.parameters)))
        dependencies = tuple(self.dependencies)
        if not all(isinstance(value, EffectAssetReference) for value in dependencies):
            raise TypeError("dependencies must contain EffectAssetReference values")
        object.__setattr__(self, "dependencies", dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": RENDER_EFFECT_SCHEMA,
            "feature_type": self.feature_type,
            "parameters": _json_clone(dict(self.parameters)),
            "dependencies": [reference.to_dict() for reference in self.dependencies],
        }


@dataclass(frozen=True)
class RenderEffectGroupEntry:
    """One stable entry in a reusable ordered effect group."""

    entry_id: str
    asset: EffectAssetReference
    enabled: bool = True
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entry_id = str(self.entry_id or "").strip()
        if not entry_id:
            raise ValueError("effect group entry_id cannot be empty")
        if not isinstance(self.asset, EffectAssetReference):
            raise TypeError("effect group asset must be an EffectAssetReference")
        if not isinstance(self.enabled, bool):
            raise TypeError("effect group enabled must be a bool")
        _require_json_object(self.overrides, "overrides")
        object.__setattr__(self, "entry_id", entry_id)
        object.__setattr__(self, "overrides", _json_clone(dict(self.overrides)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "asset": self.asset.to_dict(),
            "enabled": self.enabled,
            "overrides": _json_clone(dict(self.overrides)),
        }


@dataclass(frozen=True)
class RenderEffectGroupAsset:
    """An ordered reusable chain of effects or nested groups."""

    entries: tuple[RenderEffectGroupEntry, ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not all(isinstance(value, RenderEffectGroupEntry) for value in entries):
            raise TypeError("entries must contain RenderEffectGroupEntry values")
        entry_ids = [entry.entry_id for entry in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("effect group entry_id values must be unique")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": RENDER_EFFECT_GROUP_SCHEMA,
            "entries": [entry.to_dict() for entry in self.entries],
        }


RenderEffectDocument = Union[RenderEffectAsset, RenderEffectGroupAsset]


def parse_render_effect_document(value: str | bytes | Mapping[str, Any]) -> RenderEffectDocument:
    """Parse and strictly validate an effect or effect-group source document."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        root = json.loads(value)
    elif isinstance(value, Mapping):
        root = dict(value)
    else:
        raise TypeError("render effect document must be JSON text or an object")
    if type(root) is not dict:
        raise TypeError("render effect document root must be an object")

    schema = root.get("$schema")
    if schema == RENDER_EFFECT_SCHEMA:
        _require_required_keys(
            root,
            {"$schema", "feature_type", "parameters", "dependencies"},
            "render effect",
        )
        dependencies = _parse_references(root["dependencies"], "dependencies")
        return RenderEffectAsset(
            feature_type=root["feature_type"],
            parameters=root["parameters"],
            dependencies=dependencies,
        )
    if schema == RENDER_EFFECT_GROUP_SCHEMA:
        _require_required_keys(root, {"$schema", "entries"}, "render effect group")
        raw_entries = root["entries"]
        if type(raw_entries) is not list:
            raise TypeError("entries must be an array")
        entries: list[RenderEffectGroupEntry] = []
        for index, raw_entry in enumerate(raw_entries):
            if type(raw_entry) is not dict:
                raise TypeError(f"entries[{index}] must be an object")
            _require_required_keys(raw_entry, {"asset"}, f"entries[{index}]")
            asset = _parse_reference(raw_entry["asset"], f"entries[{index}].asset")
            if asset is None:
                continue
            _require_required_keys(
                raw_entry, {"entry_id", "enabled", "overrides"}, f"entries[{index}]"
            )
            if type(raw_entry["entry_id"]) is not str or not raw_entry["entry_id"].strip():
                raise ValueError(f"entries[{index}].entry_id must be a non-empty string")
            if type(raw_entry["enabled"]) is not bool:
                raise TypeError(f"entries[{index}].enabled must be a bool")
            _require_json_object(raw_entry["overrides"], f"entries[{index}].overrides")
            entries.append(
                RenderEffectGroupEntry(
                    entry_id=raw_entry["entry_id"],
                    asset=asset,
                    enabled=raw_entry["enabled"],
                    overrides=raw_entry["overrides"],
                )
            )
        return RenderEffectGroupAsset(entries=tuple(entries))
    raise ValueError(f"unsupported render effect schema: {schema!r}")


def dump_render_effect_document(document: RenderEffectDocument, *, indent: int = 2) -> str:
    """Return deterministic UTF-8 JSON suitable for source control and LLM editing."""
    if not isinstance(document, (RenderEffectAsset, RenderEffectGroupAsset)):
        raise TypeError("document must be RenderEffectAsset or RenderEffectGroupAsset")
    return json.dumps(document.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n"


def direct_effect_dependencies(document: RenderEffectDocument) -> tuple[EffectAssetReference, ...]:
    """Return direct asset edges for importer dependency tracking."""
    if isinstance(document, RenderEffectAsset):
        return document.dependencies
    return tuple(entry.asset for entry in document.entries)


def _parse_references(value: Any, location: str) -> tuple[EffectAssetReference, ...]:
    if type(value) is not list:
        raise TypeError(f"{location} must be an array")
    references = (
        _parse_reference(item, f"{location}[{index}]")
        for index, item in enumerate(value)
    )
    return tuple(reference for reference in references if reference is not None)


def _parse_reference(value: Any, location: str) -> EffectAssetReference | None:
    if type(value) is not dict:
        raise TypeError(f"{location} must be an object")
    guid = value.get("guid", "")
    if type(guid) is not str:
        raise TypeError(f"{location}.guid must be a string")
    if not guid.strip():
        return None
    return EffectAssetReference(guid=guid)


def _require_required_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    missing = sorted(expected - set(value))
    if missing:
        raise ValueError(f"{location} is missing required keys: {missing}")


def _require_json_object(value: Mapping[str, Any], location: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{location} must be an object")
    _json_clone(dict(value))


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("render effect values must be finite JSON data") from exc
