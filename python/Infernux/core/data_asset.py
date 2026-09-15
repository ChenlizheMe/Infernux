"""Typed project data stored as a GUID-addressed Infernux asset."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import struct
from typing import Any
import zlib

from Infernux.components.serializable_object import SerializableObject


DATA_ASSET_EXTENSION = ".inxdata"
DATA_ASSET_ARTIFACT_EXTENSION = ".inxasset"
DATA_ASSET_DOCUMENT_TYPE = "data_asset"
_DATA_ASSET_ARTIFACT_MAGIC = b"INXDATA\0"
_DATA_ASSET_ARTIFACT_VERSION = 1
_DATA_ASSET_ARTIFACT_HEADER = struct.Struct("<8sII")


def encode_data_asset_artifact(document: dict[str, Any]) -> bytes:
    """Serialize one DataAsset document into the Player binary format."""
    # Cook is the compatibility boundary: Player artifacts always contain the
    # current schema even when the authored source predates versioned data.
    document = DataAsset.from_document(document).serialize_document()
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _DATA_ASSET_ARTIFACT_HEADER.pack(
        _DATA_ASSET_ARTIFACT_MAGIC,
        _DATA_ASSET_ARTIFACT_VERSION,
        len(payload),
    ) + zlib.compress(payload, level=9)


def decode_data_asset_artifact(payload: bytes) -> dict[str, Any]:
    """Read the current Player binary format into its authored document."""
    if len(payload) < _DATA_ASSET_ARTIFACT_HEADER.size:
        raise ValueError("truncated DataAsset artifact")
    magic, version, document_size = _DATA_ASSET_ARTIFACT_HEADER.unpack_from(payload)
    if magic != _DATA_ASSET_ARTIFACT_MAGIC or version != _DATA_ASSET_ARTIFACT_VERSION:
        raise ValueError("unsupported DataAsset artifact")
    try:
        document_bytes = zlib.decompress(payload[_DATA_ASSET_ARTIFACT_HEADER.size :])
    except zlib.error as exc:
        raise ValueError("invalid DataAsset artifact payload") from exc
    if len(document_bytes) != document_size:
        raise ValueError("DataAsset artifact document size mismatch")
    try:
        document = json.loads(document_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid DataAsset artifact document") from exc
    DataAsset.from_document(document)
    return document


class DataAsset(SerializableObject):
    """Base class for authored, shared project data.

    Subclasses must declare a stable ``__serialized_type_id__``. The ID is
    resolved only through the published SerializableObject catalog; loading a
    document never imports a module named by asset content.
    """

    __serialized_type_id__ = "infernux.data_asset"

    def __init_subclass__(cls, **kwargs):
        explicit_type_id = cls.__dict__.get("__serialized_type_id__", "")
        if not isinstance(explicit_type_id, str) or not explicit_type_id.strip():
            raise TypeError(
                "DataAsset subclasses require an explicit stable "
                "__serialized_type_id__"
            )
        cls.__serialized_type_id__ = explicit_type_id.strip()
        super().__init_subclass__(**kwargs)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._asset_guid = ""
        self._asset_path = ""
        self._play_isolated = False

    @property
    def guid(self) -> str:
        return str(getattr(self, "_asset_guid", "") or "")

    @property
    def file_path(self) -> str:
        return str(getattr(self, "_asset_path", "") or "")

    @property
    def is_persistent(self) -> bool:
        return bool(self.guid and self.file_path)

    def serialize_document(self) -> dict[str, Any]:
        document = super()._serialize()
        document["$type"] = DATA_ASSET_DOCUMENT_TYPE
        return document

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "DataAsset":
        if not isinstance(document, dict):
            raise TypeError("DataAsset document must be an object")
        legacy_keys = {"$type", "type_id", "fields"}
        current_keys = legacy_keys | {"schema_version"}
        if set(document) not in (legacy_keys, current_keys) or document.get("$type") != DATA_ASSET_DOCUMENT_TYPE:
            raise ValueError("invalid DataAsset document")
        nested = dict(document)
        nested["$type"] = "serializable_object"
        value = SerializableObject._deserialize(nested)
        if not isinstance(value, DataAsset):
            raise TypeError("DataAsset document type must derive from DataAsset")
        if cls is not DataAsset and not isinstance(value, cls):
            raise TypeError(
                f"DataAsset document contains {type(value).__name__}, expected {cls.__name__}"
            )
        return value

    def deserialize_document(self, document: dict[str, Any]) -> bool:
        """Apply an authored document without replacing this asset identity."""
        from Infernux.components.fields import (
            get_raw_field_value,
            get_serialized_fields,
        )

        replacement = type(self).from_document(document)
        for field_name in get_serialized_fields(type(self)):
            setattr(
                self,
                field_name,
                copy.deepcopy(get_raw_field_value(replacement, field_name)),
            )
        return True

    @staticmethod
    def _resolve_project_path(path: str, database: Any = None) -> tuple[str, Any]:
        from Infernux.core.assets import AssetManager
        from Infernux.engine.path_utils import is_path_within, resolved_path

        database = database or AssetManager.require_asset_database()
        candidate = resolved_path(
            path if os.path.isabs(path) else os.path.join(database.project_root, path)
        )
        if os.path.splitext(candidate)[1].casefold() != DATA_ASSET_EXTENSION:
            raise ValueError(f"DataAsset files require the {DATA_ASSET_EXTENSION} extension")
        roots = (
            os.path.join(database.project_root, "Assets"),
            os.path.join(database.project_root, "Packages"),
        )
        if not any(is_path_within(candidate, root) for root in roots):
            raise ValueError("DataAsset files must be stored under Assets or Packages")
        return candidate, database

    def _bind_asset(self, path: str, guid: str) -> None:
        from Infernux.engine.path_utils import resolved_path

        object.__setattr__(self, "_asset_path", resolved_path(path) if path else "")
        object.__setattr__(self, "_asset_guid", str(guid or ""))

    @classmethod
    def load(cls, path: str) -> "DataAsset":
        from Infernux.core.assets import AssetManager
        from Infernux.engine.path_utils import resolved_path

        extension = os.path.splitext(path)[1].casefold()
        if extension == DATA_ASSET_EXTENSION:
            resolved, database = cls._resolve_project_path(path)
            document = json.loads(Path(resolved).read_text(encoding="utf-8"))
        elif extension == DATA_ASSET_ARTIFACT_EXTENSION:
            database = AssetManager.require_asset_database()
            resolved = resolved_path(
                path
                if os.path.isabs(path)
                else os.path.join(database.project_root, path)
            )
            if not database.get_guid_from_path(resolved):
                raise RuntimeError("DataAsset artifact is not registered in the AssetDatabase")
            document = decode_data_asset_artifact(Path(resolved).read_bytes())
        else:
            raise ValueError(
                "DataAsset loading requires an .inxdata source or .inxasset artifact"
            )
        value = cls.from_document(document)
        guid = str(database.get_guid_from_path(resolved) or "")
        if not guid:
            raise RuntimeError("DataAsset is not registered in the AssetDatabase")
        value._bind_asset(resolved, guid)
        return value

    def save_to(self, path: str, *, database: Any = None) -> str:
        from Infernux.application import Application
        from Infernux.core.assets import AssetManager
        from Infernux.core.document_store import write_document_text

        if Application.is_player():
            raise RuntimeError("DataAsset authoring is read-only in Player")
        if bool(getattr(self, "_play_isolated", False)):
            raise RuntimeError(
                "Play-isolated DataAsset values cannot overwrite authored assets"
            )
        resolved, database = self._resolve_project_path(path, database)
        if not os.path.isdir(os.path.dirname(resolved)):
            raise FileNotFoundError(os.path.dirname(resolved))
        content = json.dumps(
            self.serialize_document(), ensure_ascii=False, indent=2
        ) + "\n"
        write_document_text(resolved, content)

        guid = str(database.get_guid_from_path(resolved) or "")
        result = (
            AssetManager.reimport_asset(resolved, database=database)
            if guid
            else AssetManager.import_asset(resolved, database=database)
        )
        if not result:
            raise RuntimeError(str(getattr(result, "error", "DataAsset import failed")))
        guid = str(result.guid)
        AssetManager.invalidate(guid)
        self._bind_asset(resolved, guid)
        AssetManager._put_cache(guid, self)
        return resolved

    def save(self) -> None:
        if not self.file_path:
            raise RuntimeError("DataAsset has no persistent file path; call save_to(path)")
        self.save_to(self.file_path)

    def instantiate(self) -> "DataAsset":
        """Return an explicit mutable copy without persistent asset identity."""
        value = copy.deepcopy(self)
        value._bind_asset("", "")
        value._play_isolated = False
        return value

    def _clone_for_play(self) -> "DataAsset":
        value = type(self).from_document(self.serialize_document())
        value._bind_asset(self.file_path, self.guid)
        value._play_isolated = True
        return value

    def _mark_play_isolated(self) -> None:
        self._play_isolated = True


def get_registered_data_asset_types() -> tuple[tuple[str, type[DataAsset]], ...]:
    """Return authored DataAsset classes from the published semantic catalog."""
    from Infernux.components.serializable_object import get_registered_serializable_types

    return tuple(
        (type_id, value_type)
        for type_id, value_type in get_registered_serializable_types()
        if value_type is not DataAsset and issubclass(value_type, DataAsset)
    )


__all__ = [
    "DATA_ASSET_ARTIFACT_EXTENSION",
    "DATA_ASSET_DOCUMENT_TYPE",
    "DATA_ASSET_EXTENSION",
    "DataAsset",
    "decode_data_asset_artifact",
    "encode_data_asset_artifact",
    "get_registered_data_asset_types",
]
