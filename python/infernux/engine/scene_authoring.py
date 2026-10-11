"""Explicit boundaries between authored Scene files and runtime snapshots."""
from __future__ import annotations


def encode_scene_document(snapshot: dict) -> dict:
    from infernux.lib import _Infernux
    return _Infernux._encode_scene_authoring_document(snapshot)


def decode_scene_document(document: dict) -> dict:
    from infernux.lib import _Infernux
    return _Infernux._decode_scene_authoring_document(document)


def encode_runtime_scene_artifact(snapshot: dict) -> dict:
    """The cook owns resolved compact IDs; it never allocates random file GUIDs."""
    if not isinstance(snapshot, dict) or "identity_format" in snapshot:
        raise ValueError("Scene cooking requires a resolved runtime snapshot")
    from infernux.lib import _Infernux
    _Infernux._validate_resolved_scene_document(snapshot)
    result = dict(snapshot)
    result.pop("authoring_identity", None)
    result.pop("nextObjectId", None)
    result.pop("nextComponentId", None)
    result["identity_format"] = "runtime-v1"
    return result
