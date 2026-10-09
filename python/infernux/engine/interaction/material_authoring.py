"""Persist editor edits without rewriting unchanged, resolved material values."""
from __future__ import annotations

import copy
import json


def dump_material_document(document: dict) -> str:
    authored = dict(document)
    authored.pop("_shader_property_order", None)  # Inspector layout, not asset data.
    return json.dumps(authored, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _apply_local_edits(authored, before, after):
    if before == after:
        return copy.deepcopy(authored)
    if not all(isinstance(value, dict) for value in (authored, before, after)):
        return copy.deepcopy(after)
    result = copy.deepcopy(authored)
    for key in before.keys() - after.keys():
        result.pop(key, None)
    for key, value in after.items():
        if key not in before:
            result[key] = copy.deepcopy(value)
        elif before[key] != value:
            result[key] = _apply_local_edits(result.get(key), before[key], value)
    return result


class MaterialAuthoringSnapshot:
    """One editor transaction baseline, never an external-file merge.

    Native loading resolves defaults and float32 values. Preserve the authored
    representation of unchanged fields. External writes still fail the existing
    DocumentStore expected-state check; this snapshot never rereads or merges them.
    """

    def __init__(self, path: str, before: dict):
        try:
            with open(path, "r", encoding="utf-8") as stream:
                authored = json.load(stream)
        except FileNotFoundError:
            authored = before  # First publication of a newly created material.
        if not isinstance(authored, dict):
            raise ValueError("Material authoring source must contain a JSON object")
        self._authored = copy.deepcopy(authored)
        self._before = copy.deepcopy(before)

    def dump(self, current: dict) -> str:
        return dump_material_document(_apply_local_edits(self._authored, self._before, current))
