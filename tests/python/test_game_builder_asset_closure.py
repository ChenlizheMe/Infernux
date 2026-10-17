from __future__ import annotations

import json


from infernux.engine.runtime_artifact_catalog import unix_ns_to_filetime_ticks


def _content_hash(payload: bytes) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _write_asset_index(project, entries):
    current_entries = []
    for entry in entries:
        current = dict(entry)
        source = project / current["normalized_path"]
        stat = source.stat()
        current["source"] = {
            "size": stat.st_size,
            "modified_ns": unix_ns_to_filetime_ticks(stat.st_mtime_ns),
        }
        current["content_hash"] = _content_hash(source.read_bytes())
        current_entries.append(current)
    library = project / "Library"
    library.mkdir(parents=True, exist_ok=True)
    (library / "AssetIndex.json").write_text(
        json.dumps({"entries": current_entries}),
        encoding="utf-8",
    )


def _entry(guid, path, dependencies=()):
    return {
        "guid": guid,
        "normalized_path": path,
        "source": {"size": 1, "modified_ns": 1},
        "content_hash": "a" * 16,
        "dependencies": list(dependencies),
    }
