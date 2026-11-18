#!/usr/bin/env python3
"""Build an Infernux plugin using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "package"
SOURCE_MANIFEST = "inx_package.json"
ARCHIVE_PREFIX = "package/"
PACKAGE_SCHEMA = "infernux.inxpackage"
RELEASE_SCHEMA = "infernux.plugin_release"
GUID_NAMESPACE = uuid.UUID("2bd3f0e2-0e94-4a61-bfe4-146b96bb66ab")
ALIGNMENT = 64
HEADER = struct.Struct("<8sIIIIQQQQQQQ32s16s")
TOC_PREFIX = struct.Struct("<4sIQQ")
ENTRY = struct.Struct("<QIIQQQB7sII8s")
HASH_OFFSET = 80


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def _content_hash(payload: bytes) -> str:
    value = 14695981039346656037
    for byte in payload:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _meta_bytes(reference: str, logical: str, source: Path, payload: bytes) -> tuple[str, bytes]:
    meta_path = source.with_name(source.name + ".meta")
    if meta_path.is_file():
        document = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata = document["metadata"]
        guid = str(metadata["guid"]["value"])
        if not re.fullmatch(r"[0-9a-fA-F]{32}", guid):
            raise ValueError(f"invalid GUID in {meta_path}")
    else:
        guid = uuid.uuid5(GUID_NAMESPACE, f"{reference}\0{logical}").hex
        document = {"metadata": {}}
        metadata = document["metadata"]
    metadata["guid"] = {"type": "string", "value": guid}
    metadata["content_hash"] = {"type": "string", "value": _content_hash(payload)}
    return guid, (json.dumps(document, ensure_ascii=False, indent=4) + "\n").encode()


def _role(logical: str) -> str:
    first = logical.split("/", 1)[0]
    if first in {"runtime", "editor"}:
        return first
    if first == "plugin_pages" or logical == "requirements.txt":
        return "control"
    if logical.endswith(".inxpkg"):
        return "nested_package"
    return "content"


def _title(path: Path) -> str:
    if path.suffix.casefold() in {".md", ".markdown"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
            if match:
                return match.group(1).strip(" #")
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _pages() -> list[dict[str, str]]:
    root = PACKAGE_ROOT / "plugin_pages"
    if not root.is_dir():
        return []
    result: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".md", ".markdown", ".txt"}:
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        stem = path.relative_to(root).with_suffix("").as_posix()
        locale = ""
        if stem.endswith(".zh-CN"):
            stem = stem[:-6]
            locale = "zh-CN"
        page_id = re.sub(r"[^a-z0-9._-]+", "-", stem.casefold().replace("/", ".")).strip("-._") or "page"
        item = {
            "id": page_id,
            "title": _title(path),
            "path": relative,
            "format": "markdown" if path.suffix.casefold() in {".md", ".markdown"} else "text",
        }
        if locale:
            item["locale"] = locale
        result.append(item)
    return result


def _payloads() -> tuple[dict[str, object], list[tuple[str, bytes]]]:
    source = json.loads((PACKAGE_ROOT / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    reference = str(source.get("reference", ""))
    if not reference or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", reference):
        raise ValueError("inx_package.json has an invalid reference")
    allowed = {"$schema", "reference", "name", "version", "engine", "intro", "intros", "pages", "category"}
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError("unsupported inx_package.json fields: " + ", ".join(unknown))
    theme_files = sorted((PACKAGE_ROOT / "editor/themes").glob("*.theme.json"))
    if theme_files:
        schema_path = ROOT / "theme_schema.py"
        table_path = Path(__file__).resolve().parents[2] / "cpp/infernux/function/editor/EditorThemeTable.inl"
        if not schema_path.is_file():
            schema_path = Path(__file__).resolve().parents[2] / "python/infernux/theme_data.py"
        if not schema_path.is_file():
            raise ValueError("Theme packages require theme_schema.py and theme-tokens.json from the theme template")
        spec = importlib.util.spec_from_file_location("infernux_packager_theme_schema", schema_path)
        schema = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema)
        tokens_path = ROOT / "theme-tokens.json"
        tokens = json.loads(tokens_path.read_text(encoding="utf-8")) if tokens_path.is_file() else schema.theme_token_names(table_path.read_text(encoding="utf-8"))
        identities = set()
        for path in theme_files:
            try:
                theme = schema.validate_theme_document(json.loads(path.read_text(encoding="utf-8")), tokens)
                if theme["id"] in identities:
                    raise ValueError("Duplicate theme id: " + theme["id"])
                identities.add(theme["id"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}: {exc}") from exc
    records: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.name == SOURCE_MANIFEST:
            continue
        if path.suffix.casefold() == ".meta" or any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        logical = path.relative_to(PACKAGE_ROOT).as_posix()
        first = logical.split("/", 1)[0]
        if first.casefold() in {"runtime", "editor", "plugin_pages"} and first not in {"runtime", "editor", "plugin_pages"}:
            raise ValueError(f"package directory must use Pythonic lowercase naming: {logical}")
        content = path.read_bytes()
        guid, meta = _meta_bytes(reference, logical, path, content)
        archive_path = ARCHIVE_PREFIX + logical
        records.append({
            "logical_path": logical,
            "guid": guid,
            "role": _role(logical),
            "archive_path": archive_path,
            "meta_archive_path": archive_path + ".meta",
        })
        payloads.extend(((archive_path, content), (archive_path + ".meta", meta)))
    if not records:
        raise ValueError("package contains no distributable files")
    explicit_pages = source.get("pages")
    pages = _pages() if explicit_pages is None else explicit_pages
    metadata = {
        "$schema": PACKAGE_SCHEMA,
        "reference": reference,
        "name": str(source.get("name") or reference.rsplit("/", 1)[-1]),
        "version": str(source.get("version") or "0.0.0"),
        "intro": str(source.get("intro") or ""),
        "intros": dict(source.get("intros") or {}),
        "engine": str(source.get("engine") or ""),
        "pages": pages,
        "control_guid": uuid.uuid5(GUID_NAMESPACE, f"{reference}\0{SOURCE_MANIFEST}").hex,
        "files": records,
    }
    if "category" in source:
        if not isinstance(source["category"], str):
            raise ValueError("inx_package.json category must be a string")
        metadata["category"] = source["category"]
    manifest = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    return metadata, [(SOURCE_MANIFEST, manifest), *payloads]


def _write_pack(destination: Path, payloads: list[tuple[str, bytes]]) -> None:
    payloads = sorted(payloads, key=lambda item: item[0].encode())
    strings = b"".join(path.encode() for path, _ in payloads)
    toc_bytes = _align(TOC_PREFIX.size + len(payloads) * ENTRY.size + len(strings))
    payload_start = _align(HEADER.size + toc_bytes)
    entries = bytearray()
    body = bytearray()
    path_offset = 0
    raw_bytes = 0
    for path, payload in payloads:
        encoded = path.encode()
        offset = len(body)
        entries += ENTRY.pack(path_offset, len(encoded), 0, offset, len(payload), len(payload), 0, b"\0" * 7, ALIGNMENT, 0, b"\0" * 8)
        body += payload
        body += b"\0" * (_align(len(body)) - len(body))
        path_offset += len(encoded)
        raw_bytes += len(payload)
    toc = TOC_PREFIX.pack(b"TOC0", 0, len(payloads), len(strings)) + entries + strings
    toc += b"\0" * (toc_bytes - len(toc))
    header = HEADER.pack(b"INXPKG\0\0", HEADER.size, 0, ENTRY.size, 0, HEADER.size, toc_bytes, payload_start, len(body), len(payloads), len(strings), raw_bytes, b"\0" * 32, b"\0" * 16)
    archive = bytearray(header + toc + body)
    digest = hashlib.sha256(archive).digest()
    archive[HASH_OFFSET:HASH_OFFSET + 32] = digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(archive)
    os.replace(temporary, destination)


def _read_metadata(path: Path) -> dict[str, object]:
    data = bytearray(path.read_bytes())
    values = HEADER.unpack_from(data)
    if values[0] != b"INXPKG\0\0" or values[1] != HEADER.size or values[3] != ENTRY.size:
        raise ValueError("unsupported InxPack header")
    expected = values[12]
    data[HASH_OFFSET:HASH_OFFSET + 32] = b"\0" * 32
    if hashlib.sha256(data).digest() != expected:
        raise ValueError("InxPack archive identity does not match")
    toc_offset, _toc_bytes, payload_start, _payload_bytes, count, string_bytes = values[5:11]
    magic, _flags, toc_count, toc_strings = TOC_PREFIX.unpack_from(data, toc_offset)
    if magic != b"TOC0" or toc_count != count or toc_strings != string_bytes:
        raise ValueError("invalid InxPack table of contents")
    records = toc_offset + TOC_PREFIX.size
    strings = records + count * ENTRY.size
    for index in range(count):
        entry = ENTRY.unpack_from(data, records + index * ENTRY.size)
        path_bytes = bytes(data[strings + entry[0]:strings + entry[0] + entry[1]])
        if path_bytes.decode() == SOURCE_MANIFEST:
            if entry[6] != 0:
                raise ValueError("standalone verifier only accepts stored manifests")
            start = payload_start + entry[3]
            return json.loads(bytes(data[start:start + entry[5]]).decode())
    raise ValueError("InxPackage is missing inx_package.json")


def build(output: Path) -> None:
    metadata, payloads = _payloads()
    _write_pack(output, payloads)
    print(json.dumps({"path": str(output), "reference": metadata["reference"], "version": metadata["version"], "bytes": output.stat().st_size}))


def verify(path: Path) -> dict[str, object]:
    metadata = _read_metadata(path)
    if metadata.get("$schema") != PACKAGE_SCHEMA:
        raise ValueError("unsupported InxPackage metadata")
    print(json.dumps({"path": str(path), "reference": metadata.get("reference"), "version": metadata.get("version"), "verified": True}))
    return metadata


def release_manifest(path: Path, output: Path, tag: str) -> None:
    metadata = verify(path)
    expected = f"v{metadata['version']}"
    if tag and tag != expected:
        raise ValueError(f"release tag {tag!r} does not match {expected!r}")
    document = {
        "$schema": RELEASE_SCHEMA,
        "reference": metadata["reference"],
        "version": metadata["version"],
        "engine": metadata["engine"],
        "artifact": {"name": path.name},
        "generator": {"name": "package.py", "version": "1"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global ROOT, PACKAGE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT, help="Plugin repository containing package/")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("output", nargs="?", type=Path, default=Path("dist/plugin.inxpkg"))
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("package", type=Path)
    release_parser = commands.add_parser("release-manifest")
    release_parser.add_argument("package", type=Path)
    release_parser.add_argument("--output", required=True, type=Path)
    release_parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ROOT = args.source.resolve()
    PACKAGE_ROOT = ROOT / "package"
    if args.command == "build":
        build(args.output)
    elif args.command == "verify":
        verify(args.package)
    else:
        release_manifest(args.package, args.output, args.tag)


if __name__ == "__main__":
    main()
