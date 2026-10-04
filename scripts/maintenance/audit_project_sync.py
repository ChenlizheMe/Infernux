"""Audit a project checkout before sharing or opening it from Git.

The engine deliberately keeps imported state in ``Library`` and keeps stable
asset identity in tracked ``.meta`` sidecars.  This command checks that the
boundary is intact: generated state is ignored, authored assets have metadata,
GUIDs are unique, package registry paths are portable, and structured asset
documents contain no machine-local absolute paths.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


GUID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
ABSOLUTE_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{2}|/)")

REQUIRED_IGNORE = (
    "/Library/",
    "/Temp/",
    "/Logs/",
    "/Cache/",
    "/.runtime/",
    "/.venv/",
    "/Build/",
    "/Builds/",
    "/Dist/",
    "/Export/",
    "/Exports/",
    "/ProjectSettings/.infernux-engine-lock.json",
    "*.meta.tmp",
)

REQUIRED_ATTRIBUTES = (
    ("*.scene", ("text", "eol=lf", "merge=infernux")),
    ("*.prefab", ("text", "eol=lf", "merge=infernux")),
    ("*.mat", ("text", "eol=lf", "merge=infernux")),
    ("*.meta", ("text", "eol=lf", "merge=infernux")),
    ("*.json", ("text", "eol=lf")),
    ("*.png", ("binary",)),
    ("*.fbx", ("binary",)),
    ("*.wav", ("binary",)),
)

STRUCTURED_SUFFIXES = {
    ".scene",
    ".prefab",
    ".mat",
    ".effect",
    ".effectgroup",
    ".particlegraph",
}

SKIP_META_NAMES = {".gitignore", ".gitattributes"}
SKIP_DIRECTORIES = {"__pycache__", ".staging", ".cache", ".pytest_cache"}
PATH_KEYS = {"file_path", "path", "path_hint", "scene_path", "asset_path"}


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    asset_count: int = 0
    package_count: int = 0
    guid_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _relative(project: Path, path: Path) -> str:
    return path.relative_to(project).as_posix()


def _read_lines(path: Path, report: AuditReport) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        report.errors.append(f"{_relative(path.parent, path)}: cannot read: {exc}")
        return []


def _has_ignore(lines: Iterable[str], expected: str) -> bool:
    return any(line.strip() == expected for line in lines)


def _has_attribute(lines: Iterable[str], pattern: str, tokens: tuple[str, ...]) -> bool:
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields and fields[0] == pattern and all(token in fields[1:] for token in tokens):
            return True
    return False


def _json(path: Path, report: AuditReport) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.errors.append(f"{path}: invalid JSON: {exc}")
        return None


def _metadata_value(document: Any, key: str) -> Any | None:
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        return None
    entry = document["metadata"].get(key)
    if not isinstance(entry, dict) or "value" not in entry:
        return None
    return entry["value"]


def _is_absolute(value: str) -> bool:
    return bool(ABSOLUTE_RE.match(value.replace("\\", "/")))


def _walk_absolute_strings(value: Any, location: str, report: AuditReport, *, path_field: bool = False) -> None:
    if path_field and isinstance(value, str) and _is_absolute(value):
        report.errors.append(f"{location}: machine-local absolute path '{value}'")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _walk_absolute_strings(child, f"{location}.{key}", report, path_field=key in PATH_KEYS)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_absolute_strings(child, f"{location}[{index}]", report)


def _audit_asset_roots(project: Path, report: AuditReport) -> dict[str, str]:
    guids: dict[str, str] = {}
    paths: dict[str, str] = {}
    for root_name in ("Assets", "Packages"):
        root = project / root_name
        if not root.is_dir():
            if root_name == "Assets":
                report.errors.append(f"{root_name}/ is required")
            continue
        for path in sorted(root.rglob("*")):
            relative_parts = path.relative_to(root).parts
            if any(part in SKIP_DIRECTORIES for part in relative_parts):
                continue
            if path.is_symlink():
                report.errors.append(f"{_relative(project, path)}: symlink assets are not portable")
                continue
            if not path.is_file() or path.name in SKIP_META_NAMES:
                continue
            relative = _relative(project, path)
            key = relative.casefold()
            if key in paths:
                report.errors.append(f"{relative}: case-only path collision with {paths[key]}")
            paths[key] = relative
            if path.suffix.casefold() == ".meta":
                if not Path(str(path)[:-5]).is_file():
                    report.errors.append(f"{relative}: orphan .meta sidecar")
                continue
            if path.suffix.casefold() in {".pyc", ".pyo", ".tmp", ".bak", ".orig", ".log"}:
                continue
            with path.open("rb") as stream:
                if stream.readline(128).rstrip(b"\r\n") == b"version https://git-lfs.github.com/spec/v1":
                    report.errors.append(f"{relative}: Git LFS payload has not been downloaded")
            report.asset_count += 1
            if root_name == "Packages":
                report.package_count += 1
            meta_path = Path(str(path) + ".meta")
            if not meta_path.is_file():
                report.errors.append(f"{_relative(project, path)}: missing tracked .meta sidecar")
                continue
            document = _json(meta_path, report)
            _audit_metadata_paths(document, _relative(project, meta_path), report)
            guid = _metadata_value(document, "guid")
            if not isinstance(guid, str) or not GUID_RE.fullmatch(guid):
                report.errors.append(f"{_relative(project, meta_path)}: invalid canonical GUID")
            elif guid.casefold() in guids:
                report.errors.append(
                    f"{_relative(project, meta_path)}: duplicate GUID {guid} also used by {guids[guid.casefold()]}"
                )
            else:
                guids[guid.casefold()] = _relative(project, path)
                report.guid_count += 1

            file_path = _metadata_value(document, "file_path")
            if isinstance(file_path, str):
                if _is_absolute(file_path):
                    report.errors.append(
                        f"{_relative(project, meta_path)}: metadata.file_path must be project-relative"
                    )
                else:
                    expected = _relative(project, path)
                    actual = file_path.replace("\\", "/")
                    if actual != expected:
                        report.errors.append(
                            f"{_relative(project, meta_path)}: metadata.file_path '{file_path}' "
                            f"does not identify {expected}"
                        )

            if path.suffix.casefold() in STRUCTURED_SUFFIXES:
                structured = _json(path, report)
                if structured is not None:
                    _walk_absolute_strings(structured, _relative(project, path), report)
    return guids


def _audit_metadata_paths(document: Any, location: str, report: AuditReport) -> None:
    if isinstance(document, list):
        for index, item in enumerate(document):
            _audit_metadata_paths(item, f"{location}[{index}]", report)
    elif isinstance(document, dict):
        fields = document.get("metadata")
        if isinstance(fields, dict):
            path = _metadata_value(document, "file_path")
            if isinstance(path, str) and _is_absolute(path):
                report.errors.append(f"{location}: metadata.file_path must be project-relative")
            for key in ("model_textures", "model_animations"):
                table = _metadata_value(document, key)
                if isinstance(table, str):
                    try:
                        _audit_metadata_paths(json.loads(table), f"{location}.{key}", report)
                    except json.JSONDecodeError:
                        report.errors.append(f"{location}.{key}: invalid nested metadata JSON")
        for key, item in document.items():
            _audit_metadata_paths(item, f"{location}.{key}", report)


def _audit_plugin_registry(project: Path, guids: dict[str, str], report: AuditReport) -> None:
    registry_path = project / "ProjectSettings" / "InxPlugins.json"
    if not registry_path.is_file():
        return
    document = _json(registry_path, report)
    if not isinstance(document, dict) or document.get("$schema") != "infernux.plugin_registry":
        report.errors.append(f"{_relative(project, registry_path)}: invalid plugin registry schema")
        return
    for section in ("packages", "installed"):
        records = document.get(section)
        if not isinstance(records, list):
            report.errors.append(f"{_relative(project, registry_path)}: {section} must be an array")
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                report.errors.append(f"{_relative(project, registry_path)}:{section}[{index}] is not an object")
                continue
            source = record.get("source")
            if isinstance(source, dict) and str(source.get("cache_scope", "")).casefold() == "hub":
                location = str(source.get("cache_location", "")).replace("\\", "/")
                parts = location.split("/")
                if (len(parts) < 3 or parts[0] != "packages" or
                    any(part in {"", ".", ".."} for part in parts) or
                    not location.endswith(".inxpkg") or _is_absolute(location)):
                    report.errors.append(
                        f"{_relative(project, registry_path)}:{section}[{index}] has a non-portable Hub cache_location"
                    )
            if isinstance(source, dict) and source.get("type") == "local" and _is_absolute(str(source.get("location", ""))):
                report.errors.append(f"{section}[{index}]: local package source must be portable")
            if _is_absolute(str(record.get("package_path", ""))):
                report.errors.append(f"{section}[{index}]: package_path must be cache-relative")
            files = record.get("files", [])
            if not isinstance(files, list):
                report.errors.append(f"{section}[{index}]: files must be an array")
                continue
            control = record.get("control")
            ownership = [*files, control] if isinstance(control, dict) else files
            for file_index, file_record in enumerate(ownership):
                if not isinstance(file_record, dict):
                    report.errors.append(
                        f"{_relative(project, registry_path)}:{section}[{index}].files[{file_index}] is not an object"
                    )
                    continue
                hint = str(file_record.get("path_hint", "")).replace("\\", "/")
                if hint and (not (hint.startswith("Assets/") or hint.startswith("Packages/")) or
                             any(part in {"", ".", ".."} for part in hint.split("/"))):
                    report.errors.append(
                        f"{_relative(project, registry_path)}:{section}[{index}] has an invalid path_hint {hint!r}"
                    )
                guid = str(file_record.get("guid", "")).casefold()
                if hint and not (project / hint).is_file() and guid not in guids:
                    report.errors.append(
                        f"{_relative(project, registry_path)}:{section}[{index}] references missing {hint}"
                    )
                if guid and guid not in guids:
                    report.errors.append(
                        f"{_relative(project, registry_path)}:{section}[{index}] references unknown GUID {guid}"
                    )
                elif guid and hint and guids[guid] != hint:
                    report.warnings.append(f"{section}[{index}]: stale path_hint {hint}; GUID resolves to {guids[guid]}")


def audit_project(project_root: str | Path, *, require_tracked: bool = False) -> AuditReport:
    project = Path(project_root).expanduser().resolve()
    report = AuditReport()
    if not project.is_dir():
        report.errors.append(f"project does not exist: {project}")
        return report
    for directory in ("Assets", "ProjectSettings"):
        if not (project / directory).is_dir():
            report.errors.append(f"{directory}/ is required")

    gitignore = project / ".gitignore"
    gitattributes = project / ".gitattributes"
    ignore_lines = _read_lines(gitignore, report) if gitignore.is_file() else []
    attribute_lines = _read_lines(gitattributes, report) if gitattributes.is_file() else []
    if not gitignore.is_file():
        report.errors.append(".gitignore is required for a shared project")
    if not gitattributes.is_file():
        report.errors.append(".gitattributes is required for a shared project")
    for expected in REQUIRED_IGNORE:
        if not _has_ignore(ignore_lines, expected):
            report.errors.append(f".gitignore is missing {expected}")
    for pattern, tokens in REQUIRED_ATTRIBUTES:
        if not _has_attribute(attribute_lines, pattern, tokens):
            report.errors.append(
                f".gitattributes is missing: {pattern} {' '.join(tokens)}"
            )

    guids = _audit_asset_roots(project, report)
    for path in sorted((project / "ProjectSettings").glob("*.json")):
        if path.name.startswith(".") or path.name == "InxPlugins.json":
            continue
        document = _json(path, report)
        _walk_absolute_strings(document, _relative(project, path), report)
    _audit_plugin_registry(project, guids, report)
    if require_tracked:
        _audit_git_index(project, guids, report)
    return report


def _audit_git_index(project: Path, guids: dict[str, str], report: AuditReport) -> None:
    completed = subprocess.run(
        ["git", "-C", str(project), "ls-files", "-z"], capture_output=True, check=False,
    )
    if completed.returncode:
        report.errors.append("--tracked requires a Git checkout")
        return
    driver = subprocess.run(
        ["git", "-C", str(project), "config", "--get", "merge.infernux.driver"], capture_output=True,
    )
    if driver.returncode or not driver.stdout.strip():
        report.errors.append("Infernux semantic merge driver is not installed; open this project with its engine version first")
    tracked = set(completed.stdout.decode("utf-8").split("\0"))
    for path in [*guids.values(), *(path + ".meta" for path in guids.values()), ".gitignore", ".gitattributes"]:
        if path not in tracked:
            report.errors.append(f"{path}: not tracked in Git")
    for path in tracked:
        if path.split("/", 1)[0] in {"Library", "Temp", "Logs", "Cache", ".runtime", ".venv", "Build", "Builds", "Dist", "Export", "Exports"}:
            report.errors.append(f"{path}: generated state is tracked in Git")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", help="project root containing Assets and ProjectSettings")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit a machine-readable report")
    parser.add_argument("--tracked", action="store_true", help="require source assets and sidecars in the Git index")
    args = parser.parse_args(argv)
    report = audit_project(args.project, require_tracked=args.tracked)
    if args.as_json:
        print(json.dumps({
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "asset_count": report.asset_count,
            "package_count": report.package_count,
            "guid_count": report.guid_count,
        }, ensure_ascii=False, indent=2))
    else:
        for message in report.errors:
            print(f"ERROR: {message}", file=sys.stderr)
        for message in report.warnings:
            print(f"WARNING: {message}", file=sys.stderr)
        if report.ok:
            print(
                f"Project sync audit passed: assets={report.asset_count} "
                f"packages={report.package_count} guids={report.guid_count}"
            )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
