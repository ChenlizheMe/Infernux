from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import zipfile
# LZMA payload members need the stdlib decoder; import it explicitly so the
# frozen installer can never be built without it.
import lzma  # noqa: F401
from pathlib import Path, PurePosixPath
from typing import Mapping


HUB_EXECUTABLE = "Infernux Hub.exe" if sys.platform == "win32" else "Infernux Hub"
HUB_PAYLOAD_ARCHIVE = "infernux-hub-payload.zip"
BUNDLED_ENGINES_DIR = PurePosixPath("InfernuxHubData/engines")
# Members that are already compressed archives (the LZMA runtime bundle and
# engine wheels) gain nothing from a second compression pass.
_STORED_SUFFIXES = frozenset({".zip", ".whl"})


def payload_compression(archive_name: str) -> int:
    """Return the ZIP compression used for one installer payload member."""
    if PurePosixPath(archive_name).suffix.casefold() in _STORED_SUFFIXES:
        return zipfile.ZIP_STORED
    return zipfile.ZIP_LZMA


def create_payload_archive(
    source_dir: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    extra_files: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Archive the staged Hub directory for the installer.

    ``extra_files`` maps payload-relative POSIX names to files that are
    installed with the Hub but are not part of the Hub update manifest, such as
    the bundled engine wheel under ``InfernuxHubData/engines``.
    """
    source = Path(source_dir).resolve()
    output = Path(destination).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Hub payload directory not found: {source}")
    if not (source / HUB_EXECUTABLE).is_file():
        raise RuntimeError(f"Hub payload is missing {HUB_EXECUTABLE}: {source}")

    members: dict[str, Path] = {}
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        members[source_file.relative_to(source).as_posix()] = source_file
    for name, extra in (extra_files or {}).items():
        relative = _safe_archive_path(name).as_posix()
        extra_path = Path(extra)
        if not extra_path.is_file():
            raise RuntimeError(f"Extra Hub payload file not found: {extra_path}")
        if any(existing.casefold() == relative.casefold() for existing in members):
            raise RuntimeError(f"Duplicate path in Hub payload archive: {relative}")
        members[relative] = extra_path

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as archive:
            for name, member in members.items():
                archive.write(member, name, compress_type=payload_compression(name))
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _safe_archive_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or (relative.parts and ":" in relative.parts[0])
    ):
        raise RuntimeError(f"Unsafe path in Hub payload archive: {name!r}")
    return relative


def extract_payload_archive(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    archive_file = Path(archive_path).resolve()
    output = Path(destination).resolve()
    if not archive_file.is_file():
        raise RuntimeError(f"Hub payload archive not found: {archive_file}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_file, mode="r") as archive:
            for entry in archive.infolist():
                relative = _safe_archive_path(entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError(
                        f"Links are not allowed in the Hub payload archive: {entry.filename!r}"
                    )
                target = output.joinpath(*relative.parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(entry, mode="r") as source,
                    target.open("wb") as destination_file,
                ):
                    shutil.copyfileobj(source, destination_file, length=1024 * 1024)
                archived_mode = mode & 0o777
                if archived_mode:
                    target.chmod(archived_mode)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid Hub payload archive: {archive_file}") from exc

    if not (output / HUB_EXECUTABLE).is_file():
        raise RuntimeError(f"The Hub payload archive is missing {HUB_EXECUTABLE}.")
