"""Authoritative Python filesystem path operations.

Display paths, persistent portable paths, and dictionary identity keys are
different concepts. Callers must choose the operation matching their intent
instead of assembling identity checks from ``abspath``/``normcase``.
"""

from __future__ import annotations

import hashlib
import os
from typing import TypeAlias

PathLike: TypeAlias = str | os.PathLike[str]


def lexical_path(path: PathLike) -> str:
    """Return an absolute normalized path without consulting the filesystem."""
    if not path:
        return ""
    return os.path.abspath(os.path.normpath(os.fspath(path)))


def resolved_path(path: PathLike) -> str:
    """Return an absolute display/storage path with existing aliases resolved."""
    if not path:
        return ""
    lexical = lexical_path(path)
    # CPython's Windows realpath implementation already resolves reparse
    # points, expands 8.3 names, and preserves a non-existent suffix after the
    # nearest existing ancestor.  Repeating that work through
    # GetLongPathNameW made this identity primitive perform another filesystem
    # query on every call, which is especially costly during plugin discovery.
    return os.path.normpath(os.path.realpath(lexical))


def path_key(path: PathLike) -> str:
    """Return the cross-platform identity key for a filesystem path."""
    return os.path.normcase(resolved_path(path))


def path_fingerprint(path: PathLike) -> str:
    """Return a stable, non-reversible identity for a filesystem path."""
    identity = path_key(path)
    if not identity:
        return ""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def lexical_path_key(path: PathLike) -> str:
    """Return a disk-independent key for a path that has no known identity."""
    if not path:
        return ""
    return os.path.normcase(lexical_path(path))


def same_path(left: PathLike, right: PathLike) -> bool:
    """Return whether two path spellings refer to the same filesystem identity."""
    if not left or not right:
        return False
    try:
        if os.path.exists(left) and os.path.exists(right):
            return os.path.samefile(left, right)
    except OSError:
        pass
    return path_key(left) == path_key(right)


def is_path_within(path: PathLike, root: PathLike, *, allow_root: bool = True) -> bool:
    """Return whether *path* resolves inside *root* using path components."""
    candidate = path_key(path)
    parent = path_key(root)
    if not candidate or not parent:
        return False
    try:
        inside = os.path.commonpath((candidate, parent)) == parent
    except ValueError:
        return False
    return inside and (allow_root or candidate != parent)


def is_lexical_path_within(
    path: PathLike,
    root: PathLike,
    *,
    allow_root: bool = True,
) -> bool:
    """Return whether *path* is lexically below *root* without following links.

    This is for capability code that already owns a canonical root and must
    compare an opened handle's final path against that immutable spelling. It
    must not be used as a substitute for :func:`is_path_within` when aliases
    have not already been rejected or resolved.
    """
    candidate = lexical_path_key(path)
    parent = lexical_path_key(root)
    if not candidate or not parent:
        return False
    try:
        inside = os.path.commonpath((candidate, parent)) == parent
    except ValueError:
        return False
    return inside and (allow_root or candidate != parent)


def relative_path(
    path: PathLike,
    root: PathLike,
    *,
    resolve: bool = True,
    allow_root: bool = False,
) -> str:
    """Return a portable relative path, rejecting paths outside *root*."""
    normalize = resolved_path if resolve else lexical_path
    key = path_key if resolve else lexical_path_key
    candidate = normalize(path)
    parent = normalize(root)
    candidate_key = key(candidate)
    parent_key = key(parent)
    try:
        inside = os.path.commonpath((candidate_key, parent_key)) == parent_key
    except ValueError:
        inside = False
    if not inside or (not allow_root and candidate_key == parent_key):
        raise ValueError(f"Path is outside root: {candidate!r} is not under {parent!r}")
    relative = os.path.relpath(candidate, parent)
    if relative == "." and not allow_root:
        raise ValueError("Path must name an entry below the root")
    return portable_path(relative)


def portable_path(path: PathLike) -> str:
    """Normalize separators for a project-relative path stored in an asset."""
    if not path:
        return ""
    return os.path.normpath(os.fspath(path)).replace("\\", "/")


def portable_relative_path(path: PathLike, *, allow_root: bool = False) -> str:
    """Normalize and validate an engine-owned portable relative path."""
    normalized = portable_path(path)
    if not normalized or os.path.isabs(normalized):
        raise ValueError(f"Path must be relative: {path!r}")
    parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise ValueError(f"Path escapes its logical root: {path!r}")
    if not parts:
        if allow_root:
            return "."
        raise ValueError("Path must name an entry below the logical root")
    return "/".join(parts)


def safe_path(path: PathLike) -> str:
    """Normalize a Python path before crossing the UTF-8 C++ boundary."""
    return resolved_path(path)
