"""Typed GPU source identity and discovery shared by authoring and tooling.

This module never imports Inspector widgets or starts an Engine. A running
asset registry owns imported references; standalone compilation uses the
same static declarations in the configured source roots.
"""
from __future__ import annotations

from functools import lru_cache
import os

from infernux.engine.path_utils import lexical_path_key, path_key, portable_path
from infernux._source_kinds import SHADER_SOURCE_KINDS, source_extension

_shader_property_generation = 0
_shader_catalog_cache = {}
_shader_visibility_cache = {}


def _matches_stage(filepath: str, ext: str) -> bool:
    return source_extension(filepath) == ext + ".py"


def _read_compiled_shader_metadata(filepath: str) -> dict | None:
    from infernux.core.asset_types import read_published_asset_metadata
    metadata = read_published_asset_metadata(filepath)
    return metadata if metadata and isinstance(metadata.get("shader_id"), str) else None


def _read_source_shader_metadata(filepath: str) -> dict:
    try:
        stat = os.stat(filepath)
    except OSError:
        return {}
    return _read_source_shader_metadata_revision(filepath, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=512)
def _read_source_shader_metadata_revision(filepath: str, _mtime_ns: int, _size: int) -> dict:
    from infernux._compiler.shader_info import read_shader_info
    try:
        with open(filepath, "r", encoding="utf-8-sig") as stream:
            info = read_shader_info(stream.read(), filepath)
    except (OSError, SyntaxError):
        # Invalid drafts stay absent from choices until their next source edit;
        # their importer publishes the actual source diagnostic.
        return {}
    return {"shader_id": info["shader_id"], "shader_hidden": info["hidden"],
            "queue": info["render_queue"], "domain": info["domain"]}


def bump_shader_property_generation():
    global _shader_property_generation
    _shader_property_generation += 1
    _shader_catalog_cache.clear()
    _shader_visibility_cache.clear()
    _read_source_shader_metadata_revision.cache_clear()
    _canonical_shader_roots.cache_clear()


def get_shader_property_generation() -> int:
    return _shader_property_generation


@lru_cache(maxsize=32)
def _canonical_shader_roots(roots: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path_key(root) for root in roots)


def get_shader_file_path(shader_id: str, ext: str) -> str | None:
    if not shader_id:
        return None
    from infernux.lib import AssetRegistry
    database = AssetRegistry.instance().get_asset_database()
    if database is not None:
        stage = SHADER_SOURCE_KINDS[ext + ".py"][0]
        return database.find_shader_path_by_id(shader_id, stage) or None
    paths = _get_shader_catalog(ext)["paths"]
    if shader_id in paths and paths[shader_id] is None:
        raise ValueError("Shader ID is ambiguous: " + shader_id + "; use a source path")
    return paths.get(shader_id)


def _get_shader_search_roots() -> list[str]:
    """Return the project and built-in shader roots."""
    from infernux.engine.project_context import get_project_root

    project_root = get_project_root()
    search_roots = []
    if project_root:
        search_roots.append(os.path.join(project_root, "Assets"))
        search_roots.append(os.path.join(project_root, "Packages"))

    from infernux.resources import resources_path
    builtin_root = os.path.join(resources_path, "shaders")
    search_roots.append(builtin_root)
    return search_roots

def _is_builtin_shader_path(filepath: str) -> bool:
    """Return whether a shader path is generated/package-owned engine data."""
    if not filepath:
        return False
    from infernux.engine.project_context import get_project_root
    from infernux.resources import resources_path

    candidate = lexical_path_key(filepath)
    portable_candidate = portable_path(candidate).casefold()
    if "/library/resources/shaders/" in portable_candidate:
        # Library is generated project data.  This remains identifiable while
        # opening/materializing a document before ProjectContext is published.
        return True
    project_root = get_project_root()
    roots = [os.path.join(resources_path, "shaders")]
    if project_root:
        roots.append(os.path.join(project_root, "Library", "Resources", "shaders"))
    for root in roots:
        root_key = lexical_path_key(root)
        prefix = root_key.rstrip("\\/") + os.sep
        if candidate == root_key or candidate.startswith(prefix):
            return True
    return False

def shader_menu_folders(shader_id: str) -> tuple[str, ...]:
    """Logical Shader ID folders; physical storage never participates in the menu."""
    return tuple(shader_id.split("/")[:-1])

def _scan_shader_catalog(ext: str, search_roots: list[str] | None = None) -> dict:
    """Discover typed sources; ambiguous IDs follow the native registry rule."""
    paths = {}
    for root in search_roots if search_roots is not None else _get_shader_search_roots():
        if not root or not os.path.isdir(root):
            continue
        for directory, _, filenames in os.walk(root):
            for name in filenames:
                if not _matches_stage(name, ext):
                    continue
                path = os.path.join(directory, name)
                shader_id = parse_shader_id(path)
                if shader_id:
                    paths[shader_id] = path if shader_id not in paths or paths[shader_id] == path else None
    items = [(shader_id, shader_id) for shader_id, path in paths.items()
             if path is not None and shader_picker_name(path, material_fragment=ext == ".frag") is not None]
    return {"items": items or [("(No shaders found)", "")], "paths": paths}


def _get_shader_catalog(ext: str) -> dict[str, object]:
    """Return cached shader catalog for the requested extension."""
    search_roots = _get_shader_search_roots()
    root_signature = _canonical_shader_roots(tuple(root for root in search_roots if root))
    cache_key = (ext, root_signature)
    catalog = _shader_catalog_cache.get(cache_key)
    if catalog is None:
        catalog = _scan_shader_catalog(ext, search_roots)
        _shader_catalog_cache[cache_key] = catalog
    return catalog

def parse_shader_id(filepath: str) -> str | None:
    """Return the canonical imported GPU source ID."""
    metadata = _read_compiled_shader_metadata(filepath)
    if metadata is not None:
        shader_id = metadata.get("shader_id", "").strip()
        if shader_id:
            return shader_id
    source_metadata = _read_source_shader_metadata(filepath)
    shader_id = source_metadata.get("shader_id", "")
    if isinstance(shader_id, str) and shader_id:
        return shader_id
    return None

def shader_picker_name(filepath: str, *, material_fragment: bool = False) -> str | None:
    """Read picker identity and domain without compiling or executing the source."""
    if is_shader_hidden(filepath):
        return None
    if material_fragment and source_extension(filepath) == ".frag.py":
        if _read_source_shader_metadata(filepath).get("domain") != "mesh":
            return None
    return parse_shader_id(filepath)

def is_shader_hidden(filepath: str) -> bool:
    """Check imported visibility, and the current static source declaration."""
    cache_key = path_key(filepath)
    try:
        stat = os.stat(filepath)
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        signature = (-1, -1)
    cached = _shader_visibility_cache.get(cache_key)
    if cached is not None and cached[:2] == signature:
        return cached[2]

    metadata = _read_compiled_shader_metadata(filepath)
    if metadata is not None and isinstance(metadata.get("shader_hidden"), bool):
        hidden = metadata["shader_hidden"]
    else:
        source_metadata = _read_source_shader_metadata(filepath)
        hidden = bool(source_metadata.get("shader_hidden", False))
    _shader_visibility_cache[cache_key] = (*signature, hidden)
    return hidden

def get_shader_candidates(ext: str, cache: dict = None, *, domain: str | None = None):
    """Collect shader files from project and built-in shader folders.
    Only typed sources with valid declared IDs are listed.
    Each unique shader_id appears only once in the list.

    A typed consumer can request its domain from the same source index.
    The default fragment menu remains the mesh-material menu.
    """
    cache_key = ext if domain is None else (ext, domain)
    if cache is not None and cache.get(cache_key) is not None:
        return cache[cache_key]

    catalog = _get_shader_catalog(ext)
    items = catalog["items"] if domain is None else [
        (shader_id, shader_id) for shader_id, path in catalog["paths"].items()
        if path is not None and _read_source_shader_metadata(path).get("domain") == domain and shader_picker_name(path) is not None
    ]

    if cache is not None:
        cache[cache_key] = items
    return items
