"""Numba disk-cache adapter for engine-owned publication identities.

Loaded only when CPU JIT disk caching is requested. Serialization, target
matching and atomic writes remain Numba-owned; identity and location are ours.
"""

from functools import partial
import os
from pathlib import Path
import shutil
import tempfile

from numba.core.caching import CompileResultCacheImpl, FunctionCache, _CacheLocator

from infernux._compiler.cache import compiler_cache_root, prune_cache_files
from infernux.engine.path_utils import resolved_path


_CACHE_FILE_LIMIT = 512
_CACHE_BYTE_LIMIT = 256 * 1024 * 1024


def cpu_cache_root() -> Path:
    from infernux.application import Application

    if Application.is_player():
        persistent_root = Application.persistent_data_path()
    elif os.environ.get("_INFERNUX_PLAYER_MODE", "") == "1":
        persistent_root = os.environ.get(
            "_INFERNUX_PLAYER_PERSISTENT_DATA_ROOT", ""
        ).strip()
        if not persistent_root:
            raise RuntimeError("Player host did not provide a writable persistent data root")
    else:
        persistent_root = ""
    if persistent_root:
        persistent = Path(resolved_path(persistent_root)) / "Cache" / "Compute" / "CPU"
        _materialize_packaged_cache(persistent)
        return persistent
    if Application.data_path():
        return compiler_cache_root() / "CPU"
    # Standalone compiler tooling may explicitly choose its storage. An Editor
    # or Player always uses its owned root, never a process-wide override.
    configured = os.environ.get("NUMBA_CACHE_DIR", "").strip()
    if configured:
        return Path(resolved_path(configured))
    raise RuntimeError("CPU disk caching requires an active project or explicit NUMBA_CACHE_DIR")


def _materialize_packaged_cache(destination: Path) -> None:
    """Copy same-platform CPU cache entries from the sealed Player package."""
    if destination.exists() and any(destination.glob("inx-*.nb[ci]")):
        return
    data_root_value = os.environ.get("_INFERNUX_PLAYER_DATA_ROOT", "").strip()
    if not data_root_value:
        raise RuntimeError("Player host did not provide its sealed data root")
    data_root = Path(resolved_path(data_root_value))
    archive = data_root / "Content.inxpkg"
    if not archive.is_file():
        raise RuntimeError("Player CPU JIT cache requires Content.inxpkg")
    from infernux.engine.player_package_native import read_manifest, read_entry
    manifest = read_manifest(archive)
    names = tuple(
        str(item["path"])
        for item in manifest["files"]
        if isinstance(item, dict) and "path" in item
    )
    selected = tuple(
        name for name in names
        if name.startswith("Library/Artifacts/Compute/CPU/inx-")
        and name.endswith((".nbi", ".nbc"))
    )
    if not selected:
        raise RuntimeError(
            "Player CPU JIT cache is missing; build the Player after startup warmup"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="infernux-cpu-cache-", dir=str(destination.parent)))
    try:
        for name in selected:
            target = temporary / Path(name).name
            target.write_bytes(read_entry(archive, name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


class _PublicationLocator(_CacheLocator):
    def __init__(self, function, root, identity):
        self._py_file = function.__code__.co_filename
        self._root = root
        self._identity = identity

    def ensure_cache_path(self):
        self._root.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self):
        return str(self._root)

    def get_source_stamp(self):
        return self._identity

    def get_disambiguator(self):
        return self._identity


class _PublicationCacheImpl(CompileResultCacheImpl):
    def __init__(self, function, *, root, identity):
        self._lineno = function.__code__.co_firstlineno
        self._locator = _PublicationLocator(function, root, identity)
        self._filename_base = "inx-" + identity


class PublicationCache(FunctionCache):
    def __init__(self, function, identity: str):
        self._publication_identity = identity
        self._root = cpu_cache_root()
        self._impl_class = partial(_PublicationCacheImpl, root=self._root, identity=identity)
        super().__init__(function)

    def _index_key(self, signature, codegen):
        return super()._index_key(signature, codegen), self._publication_identity

    def save_overload(self, signature, data):
        super().save_overload(signature, data)
        prune_cache_files(self._root, "inx-*.nb[ci]",
                          file_limit=_CACHE_FILE_LIMIT, byte_limit=_CACHE_BYTE_LIMIT)
