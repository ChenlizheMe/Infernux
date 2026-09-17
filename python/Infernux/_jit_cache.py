"""Numba disk-cache adapter for engine-owned publication identities.

Loaded only when CPU JIT disk caching is requested. Serialization, target
matching and atomic writes remain Numba-owned; identity and location are ours.
"""

from functools import partial
import os
from pathlib import Path

from numba.core.caching import CompileResultCacheImpl, FunctionCache, _CacheLocator

from Infernux._compiler.cache import compiler_cache_root, prune_cache_files


_CACHE_FILE_LIMIT = 512
_CACHE_BYTE_LIMIT = 256 * 1024 * 1024


def cpu_cache_root() -> Path:
    from Infernux.application import Application

    if Application.is_player() or Application.data_path():
        return compiler_cache_root() / "CPU"
    # Standalone compiler tooling may explicitly choose its storage. An Editor
    # or Player always uses its owned root, never a process-wide override.
    configured = os.environ.get("NUMBA_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    raise RuntimeError("CPU disk caching requires an active project or explicit NUMBA_CACHE_DIR")


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
