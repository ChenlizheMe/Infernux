"""Numba disk-cache adapter for engine-owned publication identities.

Loaded only when CPU JIT disk caching is requested. Storage, target matching
and atomic writes remain Numba-owned; authored dependency identity is ours.
"""

from numba.core.caching import FunctionCache


class PublicationCache(FunctionCache):
    def __init__(self, function, identity: str):
        self._publication_identity = identity
        super().__init__(function)

    def _index_key(self, signature, codegen):
        return super()._index_key(signature, codegen), self._publication_identity
