from __future__ import annotations
from typing import Mapping
from Infernux.rendergraph.graph import BufferHandle as GraphBufferHandle, TextureHandle

GraphResourceHandle = TextureHandle | GraphBufferHandle

class BufferHandle:
    name: str
    def __init__(self, name: str) -> None: ...

class PassResult:
    source: str
    revision: int
    buffers: Mapping[str, GraphResourceHandle]
    def has(self, name: str | BufferHandle) -> bool: ...
    def sample(self, name: str | BufferHandle) -> GraphResourceHandle: ...
    @property
    def snapshot(self) -> Mapping[str, GraphResourceHandle]: ...

def normalize_buffer_name(value: str) -> str: ...
