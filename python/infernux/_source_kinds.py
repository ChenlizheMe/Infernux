"""Native author-source classification shared by import, script discovery and cook."""

from os import fspath
from os.path import splitext

from infernux.lib import _Infernux


SHADER_SOURCE_KINDS = {row[0]: row[1:] for row in _Infernux._shader_source_kinds}
GPU_SOURCE_SUFFIXES = tuple(SHADER_SOURCE_KINDS)
_STATIC_AUTHOR_SUFFIXES = (*GPU_SOURCE_SUFFIXES, ".particle.py")


def source_extension(path) -> str:
    name = fspath(path).casefold()
    for suffix in GPU_SOURCE_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return splitext(name)[1]


def is_gpu_source(path) -> bool:
    return fspath(path).casefold().endswith(GPU_SOURCE_SUFFIXES)


def is_host_python_source(path) -> bool:
    name = fspath(path).casefold()
    return name.endswith(".py") and not name.endswith(_STATIC_AUTHOR_SUFFIXES)
