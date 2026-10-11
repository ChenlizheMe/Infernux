"""Authored package comparisons across Git text checkouts."""
from pathlib import PurePosixPath


_TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".txt", ".md", ".markdown", ".html", ".css", ".js", ".ts", ".svg",
    ".xml", ".csv", ".sh", ".bat", ".ps1", ".c", ".cpp", ".h", ".hpp",
    ".glsl", ".vert", ".frag", ".comp", ".shadingmodel", ".scene", ".prefab",
    ".mat", ".effect", ".effectgroup", ".particlegraph", ".physicmaterial",
    ".rendertexture", ".inxdata", ".animclip2d", ".animclip3d", ".animfsm",
    ".timelinefsm", ".animtimeline", ".meta",
})


def same_source_content(path: str, left: bytes | None, right: bytes | None) -> bool:
    """Ignore CRLF/LF only for known UTF-8 text; preserve every other byte."""
    if left == right:
        return True
    if left is None or right is None:
        return False
    if PurePosixPath(path.replace("\\", "/")).suffix.casefold() not in _TEXT_SUFFIXES:
        return False
    if b"\0" in left or b"\0" in right:
        return False
    try:
        left.decode("utf-8")
        right.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return left.replace(b"\r\n", b"\n") == right.replace(b"\r\n", b"\n")
