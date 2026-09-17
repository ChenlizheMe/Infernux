"""Project-owned compiler artifact paths and write-boundary capacity limits."""

from pathlib import Path


def compiler_cache_root() -> Path:
    from Infernux.application import Application

    if Application.is_player():
        return Path(Application.persistent_data_path()) / "Cache" / "Compute"
    root = Application.data_path()
    if not root:
        raise RuntimeError("Compiler caching requires an active project")
    return Path(root) / "Library" / "Artifacts" / "Compute"


def prune_cache_files(root: Path, pattern: str, *, file_limit: int, byte_limit: int) -> None:
    """Evict oldest writes, only within the caller's reserved flat namespace.

    No traversal, symlink following, whole-directory deletion or per-frame scan.
    Concurrent cache eviction is normal; permission and other I/O failures are
    not hidden. Missing cache entries are recompiled by their existing loader.
    """
    entries = []
    for path in root.glob(pattern):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        entries.append((stat.st_mtime_ns, path.name, stat.st_size, path))
    entries.sort(reverse=True)
    count = size = 0
    for _, _, entry_size, path in entries:
        if count < file_limit and size + entry_size <= byte_limit:
            count += 1
            size += entry_size
        else:
            path.unlink(missing_ok=True)
