"""Filesystem publication at build/Player IO boundaries, never in the frame loop."""
import os
import shutil
import sys
import time

from infernux.engine.path_utils import lexical_path


def remove_directory_tree(path: str | os.PathLike[str]) -> None:
    """Delete one owned tree with literal paths and observable failure.

    Do not resolve the final path through a link before checking it: a linked
    root is not the directory the caller owns. Descendant links are handled
    without traversal by shutil.rmtree, including Windows junctions.
    """
    if not path:
        raise ValueError("Directory cleanup requires an explicit path")
    target = lexical_path(path)
    if target == os.path.dirname(target):
        raise ValueError("Directory cleanup cannot remove a filesystem root")
    try:
        os.lstat(target)
    except FileNotFoundError:
        return
    if os.path.islink(target) or os.path.isjunction(target):
        raise ValueError(f"Directory cleanup cannot remove a linked root: {target}")
    shutil.rmtree(target)


def replace_path(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Publish a prepared file/directory; fail without copy/merge fallbacks.

    Windows indexers/scanners can briefly hold a child without FILE_SHARE_DELETE.
    Match AtomicFile's bounded compatibility policy: at most 8 attempts / 254 ms
    of waiting, only for Windows access/sharing/lock errors. Normal success and
    other platforms perform exactly one rename. Caller owns target selection.
    """
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (sys.platform != "win32" or getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == 7):
                raise
            time.sleep(0.002 * (2 ** attempt))
