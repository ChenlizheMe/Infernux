"""Filesystem publication at build/Player IO boundaries, never in the frame loop."""
import os
import sys
import time


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
