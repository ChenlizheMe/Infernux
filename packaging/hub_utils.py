"""Utility helpers shared across the Hub codebase."""

import os
import sys


def is_frozen() -> bool:
    """Return *True* when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> str:
    """Return the directory containing bundled data files."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


def get_app_dir() -> str:
    """Return the executable directory for the running Hub."""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def get_inner_dir() -> str:
    """Return the Hub private data directory.

    In packaged builds this resolves next to the Hub executable. In source mode
    it resolves under packaging/_inner so dev runs behave the same way.
    """
    return os.path.join(get_app_dir(), "_inner")
