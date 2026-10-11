"""Shared cancellation signal for the standalone build pipeline."""


class BuildCancelled(Exception):
    """Raised when the user requests cancellation of an active build."""
