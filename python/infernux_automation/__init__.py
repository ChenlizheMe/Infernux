"""Lightweight automation client; importing this package never imports the engine."""

from .client import AutomationClient, AutomationError

__all__ = ["AutomationClient", "AutomationError"]
