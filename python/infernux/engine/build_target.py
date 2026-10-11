"""Runtime-neutral identity contract for Player build targets."""

from __future__ import annotations

import re


_TARGET_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class BuildTargetId(str):
    """Stable, portable identifier for one Player build target."""

    def __new__(cls, value: str) -> "BuildTargetId":
        normalized = str(value or "").strip()
        if not _TARGET_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Build target IDs must be lowercase dash-separated tokens: "
                f"{value!r}"
            )
        return str.__new__(cls, normalized)


__all__ = ["BuildTargetId"]
