"""Versioned Python runtime identities and download catalog for Infernux Hub."""

from __future__ import annotations

import re
from dataclasses import dataclass


_VERSION_PATTERN = re.compile(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\s*$")


@dataclass(frozen=True, order=True)
class PythonRuntimeId:
    """A Python ABI identity at the major/minor boundary."""

    major: int
    minor: int

    @classmethod
    def parse(cls, value: str | "PythonRuntimeId") -> "PythonRuntimeId":
        if isinstance(value, cls):
            return value
        match = _VERSION_PATTERN.fullmatch(str(value or ""))
        if match is None:
            raise ValueError(
                f"Invalid Python runtime version {value!r}; expected major.minor."
            )
        return cls(int(match.group(1)), int(match.group(2)))

    @property
    def series(self) -> str:
        return f"{self.major}.{self.minor}"

    @property
    def directory_name(self) -> str:
        return f"python{self.major}{self.minor}"

    @property
    def cp_tag(self) -> str:
        return f"cp{self.major}{self.minor}"

    @property
    def unix_library_stem(self) -> str:
        return f"python{self.series}"

    @property
    def windows_library_stem(self) -> str:
        return f"python{self.major}{self.minor}"


@dataclass(frozen=True)
class PythonRuntimeRelease:
    runtime_id: PythonRuntimeId
    patch_version: str
    build_release: str

    def __post_init__(self) -> None:
        match = _VERSION_PATTERN.fullmatch(self.patch_version)
        if match is None or match.group(3) is None:
            raise ValueError(
                f"Runtime patch version must be major.minor.patch: {self.patch_version!r}"
            )
        if PythonRuntimeId(int(match.group(1)), int(match.group(2))) != self.runtime_id:
            raise ValueError(
                f"Runtime {self.runtime_id.series} cannot use patch {self.patch_version}."
            )


_PYTHON_312 = PythonRuntimeRelease(
    runtime_id=PythonRuntimeId(3, 12),
    patch_version="3.12.13",
    build_release="20260805",
)

_PYTHON_313 = PythonRuntimeRelease(
    runtime_id=PythonRuntimeId(3, 13),
    patch_version="3.13.15",
    build_release="20260825",
)


DEFAULT_PYTHON_RUNTIME = PythonRuntimeId(3, 13)
SUPPORTED_PYTHON_RUNTIMES: tuple[PythonRuntimeId, ...] = (
    DEFAULT_PYTHON_RUNTIME,
    PythonRuntimeId(3, 12),
)
_RELEASES = {
    release.runtime_id: release for release in (_PYTHON_313, _PYTHON_312)
}


def runtime_release(
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> PythonRuntimeRelease:
    runtime_id = PythonRuntimeId.parse(runtime)
    try:
        return _RELEASES[runtime_id]
    except KeyError as exc:
        supported = ", ".join(item.series for item in SUPPORTED_PYTHON_RUNTIMES)
        raise ValueError(
            f"Python {runtime_id.series} is not in the Hub runtime catalog. "
            f"Supported versions: {supported}."
        ) from exc


def runtime_directory_name(
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> str:
    return PythonRuntimeId.parse(runtime).directory_name


__all__ = [
    "DEFAULT_PYTHON_RUNTIME",
    "PythonRuntimeId",
    "PythonRuntimeRelease",
    "SUPPORTED_PYTHON_RUNTIMES",
    "runtime_directory_name",
    "runtime_release",
]
