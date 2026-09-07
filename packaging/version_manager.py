"""Infernux version manager — discovers wheels on PyPI and GitHub Releases.

Layout on disk::

    <Infernux data root>/
        Engines/
            0.2.9/
                infernux-0.4.0-cp313-cp313-win_amd64.whl
            0.3.0/
                infernux-0.4.1-cp313-cp313-win_amd64.whl
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional
import logging

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from python_runtime_catalog import DEFAULT_PYTHON_RUNTIME, PythonRuntimeId
from hub_utils import get_hub_shared_data_dir


class DownloadCancelled(Exception):
    """Raised when a version download is cancelled by the user."""


# ── Configuration ────────────────────────────────────────────────────

GITHUB_OWNER = "ChenlizheMe"
GITHUB_REPO = "Infernux"
_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
_PYPI_API = "https://pypi.org/pypi/Infernux/json"
_VERSIONS_DIR = Path(get_hub_shared_data_dir()) / "Engines"
_CACHE_TTL = 300  # seconds before re-fetching release list


@dataclass
class EngineWheel:
    filename: str
    url: str
    size: int
    python_version: str
    source: str = "github"


@dataclass
class EngineVersion:
    """Represents a single Infernux release."""

    tag: str  # e.g. "v0.3.0"
    version: str  # e.g. "0.3.0"
    wheel_url: str = ""
    wheel_size: int = 0
    published_at: str = ""
    prerelease: bool = False
    installed: bool = False
    python_version: str = ""
    wheel_options: tuple[EngineWheel, ...] = ()
    sources: tuple[str, ...] = ()
    compatibility_error: str = ""

    @property
    def display_name(self) -> str:
        suffix = " (pre-release)" if self.prerelease else ""
        return f"{self.version}{suffix}"

class VersionManager:
    """Discovers, downloads, and manages Infernux engine versions."""

    def __init__(self, runtime_manager=None) -> None:
        _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_file = _VERSIONS_DIR / "_releases_cache.json"
        self._cached_releases: list[dict] | None = None
        self._cached_at: float = 0.0
        self._runtime_manager = runtime_manager

    # ── Public API ───────────────────────────────────────────────────

    def list_versions(self, *, include_prerelease: bool = False) -> List[EngineVersion]:
        """Return available versions (remote + local), newest first."""
        remote = self._fetch_releases()
        versions: dict[str, EngineVersion] = {}

        for rel in remote:
            tag = rel.get("tag_name", "")
            ver = _tag_to_version(tag)
            if not ver:
                continue
            pre = rel.get("prerelease", False)
            if pre and not include_prerelease:
                continue

            wheel_options = _find_wheel_assets(rel)
            python_versions = {
                wheel.python_version for wheel in wheel_options
            }
            compatibility_error = ""
            if len(python_versions) > 1:
                compatibility_error = (
                    f"Infernux {ver} publishes conflicting Python ABIs: "
                    f"{', '.join(sorted(python_versions))}. Each Infernux version "
                    "must target exactly one Python minor version."
                )
            wheel = (
                self._preferred_wheel(wheel_options)
                if not compatibility_error
                else None
            )
            ev = EngineVersion(
                tag=tag,
                version=ver,
                wheel_url=wheel.url if wheel else "",
                wheel_size=wheel.size if wheel else 0,
                published_at=rel.get("published_at", ""),
                prerelease=pre,
                installed=self.is_installed(ver),
                python_version=wheel.python_version if wheel else "",
                wheel_options=wheel_options,
                sources=tuple(dict.fromkeys(item.source for item in wheel_options)),
                compatibility_error=compatibility_error,
            )
            versions[ver] = ev

        # Add locally-installed versions not on remote (e.g. manually copied)
        for local_ver in self._local_versions():
            if local_ver not in versions:
                versions[local_ver] = EngineVersion(
                    tag=f"v{local_ver}",
                    version=local_ver,
                    installed=True,
                )

        result = sorted(versions.values(), key=lambda v: _version_tuple(v.version), reverse=True)
        return result

    def installed_versions(self, python_version: str | None = None) -> List[str]:
        """Return list of locally-installed version strings, newest first."""
        vers = self._local_versions(python_version)
        vers.sort(key=_version_tuple, reverse=True)
        return vers

    def is_installed(self, version: str, python_version: str | None = None) -> bool:
        return bool(self.get_wheel_path(version, python_version))

    @staticmethod
    def _is_valid_wheel(path: str) -> bool:
        """A wheel is a zip — reject truncated/corrupted files outright."""
        try:
            return os.path.getsize(path) > 0 and zipfile.is_zipfile(path)
        except OSError:
            return False

    def get_wheel_path(
        self, version: str, python_version: str | None = None
    ) -> Optional[str]:
        """Return path to a VALID cached wheel for *version*, or None.

        Corrupted wheels (e.g. left over from an interrupted install before
        atomic installs existed) are deleted on sight so they neither appear
        in the version list nor block a clean re-download (issue #43).
        """
        ver_dir = _VERSIONS_DIR / version
        if not ver_dir.is_dir():
            return None
        target_python = (
            PythonRuntimeId.parse(python_version).series if python_version else ""
        )
        valid_wheels: list[str] = []
        for wheel in glob.glob(str(ver_dir / "infernux-*.whl")):
            if self._is_valid_wheel(wheel):
                if not wheel_platform_compatible(wheel):
                    continue
                if target_python and wheel_python_version(wheel) != target_python:
                    continue
                valid_wheels.append(wheel)
                continue
            try:
                os.remove(wheel)
                logging.getLogger(__name__).warning(
                    "Removed corrupted cached wheel: %s", wheel
                )
            except OSError:
                pass
        if not valid_wheels:
            return None
        preferred = self._preferred_local_wheel(valid_wheels)
        return preferred or sorted(valid_wheels)[0]

    def installed_python_versions(self, version: str) -> list[str]:
        ver_dir = _VERSIONS_DIR / version
        if not ver_dir.is_dir():
            return []
        versions = {
            wheel_python_version(path)
            for path in glob.glob(str(ver_dir / "infernux-*.whl"))
            if (
                self._is_valid_wheel(path)
                and wheel_platform_compatible(path)
                and wheel_python_version(path)
            )
        }
        return sorted(
            versions,
            key=lambda item: PythonRuntimeId.parse(item),
            reverse=True,
        )

    def python_version_for_engine(self, version: str) -> str:
        versions = self.installed_python_versions(version)
        if not versions:
            return ""
        if len(versions) != 1:
            raise ValueError(
                f"Infernux {version} has wheels for conflicting Python ABIs: "
                f"{', '.join(versions)}. Remove the conflicting engine install."
            )
        return versions[0]

    def is_python_runtime_installed(self, python_version: str) -> bool:
        return bool(
            self._runtime_manager is not None
            and self._runtime_manager.has_runtime(python_version)
        )

    def installation_block_reason(self, engine: EngineVersion) -> str:
        """Explain why a visible online engine release cannot be installed."""
        if engine.compatibility_error:
            return engine.compatibility_error
        if not engine.wheel_url or not engine.python_version:
            return f"Infernux {engine.version} has no compatible wheel for this platform."
        if not self.is_python_runtime_installed(engine.python_version):
            return (
                f"Infernux {engine.version} requires Python {engine.python_version}. "
                f"Please install Python {engine.python_version} first."
            )
        return ""

    def download_version(
        self,
        version: str,
        *,
        on_progress: Optional[Callable] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Download a specific version's wheel.  Returns the local wheel path.

        Cancellation-safe and concurrency-safe (issue #43):
        - data streams into a unique ``*.tmp-<uuid>`` file, so a second
          download attempt can never interleave writes with an abandoned one;
        - the final ``os.replace`` is atomic — the destination either does
          not exist or is a complete wheel;
        - when *should_cancel* returns True the partial file is deleted and
          ``DownloadCancelled`` is raised.
        """
        versions = self.list_versions(include_prerelease=True)
        ev = next((v for v in versions if v.version == version), None)
        if ev is None:
            raise ValueError(f"Version {version} not found in releases")
        if ev.compatibility_error:
            raise ValueError(ev.compatibility_error)
        wheels = self._preferred_wheels(ev.wheel_options)
        if not wheels:
            raise ValueError(
                f"No wheel asset found for Infernux {version} on this platform"
            )
        wheel = wheels[0]
        self._require_installed_python(wheel.python_version, engine_version=version)

        ver_dir = _VERSIONS_DIR / version
        ver_dir.mkdir(parents=True, exist_ok=True)

        filename = wheel.filename or wheel.url.rsplit("/", 1)[-1]
        dest = ver_dir / filename

        if dest.exists():
            if self._is_valid_wheel(str(dest)):
                return str(dest)
            dest.unlink(missing_ok=True)  # heal corrupted leftovers

        # PyPI wheels sort first. A transport failure gets one deterministic
        # chance to use the matching GitHub Release asset; invalid content is
        # never treated as a reason to change sources.
        transport_error: BaseException | None = None
        for index, candidate in enumerate(wheels):
            req = urllib.request.Request(candidate.url)
            req.add_header("Accept", "application/octet-stream")
            req.add_header("User-Agent", "Infernux-Hub/1.0")
            tmp_path = f"{dest}.tmp-{uuid.uuid4().hex[:8]}"
            try:
                try:
                    response = urllib.request.urlopen(req)
                    with response as resp, open(tmp_path, "wb") as stream:
                        total = int(resp.headers.get("Content-Length", 0)) or candidate.size
                        downloaded = 0
                        while True:
                            if should_cancel is not None and should_cancel():
                                raise DownloadCancelled(version)
                            chunk = resp.read(64 * 1024)
                            if not chunk:
                                break
                            stream.write(chunk)
                            downloaded += len(chunk)
                            if on_progress and total:
                                on_progress(downloaded, total)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                    transport_error = exc
                    if index + 1 < len(wheels):
                        continue
                    raise

                if not zipfile.is_zipfile(tmp_path):
                    raise ValueError(
                        f"Downloaded file for {version} is not a valid wheel "
                        "(truncated or corrupted transfer)."
                    )
                os.replace(tmp_path, str(dest))
                transport_error = None
                break
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        if transport_error is not None:
            raise transport_error

        # If the cancelled/failed download left an empty version dir, drop it
        # so it does not show up as an installed version.
        if not dest.exists():
            try:
                if not any(ver_dir.iterdir()):
                    ver_dir.rmdir()
            except OSError:
                pass

        return str(dest)

    def remove_version(self, version: str) -> bool:
        """Delete a cached version.  Returns True if it existed."""
        import shutil

        ver_dir = _VERSIONS_DIR / version
        if ver_dir.is_dir():
            shutil.rmtree(ver_dir, ignore_errors=True)
            return True
        return False

    def install_local_wheel(self, wheel_path: str) -> str:
        """Copy a local .whl into the versions cache.

        Returns the version string extracted from the filename.
        Raises ValueError if the filename doesn't match the expected pattern.
        """
        import shutil

        filename = os.path.basename(wheel_path)
        if not wheel_platform_compatible(filename):
            raise ValueError(
                f"The selected wheel is not compatible with this platform: {filename}"
            )
        match = re.match(r"infernux-([^-]+)-", filename, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Cannot determine version from wheel filename: {filename}\n"
                "Expected a file like infernux-0.4.0-cp313-cp313-win_amd64.whl"
            )
        version = match.group(1)
        python_version = wheel_python_version(filename)
        if not python_version:
            raise ValueError(
                f"Cannot determine the target Python ABI from wheel filename: {filename}"
            )
        self._require_installed_python(python_version, engine_version=version)

        existing_versions = self.installed_python_versions(version)
        if existing_versions and python_version not in existing_versions:
            raise ValueError(
                f"Infernux {version} is already bound to Python "
                f"{existing_versions[0]}; the same engine version cannot also "
                f"target Python {python_version}."
            )

        ver_dir = _VERSIONS_DIR / version
        ver_dir.mkdir(parents=True, exist_ok=True)
        dest = ver_dir / filename
        shutil.copy2(wheel_path, str(dest))
        return version

    # ── Project version binding ──────────────────────────────────────

    @staticmethod
    def read_project_version(project_dir: str) -> Optional[str]:
        """Read the engine version pinned in a project.

        Ignores comment lines (starting with ``#``) so the file can carry
        human-readable annotations without breaking version parsing.
        """
        vf = os.path.join(project_dir, ".infernux-version")
        if os.path.isfile(vf):
            for line in open(vf, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        return None

    @staticmethod
    def write_project_version(project_dir: str, version: str) -> None:
        """Pin an engine version for a project."""
        vf = os.path.join(project_dir, ".infernux-version")
        with open(vf, "w", encoding="utf-8") as f:
            f.write("# Infernux project version pin — do not edit manually.\n")
            f.write("# Format: <major>.<minor>.<patch>\n")
            f.write(version + "\n")

    # ── Internal ─────────────────────────────────────────────────────

    def _fetch_releases(self) -> list[dict]:
        """Fetch and merge the PyPI and GitHub catalogs with local caching."""
        now = time.time()

        # Try memory cache
        if self._cached_releases is not None and (now - self._cached_at) < _CACHE_TTL:
            return self._cached_releases

        # Try disk cache
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                cached_at = data.get("_ts", 0.0)
                if (now - cached_at) < _CACHE_TTL:
                    self._cached_releases = data.get("releases", [])
                    self._cached_at = cached_at
                    return self._cached_releases
            except (json.JSONDecodeError, KeyError) as _exc:
                logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                pass

        github: list[dict] = []
        pypi: dict = {}
        reached = False
        for source, url, accept in (
            ("pypi", _PYPI_API, "application/json"),
            ("github", f"{_API_BASE}/releases?per_page=50", "application/vnd.github+json"),
        ):
            try:
                req = urllib.request.Request(url)
                req.add_header("Accept", accept)
                req.add_header("User-Agent", "Infernux-Hub/1.0")
                if source == "github":
                    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
                    if token:
                        req.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req, timeout=15) as resp:
                    document = json.loads(resp.read().decode("utf-8"))
                if source == "pypi" and isinstance(document, dict):
                    pypi = document
                    reached = True
                elif source == "github" and isinstance(document, list):
                    github = document
                    reached = True
            except (urllib.error.URLError, OSError, ValueError):
                continue

        if not reached:
            # Offline — fall back to disk cache regardless of age
            if self._cache_file.exists():
                try:
                    data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                    self._cached_releases = data.get("releases", [])
                    self._cached_at = now
                    return self._cached_releases
                except (json.JSONDecodeError, KeyError) as _exc:
                    logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                    pass
            return []

        releases = _merge_release_catalogs(github, pypi)

        # Save to disk cache
        cache_data = {"_ts": now, "releases": releases}
        self._cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")

        self._cached_releases = releases
        self._cached_at = now
        return releases

    def _local_versions(self, python_version: str | None = None) -> List[str]:
        """List versions with a VALID wheel downloaded locally.

        Uses get_wheel_path() so corrupted leftovers from interrupted
        installs are healed and never listed (issue #43).
        """
        result = []
        if not _VERSIONS_DIR.is_dir():
            return result
        for entry in _VERSIONS_DIR.iterdir():
            if entry.is_dir() and not entry.name.startswith("_"):
                if self.get_wheel_path(entry.name, python_version):
                    result.append(entry.name)
        return result

    def _require_installed_python(
        self,
        python_version: str,
        *,
        engine_version: str,
    ) -> None:
        if self._runtime_manager is None:
            raise RuntimeError(
                "Infernux Hub cannot verify installed Python runtimes. "
                "Open the Installs page and try again."
            )
        if self._runtime_manager.has_runtime(python_version):
            return
        raise ValueError(
            f"Infernux {engine_version} requires Python {python_version}. "
            f"Please install Python {python_version} in Infernux Hub first."
        )

    def _preferred_wheel(
        self, wheels: tuple[EngineWheel, ...]
    ) -> EngineWheel | None:
        preferred = self._preferred_wheels(wheels)
        return preferred[0] if preferred else None

    def _preferred_wheels(
        self, wheels: tuple[EngineWheel, ...]
    ) -> tuple[EngineWheel, ...]:
        if not wheels:
            return ()
        preferred_versions: list[str] = []
        if self._runtime_manager is not None:
            preferred_versions.extend(self._runtime_manager.installed_versions())
        preferred_versions.append(DEFAULT_PYTHON_RUNTIME.series)
        for python_version in preferred_versions:
            matches = tuple(
                wheel for wheel in wheels if wheel.python_version == python_version
            )
            if matches:
                return matches
        return (wheels[0],)

    def _preferred_local_wheel(self, wheels: list[str]) -> str:
        wheel_by_python = {
            wheel_python_version(wheel): wheel
            for wheel in wheels
            if wheel_python_version(wheel)
        }
        preferred_versions: list[str] = []
        if self._runtime_manager is not None:
            preferred_versions.extend(self._runtime_manager.installed_versions())
        preferred_versions.append(DEFAULT_PYTHON_RUNTIME.series)
        for version in preferred_versions:
            if version in wheel_by_python:
                return wheel_by_python[version]
        return ""


# ── Helpers ──────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"^v?(\d+\.\d+\.\d+.*)$")


def _tag_to_version(tag: str) -> str:
    """Convert 'v0.3.0' → '0.3.0', return '' on failure."""
    m = _TAG_RE.match(tag)
    return m.group(1) if m else ""


def _version_tuple(version: str):
    """Parse '0.3.0' → (0, 3, 0) for sorting."""
    parts = []
    for p in version.split(".")[:3]:
        digits = re.match(r"\d+", p)
        parts.append(int(digits.group()) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


_CPYTHON_WHEEL_TAG = re.compile(r"(?:^|-)(cp(\d)(\d{1,2}))(?:-|_)", re.IGNORECASE)


def wheel_python_version(path_or_name: str) -> str:
    """Return the Python major/minor ABI encoded in an Infernux wheel name."""
    name = os.path.basename(path_or_name)
    match = _CPYTHON_WHEEL_TAG.search(name)
    if match is None:
        return ""
    return f"{int(match.group(2))}.{int(match.group(3))}"


@lru_cache(maxsize=1)
def supported_wheel_platforms() -> frozenset[str]:
    """Return host platform tags without tying them to Hub's Python ABI."""

    return frozenset(tag.platform for tag in sys_tags())


def wheel_platform_compatible(path_or_name: str) -> bool:
    """Whether a wheel targets this OS/architecture, independent of CPython."""

    name = os.path.basename(path_or_name)
    try:
        _distribution, _version, _build, tags = parse_wheel_filename(name)
    except InvalidWheelFilename:
        return False
    platforms = supported_wheel_platforms()
    return any(tag.platform in platforms for tag in tags)


def _find_wheel_assets(release: dict) -> tuple[EngineWheel, ...]:
    """Find host-compatible CPython wheels in a merged remote release."""
    result: list[EngineWheel] = []
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if (
            name.endswith(".whl")
            and "infernux" in name.lower()
            and wheel_platform_compatible(name)
        ):
            python_version = wheel_python_version(name)
            if not python_version:
                continue
            result.append(
                EngineWheel(
                    filename=name,
                    url=asset.get("browser_download_url", ""),
                    size=asset.get("size", 0),
                    python_version=python_version,
                    source=str(asset.get("source", "github")),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda wheel: (
                PythonRuntimeId.parse(wheel.python_version),
                1 if wheel.source == "pypi" else 0,
            ),
            reverse=True,
        )
    )


def _merge_release_catalogs(github: list[dict], pypi: dict) -> list[dict]:
    """Normalize both public catalogs; PyPI assets intentionally sort first."""

    merged: dict[str, dict] = {}
    for release in github:
        if not isinstance(release, dict):
            continue
        version = _tag_to_version(str(release.get("tag_name", "")))
        if not version:
            continue
        normalized = dict(release)
        normalized["assets"] = [
            {**asset, "source": "github"}
            for asset in release.get("assets", [])
            if isinstance(asset, dict)
        ]
        merged[version] = normalized

    releases = pypi.get("releases", {}) if isinstance(pypi, dict) else {}
    if isinstance(releases, dict):
        for version, files in releases.items():
            if not _tag_to_version(str(version)) or not isinstance(files, list):
                continue
            wheels = [
                {
                    "name": str(item.get("filename", "")),
                    "browser_download_url": str(item.get("url", "")),
                    "size": int(item.get("size", 0) or 0),
                    "source": "pypi",
                }
                for item in files
                if isinstance(item, dict)
                and item.get("packagetype") == "bdist_wheel"
                and not item.get("yanked", False)
                and str(item.get("filename", "")).casefold().endswith(".whl")
                and str(item.get("url", "")).startswith("https://")
            ]
            if not wheels:
                continue
            current = merged.setdefault(
                str(version),
                {
                    "tag_name": f"v{version}",
                    "prerelease": bool(re.search(r"[A-Za-z]", str(version))),
                    "published_at": max(
                        (str(item.get("upload_time_iso_8601", "")) for item in files if isinstance(item, dict)),
                        default="",
                    ),
                    "assets": [],
                },
            )
            current["assets"] = wheels + list(current.get("assets", []))

    return sorted(
        merged.values(),
        key=lambda release: _version_tuple(_tag_to_version(str(release.get("tag_name", "")))),
        reverse=True,
    )
