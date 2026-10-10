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
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import unquote, urlsplit
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional
import logging

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from python_runtime_catalog import DEFAULT_PYTHON_RUNTIME, PythonRuntimeId
from hub_utils import get_hub_shared_data_dir, replace_path
from wheel_identity import has_matching_wheel_identity, validate_wheel_identity


class DownloadCancelled(Exception):
    """Raised when a version download is cancelled by the user."""


# ── Configuration ────────────────────────────────────────────────────

GITHUB_OWNER = "ChenlizheMe"
GITHUB_REPO = "Infernux"
_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
_PYPI_API = "https://pypi.org/pypi/infernux/json"
_VERSIONS_DIR = Path(get_hub_shared_data_dir()) / "Engines"
_CACHE_TTL = 300  # seconds before re-fetching release list


@dataclass
class EngineWheel:
    filename: str
    url: str
    size: int
    python_version: str
    source: str = "github"
    sha256: str = ""


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
    update_available: bool = False
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

    def catalog_timestamp(self) -> float | None:
        """When the newest known release catalog was fetched, or None."""
        if self._cached_releases is not None:
            return self._cached_at
        cached = self._read_catalog_cache()
        return cached[0] if cached is not None else None

    def cached_versions(self, *, include_prerelease: bool = False) -> List[EngineVersion]:
        """Same as list_versions(), but from the last catalog without any network.

        The Installs UI shows this immediately and refreshes in the background.
        """
        return self.list_versions(include_prerelease=include_prerelease, offline=True)

    def list_versions(self, *, include_prerelease: bool = False, offline: bool = False) -> List[EngineVersion]:
        """Return available versions (remote + local), newest first."""
        remote = self._offline_releases() if offline else self._fetch_releases()
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
            local = self.get_wheel_path(ver, ev.python_version or None)
            ev.update_available = bool(
                local and wheel and wheel_build(wheel.filename) > wheel_build(local)
            )

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
        if _tag_to_version(version) != version:
            return None
        ver_dir = _VERSIONS_DIR / _base_version(version)
        if not ver_dir.is_dir():
            return None
        target_python = (
            PythonRuntimeId.parse(python_version).series if python_version else ""
        )
        valid_wheels: list[str] = []
        catalog_wheels = self._cached_wheel_assets(version)
        for wheel in glob.glob(str(ver_dir / "infernux-*.whl")):
            expected = tuple(
                item for item in catalog_wheels if item.filename == os.path.basename(wheel)
            )
            verdict = _cached_wheel_verdict(wheel, expected)
            if verdict == "corrupt":
                try:
                    os.remove(wheel)
                    logging.getLogger(__name__).warning(
                        "Removed corrupted cached wheel: %s", wheel
                    )
                except OSError:
                    pass
                continue
            if wheel_release(wheel) != version or not wheel_platform_compatible(wheel):
                continue
            if not wheel_python_version(wheel):
                continue
            if target_python and wheel_python_version(wheel) != target_python:
                continue
            if verdict == "valid":
                valid_wheels.append(wheel)
        if not valid_wheels:
            return None
        preferred = self._preferred_local_wheel(valid_wheels)
        return preferred or max(valid_wheels, key=wheel_build)

    def installed_python_versions(self, version: str) -> list[str]:
        if _tag_to_version(version) != version:
            return []
        ver_dir = _VERSIONS_DIR / _base_version(version)
        if not ver_dir.is_dir():
            return []
        versions = {
            wheel_python_version(path)
            for path in glob.glob(str(ver_dir / "infernux-*.whl"))
            if (
                self._is_valid_wheel(path)
                and wheel_release(path) == version
                and wheel_platform_compatible(path)
                and wheel_python_version(path)
                and has_matching_wheel_identity(path)
                and self.get_wheel_path(version, wheel_python_version(path)) is not None
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
        hashes = {candidate.sha256 for candidate in wheels if candidate.sha256}
        if len(hashes) > 1:
            raise ValueError(f"Release sources disagree on wheel SHA-256: {wheel.filename}")
        if wheel_release(wheel.filename) != version or not wheel_platform_compatible(wheel.filename):
            raise ValueError(f"Incompatible engine download for {version}: {wheel.filename}")
        self._require_installed_python(wheel.python_version, engine_version=version)

        ver_dir = _VERSIONS_DIR / _base_version(version)
        ver_dir.mkdir(parents=True, exist_ok=True)

        filename = wheel.filename or wheel.url.rsplit("/", 1)[-1]
        dest = ver_dir / filename

        if dest.exists():
            if self._is_valid_wheel(str(dest)) and has_matching_wheel_identity(str(dest)):
                try:
                    for expected in wheels:
                        _verify_download(dest, expected)
                except ValueError:
                    logging.getLogger(__name__).warning("Cached wheel differs from release: %s", dest)
                else:
                    logging.getLogger(__name__).info("Engine cache hit: release=%s wheel=%s", version, dest)
                    return str(dest)

        # PyPI wheels sort first. A transport failure gets one deterministic
        # chance to use the matching GitHub Release asset; invalid content is
        # never treated as a reason to change sources.
        transport_error: BaseException | None = None
        for index, candidate in enumerate(wheels):
            logging.getLogger(__name__).info(
                "Engine download: release=%s platform=%s Python=%s source=%s wheel=%s destination=%s",
                version, sys.platform, candidate.python_version, candidate.source, filename, dest,
            )
            req = urllib.request.Request(candidate.url)
            req.add_header("Accept", "application/octet-stream")
            req.add_header("User-Agent", "Infernux-Hub/1.0")
            tmp_path = f"{dest}.tmp-{uuid.uuid4().hex[:8]}"
            try:
                try:
                    response = urllib.request.urlopen(req, timeout=120)
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
                validate_wheel_identity(tmp_path, filename=filename)
                # A mirror without a digest must not bypass the hash published
                # by another source for this exact same wheel filename.
                for expected in wheels:
                    _verify_download(Path(tmp_path), expected)
                replace_path(tmp_path, str(dest))
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
        ver_dir = _VERSIONS_DIR / _base_version(version)
        if not ver_dir.is_dir():
            return False
        removed = False
        for path in ver_dir.glob("infernux-*.whl"):
            if wheel_release(str(path)) == version:
                path.unlink()
                removed = True
        if not any(ver_dir.iterdir()):
            ver_dir.rmdir()
        return removed

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
        version = wheel_release(filename)
        if not version:
            raise ValueError(
                f"Cannot determine version from wheel filename: {filename}\n"
                "Expected a file like infernux-0.4.0-cp313-cp313-win_amd64.whl"
            )
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

        ver_dir = _VERSIONS_DIR / _base_version(version)
        ver_dir.mkdir(parents=True, exist_ok=True)
        dest = ver_dir / filename
        temporary = ver_dir / f"{filename}.tmp-{uuid.uuid4().hex}"
        try:
            shutil.copyfile(wheel_path, temporary)
            validate_wheel_identity(str(temporary), filename=filename)
            for expected in self._cached_wheel_assets(version):
                if expected.filename == filename:
                    _verify_download(temporary, expected)
            replace_path(temporary, dest)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove temporary wheel: %s", temporary)
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
            with open(vf, encoding="utf-8") as stream:
                versions = [line.strip() for line in stream
                            if line.strip() and not line.lstrip().startswith("#")]
            return versions[0] if len(versions) == 1 else None
        return None

    @staticmethod
    def write_project_version(project_dir: str, version: str) -> None:
        """Pin an engine version for a project."""
        vf = os.path.join(project_dir, ".infernux-version")
        with open(vf, "w", encoding="utf-8") as f:
            f.write("# Infernux project version pin — do not edit manually.\n")
            f.write("# Exact release: <package-version>[-v<revision>]; no automatic upgrades.\n")
            f.write(version + "\n")

    # ── Internal ─────────────────────────────────────────────────────

    def _fetch_releases(self) -> list[dict]:
        """Fetch and merge the PyPI and GitHub catalogs with local caching."""
        now = time.time()

        # Try memory cache
        if self._cached_releases is not None and (now - self._cached_at) < _CACHE_TTL:
            return self._cached_releases

        # Cache the raw catalog, never a host's selected download URL.
        cached = self._read_catalog_cache()
        if cached is not None and 0 <= now - cached[0] < _CACHE_TTL:
            self._cached_at, self._cached_releases = cached
            return self._cached_releases

        github: list[dict] = []
        pypi: dict = {}
        reached = False
        failures = []

        def request(source: str, url: str, accept: str):
            req = urllib.request.Request(url)
            req.add_header("Accept", accept)
            req.add_header("User-Agent", "Infernux-Hub/1.0")
            if source == "github":
                token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
                if token:
                    req.add_header("Authorization", "Bearer " + token)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))

        sources = (
            ("pypi", _PYPI_API, "application/json"),
            ("github", f"{_API_BASE}/releases?per_page=50", "application/vnd.github+json"),
        )
        # The catalogs are independent: fetching them together halves the
        # wait, and one slow source no longer delays the other.
        with ThreadPoolExecutor(max_workers=len(sources), thread_name_prefix="engine-catalog") as pool:
            futures = [(source, pool.submit(request, source, url, accept)) for source, url, accept in sources]
            for source, future in futures:
                try:
                    document = future.result()
                    if source == "pypi" and isinstance(document, dict):
                        pypi = document
                        reached = True
                    elif source == "github" and isinstance(document, list):
                        github = document
                        reached = True
                    else:
                        raise ValueError(f"Unexpected {source} catalog response")
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    failures.append(f"{source}: {type(exc).__name__}: {exc}")
                    logging.getLogger(__name__).warning("Engine catalog request failed: %s", source, exc_info=True)

        if not reached:
            # Offline — fall back to disk cache regardless of age
            if cached is not None:
                self._cached_releases = cached[1]
                self._cached_at = now
                logging.getLogger(__name__).warning("Using offline engine catalog: %s", self._cache_file)
                return self._cached_releases
            raise RuntimeError("Unable to fetch engine versions.\n" + "\n".join(failures))

        releases = _merge_release_catalogs(github, pypi)

        # Save to disk cache
        cache_data = {"_ts": now, "releases": releases}
        temporary = self._cache_file.with_name(f"{self._cache_file.name}.tmp-{uuid.uuid4().hex}")
        try:
            temporary.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
            replace_path(temporary, self._cache_file)
        except OSError:
            logging.getLogger(__name__).warning("Could not save engine catalog cache: %s", self._cache_file, exc_info=True)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove temporary catalog: %s", temporary)

        self._cached_releases = releases
        self._cached_at = now
        return releases

    def _offline_releases(self) -> list[dict]:
        if self._cached_releases is not None:
            return self._cached_releases
        cached = self._read_catalog_cache()
        if cached is None:
            return []
        self._cached_at, self._cached_releases = cached
        return self._cached_releases

    def _read_catalog_cache(self) -> tuple[float, list[dict]] | None:
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("_ts"), (int, float)):
                return None
            releases = data.get("releases")
            if not isinstance(releases, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("tag_name"), str)
                or not isinstance(item.get("assets", []), list) for item in releases
            ):
                return None
            return data["_ts"], releases
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Ignoring unreadable engine catalog: %s", self._cache_file)
            return None

    def _cached_wheel_assets(self, version: str) -> tuple[EngineWheel, ...]:
        """Use known release digests for offline cache reads, without networking."""
        releases = self._cached_releases
        if releases is None:
            cached = self._read_catalog_cache()
            releases = cached[1] if cached is not None else []
        return tuple(
            wheel for release in releases
            if _tag_to_version(release.get("tag_name", "")) == version
            for wheel in _find_wheel_assets(release)
        )
    def _local_versions(self, python_version: str | None = None) -> List[str]:
        """List versions with a VALID wheel downloaded locally.

        Uses get_wheel_path() so corrupted leftovers from interrupted
        installs are healed and never listed (issue #43).
        """
        result = set()
        if not _VERSIONS_DIR.is_dir():
            return []
        for entry in _VERSIONS_DIR.iterdir():
            if entry.is_dir() and not entry.name.startswith("_"):
                for path in entry.glob("infernux-*.whl"):
                    release = wheel_release(str(path))
                    if release and _base_version(release) == entry.name and self.get_wheel_path(release, python_version):
                        result.add(release)
        return list(result)

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
                newest = max(matches, key=lambda item: wheel_build(item.filename))
                return tuple(item for item in matches if item.filename == newest.filename)
        newest = max(wheels, key=lambda item: wheel_build(item.filename))
        return tuple(item for item in wheels if item.filename == newest.filename)

    def _preferred_local_wheel(self, wheels: list[str]) -> str:
        wheel_by_python = {
            wheel_python_version(wheel): wheel
            for wheel in sorted(wheels, key=wheel_build)
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

_RELEASE_RE = re.compile(r"([0-9][A-Za-z0-9.!+]*)(?:-v([1-9]\d*))?")


def _base_version(release: str) -> str:
    match = _RELEASE_RE.fullmatch(release)
    if match is None:
        raise ValueError(f"Invalid Infernux release identity: {release!r}")
    return str(Version(match.group(1)))


def _release_for(version: str, build: int) -> str:
    return version if build == 1 else f"{version}-v{build}"


def _tag_to_version(tag: str) -> str:
    """Convert 'v0.3.0' → '0.3.0', return '' on failure."""
    value = tag.removeprefix("v")
    match = _RELEASE_RE.fullmatch(value)
    if match is None:
        return ""
    try:
        return _release_for(_base_version(value), int(match.group(2) or 1))
    except InvalidVersion:
        return ""


def _version_tuple(version: str):
    """Sort package versions and their independent wheel revisions numerically."""
    match = _RELEASE_RE.fullmatch(version)
    return (Version(_base_version(version)), int(match.group(2) or 1))


# ── Hotfix presentation ──────────────────────────────────────────────
# A release identity is "<version>[-v<n>]"; n > 1 is a hotfix of <version>.
# Projects still pin the exact identity, but lists show one entry per
# version: its newest hotfix.

def hotfix_number(release: str) -> int:
    match = _RELEASE_RE.fullmatch(release or "")
    return int(match.group(2) or 1) if match else 1


def display_release(release: str) -> str:
    """The version a person reads: hotfix revisions fold into their version."""
    try:
        return _base_version(release)
    except (ValueError, InvalidVersion):
        return release


def hotfix_label(release: str) -> str:
    number = hotfix_number(release)
    return f"HOTFIX {number}" if number > 1 else ""


def latest_releases(releases) -> list[str]:
    """Keep the newest hotfix of each version, newest version first."""
    newest: dict[str, str] = {}
    for release in releases:
        try:
            base = _base_version(release)
        except (ValueError, InvalidVersion):
            continue
        current = newest.get(base)
        if current is None or _version_tuple(release) > _version_tuple(current):
            newest[base] = release
    return sorted(newest.values(), key=_version_tuple, reverse=True)


def latest_engine_versions(versions, *, keep: str = "") -> list["EngineVersion"]:
    """Collapse a catalog to one row per version: its newest usable hotfix.

    A hotfix without a wheel for this platform does not hide an older hotfix
    that has one. If an older hotfix is installed, the visible row offers the
    newer one as an update. ``keep`` (an exact identity a project pins) stays
    listed even when a newer hotfix exists.
    """
    groups: dict[str, list[EngineVersion]] = {}
    for item in versions:
        groups.setdefault(display_release(item.version), []).append(item)
    result = []
    for members in groups.values():
        members.sort(key=lambda item: _version_tuple(item.version), reverse=True)
        usable = [item for item in members if item.wheel_url and not item.compatibility_error]
        chosen = (usable or members)[0]
        if any(item.installed for item in members if item is not chosen) and not chosen.installed:
            chosen.installed = True
            chosen.update_available = True
        result.append(chosen)
        result.extend(item for item in members if item.version == keep and item is not chosen)
    return sorted(result, key=lambda item: _version_tuple(item.version), reverse=True)


def wheel_release(path_or_name: str) -> str:
    """Read the exact release identity, never infer it from a cache directory."""
    try:
        distribution, version, build, _tags = parse_wheel_filename(os.path.basename(path_or_name))
    except InvalidWheelFilename:
        return ""
    if distribution != "infernux" or (build and (build[0] < 1 or build[1])):
        return ""
    return _release_for(str(version), build[0] if build else 1)


_CPYTHON_WHEEL_TAG = re.compile(r"cp(\d)(\d{1,2})")


def wheel_python_version(path_or_name: str) -> str:
    """Return the Python major/minor ABI encoded in an Infernux wheel name."""
    try:
        _distribution, _version, _build, tags = parse_wheel_filename(os.path.basename(path_or_name))
    except InvalidWheelFilename:
        return ""
    interpreters = {tag.interpreter for tag in tags}
    if len(interpreters) != 1 or any(tag.abi != tag.interpreter for tag in tags):
        return ""
    match = _CPYTHON_WHEEL_TAG.fullmatch(next(iter(interpreters)))
    if match is None:
        return ""
    return f"{int(match.group(1))}.{int(match.group(2))}"


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
    # Do not let cross-build sysconfig overrides make a Windows Hub accept
    # Linux artifacts. ABI selection is separate, but OS identity is not.
    def same_os(platform: str) -> bool:
        if sys.platform == "win32":
            return platform.startswith("win")
        if sys.platform.startswith("linux"):
            return platform.startswith(("linux_", "manylinux", "musllinux"))
        if sys.platform == "darwin":
            return platform.startswith("macosx_")
        return False
    return any(tag.platform in platforms and same_os(tag.platform) for tag in tags)


def _verify_download(path: Path, wheel: EngineWheel) -> None:
    if wheel.size and path.stat().st_size != wheel.size:
        raise ValueError(f"Wheel size mismatch for {wheel.filename}: expected {wheel.size}, got {path.stat().st_size}")
    if wheel.sha256:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != wheel.sha256:
            raise ValueError(f"Wheel SHA-256 mismatch for {wheel.filename}: expected {wheel.sha256}, got {actual}")


# Hashing a 24-90 MB wheel took most of every Installs/Projects refresh. A
# verdict stays valid while the file's identity (size, mtime) and the published
# expectations are unchanged; any rewrite of the file invalidates it.
_WHEEL_VERDICTS: dict[tuple, str] = {}


def _cached_wheel_verdict(path: str, expected: tuple[EngineWheel, ...]) -> str:
    """Return "valid", "invalid" (keep, but do not use) or "corrupt" (delete)."""
    try:
        stat = os.stat(path)
    except OSError:
        return "invalid"
    key = (
        os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns,
        tuple((item.size, item.sha256) for item in expected),
    )
    verdict = _WHEEL_VERDICTS.get(key)
    if verdict is not None:
        return verdict
    if not VersionManager._is_valid_wheel(path):
        verdict = "corrupt"
    elif not has_matching_wheel_identity(path):
        verdict = "invalid"
    else:
        verdict = "valid"
        try:
            for item in expected:
                _verify_download(Path(path), item)
        except (OSError, ValueError) as exc:
            logging.getLogger(__name__).warning("Ignoring invalid cached wheel: %s", exc)
            verdict = "invalid"
    if len(_WHEEL_VERDICTS) > 256:
        _WHEEL_VERDICTS.clear()
    _WHEEL_VERDICTS[key] = verdict
    return verdict


def _asset_sha256(asset: dict) -> str:
    digest = asset.get("digest") or ""
    if not digest or digest == "sha256:":
        return ""
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        raise ValueError(f"Invalid wheel SHA-256 in release catalog: {asset.get('name')}")
    return digest.split(":", 1)[1].lower()


def wheel_build(path_or_name: str) -> tuple[int, str]:
    """Return the PEP 427 build tag used to order same-version wheels."""

    try:
        _distribution, _version, build, _tags = parse_wheel_filename(
            os.path.basename(path_or_name)
        )
    except InvalidWheelFilename:
        return ()
    return build


def _find_wheel_assets(release: dict) -> tuple[EngineWheel, ...]:
    """Find host-compatible CPython wheels in a merged remote release."""
    result: list[EngineWheel] = []
    for asset in release.get("assets", []):
        if not isinstance(asset, dict):
            continue
        name = asset.get("name", "")
        url = asset.get("browser_download_url", "")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        if any(character in name for character in ("/", "\\", ":")):
            continue
        try:
            target = urlsplit(url)
        except ValueError:
            continue
        url_name = unquote(target.path.rsplit("/", 1)[-1])
        if target.scheme not in {"https", "http"} or not target.netloc:
            continue
        if url_name.endswith(".whl") and url_name != name:
            logging.getLogger(__name__).warning("Ignoring mismatched wheel URL: filename=%s URL filename=%s", name, url_name)
            continue
        if (
            name.endswith(".whl")
            and wheel_release(name) == _tag_to_version(release.get("tag_name", ""))
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
                    sha256=_asset_sha256(asset),
                )
            )
    return tuple(
        sorted(
            result,
            key=lambda wheel: (
                PythonRuntimeId.parse(wheel.python_version),
                wheel_build(wheel.filename),
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
        if version in merged:
            merged[version]["assets"].extend(normalized["assets"])
        else:
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
                    "digest": "sha256:" + str(item.get("digests", {}).get("sha256", "")),
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
            for wheel in wheels:
                release = wheel_release(wheel["name"])
                if not release or _base_version(release) != str(version):
                    continue
                current = merged.setdefault(
                    release,
                    {
                        "tag_name": f"v{release}",
                        "prerelease": Version(str(version)).is_prerelease,
                        "published_at": max(
                            (str(item.get("upload_time_iso_8601", "")) for item in files if isinstance(item, dict)),
                            default="",
                        ),
                        "assets": [],
                    },
                )
                current["assets"].insert(0, wheel)

    return sorted(
        merged.values(),
        key=lambda release: _version_tuple(_tag_to_version(str(release.get("tag_name", "")))),
        reverse=True,
    )
