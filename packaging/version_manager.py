"""Infernux version manager — downloads & caches engine wheels from GitHub Releases.

Layout on disk::

    ~/.infernux/
        versions/
            0.2.0/
                infernux-0.2.0-cp312-cp312-win_amd64.whl
            0.3.0/
                infernux-0.3.0-cp312-cp312-win_amd64.whl
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging


# ── Configuration ────────────────────────────────────────────────────

GITHUB_OWNER = "ChenlizheMe"
GITHUB_REPO = "Infernux"
_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
_VERSIONS_DIR = Path.home() / ".infernux" / "versions"
_CACHE_TTL = 300  # seconds before re-fetching release list


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

    @property
    def display_name(self) -> str:
        suffix = " (pre-release)" if self.prerelease else ""
        return f"{self.version}{suffix}"


class VersionManager:
    """Discovers, downloads, and manages Infernux engine versions."""

    def __init__(self) -> None:
        _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_file = _VERSIONS_DIR / "_releases_cache.json"
        self._cached_releases: list[dict] | None = None
        self._cached_at: float = 0.0

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

            wheel_url, wheel_size = _find_wheel_asset(rel)
            ev = EngineVersion(
                tag=tag,
                version=ver,
                wheel_url=wheel_url,
                wheel_size=wheel_size,
                published_at=rel.get("published_at", ""),
                prerelease=pre,
                installed=self.is_installed(ver),
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

    def installed_versions(self) -> List[str]:
        """Return list of locally-installed version strings, newest first."""
        vers = self._local_versions()
        vers.sort(key=_version_tuple, reverse=True)
        return vers

    def is_installed(self, version: str) -> bool:
        return bool(self.get_wheel_path(version))

    def get_wheel_path(self, version: str) -> Optional[str]:
        """Return path to the cached wheel for *version*, or None."""
        ver_dir = _VERSIONS_DIR / version
        if not ver_dir.is_dir():
            return None
        wheels = glob.glob(str(ver_dir / "infernux-*.whl"))
        return wheels[0] if wheels else None

    def download_version(
        self,
        version: str,
        *,
        on_progress: Optional[callable] = None,
    ) -> str:
        """Download a specific version's wheel.  Returns the local wheel path."""
        versions = self.list_versions(include_prerelease=True)
        ev = next((v for v in versions if v.version == version), None)
        if ev is None:
            raise ValueError(f"Version {version} not found in releases")
        if not ev.wheel_url:
            raise ValueError(f"No wheel asset found for version {version}")

        ver_dir = _VERSIONS_DIR / version
        ver_dir.mkdir(parents=True, exist_ok=True)

        filename = ev.wheel_url.rsplit("/", 1)[-1]
        dest = ver_dir / filename

        if dest.exists():
            return str(dest)

        # Stream download
        req = urllib.request.Request(ev.wheel_url)
        req.add_header("Accept", "application/octet-stream")
        req.add_header("User-Agent", "Infernux-Hub/1.0")

        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or ev.wheel_size
            downloaded = 0
            chunk_size = 64 * 1024

            tmp_path = str(dest) + ".tmp"
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress and total:
                        on_progress(downloaded, total)

        os.replace(tmp_path, str(dest))
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
        match = re.match(r"infernux-([^-]+)-", filename, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"Cannot determine version from wheel filename: {filename}\n"
                "Expected a file like infernux-0.3.0-cp312-cp312-win_amd64.whl"
            )
        version = match.group(1)

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
        """Fetch releases from GitHub API with local-file caching."""
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

        # Fetch from GitHub
        url = f"{_API_BASE}/releases?per_page=50"
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github+json")
            req.add_header("User-Agent", "Infernux-Hub/1.0")
            # Optional: use a token if set in env
            token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")

            with urllib.request.urlopen(req, timeout=15) as resp:
                releases = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError):
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

        # Save to disk cache
        cache_data = {"_ts": now, "releases": releases}
        self._cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")

        self._cached_releases = releases
        self._cached_at = now
        return releases

    def _local_versions(self) -> List[str]:
        """List versions that are already downloaded locally."""
        result = []
        if not _VERSIONS_DIR.is_dir():
            return result
        for entry in _VERSIONS_DIR.iterdir():
            if entry.is_dir() and not entry.name.startswith("_"):
                wheels = glob.glob(str(entry / "infernux-*.whl"))
                if wheels:
                    result.append(entry.name)
        return result


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


def _find_wheel_asset(release: dict) -> tuple[str, int]:
    """Find the .whl asset URL and size from a GitHub release."""
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.endswith(".whl") and "infernux" in name.lower():
            return asset.get("browser_download_url", ""), asset.get("size", 0)
    return "", 0
