"""Publish one Hub release directory to the Infernux object channel."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path


UPLOAD_ENDPOINT = "https://upload.infernux-engine.com"
PUBLIC_ENDPOINT = "https://downloads.infernux-engine.com"
PART_SIZE = 64 * 1024 * 1024
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def release_assets(version: str) -> tuple[str, ...]:
    if not _VERSION.fullmatch(version):
        raise ValueError(f"Invalid release version: {version!r}")
    return (
        f"InfernuxHubInstaller-{version}-windows-x64.exe",
        f"InfernuxHub-{version}-windows-x64-full.zip",
        "InfernuxHub-windows-x64-manifest.json",
        f"InfernuxHubInstaller-{version}-linux-x64",
        f"InfernuxHub-{version}-linux-x64-full.zip",
        "InfernuxHub-linux-x64-manifest.json",
    )


class Publisher:
    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("R2_UPLOAD_TOKEN is required")
        self._token = token

    def _request(self, method: str, path: str, body: dict | bytes) -> dict:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{UPLOAD_ENDPOINT}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": (
                    "application/octet-stream"
                    if isinstance(body, bytes)
                    else "application/json"
                ),
                "User-Agent": "Infernux-Release-Publisher",
            },
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("The distribution upload endpoint returned invalid JSON")
        return result

    def upload(self, source: Path, key: str) -> None:
        is_json = source.suffix == ".json"
        started = self._request(
            "POST",
            "/start",
            {
                "key": key,
                "contentType": "application/json" if is_json else "application/octet-stream",
                "cacheControl": (
                    "public, max-age=300"
                    if is_json
                    else "public, max-age=31536000, immutable"
                ),
                "contentDisposition": None if is_json else f'attachment; filename="{source.name}"',
            },
        )
        upload_id = str(started["uploadId"])
        parts: list[dict] = []
        try:
            with source.open("rb") as stream:
                part_number = 1
                while chunk := stream.read(PART_SIZE):
                    query = urllib.parse.urlencode(
                        {"key": key, "uploadId": upload_id, "partNumber": part_number}
                    )
                    parts.append(self._request("PUT", f"/part?{query}", chunk))
                    part_number += 1
            completed = self._request(
                "POST",
                "/complete",
                {"key": key, "uploadId": upload_id, "parts": parts},
            )
        except BaseException:
            self._request("POST", "/abort", {"key": key, "uploadId": upload_id})
            raise
        actual = int(completed["size"])
        expected = source.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Published object size mismatch for {key}: {actual} != {expected}")
        print(f"Published {key} ({actual:,} bytes)")


def publish(release_dir: Path, version: str, build_number: int, token: str) -> None:
    if build_number < 1:
        raise ValueError("Wheel build number must be positive")
    publisher = Publisher(token)
    for name in release_assets(version):
        source = release_dir / name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Hub release asset is missing: {source}")
        key = f"hub/{version}/build-{build_number}/{name}"
        publisher.upload(source, key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-number", type=int, required=True)
    arguments = parser.parse_args()
    publish(
        arguments.release_dir.resolve(),
        arguments.version,
        arguments.build_number,
        os.environ.get("R2_UPLOAD_TOKEN", ""),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
