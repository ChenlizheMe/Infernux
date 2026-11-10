"""Deploy the release catalog through the repository's GitHub Pages branch source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request


MAX_DEPLOYMENT_ATTEMPTS = 90  # Branch build plus the site's ten-minute CDN cache.


def _pages_request(repository: str, path: str, *, method: str = "GET") -> dict | list:
    return json.loads(subprocess.check_output(
        ["gh", "api", "--method", method, f"repos/{repository}/pages/{path}"],
        text=True, encoding="utf-8", timeout=60,
    ))


def deploy_catalog(repository: str, commit: str, catalog: Path) -> None:
    expected = json.loads(catalog.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        "https://infernux-engine.com/hub-catalog.json",
        headers={
            "Accept": "application/json",
            "User-Agent": "InfernuxHub-Updater",
            "Cache-Control": "no-cache",
        },
    )
    builds = _pages_request(repository, "builds?per_page=20")
    existing = [build for build in builds if build["commit"] == commit]
    ignored = set()
    if not any(build["status"] in ("queued", "building", "built") for build in existing):
        ignored = {build["url"] for build in existing}
        _pages_request(repository, "builds", method="POST")
    for attempt in range(MAX_DEPLOYMENT_ATTEMPTS):
        builds = [build for build in _pages_request(repository, "builds?per_page=20")
                  if build["commit"] == commit and build["url"] not in ignored]
        if any(build["status"] == "built" for build in builds):
            # Use Hub's exact URL; another cache key could conceal stale data.
            with urllib.request.urlopen(request, timeout=15) as response:
                actual = json.load(response)
            if actual == expected:
                print(f"Published Hub catalog {expected['stable']} at {commit}")
                return
        # Cancelled duplicate builds can appear before the replacement build is
        # visible. Wait for the matching deployment within the existing deadline.
        if attempt < MAX_DEPLOYMENT_ATTEMPTS - 1:
            time.sleep(10)
    raise TimeoutError(f"The public Hub catalog did not reach release commit {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    arguments = parser.parse_args()
    deploy_catalog(arguments.repository, arguments.commit, arguments.catalog)
