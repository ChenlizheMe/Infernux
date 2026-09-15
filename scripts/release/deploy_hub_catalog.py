"""Deploy the release catalog through the repository's GitHub Pages branch source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request


MAX_DEPLOYMENT_ATTEMPTS = 90  # Branch build plus the site's ten-minute CDN cache.


def _pages_request(repository: str, path: str, *, method: str = "GET") -> dict:
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
    # GITHUB_TOKEN pushes do not trigger a build of a branch-published Pages
    # site. Request it explicitly and do not report success while still queued.
    _pages_request(repository, "builds", method="POST")
    for attempt in range(MAX_DEPLOYMENT_ATTEMPTS):
        build = _pages_request(repository, "builds/latest")
        if build["commit"] == commit:
            if build["status"] == "built":
                # Verify the URL and headers used by Hub itself. A different
                # cache key could conceal a stale catalog at the actual endpoint.
                with urllib.request.urlopen(request, timeout=15) as response:
                    actual = json.load(response)
                if actual == expected:
                    print(f"Published Hub catalog {expected['stable']} at {commit}")
                    return
            if build["status"] == "errored":
                raise RuntimeError(f"Hub catalog Pages build failed: {build['error']}")
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
