"""Reject projects authored for a different engine before touching their files."""
from pathlib import Path

from infernux.version import ENGINE_RELEASE


def validate_project_engine_version(project_path: str) -> str:
    path = Path(project_path) / ".infernux-version"
    try:
        versions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    except OSError as exc:
        raise RuntimeError(f"Project must declare its exact engine version in {path}") from exc
    if len(versions) != 1:
        raise RuntimeError(f"Project must declare exactly one engine version in {path}")
    if versions[0] != ENGINE_RELEASE:
        raise RuntimeError(
            f"Project requires Infernux {versions[0]}; the running engine is {ENGINE_RELEASE}. "
            "Install and launch the exact required version. Engine upgrades are not supported."
        )
    return versions[0]
