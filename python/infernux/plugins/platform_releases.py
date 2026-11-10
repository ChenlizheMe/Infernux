"""Shared Player build declarations and SDK source provenance."""
from infernux.version import ENGINE_COMPATIBILITY_VERSION
import json
from pathlib import Path

_declarations = json.loads((Path(__file__).resolve().parents[1] / "resources/platform-players.json").read_text(encoding="utf-8"))
PLATFORM_PLAYERS = {item["id"]: item for item in _declarations["platforms"]}
PLATFORM_REFERENCES = frozenset(item["reference"] for item in PLATFORM_PLAYERS.values())
PLATFORM_TARGETS = frozenset(target for item in PLATFORM_PLAYERS.values() for target in item["targets"])
PLAYER_PYTHON_ABI = _declarations["python_abi"]
_provenance = Path(__file__).resolve().parents[1] / "resources/player-build-source.json"
PLAYER_SOURCE_REVISION = json.loads(_provenance.read_text(encoding="utf-8"))["revision"] if _provenance.is_file() else None
PLATFORM_PACKAGE_VERSION = ENGINE_COMPATIBILITY_VERSION
