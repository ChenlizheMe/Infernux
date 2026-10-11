"""A package rename must not rewrite published documentation snapshots."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("changed_signature", [False, True])
def test_published_api_snapshot_accepts_only_namespace_spelling_migration(tmp_path, changed_signature):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the documentation generator")
    root = Path(__file__).parents[2]
    original = json.loads((root / "docs/api-snapshots/0.4.1.json").read_text(encoding="utf-8"))
    symbol = next(s for s in original["symbols"] if s["module"].startswith("Infernux."))
    snapshot = {"schema_version": 1, "release": "0.4.1", "symbol_count": 1, "fingerprint_sha256": "published", "symbols": [symbol]}
    docs = tmp_path / "docs"
    (docs / "api-snapshots").mkdir(parents=True)
    frozen = json.dumps(snapshot, indent=2) + "\n"
    snapshot_path = docs / "api-snapshots/0.4.1.json"
    snapshot_path.write_text(frozen, encoding="utf-8")
    current = dict(symbol, language="en", symbol_key=symbol["symbol_key"].replace("Infernux.", "infernux."), module=symbol["module"].replace("Infernux.", "infernux."), signatures=[s.replace("Infernux.", "infernux.") for s in symbol["signatures"]])
    if changed_signature:
        current["signatures"] = ["changed(required_argument)"]
    (docs / "api-index.json").write_text(json.dumps({"symbols": [current]}), encoding="utf-8")
    (docs / "docs-manifest.json").write_text(json.dumps({"documented_release": "0.4.1"}), encoding="utf-8")
    result = subprocess.run([node, str(root / "docs/tools/build-api-diff.mjs")], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert snapshot_path.read_text(encoding="utf-8") == frozen
    if changed_signature:
        assert result.returncode != 0
        assert "immutable release" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
