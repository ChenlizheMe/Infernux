from __future__ import annotations

import json
import sys
from pathlib import Path


PACKAGING_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PACKAGING_DIR / "model"
for directory in (PACKAGING_DIR, MODEL_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from project_model import _create_default_project_content


def test_default_render_assets_seed_identity_without_derived_fingerprints(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    (staging / "ProjectSettings").mkdir(parents=True)

    _create_default_project_content(str(staging), "Project")

    rendering = staging / "Assets" / "Rendering"
    for asset in (
        rendering / "Bloom.effect",
        rendering / "ACES Tone Mapping.effect",
        rendering / "Default Post Processing.effectgroup",
    ):
        metadata = json.loads(asset.with_name(asset.name + ".meta").read_text(encoding="utf-8"))
        assert "content_hash" not in metadata["metadata"]
        assert len(metadata["metadata"]["guid"]["value"]) == 32
        assert metadata["metadata"]["resource_type"]["value"] == "RenderEffect"
