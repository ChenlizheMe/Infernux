from __future__ import annotations

import pytest

from Infernux.engine.runtime_artifact_catalog import (
    RuntimeArtifactError,
    artifact_source_hash,
)


@pytest.mark.parametrize(
    ("suffix", "magic", "schema"),
    (
        (".inxmesh", b"INXMESHART", b"MSH1"),
        (".inxskin", b"INXSKINAR", b"SKN1"),
    ),
)
def test_current_mesh_artifact_source_hash_follows_schema(
    tmp_path, suffix, magic, schema
):
    source_hash = b"0123456789abcdef"
    artifact = tmp_path / f"fixture{suffix}"
    artifact.write_bytes(
        magic
        + b"\x04\x03\x02\x01"
        + schema
        + len(source_hash).to_bytes(4, "little")
        + source_hash
    )

    assert artifact_source_hash(artifact) == source_hash.decode("ascii")


@pytest.mark.parametrize(
    ("suffix", "magic"),
    (
        (".inxmesh", b"INXMESHART"),
        (".inxskin", b"INXSKINAR"),
    ),
)
def test_mesh_artifact_without_current_schema_is_rejected(tmp_path, suffix, magic):
    artifact = tmp_path / f"fixture{suffix}"
    artifact.write_bytes(
        magic
        + b"\x04\x03\x02\x01"
        + (16).to_bytes(4, "little")
        + b"0123456789abcdef"
    )

    with pytest.raises(RuntimeArtifactError, match="current format marker"):
        artifact_source_hash(artifact)
