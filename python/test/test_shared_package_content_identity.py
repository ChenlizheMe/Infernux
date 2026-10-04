"""Real native archives preserve authored identity across sidecar migrations."""
import json
from pathlib import Path

import pytest

from Infernux.engine.player_package_native import read_entry, write_pack
from Infernux.plugins.cache import SharedPackageCache
from Infernux.plugins.package import InxPackage, PACKAGE_MANIFEST


@pytest.mark.parametrize("change", ["derived", "source", "guid", "mesh_readability"])
def test_shared_version_identity_uses_source_guids_and_authored_settings(tmp_path, change):
    source = tmp_path / "source"
    source.mkdir()
    (source / PACKAGE_MANIFEST).write_text(json.dumps({
        "reference": "vendor/identity", "name": "Identity", "version": "1.0.0",
        "engine": "",
    }), encoding="utf-8")
    (source / "shape.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")
    initial = tmp_path / "initial.inxpkg"
    preview = InxPackage.export_source(str(source), str(initial))
    record = next(item for item in preview.file_records if item["logical_path"] == "shape.obj")
    entries = {entry["path"]: read_entry(initial, entry["path"]) for entry in preview.entries}
    sidecar = json.loads(entries[record["meta_archive_path"]])
    sidecar["metadata"]["resource_type"] = {"type": "string", "value": "Mesh"}
    sidecar["metadata"]["is_readable"] = {"type": "bool", "value": False}

    def pack(name, payloads):
        files = tmp_path / (name + "-files")
        files.mkdir()
        pairs = []
        for index, (logical, payload) in enumerate(payloads.items()):
            path = files / str(index)
            path.write_bytes(payload)
            pairs.append((logical, path))
        destination = tmp_path / (name + ".inxpkg")
        write_pack(pairs, destination)
        return destination

    legacy = dict(entries)
    legacy_sidecar = json.loads(json.dumps(sidecar))
    legacy_sidecar["metadata"]["content_hash"] = {"type": "string", "value": "legacy-observation"}
    legacy[record["meta_archive_path"]] = json.dumps(legacy_sidecar).encode("utf-8")
    old = pack("legacy", legacy)
    current = dict(entries)
    if change == "source":
        current[record["archive_path"]] += b"# authored source changed\n"
    elif change == "guid":
        guid = "1" * 32
        metadata = json.loads(current[PACKAGE_MANIFEST])
        next(item for item in metadata["files"] if item["logical_path"] == "shape.obj")["guid"] = guid
        current[PACKAGE_MANIFEST] = json.dumps(metadata).encode("utf-8")
        sidecar["metadata"]["guid"]["value"] = guid
    elif change == "mesh_readability":
        sidecar["metadata"]["is_readable"]["value"] = True
    current[record["meta_archive_path"]] = json.dumps(sidecar).encode("utf-8")
    new = pack("current", current)
    cache = SharedPackageCache(tmp_path / "cache")
    baseline = Path(cache.store(str(old), reference="vendor/identity", version="1.0.0"))
    if change == "derived":
        assert cache.store(str(new), reference="vendor/identity", version="1.0.0") == str(baseline)
    else:
        with pytest.raises(ValueError, match="Plugin version is immutable"):
            cache.store(str(new), reference="vendor/identity", version="1.0.0")
    assert baseline.read_bytes() == old.read_bytes()
    assert not list(baseline.parent.glob("*.tmp.*"))
