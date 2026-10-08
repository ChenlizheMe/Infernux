"""Real native archives preserve authored identity across sidecar migrations."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from infernux.engine.player_package_native import read_entry, write_pack
from infernux.plugins.cache import SharedPackageCache
from infernux.plugins.package import InxPackage, PACKAGE_MANIFEST


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


def test_official_mcp_exports_same_identity_from_windows_and_linux_checkouts(tmp_path):
    git = shutil.which('git')
    assert git, 'Git is required to verify cross-device plugin identity'
    source = tmp_path / 'source'
    source.mkdir()
    attributes = Path(__file__).resolve().parents[2] / 'external/plugins/infernux_mcp/.gitattributes'
    shutil.copyfile(attributes, source / '.gitattributes')
    package = source / 'package'
    package.mkdir()
    (package / PACKAGE_MANIFEST).write_text(json.dumps(dict(
        reference='vendor/line-endings', name='Line endings', version='1.0.0', engine='')), encoding='utf-8')
    authored = {'entry.py': b'VALUE = "stable"\n', 'page.md': b'# Usage\nSame package.\n',
                'requirements.txt': b'fastmcp\n', 'binary.png': b'\x89PNG\r\n\x1a\n\x00\r\n'}
    for name, payload in authored.items():
        (package / name).write_bytes(payload)
    def run(*args, cwd=source):
        subprocess.run([git, *args], cwd=cwd, check=True, capture_output=True)
    run('init', '-q')
    run('-c', 'core.autocrlf=false', 'add', '.')
    run('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Fixture')
    cache = SharedPackageCache(tmp_path / 'cache')
    baseline = None
    for conversion in ('true', 'false'):
        checkout = tmp_path / ('checkout-' + conversion)
        run('clone', '-q', '--no-checkout', str(source), str(checkout))
        run('-c', 'core.autocrlf=' + conversion, 'checkout', '-q', 'HEAD', cwd=checkout)
        for name, payload in authored.items():
            assert (checkout / 'package' / name).read_bytes() == payload
        archive = tmp_path / (conversion + '.inxpkg')
        InxPackage.export_source(str(checkout / 'package'), str(archive))
        stored = cache.store(str(archive), reference='vendor/line-endings', version='1.0.0')
        if baseline is None:
            baseline = stored
        else:
            assert stored == baseline
