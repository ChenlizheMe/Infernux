"""Git checkout newlines do not become plugin edits or binary equivalence."""
import json
from pathlib import Path

import pytest

from infernux.plugins import InxPackage, PackageUpdateConflict
from test_plugin_updates import installed


@pytest.fixture(autouse=True)
def isolated_package_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "hub-cache"))


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("remove", [False, True])
def test_git_checkout_line_endings_allow_unmodified_plugin_update(installed, ending, remove):
    manager, source, root = installed
    target = root / "runtime/retired.py"
    target.write_bytes(b"value = 1" + ending)
    control = root / "inx_package.json"
    control.write_bytes(control.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending))
    metadata = json.loads((source / "inx_package.json").read_text(encoding="utf-8"))
    metadata["version"] = "2.0.0"
    (source / "inx_package.json").write_text(json.dumps(metadata), encoding="utf-8")
    if remove:
        (source / "runtime/retired.py").unlink()
    else:
        (source / "runtime/retired.py").write_bytes(b"value = 2\n")
    archive = source.parent / "next.inxpkg"
    InxPackage.export_source(str(source), str(archive))
    state = manager.install_package(str(archive), update=True, install_dependencies=False)
    assert state.loaded
    if remove:
        assert not target.exists()
    else:
        assert target.read_bytes() == b"value = 2\n"


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_local_script_edit_remains_a_conflict_after_git_checkout(installed, ending):
    manager, source, root = installed
    target = root / "runtime/retired.py"
    target.write_bytes(b"value = 99" + ending)
    before = Path(manager.registry.path).read_bytes()
    from test_plugin_updates import next_package
    with pytest.raises(PackageUpdateConflict):
        manager.install_package(next_package(source), update=True, install_dependencies=False)
    assert target.read_bytes() == b"value = 99" + ending
    assert Path(manager.registry.path).read_bytes() == before


@pytest.mark.parametrize("path,left,right,equal", [
    ("runtime/value.py", b"a = 1\r\n", b"a = 1\n", True),
    ("runtime/data.bin", b"a = 1\r\n", b"a = 1\n", False),
    ("runtime/corrupt.txt", b"\xff\r\n", b"\xff\n", False),
    ("runtime/utf16.txt", b"a\0\r\n", b"a\0\n", False),
    ("runtime/value.py", b"a = 1\r\n", b"a = 2\n", False),
    ("runtime/value.py", b"a = 1\r\n", b"a = 1", False),
    ("runtime/value.py", b"", None, False),
])
def test_package_source_comparison_preserves_binary_and_authored_changes(path, left, right, equal):
    from infernux.plugins.source_content import same_source_content
    assert same_source_content(path, left, right) is equal


def test_shared_archive_identity_survives_text_checkout_endings(installed, tmp_path):
    manager, source, _root = installed
    record = manager.registry.installed_record("vendor/plugin")
    baseline = Path(manager._installed_archive_path(record))
    original = baseline.read_bytes()
    script = source / "runtime/retired.py"
    script.write_bytes(b"value = 1\n" if b"\r\n" in script.read_bytes() else b"value = 1\r\n")
    archive = tmp_path / "other-checkout.inxpkg"
    InxPackage.export_source(str(source), str(archive))
    cache = manager._package_cache()
    assert cache.store(str(archive), reference="vendor/plugin", version=record["version"]) == str(baseline)
    assert baseline.read_bytes() == original
